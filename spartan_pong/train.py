from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange

from spartan_pong.config import ModelConfig, TrainConfig
from spartan_pong.data import load_npz
from spartan_pong.evaluate import evaluate_arrays, representative_eval_indices
from spartan_pong.models import build_model, checkpoint_payload


def _batch(
    data: dict[str, np.ndarray], batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = np.random.randint(0, len(data["x"]), size=batch_size)
    x = torch.as_tensor(data["x"][idx], dtype=torch.float32, device=device)
    y = torch.as_tensor(data["y"][idx], dtype=torch.float32, device=device)
    env = torch.as_tensor(data["env"][idx], dtype=torch.long, device=device)
    return x, y, env


def _load_log_lambda(
    payload: dict[str, Any], train_cfg: TrainConfig, device: torch.device
) -> torch.Tensor:
    if "log_lambda" in payload:
        value = float(payload["log_lambda"])
    else:
        raw_lambda = float(payload.get("lambda", train_cfg.lambda_init))
        if raw_lambda <= 0.0:
            raw_lambda = train_cfg.lambda_init
        value = math.log(raw_lambda)
    return torch.tensor(value, dtype=torch.float32, device=device)


def _safe_lambda_value(log_lambda: torch.Tensor) -> float | None:
    value = float(log_lambda.detach().cpu())
    if not math.isfinite(value) or value > 700.0:
        return None
    lambda_value = math.exp(value)
    return lambda_value if math.isfinite(lambda_value) else None


def _lagrange_state(log_lambda: torch.Tensor, moving_gap: torch.Tensor) -> dict[str, float]:
    state = {
        "log_lambda": float(log_lambda.detach().cpu()),
        "moving_gap": float(moving_gap.detach().cpu()),
    }
    lambda_value = _safe_lambda_value(log_lambda)
    if lambda_value is not None:
        state["lambda"] = lambda_value
    return state


def _update_log_lambda(
    log_lambda: torch.Tensor,
    moving_gap: torch.Tensor,
    mse: torch.Tensor,
    target_loss: float,
    alpha: float,
) -> None:
    gap = mse.detach() - target_loss
    moving_gap.mul_(0.99).add_(0.01 * gap)
    log_lambda.add_(math.log(alpha) + moving_gap)


def train_model(
    model_type: str,
    train_path: str | Path,
    val_path: str | Path,
    out_dir: str | Path,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    resume: bool = False,
) -> dict[str, float]:
    # Set seed
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    # Create output directory
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Load data, build model and optimizer
    train = load_npz(train_path)
    val = load_npz(val_path)
    model = build_model(model_type, model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    # Initialize training state
    if train_cfg.lambda_init <= 0.0:
        raise ValueError("lambda_init must be positive")
    if train_cfg.lagrangian_alpha <= 0.0:
        raise ValueError("lagrangian_alpha must be positive")
    log_lambda = torch.tensor(math.log(train_cfg.lambda_init), dtype=torch.float32, device=device)
    moving_gap = torch.tensor(0.0, dtype=torch.float32, device=device)
    checkpoint_path = out / "checkpoint.pt"
    start_step = 0
    last_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    # Resume from checkpoint if requested and available
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        if payload.get("model_type") != model_type:
            raise ValueError(
                f"checkpoint model type {payload.get('model_type')} does not match {model_type}"
            )
        model.load_state_dict(payload["state_dict"])
        if "optimizer_state" in payload:
            opt.load_state_dict(payload["optimizer_state"])
        start_step = int(payload.get("step", -1)) + 1
        last_metrics = dict(payload.get("metrics", {}))
        history = list(payload.get("history", []))
        log_lambda = _load_log_lambda(payload, train_cfg, device)
        moving_gap = torch.tensor(
            float(payload.get("moving_gap", 0.0)), dtype=torch.float32, device=device
        )
        if start_step >= train_cfg.steps:
            return last_metrics
    config_payload = {
        "model_type": model_type,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
    }
    (out / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True))
    history_eval_indices = representative_eval_indices(
        val, max_samples=train_cfg.batch_size * 8, seed=train_cfg.seed
    )

    # Initialize best validation metric
    best_val = min(
        (float(row.get("one_step_l2", float("inf"))) for row in history), default=float("inf")
    )
    # Training loop
    pbar = trange(start_step, train_cfg.steps, desc=f"train {model_type}")
    for step in pbar:
        model.train()
        # Sample batch and compute loss
        x, y, env = _batch(train, train_cfg.batch_size, device)
        out_dict = model(x, env=env, hard=True)
        mse = F.mse_loss(out_dict["pred"], y)
        # For Spartan, add sparsity penalty weighted by the Lagrange multiplier
        if model_type == "spartan":
            loss = (mse - train_cfg.target_loss) + train_cfg.sparsity_weight * out_dict[
                "sparsity"
            ] * torch.exp(-log_lambda)
        else:
            loss = mse
        # Backprop and optimize
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Update Lagrange multiplier for Spartan to encourage meeting the target loss
        if model_type == "spartan":
            with torch.no_grad():
                _update_log_lambda(
                    log_lambda,
                    moving_gap,
                    mse,
                    train_cfg.target_loss,
                    train_cfg.lagrangian_alpha,
                )

        # Periodically evaluate on validation set and save checkpoints
        if step % max(1, train_cfg.eval_every) == 0 or step == train_cfg.steps - 1:
            last_metrics = evaluate_arrays(
                model,
                val,
                device=device,
                batch_size=train_cfg.batch_size,
                rollout=False,
                indices=history_eval_indices,
            )
            last_metrics.update(
                {
                    **_lagrange_state(log_lambda, moving_gap),
                    "eval_subset_size": float(len(history_eval_indices)),
                    "train_mse": float(mse.detach().cpu()),
                    "loss": float(loss.detach().cpu()),
                    "sparsity": float(out_dict["sparsity"].detach().cpu()),
                    "step": float(step),
                }
            )
            # Update history and save checkpoints
            history.append(last_metrics.copy())
            with (out / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            if last_metrics["one_step_l2"] < best_val:
                best_val = last_metrics["one_step_l2"]
                torch.save(
                    checkpoint_payload(
                        model,
                        model_cfg,
                        model_type,
                        {
                            "train_config": asdict(train_cfg),
                            "metrics": last_metrics,
                            "config": config_payload,
                            "optimizer_state": opt.state_dict(),
                            "step": step,
                            **_lagrange_state(log_lambda, moving_gap),
                        },
                    ),
                    out / "best_checkpoint.pt",
                )
            # Save the latest checkpoint regardless of improvement to allow resuming from the most recent state
            torch.save(
                checkpoint_payload(
                    model,
                    model_cfg,
                    model_type,
                    {
                        "train_config": asdict(train_cfg),
                        "metrics": last_metrics,
                        "history": history,
                        "config": config_payload,
                        "optimizer_state": opt.state_dict(),
                        "step": step,
                        **_lagrange_state(log_lambda, moving_gap),
                    },
                ),
                checkpoint_path,
            )
            pbar.set_postfix(
                {
                    k: f"{v:.4g}"
                    for k, v in last_metrics.items()
                    if k in {"pred_l2", "shd", "robust_pct"}
                }
            )

    # Final save after training loop completes
    torch.save(
        checkpoint_payload(
            model,
            model_cfg,
            model_type,
            {
                "train_config": asdict(train_cfg),
                "metrics": last_metrics,
                "history": history,
                "config": config_payload,
                "optimizer_state": opt.state_dict(),
                "step": train_cfg.steps - 1,
                **_lagrange_state(log_lambda, moving_gap),
            },
        ),
        checkpoint_path,
    )
    (out / "metrics.json").write_text(json.dumps(last_metrics, indent=2, sort_keys=True))
    print(f"  {model_type} done — one_step_l2={last_metrics.get('one_step_l2', 'N/A'):.4f}, shd={last_metrics.get('shd', 'N/A')}")
    return last_metrics
