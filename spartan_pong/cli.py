from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from spartan_pong.config import ModelConfig, TrainConfig
from spartan_pong.data import generate_dataset
from spartan_pong.evaluate import evaluate_checkpoint, one_step_mse_checkpoint
from spartan_pong.preflight import PreflightConfig, estimate, format_preflight
from spartan_pong.train import train_model
from spartan_pong.vae import encode_dataset_tokens, export_reconstruction_sheet, train_vae
from spartan_pong.visualize import visualize_outputs

# Metrics from the SPARTAN paper for reference in report.
PAPER_TARGETS = {
    "spartan": {"pred_l2": 8.60, "shd": 1.51, "robust_pct": 24.5},
    "transformer": {"pred_l2": 8.83, "shd": 6.37, "robust_pct": 1140.2},
}

def _model_cfg(args: argparse.Namespace) -> ModelConfig:
    # Returns the model config using the cli args.
    return ModelConfig(
        embed_dim=args.embed_dim,
        layers=args.layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_layers=args.mlp_layers,
        attention_threshold=args.attention_threshold,
        n_envs=args.n_envs,
    )


def cmd_generate_data(args: argparse.Namespace) -> None:
    # Generate the Interventional Pong dataset using the cli args.
    generate_dataset(
        args.out,
        episodes_per_env=args.episodes_per_env,
        horizon=args.horizon,
        split=args.split,
        seed=args.seed,
        include_unseen=args.include_unseen,
    )


def cmd_train(args: argparse.Namespace) -> None:
    # Train the model specified in the cli args and save checkpoints and metrics to the output directory.
    target_loss = args.target_loss
    if args.target_loss_from:
        # Read the metrics from the specified file and extract the target loss from it, falling back to the cli arg if the expected keys are not found.
        # This is needed to set the SPARTAN target loss based on the trained transformer performance, as reported in the paper, to ensure a fair comparison.
        metrics = _read_metrics(args.target_loss_from)
        target_loss = _target_loss_from_metrics(metrics, target_loss)
    # Define the model and training configs based on the cli args, then train the model.
    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        eval_every=args.eval_every,
        target_loss=target_loss,
        lagrangian_alpha=args.lagrangian_alpha,
        lambda_init=args.lambda_init,
        sparsity_weight=args.sparsity_weight,
        device=args.device,
    )
    # Train the model
    train_model(
        args.model, args.train, args.val, args.out, _model_cfg(args), cfg, resume=args.resume
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    # Evaluate the models in the specified checkpoint on the specified data and save the metrics to the output file if provided.
    metrics = evaluate_checkpoint(
        args.checkpoint,
        args.data,
        args.out,
        device=args.device,
        batch_size=args.batch_size,
        max_rollout_episodes=args.max_rollout_episodes,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_write_target_loss(args: argparse.Namespace) -> None:
    # Evaluate the one-step MSE of the model in the specified checkpoint on the specified data and write it to the output file as a JSON object with additional metadata.
    # This is used to set the SPARTAN target loss based on the trained transformer performance, as reported in the paper, to ensure a fair comparison.
    target_loss = _write_target_loss(
        Path(args.checkpoint),
        Path(args.data),
        Path(args.out),
        args.device,
        args.batch_size,
    )
    print(f"wrote {args.out} target_loss={target_loss:.8g}")


def _read_metrics(path: str | Path) -> dict[str, Any]:
    # Read the metrics from a specified JSON file
    payload = json.loads(Path(path).read_text())
    return payload.get("metrics", payload)


def _target_loss_from_metrics(metrics: dict[str, float], default: float) -> float:
    # Extract target loss informatin from the metrics dictionary.
    for key in ("target_loss", "one_step_mse", "train_mse"):
        if key in metrics:
            return float(metrics[key])
    if "pred_l2" in metrics:
        return float(metrics["pred_l2"]) ** 2
    return default


def _vae_finished(checkpoint: str | Path, target_steps: int) -> bool:
    # Checks whether the VAE finished training
    payload = torch.load(checkpoint, map_location="cpu")
    return int(payload.get("step", -1)) + 1 >= target_steps


def cmd_smoke(args: argparse.Namespace) -> None:
    # Runs a quick smoke test of the training and evaluation pipeline.
    work = Path(args.work_dir)
    train_pixels = work / "train_pixels.npz"
    val_pixels = work / "val_pixels.npz"
    train_path = work / "train_tokens.npz"
    val_path = work / "val_tokens.npz"
    generate_dataset(train_pixels, episodes_per_env=8, horizon=16, split="train", seed=0)
    generate_dataset(val_pixels, episodes_per_env=3, horizon=12, split="val", seed=1)
    train_vae(
        train_pixels,
        work / "object_vae",
        steps=max(1, args.vae_steps),
        batch_size=64,
        device=args.device,
    )
    export_reconstruction_sheet(
        work / "object_vae" / "vae.pt",
        train_pixels,
        work / "object_vae" / "reconstruction_qa",
        num_samples=4,
        device=args.device,
        batch_size=64,
    )
    encode_dataset_tokens(
        work / "object_vae" / "vae.pt", train_pixels, train_path, device=args.device
    )
    encode_dataset_tokens(work / "object_vae" / "vae.pt", val_pixels, val_path, device=args.device)
    model_cfg = ModelConfig(embed_dim=64, layers=2, mlp_hidden_dim=128, mlp_layers=2)
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=64,
        lr=1e-3,
        eval_every=max(1, args.steps // 2),
        device=args.device,
    )
    for model_type in ("transformer", "spartan"):
        train_model(model_type, train_path, val_path, work / model_type, model_cfg, train_cfg)
        metrics = evaluate_checkpoint(
            work / model_type / "checkpoint.pt",
            val_path,
            work / model_type / "eval.json",
            device=args.device,
        )
        print(model_type, json.dumps(metrics, sort_keys=True))


def cmd_report(args: argparse.Namespace) -> None:
    # Reads the evaluation metrics for SPARTAN and the Transformer from specified files,
    # compares them to the target metrics from the paper,
    # and generates a markdown report summarizing the results, including per-environment diagnostics if available.
    transformer = _read_metrics(args.transformer)
    spartan = _read_metrics(args.spartan)
    spartan_targets = PAPER_TARGETS["spartan"]
    transformer_targets = PAPER_TARGETS["transformer"]
    lines = [
        "# Interventional Pong Reproduction Report",
        "",
        "## Table 1 Subset",
        "",
        "| Model | Rollout prediction error | One-step error | SHD |",
        "|---|---:|---:|---:|",
        f"| SPARTAN | {spartan['pred_l2']:.4f} | {spartan['one_step_l2']:.4f} | {spartan['shd']:.4f} |",
        f"| Transformer | {transformer['pred_l2']:.4f} | {transformer['one_step_l2']:.4f} | {transformer['shd']:.4f} |",
        "",
        "## Graph Edge Diagnostics",
        "",
        "| Model | Predicted active edges | Target active edges | Edge surplus |",
        "|---|---:|---:|---:|",
        _graph_edge_row("SPARTAN", spartan),
        _graph_edge_row("Transformer", transformer),
        "",
        "## Graph Threshold Selection",
        "",
        "| Model | Threshold | Selection method |",
        "|---|---:|---|",
        f"| SPARTAN | {_fmt_optional(spartan.get('graph_threshold'))} | Bernoulli/path threshold from model config |",
        f"| Transformer | {_fmt_optional(transformer.get('graph_threshold'))} | Best validation SHD over 101 attention thresholds |",
        "",
        "## Table 2 Subset",
        "",
        "| Model | Non-causal removal L2 percent change | Non-causal removal MSE percent change |",
        "|---|---:|---:|",
        _robustness_row("SPARTAN", spartan),
        _robustness_row("Transformer", transformer),
        "",
        "## Paper Target Delta",
        "",
        "| Model | Metric | Run | Paper | Delta |",
        "|---|---|---:|---:|---:|",
        f"| SPARTAN | Rollout prediction error | {spartan['pred_l2']:.4f} | {spartan_targets['pred_l2']:.4f} | {spartan['pred_l2'] - spartan_targets['pred_l2']:.4f} |",
        f"| SPARTAN | SHD | {spartan['shd']:.4f} | {spartan_targets['shd']:.4f} | {spartan['shd'] - spartan_targets['shd']:.4f} |",
        f"| SPARTAN | Non-causal removal pct | {spartan['robust_pct']:.4f} | {spartan_targets['robust_pct']:.4f} | {spartan['robust_pct'] - spartan_targets['robust_pct']:.4f} |",
        f"| Transformer | Rollout prediction error | {transformer['pred_l2']:.4f} | {transformer_targets['pred_l2']:.4f} | {transformer['pred_l2'] - transformer_targets['pred_l2']:.4f} |",
        f"| Transformer | SHD | {transformer['shd']:.4f} | {transformer_targets['shd']:.4f} | {transformer['shd'] - transformer_targets['shd']:.4f} |",
        f"| Transformer | Non-causal removal pct | {transformer['robust_pct']:.4f} | {transformer_targets['robust_pct']:.4f} | {transformer['robust_pct'] - transformer_targets['robust_pct']:.4f} |",
        "",
        "## Per-Environment Diagnostics",
        "",
        "| Model | Env | One-step error | Rollout error | SHD | Active edges | Target edges | Non-causal removal L2 pct | Non-causal removal MSE pct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    spartan_per_env = spartan.get("per_env", {})
    transformer_per_env = transformer.get("per_env", {})
    if not isinstance(spartan_per_env, dict):
        spartan_per_env = {}
    if not isinstance(transformer_per_env, dict):
        transformer_per_env = {}
    envs = sorted(
        set(_metric_by_env(spartan_per_env, "one_step_l2"))
        | set(_metric_by_env(transformer_per_env, "one_step_l2")),
        key=int,
    )
    for model_name, metrics in (("SPARTAN", spartan), ("Transformer", transformer)):
        per_env = metrics.get("per_env", {})
        if not isinstance(per_env, dict):
            per_env = {}
        for env_id in envs:
            one_step = _metric_by_env(per_env, "one_step_l2").get(env_id)
            rollout = _metric_by_env(per_env, "rollout_l2").get(env_id)
            shd = _metric_by_env(per_env, "shd").get(env_id)
            active_edges = _metric_by_env(per_env, "active_edges").get(env_id)
            target_edges = _metric_by_env(per_env, "target_edges").get(env_id)
            robust_by_env = _metric_by_env(per_env, "robust_pct_l2", "robust_pct")
            robust = robust_by_env.get(env_id)
            robust_mse = _metric_by_env(per_env, "robust_pct_mse").get(env_id)
            lines.append(
                f"| {model_name} | {env_id} | {_fmt_optional(one_step)} | {_fmt_optional(rollout)} | {_fmt_optional(shd)} | "
                f"{_fmt_optional(active_edges)} | {_fmt_optional(target_edges)} | {_fmt_optional(robust)} | {_fmt_optional(robust_mse)} |"
            )
    lines.append("")
    report = "\n".join(lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report)
    print(report)


def _fmt_optional(value: Any) -> str:
    # Formats the value as a float with 4 decimal places if it is not None, otherwise returns an empty string.
    return "" if value is None else f"{float(value):.4f}"


def _graph_edge_row(model_name: str, metrics: dict[str, float]) -> str:
    # Formats a row for the graph edge diagnostics table, calculating the surplus of active edges over target edges if both are available.
    active = metrics.get("active_edges")
    target = metrics.get("target_edges")
    surplus = None if active is None or target is None else float(active) - float(target)
    return f"| {model_name} | {_fmt_optional(active)} | {_fmt_optional(target)} | {_fmt_optional(surplus)} |"


def _metric_by_env(per_env: dict[str, Any], key: str, fallback: str | None = None) -> dict[str, Any]:
    values = per_env.get(key)
    if not isinstance(values, dict) and fallback is not None:
        values = per_env.get(fallback)
    return values if isinstance(values, dict) else {}


def _robustness_row(model_name: str, metrics: dict[str, Any]) -> str:
    robust_l2 = metrics.get("robust_pct_l2", metrics.get("robust_pct"))
    robust_mse = metrics.get("robust_pct_mse")
    return f"| {model_name} | {_fmt_optional(robust_l2)} | {_fmt_optional(robust_mse)} |"


def cmd_train_vae(args: argparse.Namespace) -> None:
    # Trains a VAE on specified data and saves the trained model and evaluation metrics to the output directory.
    metrics = train_vae(
        args.data,
        args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_encode_tokens(args: argparse.Namespace) -> None:
    # Encode tokens for the specified dataset using a trained VAE checkpoint and save the encoded tokens to the output file.
    encode_dataset_tokens(
        args.checkpoint,
        args.data,
        args.out,
        device=args.device,
        batch_size=args.batch_size,
    )


def cmd_export_vae_reconstructions(args: argparse.Namespace) -> None:
    # Exports a reconstruction sheet for the specified VAE checkpoint and dataset, saving the resulting metrics to the output directory.
    metrics = export_reconstruction_sheet(
        args.checkpoint,
        args.data,
        args.out,
        num_samples=args.num_samples,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_visualize(args: argparse.Namespace) -> None:
    metadata = visualize_outputs(
        args.pixels,
        args.tokens,
        args.vae_checkpoint,
        args.transformer,
        args.spartan,
        args.out_dir,
        num_samples=args.num_samples,
        episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        fps=args.fps,
        device=args.device,
    )
    print(json.dumps(metadata["outputs"], indent=2, sort_keys=True))


def _preflight_config(args: argparse.Namespace) -> PreflightConfig:
    # This create a prerun configuation, which is used to estimate the expected performance and potential issues before actually running the training, based on the cli args.
    return PreflightConfig(
        train_episodes_per_env=args.train_episodes_per_env,
        test_episodes_per_env=args.test_episodes_per_env,
        horizon=args.horizon,
        vae_steps=args.vae_steps,
        dynamics_steps=args.dynamics_steps,
        batch_size=args.batch_size,
        vae_batch_size=args.vae_batch_size,
        embed_dim=args.embed_dim,
        layers=args.layers,
        device=args.device,
        lagrangian_alpha=args.lagrangian_alpha,
        lambda_init=args.lambda_init,
        sparsity_weight=args.sparsity_weight,
    )


def _evaluate_model_checkpoints(
    model_dir: Path,
    data_path: Path,
    device: str,
    batch_size: int,
    max_rollout_episodes: int | None,
) -> None:
    evaluate_checkpoint(
        model_dir / "checkpoint.pt",
        data_path,
        model_dir / "eval.json",
        device=device,
        batch_size=batch_size,
        max_rollout_episodes=max_rollout_episodes,
    )
    best_checkpoint = model_dir / "best_checkpoint.pt"
    if best_checkpoint.exists():
        evaluate_checkpoint(
            best_checkpoint,
            data_path,
            model_dir / "best_eval.json",
            device=device,
            batch_size=batch_size,
            max_rollout_episodes=max_rollout_episodes,
        )


def _write_target_loss(
    checkpoint: Path,
    data_path: Path,
    out_path: Path,
    device: str,
    batch_size: int,
) -> float:
    target_loss = one_step_mse_checkpoint(
        checkpoint,
        data_path,
        device=device,
        batch_size=batch_size,
    )
    payload = {
        "checkpoint": str(checkpoint),
        "data": str(data_path),
        "target_loss": target_loss,
        "source_metric": "one_step_mse",
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return target_loss


def _write_report(spartan_eval: Path, transformer_eval: Path, out: Path) -> None:
    args = argparse.Namespace(
        spartan=str(spartan_eval), transformer=str(transformer_eval), out=str(out)
    )
    cmd_report(args)


def cmd_run_reproduction(args: argparse.Namespace) -> None:
    # Runs the full reproduction of the SPARTAN paper (pong part)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.pop("func", None)
    (work / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    preflight = estimate(_preflight_config(args))
    (work / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True))
    if preflight["warnings"] and not args.yes:
        print(format_preflight(preflight))
        raise SystemExit("Preflight warnings found. Re-run with --yes to start anyway.")

    train_pixels = work / "data" / "train_pixels.npz"
    test_pixels = work / "data" / "test_pixels.npz"
    train_tokens = work / "data" / "train_tokens.npz"
    test_tokens = work / "data" / "test_tokens.npz"

    if not (args.resume and train_pixels.exists()):
        generate_dataset(
            train_pixels,
            episodes_per_env=args.train_episodes_per_env,
            horizon=args.horizon,
            split="train",
            seed=args.seed,
        )
    if not (args.resume and test_pixels.exists()):
        generate_dataset(
            test_pixels,
            episodes_per_env=args.test_episodes_per_env,
            horizon=args.horizon,
            split="test",
            seed=args.seed + 1,
        )
    if not (
        args.resume
        and (work / "object_vae" / "vae.pt").exists()
        and _vae_finished(work / "object_vae" / "vae.pt", args.vae_steps)
    ):
        train_vae(
            train_pixels,
            work / "object_vae",
            steps=args.vae_steps,
            batch_size=args.vae_batch_size,
            lr=args.vae_lr,
            beta=args.vae_beta,
            seed=args.seed,
            device=args.device,
            resume=args.resume,
        )
    export_reconstruction_sheet(
        work / "object_vae" / "vae.pt",
        train_pixels,
        work / "object_vae" / "reconstruction_qa",
        num_samples=8,
        device=args.device,
        batch_size=args.vae_batch_size,
    )
    if not (args.resume and train_tokens.exists()):
        encode_dataset_tokens(
            work / "object_vae" / "vae.pt", train_pixels, train_tokens, device=args.device
        )
    if not (args.resume and test_tokens.exists()):
        encode_dataset_tokens(
            work / "object_vae" / "vae.pt", test_pixels, test_tokens, device=args.device
        )

    model_cfg = ModelConfig(
        embed_dim=args.embed_dim,
        layers=args.layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_layers=args.mlp_layers,
        n_envs=args.n_envs,
    )
    transformer_cfg = TrainConfig(
        steps=args.dynamics_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        eval_every=args.eval_every,
        device=args.device,
    )
    train_model(
        "transformer",
        train_tokens,
        test_tokens,
        work / "transformer",
        model_cfg,
        transformer_cfg,
        resume=args.resume,
    )
    transformer_target_loss = _write_target_loss(
        work / "transformer" / "checkpoint.pt",
        train_tokens,
        work / "transformer" / "target_loss.json",
        args.device,
        args.batch_size,
    )
    spartan_cfg = TrainConfig(
        steps=args.dynamics_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        eval_every=args.eval_every,
        target_loss=transformer_target_loss,
        lagrangian_alpha=args.lagrangian_alpha,
        lambda_init=args.lambda_init,
        sparsity_weight=args.sparsity_weight,
        device=args.device,
    )
    train_model(
        "spartan",
        train_tokens,
        test_tokens,
        work / "spartan",
        model_cfg,
        spartan_cfg,
        resume=args.resume,
    )

    _evaluate_model_checkpoints(
        work / "transformer",
        test_tokens,
        args.device,
        args.batch_size,
        args.max_rollout_episodes,
    )
    _evaluate_model_checkpoints(
        work / "spartan",
        test_tokens,
        args.device,
        args.batch_size,
        args.max_rollout_episodes,
    )
    _write_report(
        work / "spartan" / "eval.json", work / "transformer" / "eval.json", work / "report.md"
    )
    if (work / "spartan" / "best_eval.json").exists():
        _write_report(
            work / "spartan" / "best_eval.json",
            work / "transformer" / "eval.json",
            work / "best_spartan_report.md",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPARTAN Interventional Pong reproduction")
    sub = parser.add_subparsers(required=True)

    gen = sub.add_parser("generate-data")
    gen.add_argument("--out", required=True)
    gen.add_argument("--episodes-per-env", type=int, default=100)
    gen.add_argument("--horizon", type=int, default=32)
    gen.add_argument("--split", default="train")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--include-unseen", action="store_true")
    gen.set_defaults(func=cmd_generate_data)

    train = sub.add_parser("train")
    train.add_argument("--model", choices=["spartan", "transformer"], required=True)
    train.add_argument("--train", required=True)
    train.add_argument("--val", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--steps", type=int, default=100_000)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--eval-every", type=int, default=2_000)
    train.add_argument("--target-loss", type=float, default=1e-3)
    train.add_argument("--target-loss-from")
    train.add_argument("--lagrangian-alpha", type=float, default=1.001)
    train.add_argument("--lambda-init", type=float, default=20.0)
    train.add_argument("--sparsity-weight", type=float, default=1.0)
    train.add_argument("--device", default="cpu")
    train.add_argument("--embed-dim", type=int, default=512)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--mlp-hidden-dim", type=int, default=512)
    train.add_argument("--mlp-layers", type=int, default=3)
    train.add_argument("--attention-threshold", type=float, default=0.5)
    train.add_argument("--n-envs", type=int, default=11)
    train.add_argument("--resume", action="store_true")
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--out")
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--batch-size", type=int, default=256)
    ev.add_argument("--max-rollout-episodes", type=int)
    ev.set_defaults(func=cmd_evaluate)

    target = sub.add_parser("write-target-loss")
    target.add_argument("--checkpoint", required=True)
    target.add_argument("--data", required=True)
    target.add_argument("--out", required=True)
    target.add_argument("--device", default="cpu")
    target.add_argument("--batch-size", type=int, default=256)
    target.set_defaults(func=cmd_write_target_loss)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--work-dir", default="runs/smoke")
    smoke.add_argument("--steps", type=int, default=20)
    smoke.add_argument("--vae-steps", type=int, default=2)
    smoke.add_argument("--device", default="cpu")
    smoke.set_defaults(func=cmd_smoke)

    vae = sub.add_parser("train-vae")
    vae.add_argument("--data", required=True)
    vae.add_argument("--out", required=True)
    vae.add_argument("--steps", type=int, default=10_000)
    vae.add_argument("--batch-size", type=int, default=256)
    vae.add_argument("--lr", type=float, default=1e-3)
    vae.add_argument("--beta", type=float, default=1e-4)
    vae.add_argument("--seed", type=int, default=0)
    vae.add_argument("--device", default="cpu")
    vae.add_argument("--resume", action="store_true")
    vae.set_defaults(func=cmd_train_vae)

    encode = sub.add_parser("encode-tokens")
    encode.add_argument("--checkpoint", required=True)
    encode.add_argument("--data", required=True)
    encode.add_argument("--out", required=True)
    encode.add_argument("--batch-size", type=int, default=512)
    encode.add_argument("--device", default="cpu")
    encode.set_defaults(func=cmd_encode_tokens)

    recon = sub.add_parser("export-vae-reconstructions")
    recon.add_argument("--checkpoint", required=True)
    recon.add_argument("--data", required=True)
    recon.add_argument("--out", required=True)
    recon.add_argument("--num-samples", type=int, default=8)
    recon.add_argument("--batch-size", type=int, default=128)
    recon.add_argument("--device", default="cpu")
    recon.set_defaults(func=cmd_export_vae_reconstructions)

    visualize = sub.add_parser("visualize")
    visualize.add_argument("--pixels", required=True)
    visualize.add_argument("--tokens", required=True)
    visualize.add_argument("--vae-checkpoint", required=True)
    visualize.add_argument("--transformer", required=True)
    visualize.add_argument("--spartan", required=True)
    visualize.add_argument("--out-dir", required=True)
    visualize.add_argument("--num-samples", type=int, default=8)
    visualize.add_argument("--episodes", type=int, default=1)
    visualize.add_argument("--rollout-steps", type=int, default=16)
    visualize.add_argument("--fps", type=int, default=4)
    visualize.add_argument("--device", default="cpu")
    visualize.set_defaults(func=cmd_visualize)

    report = sub.add_parser("report")
    report.add_argument("--spartan", required=True)
    report.add_argument("--transformer", required=True)
    report.add_argument("--out")
    report.set_defaults(func=cmd_report)

    repro = sub.add_parser("run-reproduction")
    _add_reproduction_args(repro)
    repro.add_argument(
        "--yes", action="store_true", help="Start even if preflight reports warnings."
    )
    repro.set_defaults(func=cmd_run_reproduction)
    return parser


def _add_reproduction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", default="runs/interventional_pong")
    parser.add_argument("--train-episodes-per-env", type=int, default=2000)
    parser.add_argument("--test-episodes-per-env", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--vae-steps", type=int, default=200_000)
    parser.add_argument("--vae-batch-size", type=int, default=512)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--vae-beta", type=float, default=1e-4)
    parser.add_argument("--dynamics-steps", type=int, default=4_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lagrangian-alpha", type=float, default=1.001)
    parser.add_argument("--lambda-init", type=float, default=20.0)
    parser.add_argument("--sparsity-weight", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-layers", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=11)
    parser.add_argument("--max-rollout-episodes", type=int)
    parser.add_argument("--resume", action="store_true")


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
