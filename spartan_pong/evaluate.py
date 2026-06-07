from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spartan_pong.config import N_OBJECTS
from spartan_pong.data import load_npz
from spartan_pong.metrics import percent_change, structural_hamming_distance
from spartan_pong.models import load_checkpoint, path_matrix


def representative_eval_indices(
    data: dict[str, np.ndarray], max_samples: int, seed: int = 0
) -> np.ndarray:
    """Return a deterministic env-stratified subset for lightweight history eval."""
    n = len(data["x"])
    if max_samples <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n <= max_samples:
        return np.arange(n, dtype=np.int64)
    envs = data.get("env", np.zeros(n, dtype=np.int64)).astype(int)
    unique_envs = sorted(set(envs.tolist()))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    per_env = max(1, max_samples // max(1, len(unique_envs)))
    remainder = max(0, max_samples - per_env * len(unique_envs))
    for rank, env_id in enumerate(unique_envs):
        candidates = np.flatnonzero(envs == env_id)
        take = min(len(candidates), per_env + int(rank < remainder))
        if take == len(candidates):
            chosen = candidates
        else:
            chosen = rng.choice(candidates, size=take, replace=False)
        selected.extend(int(idx) for idx in chosen.tolist())
    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(n, dtype=np.int64), np.asarray(selected, dtype=np.int64))
        fill = min(len(remaining), max_samples - len(selected))
        if fill > 0:
            selected.extend(int(idx) for idx in rng.choice(remaining, size=fill, replace=False))
    return np.asarray(sorted(selected[:max_samples]), dtype=np.int64)


def _iter_batch_indices(
    data: dict[str, np.ndarray],
    batch_size: int,
    max_batches: int | None = None,
    indices: np.ndarray | None = None,
):
    n = len(data["x"])
    order = np.arange(n, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
    count = 0
    for start in range(0, len(order), batch_size):
        if max_batches is not None and count >= max_batches:
            break
        yield order[start : start + batch_size]
        count += 1


def _iter_batches(
    data: dict[str, np.ndarray],
    batch_size: int,
    max_batches: int | None = None,
    indices: np.ndarray | None = None,
):
    # Iterate over the data in batches, yield (x, y, env, graph)
    graph_key = "graph_with_env" if "graph_with_env" in data else "graph"
    for idx in _iter_batch_indices(data, batch_size, max_batches=max_batches, indices=indices):
        yield data["x"][idx], data["y"][idx], data["env"][idx], data[graph_key][idx]


def _episode_keys(data: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    # Get unique game from an environment. Each game is identified by (env, episode) pair. We sort them to ensure consistent order.
    if "episode" not in data:
        return []
    envs = data["env"].astype(int)
    episodes = data["episode"].astype(int)
    return sorted(set(zip(envs.tolist(), episodes.tolist(), strict=True)))


def _mean_by_env(records: list[tuple[int, float]]) -> dict[str, float]:
    # Compute mean of values grouped by environment ID to evaluate models performance on each environment seperately.
    grouped: dict[int, list[float]] = {}
    for env_id, value in records:
        grouped.setdefault(env_id, []).append(value)
    return {str(env_id): float(np.mean(values)) for env_id, values in sorted(grouped.items())}


def _mean_record(records: list[tuple[int, float]]) -> float:
    values = [value for _, value in records]
    return float(np.mean(values)) if values else float("nan")


def _edge_counts(graph: np.ndarray) -> np.ndarray:
    # Count the number of active edges for each sample in the graph, exclude self-edges.
    graph_copy = graph.copy()
    diag = np.arange(N_OBJECTS)
    graph_copy[:, diag, diag] = 0
    return graph_copy.sum(axis=(1, 2)).astype(np.float64)


def _edge_counts_tensor(graph: torch.Tensor) -> torch.Tensor:
    graph_copy = graph.clone()
    diag = torch.arange(N_OBJECTS, device=graph.device)
    graph_copy[:, diag, diag] = 0
    return graph_copy.sum(dim=(1, 2)).to(torch.float64)


def _graph_confusion_counts(
    pred: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_bool = np.asarray(pred).astype(bool)
    target_bool = np.asarray(target).astype(bool)
    if pred_bool.ndim == 2:
        pred_bool = pred_bool[None]
        target_bool = target_bool[None]
    valid = np.ones_like(target_bool, dtype=bool)
    diag = np.arange(min(N_OBJECTS, target_bool.shape[1], target_bool.shape[2]))
    valid[:, diag, diag] = False
    tp = (pred_bool & target_bool & valid).sum(axis=(1, 2)).astype(np.float64)
    fp = (pred_bool & ~target_bool & valid).sum(axis=(1, 2)).astype(np.float64)
    fn = (~pred_bool & target_bool & valid).sum(axis=(1, 2)).astype(np.float64)
    return tp, fp, fn


def _graph_confusion_counts_tensor(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_bool = pred.to(torch.bool)
    target_bool = target.to(torch.bool)
    valid = torch.ones_like(target_bool, dtype=torch.bool)
    diag = torch.arange(
        min(N_OBJECTS, target_bool.shape[1], target_bool.shape[2]), device=target.device
    )
    valid[:, diag, diag] = False
    tp = (pred_bool & target_bool & valid).sum(dim=(1, 2)).to(torch.float64)
    fp = (pred_bool & ~target_bool & valid).sum(dim=(1, 2)).to(torch.float64)
    fn = (~pred_bool & target_bool & valid).sum(dim=(1, 2)).to(torch.float64)
    return tp, fp, fn


def _threshold_candidates() -> np.ndarray:
    # Generate candidate thresholds for transformer attention to graph conversion. We include 0.0 and 1.0 as edge cases, and 99 values evenly spaced between them.
    return np.concatenate(([0.0], np.linspace(0.01, 0.99, 99), [1.0])).astype(np.float32)


def empty_graph_baseline(
    data: dict[str, np.ndarray], graph_key: str | None = None
) -> dict[str, Any]:
    # Compute the SHD and edge counts for an empty graph baseline, where the predicted graph has no edges.
    # This serves as a reference point to evaluate the model's predicted graphs against.
    resolved_graph_key = graph_key or ("graph_with_env" if "graph_with_env" in data else "graph")
    target = data[resolved_graph_key].astype(np.uint8)
    env = data["env"].astype(int)
    empty = np.zeros_like(target, dtype=np.uint8)
    tp, fp, fn = _graph_confusion_counts(empty, target)
    shd_records = [
        (int(env_id), float(structural_hamming_distance(pred, truth)))
        for env_id, pred, truth in zip(env.tolist(), empty, target, strict=True)
    ]
    active_records = [
        (int(env_id), float(value))
        for env_id, value in zip(env.tolist(), _edge_counts(empty).tolist(), strict=True)
    ]
    target_records = [
        (int(env_id), float(value))
        for env_id, value in zip(env.tolist(), _edge_counts(target).tolist(), strict=True)
    ]
    shd_values = [value for _, value in shd_records]
    active_values = [value for _, value in active_records]
    target_values = [value for _, value in target_records]
    return {
        "active_edges": float(np.mean(active_values)) if active_values else float("nan"),
        "fn": float(np.mean(fn)) if len(fn) else float("nan"),
        "fp": float(np.mean(fp)) if len(fp) else float("nan"),
        "graph_key": resolved_graph_key,
        "num_transitions": float(len(target)),
        "per_env": {
            "active_edges": _mean_by_env(active_records),
            "fn": _mean_by_env(
                [(int(env_id), float(value)) for env_id, value in zip(env, fn, strict=True)]
            ),
            "fp": _mean_by_env(
                [(int(env_id), float(value)) for env_id, value in zip(env, fp, strict=True)]
            ),
            "shd": _mean_by_env(shd_records),
            "target_edges": _mean_by_env(target_records),
            "tp": _mean_by_env(
                [(int(env_id), float(value)) for env_id, value in zip(env, tp, strict=True)]
            ),
        },
        "shd": float(np.mean(shd_values)) if shd_values else float("nan"),
        "target_edges": float(np.mean(target_values)) if target_values else float("nan"),
        "tp": float(np.mean(tp)) if len(tp) else float("nan"),
    }


@torch.no_grad()
def graph_threshold_sweep_arrays(
    model: nn.Module,
    data: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 256,
    thresholds: np.ndarray | None = None,
    max_batches: int | None = None,
    indices: np.ndarray | None = None,
) -> dict[str, Any]:
    # Helper function to evaluate multiple candidate thresholds for transformer and spartan models.
    # It calculates the SHD and edge counts for each candidate threshold and returns the results along with the best threshold based on SHD.
    model.eval()
    candidates = (
        _threshold_candidates() if thresholds is None else np.asarray(thresholds, dtype=np.float32)
    )
    if candidates.ndim != 1:
        raise ValueError(f"expected one-dimensional thresholds, got {candidates.shape}")
    shd_totals = np.zeros_like(candidates, dtype=np.float64)
    active_totals = np.zeros_like(candidates, dtype=np.float64)
    tp_totals = np.zeros_like(candidates, dtype=np.float64)
    fp_totals = np.zeros_like(candidates, dtype=np.float64)
    fn_totals = np.zeros_like(candidates, dtype=np.float64)
    target_total = 0.0
    count = 0
    for x_np, _, env_np, graph_np in _iter_batches(
        data, batch_size, max_batches=max_batches, indices=indices
    ):
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        env = torch.as_tensor(env_np, dtype=torch.long, device=device)
        attention = model(x, env=env, hard=False)["attention"]
        include_env = graph_np.shape[-1] == N_OBJECTS + 1
        parent_count = N_OBJECTS + int(include_env)
        target = torch.as_tensor(graph_np, dtype=torch.uint8, device=device)
        target_total += float(_edge_counts_tensor(target).sum().cpu().item())
        count += int(graph_np.shape[0])
        diag = torch.arange(N_OBJECTS, device=device)
        for idx, threshold in enumerate(candidates):
            adj = (attention > float(threshold)).to(x.dtype)
            graph = (path_matrix(adj)[:, :N_OBJECTS, :parent_count] >= 1.0).to(torch.uint8)
            mismatch = graph != target
            mismatch[:, diag, diag] = False
            tp, fp, fn = _graph_confusion_counts_tensor(graph, target)
            shd_totals[idx] += float(mismatch.sum().cpu().item())
            active_totals[idx] += float(_edge_counts_tensor(graph).sum().cpu().item())
            tp_totals[idx] += float(tp.sum().cpu().item())
            fp_totals[idx] += float(fp.sum().cpu().item())
            fn_totals[idx] += float(fn.sum().cpu().item())
    target_edges = float(target_total / count) if count else float("nan")
    records = []
    for threshold, shd_total, active_total, tp_total, fp_total, fn_total in zip(
        candidates.tolist(),
        shd_totals.tolist(),
        active_totals.tolist(),
        tp_totals.tolist(),
        fp_totals.tolist(),
        fn_totals.tolist(),
        strict=True,
    ):
        active_edges = float(active_total / count) if count else float("nan")
        records.append(
            {
                "active_edges": active_edges,
                "edge_surplus": active_edges - target_edges,
                "fn": float(fn_total / count) if count else float("nan"),
                "fp": float(fp_total / count) if count else float("nan"),
                "shd": float(shd_total / count) if count else float("nan"),
                "target_edges": target_edges,
                "threshold": float(threshold),
                "tp": float(tp_total / count) if count else float("nan"),
            }
        )
    best = min(records, key=lambda item: item["shd"]) if records else None
    return {
        "best": best,
        "max_batches": max_batches,
        "model_type": str(getattr(model, "model_type", "unknown")),
        "num_transitions": float(count),
        "thresholds": records,
    }


@torch.no_grad()
def one_step_mse_arrays(
    model: nn.Module,
    data: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 256,
    max_batches: int | None = None,
    indices: np.ndarray | None = None,
) -> float:
    # Compute the mean squared error of the model's one-step predictions.
    model.eval()
    total = 0.0
    count = 0
    for x_np, y_np, env_np, _ in _iter_batches(
        data, batch_size, max_batches=max_batches, indices=indices
    ):
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
        env = torch.as_tensor(env_np, dtype=torch.long, device=device)
        pred = model(x, env=env, hard=False)["pred"]
        per_sample_mse = F.mse_loss(pred, y, reduction="none").mean(dim=(1, 2))
        total += float(per_sample_mse.sum().cpu().item())
        count += int(per_sample_mse.numel())
    return float(total / count) if count else float("nan")


@torch.no_grad()
def _select_transformer_threshold(
    model: nn.Module,
    data: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    max_batches: int | None,
    indices: np.ndarray | None = None,
) -> dict[str, float] | None:
    # For the transformer model, we need to select a threshold to convert attention weights to binary graph edges.
    # We evaluate multiple candidate thresholds on the validation data and select the one that minimizes the structural hamming distance (SHD)
    # between the predicted and target graphs.
    if getattr(model, "model_type", None) != "transformer":
        return None
    candidates = _threshold_candidates()
    totals = np.zeros_like(candidates, dtype=np.float64)
    count = 0
    # Iterate over the data in batches, compute attention based graphs for each candidate threshold, and accumulate the total SHD for each threshold.
    for x_np, _, env_np, graph_np in _iter_batches(
        data, batch_size, max_batches=max_batches, indices=indices
    ):
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        env = torch.as_tensor(env_np, dtype=torch.long, device=device)
        # Compute attention weights from the model. We set hard=False to get soft attention values.
        attention = model(x, env=env, hard=False)["attention"]
        include_env = graph_np.shape[-1] == N_OBJECTS + 1
        parent_count = N_OBJECTS + int(include_env)
        target = torch.as_tensor(graph_np, dtype=torch.uint8, device=device)
        for idx, threshold in enumerate(candidates):
            adj = (attention > float(threshold)).to(x.dtype)
            graph = (path_matrix(adj)[:, :N_OBJECTS, :parent_count] >= 1.0).to(torch.uint8)
            mismatch = graph != target
            diag = torch.arange(N_OBJECTS, device=device)
            mismatch[:, diag, diag] = False
            totals[idx] += float(mismatch.sum().cpu().item())
        count += int(graph_np.shape[0])
    if count == 0:
        return None
    best_idx = int(np.argmin(totals))
    return {
        "threshold": float(candidates[best_idx]),
        "mean_shd": float(totals[best_idx] / count),
        "candidates": float(len(candidates)),
    }


@torch.no_grad()
def evaluate_arrays(
    model: nn.Module,
    data: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 256,
    max_batches: int | None = None,
    rollout: bool = False,
    max_rollout_episodes: int | None = None,
    indices: np.ndarray | None = None,
) -> dict[str, Any]:
    # Computes all evaluation metrics for the given model and data.
    # This includes one-step L2 error, one-step MSE, rollout L2 error (if rollout=True), structural hamming distance (SHD) of the predicted graphs,
    # number of active edges in the predicted graphs, number of target edges in the ground truth graphs, and robustness to object removal.
    model.eval()
    transformer_threshold = _select_transformer_threshold(
        model, data, device, batch_size, max_batches, indices=indices
    )
    main_graph_key = "graph_with_env" if "graph_with_env" in data else "graph"
    main_has_env = main_graph_key == "graph_with_env"
    # Create lists to store
    one_step_records: list[tuple[int, float]] = []
    one_step_mse_records: list[tuple[int, float]] = []
    shd_records: list[tuple[int, float]] = []
    object_shd_records: list[tuple[int, float]] = []
    fixed_shd_records: list[tuple[int, float]] = []
    fixed_object_shd_records: list[tuple[int, float]] = []
    active_edge_records: list[tuple[int, float]] = []
    target_edge_records: list[tuple[int, float]] = []
    fixed_active_edge_records: list[tuple[int, float]] = []
    tp_records: list[tuple[int, float]] = []
    fp_records: list[tuple[int, float]] = []
    fn_records: list[tuple[int, float]] = []
    fixed_tp_records: list[tuple[int, float]] = []
    fixed_fp_records: list[tuple[int, float]] = []
    fixed_fn_records: list[tuple[int, float]] = []
    robust_records: list[tuple[int, float]] = []
    robust_mse_records: list[tuple[int, float]] = []
    evaluated = 0
    # Calculate the metrics
    for batch_idx in _iter_batch_indices(
        data, batch_size, max_batches=max_batches, indices=indices
    ):
        x_np = data["x"][batch_idx]
        y_np = data["y"][batch_idx]
        env_np = data["env"][batch_idx]
        graph_np = data[main_graph_key][batch_idx]
        object_graph_np = data["graph"][batch_idx] if "graph" in data else graph_np[:, :, :N_OBJECTS]
        evaluated += int(len(batch_idx))
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
        env = torch.as_tensor(env_np, dtype=torch.long, device=device)
        pred = model(x, env=env, hard=False)["pred"]
        # Per sample L2 and MSE
        per_sample = torch.linalg.vector_norm(pred - y, dim=-1).mean(dim=-1)
        per_sample_mse = F.mse_loss(pred, y, reduction="none").mean(dim=(1, 2))
        one_step_records.extend(
            (int(env_id), float(error))
            for env_id, error in zip(
                env_np.tolist(), per_sample.cpu().numpy().tolist(), strict=True
            )
        )
        one_step_mse_records.extend(
            (int(env_id), float(error))
            for env_id, error in zip(
                env_np.tolist(), per_sample_mse.cpu().numpy().tolist(), strict=True
            )
        )
        # For transformer models, we use the selected threshold to compute binary graphs from attention weights. For other models, we directly use the model's graph output.
        graph_kwargs: dict[str, Any] = {}
        if transformer_threshold is not None:
            graph_kwargs["threshold"] = transformer_threshold["threshold"]
        # Compute the predicted graph from the model
        pred_graph = (
            model.graph(x, env=env, include_env=main_has_env, **graph_kwargs)
            .cpu()
            .numpy()
        )
        object_pred_graph = model.graph(x, env=env, include_env=False, **graph_kwargs).cpu().numpy()
        fixed_pred_graph = (
            model.graph(x, env=env, include_env=main_has_env, threshold=0.5).cpu().numpy()
        )
        fixed_object_pred_graph = (
            model.graph(x, env=env, include_env=False, threshold=0.5).cpu().numpy()
        )
        # Compute SHD, active edges, target edges, and robustness for each sample and store the records with environment IDs for per-environment analysis.
        shd_records.extend(
            (int(env_id), float(structural_hamming_distance(p, g)))
            for env_id, p, g in zip(env_np.tolist(), pred_graph, graph_np, strict=True)
        )
        object_shd_records.extend(
            (int(env_id), float(structural_hamming_distance(p, g)))
            for env_id, p, g in zip(
                env_np.tolist(), object_pred_graph, object_graph_np, strict=True
            )
        )
        fixed_shd_records.extend(
            (int(env_id), float(structural_hamming_distance(p, g)))
            for env_id, p, g in zip(env_np.tolist(), fixed_pred_graph, graph_np, strict=True)
        )
        fixed_object_shd_records.extend(
            (int(env_id), float(structural_hamming_distance(p, g)))
            for env_id, p, g in zip(
                env_np.tolist(), fixed_object_pred_graph, object_graph_np, strict=True
            )
        )
        active_edge_records.extend(
            (int(env_id), float(value))
            for env_id, value in zip(
                env_np.tolist(), _edge_counts(pred_graph).tolist(), strict=True
            )
        )
        target_edge_records.extend(
            (int(env_id), float(value))
            for env_id, value in zip(env_np.tolist(), _edge_counts(graph_np).tolist(), strict=True)
        )
        fixed_active_edge_records.extend(
            (int(env_id), float(value))
            for env_id, value in zip(
                env_np.tolist(), _edge_counts(fixed_pred_graph).tolist(), strict=True
            )
        )
        tp, fp, fn = _graph_confusion_counts(pred_graph, graph_np)
        fixed_tp, fixed_fp, fixed_fn = _graph_confusion_counts(fixed_pred_graph, graph_np)
        tp_records.extend(
            (int(env_id), float(value)) for env_id, value in zip(env_np, tp, strict=True)
        )
        fp_records.extend(
            (int(env_id), float(value)) for env_id, value in zip(env_np, fp, strict=True)
        )
        fn_records.extend(
            (int(env_id), float(value)) for env_id, value in zip(env_np, fn, strict=True)
        )
        fixed_tp_records.extend(
            (int(env_id), float(value)) for env_id, value in zip(env_np, fixed_tp, strict=True)
        )
        fixed_fp_records.extend(
            (int(env_id), float(value)) for env_id, value in zip(env_np, fixed_fp, strict=True)
        )
        fixed_fn_records.extend(
            (int(env_id), float(value)) for env_id, value in zip(env_np, fixed_fn, strict=True)
        )
        robust, robust_mse, robust_env = _removal_robustness(model, x, y, env, graph_np)
        robust_records.extend(
            (int(env_id), float(value))
            for env_id, value in zip(
                robust_env.cpu().numpy().tolist(), robust.cpu().numpy().tolist(), strict=True
            )
        )
        robust_mse_records.extend(
            (int(env_id), float(value))
            for env_id, value in zip(
                robust_env.cpu().numpy().tolist(), robust_mse.cpu().numpy().tolist(), strict=True
            )
        )
    rollout_l2, rollout_by_env = (
        _rollout_l2(model, data, device=device, max_episodes=max_rollout_episodes)
        if rollout
        else (None, {})
    )
    metrics: dict[str, Any] = {
        "pred_l2": float(
            rollout_l2 if rollout_l2 is not None else _mean_record(one_step_records)
        ),
        "one_step_l2": _mean_record(one_step_records),
        "one_step_mse": _mean_record(one_step_mse_records),
        "rollout_l2": float(rollout_l2) if rollout_l2 is not None else None,
        "shd": _mean_record(shd_records),
        "shd_env_inclusive": _mean_record(shd_records) if main_has_env else None,
        "shd_object_only": _mean_record(object_shd_records),
        "fixed_shd_0_5": _mean_record(fixed_shd_records),
        "fixed_shd_0_5_env_inclusive": _mean_record(fixed_shd_records) if main_has_env else None,
        "fixed_shd_0_5_object_only": _mean_record(fixed_object_shd_records),
        "active_edges": _mean_record(active_edge_records),
        "target_edges": _mean_record(target_edge_records),
        "fixed_active_edges_0_5": _mean_record(fixed_active_edge_records),
        "tp": _mean_record(tp_records),
        "fp": _mean_record(fp_records),
        "fn": _mean_record(fn_records),
        "fixed_tp_0_5": _mean_record(fixed_tp_records),
        "fixed_fp_0_5": _mean_record(fixed_fp_records),
        "fixed_fn_0_5": _mean_record(fixed_fn_records),
        "robust_pct": _mean_record(robust_records),
        "robust_pct_l2": _mean_record(robust_records),
        "robust_pct_mse": _mean_record(robust_mse_records),
        "num_transitions": float(evaluated),
        "graph_threshold": (
            transformer_threshold["threshold"]
            if transformer_threshold is not None
            else float(getattr(getattr(model, "cfg", None), "attention_threshold", 0.5))
        ),
        "graph_threshold_selection": transformer_threshold,
        "per_env": {
            "one_step_l2": _mean_by_env(one_step_records),
            "one_step_mse": _mean_by_env(one_step_mse_records),
            "rollout_l2": rollout_by_env,
            "shd": _mean_by_env(shd_records),
            "shd_object_only": _mean_by_env(object_shd_records),
            "fixed_shd_0_5": _mean_by_env(fixed_shd_records),
            "fixed_shd_0_5_object_only": _mean_by_env(fixed_object_shd_records),
            "active_edges": _mean_by_env(active_edge_records),
            "target_edges": _mean_by_env(target_edge_records),
            "fixed_active_edges_0_5": _mean_by_env(fixed_active_edge_records),
            "tp": _mean_by_env(tp_records),
            "fp": _mean_by_env(fp_records),
            "fn": _mean_by_env(fn_records),
            "fixed_tp_0_5": _mean_by_env(fixed_tp_records),
            "fixed_fp_0_5": _mean_by_env(fixed_fp_records),
            "fixed_fn_0_5": _mean_by_env(fixed_fn_records),
            "robust_pct": _mean_by_env(robust_records),
            "robust_pct_l2": _mean_by_env(robust_records),
            "robust_pct_mse": _mean_by_env(robust_mse_records),
        },
    }
    return metrics


@torch.no_grad()
def _removal_robustness(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    env: torch.Tensor,
    graph_np: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Evaluate the model's robustness to object removal by comparing the prediction error before and after removing non-parent objects for each child object.
    base_pred = model(x, env=env, hard=False)["pred"]
    base_err_l2 = torch.linalg.vector_norm(base_pred - y, dim=-1)
    base_err_mse = F.mse_loss(base_pred, y, reduction="none").mean(dim=-1)
    changes = []
    mse_changes = []
    change_envs = []
    graph = torch.as_tensor(graph_np, dtype=torch.bool, device=x.device)
    # For each child object, identify the non-parent objects based on the graph.
    # Remove the non-parent objects from the input and compute the new prediction error.
    # Calculate the percentage change in error compared to the original error and store it along with the corresponding environment IDs for analysis.
    for child in range(N_OBJECTS):
        non_parents = ~graph[:, child, :]
        non_parents = non_parents[:, :N_OBJECTS]
        non_parents[:, child] = False
        # Check if there are any non-parent objects to remove for this child. If not, skip to the next child.
        has_removed = non_parents.any(dim=1)
        if not bool(has_removed.any()):
            continue
        x_removed = x.clone()
        x_removed[non_parents] = 0.0
        # Compute the model's predictions with the non-parent objects removed and calculate the new error.
        changed_pred = model(x_removed, env=env, hard=False)["pred"]
        changed_err_l2 = torch.linalg.vector_norm(changed_pred[:, child] - y[:, child], dim=-1)
        changed_err_mse = F.mse_loss(changed_pred[:, child], y[:, child], reduction="none").mean(
            dim=-1
        )
        pct_l2 = torch.as_tensor(
            percent_change(base_err_l2[:, child].cpu().numpy(), changed_err_l2.cpu().numpy()),
            dtype=torch.float32,
            device=x.device,
        )
        pct_mse = torch.as_tensor(
            percent_change(base_err_mse[:, child].cpu().numpy(), changed_err_mse.cpu().numpy()),
            dtype=torch.float32,
            device=x.device,
        )
        changes.append(pct_l2[has_removed])
        mse_changes.append(pct_mse[has_removed])
        change_envs.append(env[has_removed])
    if not changes:
        zeros = torch.zeros(x.shape[0], device=x.device)
        return zeros, zeros, env
    return torch.cat(changes), torch.cat(mse_changes), torch.cat(change_envs)


@torch.no_grad()
def _rollout_l2(
    model: nn.Module,
    data: dict[str, np.ndarray],
    device: torch.device,
    max_episodes: int | None = None,
) -> tuple[float, dict[str, float]]:
    # Evaluate the model's performance in multi-step rollouts by simulating the environment forward in time using the model's predictions as inputs for the next step.
    if "episode" not in data:
        return float("nan"), {}
    # We group the data by (env, episode) pairs to simulate each episode separately.
    # We sort the transitions within each episode by their time step to ensure correct rollout order.
    t_values = data.get("t", np.arange(len(data["x"]), dtype=np.int64))
    errors: list[float] = []
    records: list[tuple[int, float]] = []
    for count, (env_id, episode_id) in enumerate(_episode_keys(data)):
        if max_episodes is not None and count >= max_episodes:
            break
        idx = np.where((data["env"] == env_id) & (data["episode"] == episode_id))[0]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(t_values[idx])]
        env = torch.full((1,), int(env_id), dtype=torch.long, device=device)
        current = torch.as_tensor(
            data["x"][idx[0] : idx[0] + 1], dtype=torch.float32, device=device
        )
        # Simulate the episode forward in time by using the model's predictions as inputs for the next step.
        # We calculate the L2 error at each step compared to the ground truth and store the errors along with environment IDs for analysis.
        for row in idx:
            target = torch.as_tensor(data["y"][row : row + 1], dtype=torch.float32, device=device)
            current = model(current, env=env, hard=False)["pred"]
            err = torch.linalg.vector_norm(current - target, dim=-1).mean(dim=-1)
            value = float(err.cpu().item())
            errors.append(value)
            records.append((int(env_id), value))
    return (float(np.mean(errors)) if errors else float("nan")), _mean_by_env(records)


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_path: str | Path,
    out_path: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    max_rollout_episodes: int | None = None,
) -> dict[str, Any]:
    # Evaluate a model checkpoint on the given data and compute all metrics.
    dev = torch.device(device)
    model = load_checkpoint(str(checkpoint), map_location=dev).to(dev)
    data = load_npz(data_path)
    metrics = evaluate_arrays(
        model,
        data,
        device=dev,
        batch_size=batch_size,
        rollout=True,
        max_rollout_episodes=max_rollout_episodes,
    )
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint),
            "data": str(data_path),
            "batch_size": batch_size,
            "device": device,
            "metrics": metrics,
        }
        Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return metrics


def graph_threshold_sweep_checkpoint(
    checkpoint: str | Path,
    data_path: str | Path,
    device: str = "cpu",
    batch_size: int = 256,
    thresholds: np.ndarray | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    dev = torch.device(device)
    model = load_checkpoint(str(checkpoint), map_location=dev).to(dev)
    data = load_npz(data_path)
    return graph_threshold_sweep_arrays(
        model,
        data,
        device=dev,
        batch_size=batch_size,
        thresholds=thresholds,
        max_batches=max_batches,
    )


def one_step_mse_checkpoint(
    checkpoint: str | Path,
    data_path: str | Path,
    device: str = "cpu",
    batch_size: int = 256,
) -> float:
    dev = torch.device(device)
    model = load_checkpoint(str(checkpoint), map_location=dev).to(dev)
    data = load_npz(data_path)
    return one_step_mse_arrays(model, data, device=dev, batch_size=batch_size)
