from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from spartan_pong.config import N_OBJECTS, SEEN_ENVS, TOKEN_DIM


@dataclass(frozen=True)
class PreflightConfig:
    train_episodes_per_env: int
    test_episodes_per_env: int
    horizon: int
    vae_steps: int
    dynamics_steps: int
    batch_size: int
    vae_batch_size: int
    embed_dim: int
    layers: int
    device: str
    lagrangian_alpha: float = 1.001
    lambda_init: float = 20.0
    sparsity_weight: float = 1.0


def resolve_device(requested: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda", warnings
        if torch.backends.mps.is_available():
            return "mps", warnings
        warnings.append("No CUDA/MPS accelerator detected; auto selected CPU.")
        return "cpu", warnings
    if requested == "cuda" and not torch.cuda.is_available():
        warnings.append("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        warnings.append("MPS was requested but is not available.")
    return requested, warnings


def estimate(cfg: PreflightConfig) -> dict[str, object]:
    device, warnings = resolve_device(cfg.device)
    train_transitions = len(SEEN_ENVS) * cfg.train_episodes_per_env * cfg.horizon
    test_transitions = len(SEEN_ENVS) * cfg.test_episodes_per_env * cfg.horizon
    token_bytes = (train_transitions + test_transitions) * N_OBJECTS * TOKEN_DIM * 2 * 4
    graph_bytes = (train_transitions + test_transitions) * N_OBJECTS * (N_OBJECTS + 1)
    image_bytes = 32 * 32 * 3 * 4
    mask_bytes = N_OBJECTS * 32 * 32 * 4
    pixel_bytes = (train_transitions + test_transitions) * 2 * (image_bytes + mask_bytes)
    total_bytes = token_bytes + graph_bytes + pixel_bytes
    dynamics_updates = cfg.dynamics_steps * 2
    vae_object_examples = train_transitions * N_OBJECTS
    if device == "cpu" and (cfg.dynamics_steps >= 100_000 or cfg.vae_steps >= 50_000):
        warnings.append("Requested settings are large for CPU; expect a long run.")
    if cfg.embed_dim >= 512 and device == "cpu":
        warnings.append("Paper-scale embed_dim=512 on CPU is likely impractical.")
    return {
        "config": asdict(cfg),
        "resolved_device": device,
        "train_transitions": train_transitions,
        "test_transitions": test_transitions,
        "vae_object_examples": vae_object_examples,
        "vae_steps": cfg.vae_steps,
        "dynamics_steps_per_model": cfg.dynamics_steps,
        "total_dynamics_updates": dynamics_updates,
        "approx_dataset_bytes": int(total_bytes),
        "approx_dataset_gib": total_bytes / 1024**3,
        "warnings": warnings,
    }


def format_preflight(report: dict[str, object]) -> str:
    warnings = report.get("warnings", [])
    lines = [
        "SPARTAN Interventional Pong preflight",
        f"resolved_device: {report['resolved_device']}",
        f"train_transitions: {report['train_transitions']}",
        f"test_transitions: {report['test_transitions']}",
        f"vae_object_examples: {report['vae_object_examples']}",
        f"vae_steps: {report['vae_steps']}",
        f"dynamics_steps_per_model: {report['dynamics_steps_per_model']}",
        f"total_dynamics_updates: {report['total_dynamics_updates']}",
        f"approx_dataset_gib: {float(report['approx_dataset_gib']):.3f}",
    ]
    if warnings:
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("warnings: none")
    return "\n".join(lines)
