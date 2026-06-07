from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from spartan_pong.data import load_npz
from spartan_pong.evaluate import empty_graph_baseline, graph_threshold_sweep_checkpoint


def analyze_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    device: str = "cpu",
    batch_size: int | None = None,
    max_batches: int | None = None,
    skip_threshold_sweep: bool = False,
) -> dict[str, str]:
    run = Path(run_dir)
    out = Path(out_dir) if out_dir is not None else run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    resolved_batch_size = batch_size or _batch_size_from_config(run)

    histories = {
        "SPARTAN": _read_history(run / "spartan" / "history.jsonl"),
        "Transformer": _read_history(run / "transformer" / "history.jsonl"),
    }
    evals = {
        "SPARTAN final": _read_metrics(run / "spartan" / "eval.json"),
        "SPARTAN best": _read_metrics(run / "spartan" / "best_eval.json"),
        "Transformer final": _read_metrics(run / "transformer" / "eval.json"),
        "Transformer best": _read_metrics(run / "transformer" / "best_eval.json"),
    }
    artifacts: dict[str, str] = {}

    training_curves = out / "training_curves.png"
    _plot_training_curves(histories, training_curves)
    artifacts["training_curves"] = str(training_curves)

    metric_bars = out / "final_metric_bars.png"
    _plot_final_metric_bars(evals, metric_bars)
    artifacts["final_metric_bars"] = str(metric_bars)

    test_tokens = run / "data" / "test_tokens.npz"
    baseline = empty_graph_baseline(load_npz(test_tokens))
    baseline_path = out / "empty_graph_baseline.json"
    baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True))
    artifacts["empty_graph_baseline"] = str(baseline_path)

    sweeps: dict[str, Any] = {}
    if not skip_threshold_sweep:
        sweeps = _threshold_sweeps(run, test_tokens, device, resolved_batch_size, max_batches)
        sweep_path = out / "graph_threshold_sweep.json"
        sweep_path.write_text(json.dumps(sweeps, indent=2, sort_keys=True))
        artifacts["graph_threshold_sweep"] = str(sweep_path)

        sweep_plot = out / "graph_threshold_sweep.png"
        _plot_threshold_sweeps(sweeps, sweep_plot)
        artifacts["graph_threshold_sweep_plot"] = str(sweep_plot)

    summary_path = out / "summary.md"
    summary_path.write_text(
        _summary_markdown(
            evals,
            baseline,
            sweeps,
            resolved_batch_size=resolved_batch_size,
            max_batches=max_batches,
            skipped_threshold_sweep=skip_threshold_sweep,
        )
    )
    artifacts["summary"] = str(summary_path)
    return artifacts


def _batch_size_from_config(run: Path) -> int:
    config_path = run / "run_config.json"
    if not config_path.exists():
        return 256
    config = json.loads(config_path.read_text())
    return int(config.get("batch_size", 256))


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics", payload)
    return metrics if isinstance(metrics, dict) else None


def _threshold_sweeps(
    run: Path,
    test_tokens: Path,
    device: str,
    batch_size: int,
    max_batches: int | None,
) -> dict[str, Any]:
    sweeps = {}
    for model_name, checkpoint in (
        ("SPARTAN", run / "spartan" / "checkpoint.pt"),
        ("Transformer", run / "transformer" / "checkpoint.pt"),
    ):
        if checkpoint.exists():
            sweeps[model_name] = graph_threshold_sweep_checkpoint(
                checkpoint,
                test_tokens,
                device=device,
                batch_size=batch_size,
                max_batches=max_batches,
            )
    return sweeps


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def _plot_training_curves(histories: dict[str, list[dict[str, Any]]], out: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    axes_flat = axes.ravel()
    specs = [
        ("one_step_l2", None, "One-Step L2", "error"),
        ("train_mse", None, "Train MSE", "mse"),
        ("shd", None, "SHD", "shd"),
        ("active_edges", "target_edges", "Active vs Target Edges", "edges"),
        ("robust_pct_l2", "robust_pct", "Non-Causal Removal L2", "% change"),
        ("robust_pct_mse", None, "Non-Causal Removal MSE", "% change"),
        ("lambda", None, "SPARTAN Lambda", "lambda"),
        ("sparsity", None, "SPARTAN Sparsity", "path count"),
    ]
    for ax, (key, fallback, title, ylabel) in zip(axes_flat, specs, strict=True):
        _plot_history_metric(ax, histories, key, fallback=fallback)
        if key == "active_edges":
            _plot_target_edges(ax, histories)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if ax.has_data():
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_history_metric(
    ax: Any,
    histories: dict[str, list[dict[str, Any]]],
    key: str,
    fallback: str | None = None,
) -> None:
    for model_name, rows in histories.items():
        if key in {"lambda", "sparsity"} and model_name != "SPARTAN":
            continue
        points = _history_points(rows, key, fallback=fallback)
        if points:
            steps, values = zip(*points, strict=True)
            ax.plot(steps, values, label=model_name)


def _plot_target_edges(ax: Any, histories: dict[str, list[dict[str, Any]]]) -> None:
    for model_name, rows in histories.items():
        points = _history_points(rows, "target_edges")
        if points:
            steps, values = zip(*points, strict=True)
            ax.plot(steps, values, linestyle="--", linewidth=1, label=f"{model_name} target")
            return


def _history_points(
    rows: list[dict[str, Any]], key: str, fallback: str | None = None
) -> list[tuple[float, float]]:
    points = []
    for row in rows:
        value = row.get(key)
        if value is None and fallback is not None:
            value = row.get(fallback)
        step = row.get("step")
        if value is not None and step is not None:
            points.append((float(step), float(value)))
    return points


def _plot_final_metric_bars(evals: dict[str, dict[str, Any] | None], out: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    specs = [
        ("rollout_l2", "pred_l2", "Rollout L2"),
        ("one_step_l2", None, "One-Step L2"),
        ("shd", None, "SHD"),
        ("active_edges", None, "Active Edges"),
        ("robust_pct_l2", "robust_pct", "Non-Causal Removal L2"),
        ("robust_pct_mse", None, "Non-Causal Removal MSE"),
    ]
    for ax, (key, fallback, title) in zip(axes.ravel(), specs, strict=True):
        labels = []
        values = []
        for label, metrics in evals.items():
            value = _metric(metrics, key, fallback=fallback)
            if value is not None:
                labels.append(label)
                values.append(value)
        if values:
            ax.bar(labels, values)
            ax.tick_params(axis="x", labelrotation=30)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_threshold_sweeps(sweeps: dict[str, Any], out: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for model_name, sweep in sweeps.items():
        rows = sweep.get("thresholds", [])
        thresholds = [float(row["threshold"]) for row in rows]
        shd = [float(row["shd"]) for row in rows]
        active = [float(row["active_edges"]) for row in rows]
        if thresholds:
            axes[0].plot(thresholds, shd, label=model_name)
            axes[1].plot(thresholds, active, label=model_name)
    target_edges = _first_target_edges(sweeps)
    if target_edges is not None:
        axes[1].axhline(target_edges, linestyle="--", color="black", label="target edges")
    axes[0].set_title("SHD vs Threshold")
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("SHD")
    axes[1].set_title("Active Edges vs Threshold")
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("edges")
    for ax in axes:
        ax.grid(alpha=0.25)
        if ax.has_data():
            ax.legend(fontsize=8)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _first_target_edges(sweeps: dict[str, Any]) -> float | None:
    for sweep in sweeps.values():
        rows = sweep.get("thresholds", [])
        if rows:
            return float(rows[0]["target_edges"])
    return None


def _metric(
    metrics: dict[str, Any] | None, key: str, fallback: str | None = None
) -> float | None:
    if metrics is None:
        return None
    value = metrics.get(key)
    if value is None and fallback is not None:
        value = metrics.get(fallback)
    return None if value is None else float(value)


def _summary_markdown(
    evals: dict[str, dict[str, Any] | None],
    baseline: dict[str, Any],
    sweeps: dict[str, Any],
    resolved_batch_size: int,
    max_batches: int | None,
    skipped_threshold_sweep: bool,
) -> str:
    lines = [
        "# Interventional Pong Run Analysis",
        "",
        "## Inputs",
        "",
        f"- Batch size: `{resolved_batch_size}`",
        f"- Threshold sweep max batches: `{max_batches}`",
        "- Training-history curves use the lightweight validation metrics saved during training.",
        "- Final metric tables use the full evaluation JSON files when present.",
        "",
        "## Final And Best Metrics",
        "",
        "| Checkpoint | Rollout L2 | One-step L2 | SHD | Active edges | Target edges | Removal L2 pct | Removal MSE pct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in evals.items():
        if metrics is None:
            continue
        lines.append(
            f"| {label} | {_fmt(_metric(metrics, 'rollout_l2', 'pred_l2'))} | "
            f"{_fmt(_metric(metrics, 'one_step_l2'))} | {_fmt(_metric(metrics, 'shd'))} | "
            f"{_fmt(_metric(metrics, 'active_edges'))} | {_fmt(_metric(metrics, 'target_edges'))} | "
            f"{_fmt(_metric(metrics, 'robust_pct_l2', 'robust_pct'))} | "
            f"{_fmt(_metric(metrics, 'robust_pct_mse'))} |"
        )
    lines.extend(
        [
            "",
            "## Empty-Graph Baseline",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| SHD | {_fmt(float(baseline['shd']))} |",
            f"| Active edges | {_fmt(float(baseline['active_edges']))} |",
            f"| Target edges | {_fmt(float(baseline['target_edges']))} |",
            "",
        ]
    )
    if skipped_threshold_sweep:
        lines.extend(["## Threshold Sweep", "", "Threshold sweep skipped.", ""])
    else:
        lines.extend(
            [
                "## Threshold Sweep",
                "",
                "| Model | Best threshold | Best SHD | Active edges | Target edges | Edge surplus |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for model_name, sweep in sweeps.items():
            best = sweep.get("best")
            if best is None:
                continue
            lines.append(
                f"| {model_name} | {_fmt(float(best['threshold']))} | "
                f"{_fmt(float(best['shd']))} | {_fmt(float(best['active_edges']))} | "
                f"{_fmt(float(best['target_edges']))} | {_fmt(float(best['edge_surplus']))} |"
            )
        lines.append("")
    lines.extend(["## Interpretation", ""])
    lines.extend(_interpretation_lines(evals, baseline, sweeps))
    lines.append("")
    return "\n".join(lines)


def _interpretation_lines(
    evals: dict[str, dict[str, Any] | None], baseline: dict[str, Any], sweeps: dict[str, Any]
) -> list[str]:
    lines = []
    spartan = evals.get("SPARTAN final")
    transformer = evals.get("Transformer final")
    if spartan is not None and transformer is not None:
        lines.append(_winner_line("Rollout L2", spartan, transformer, "rollout_l2", "pred_l2"))
        lines.append(_winner_line("One-step L2", spartan, transformer, "one_step_l2"))
        lines.append(
            _winner_line("Non-causal removal L2", spartan, transformer, "robust_pct_l2", "robust_pct")
        )
    transformer_shd = _metric(transformer, "shd") if transformer is not None else None
    empty_shd = float(baseline["shd"])
    if transformer_shd is not None and np.isclose(transformer_shd, empty_shd):
        lines.append(
            "- Transformer final SHD matches the empty-graph baseline, so its selected graph "
            "should be treated as degenerate."
        )
    spartan_active = _metric(spartan, "active_edges") if spartan is not None else None
    target_edges = _metric(spartan, "target_edges") if spartan is not None else None
    if spartan_active is not None and target_edges is not None and spartan_active < target_edges * 0.25:
        lines.append(
            "- SPARTAN final graph is severely underconnected relative to target edges; "
            "use SHD as a diagnostic, not as successful graph recovery."
        )
    if sweeps:
        lines.append("- Use the threshold sweep plot to separate thresholding artifacts from probability collapse.")
    return lines


def _winner_line(
    name: str,
    spartan: dict[str, Any],
    transformer: dict[str, Any],
    key: str,
    fallback: str | None = None,
) -> str:
    spartan_value = _metric(spartan, key, fallback=fallback)
    transformer_value = _metric(transformer, key, fallback=fallback)
    if spartan_value is None or transformer_value is None:
        return f"- {name}: unavailable."
    winner = "SPARTAN" if spartan_value < transformer_value else "Transformer"
    return (
        f"- {name}: {winner} is lower "
        f"(SPARTAN {spartan_value:.4f} vs Transformer {transformer_value:.4f})."
    )


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"
