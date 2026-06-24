from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .geometry import circle_to_embedding, sphere_to_embedding
from .intensities import make_true_intensity
from .utils import collect_metric_json_csv, ensure_dir, load_json, save_json


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fixed_covariate(z_dim: int, value: float = 0.5) -> np.ndarray:
    if z_dim == 0:
        return np.empty((1, 0), dtype=np.float64)
    return np.full((1, z_dim), value, dtype=np.float64)


def plot_intensity(
    true_intensity: Any,
    estimator: Any,
    output_dir: str | Path,
    z_dim: int,
    covariate_value: float = 0.5,
    grid_size: int = 120,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    support = true_intensity.metadata["support"]
    if support == "euclidean1d":
        return [_plot_euclidean_1d(true_intensity, estimator, output_dir, z_dim, covariate_value, grid_size)]
    if support == "euclidean2d":
        return [_plot_euclidean_2d(true_intensity, estimator, output_dir, z_dim, covariate_value, grid_size)]
    if support == "circle":
        return [_plot_circle(true_intensity, estimator, output_dir, z_dim, covariate_value, grid_size)]
    if support == "sphere":
        return _plot_sphere(true_intensity, estimator, output_dir, z_dim, covariate_value)
    raise ValueError(f"Unknown support: {support}")


def best_intensity_plot_dir(metadata: dict[str, Any], results_dir: str | Path = "results") -> Path:
    return (
        Path(results_dir)
        / "plots"
        / "intensity"
        / str(metadata["scenario"])
        / str(metadata["support"])
        / f"zdim_{int(metadata['z_dim'])}"
        / f"n_{int(metadata['n'])}"
        / str(metadata["method"])
    )


def plot_best_repeat_intensity(
    metadata: dict[str, Any],
    results_dir: str | Path = "results",
    current_estimator: Any | None = None,
    current_repetition: int | None = None,
    true_intensity: Any | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Plot the intensity only for the currently selected best repetition."""
    output_dir = best_intensity_plot_dir(metadata, results_dir)
    marker_path = output_dir / "best_repeat_plot_metadata.json"
    best_repetition = int(metadata.get("best_repetition", -1))
    best_model_path = str(metadata.get("best_model_path") or "")
    if best_repetition < 0:
        return {}
    if not force and _best_plot_is_current(marker_path, metadata):
        existing = load_json(marker_path)
        return {
            "best_intensity_plot_dir": str(output_dir),
            "best_intensity_plot_repetition": best_repetition,
            "best_intensity_plot_metadata": str(marker_path),
            "intensity_plot_created_this_run": False,
            "best_intensity_plot_files": existing.get("plot_files", []),
        }

    if true_intensity is None:
        true_intensity = _true_intensity_from_metadata(metadata)
    estimator = current_estimator
    if current_repetition is None or int(current_repetition) != best_repetition:
        estimator = _load_estimator_for_plot(best_model_path)
    if estimator is None:
        return {}

    paths = plot_intensity(true_intensity, estimator, output_dir, z_dim=int(metadata["z_dim"]))
    payload = {
        "scenario": metadata.get("scenario"),
        "support": metadata.get("support"),
        "z_dim": metadata.get("z_dim"),
        "n": metadata.get("n"),
        "method": metadata.get("method"),
        "setting_hash": metadata.get("setting_hash"),
        "best_repetition": best_repetition,
        "best_squared_hellinger": metadata.get("best_squared_hellinger"),
        "best_model_path": best_model_path,
        "plot_files": [str(path) for path in paths],
    }
    save_json(payload, marker_path)
    return {
        "best_intensity_plot_dir": str(output_dir),
        "best_intensity_plot_repetition": best_repetition,
        "best_intensity_plot_metadata": str(marker_path),
        "best_intensity_plot_files": payload["plot_files"],
        "intensity_plot_created_this_run": True,
    }


def _best_plot_is_current(marker_path: Path, metadata: dict[str, Any]) -> bool:
    if not marker_path.exists():
        return False
    try:
        existing = load_json(marker_path)
    except Exception:
        return False
    if int(existing.get("best_repetition", -999)) != int(metadata.get("best_repetition", -1)):
        return False
    if str(existing.get("best_model_path") or "") != str(metadata.get("best_model_path") or ""):
        return False
    return all(Path(path).exists() for path in existing.get("plot_files", []))


def _true_intensity_from_metadata(metadata: dict[str, Any]) -> Any:
    cfg = metadata.get("config") or {}
    exp = cfg.get("experiment") or {}
    true_intensity = make_true_intensity(
        scenario=str(metadata["scenario"]),
        support=str(metadata["support"]),
        z_dim=int(metadata["z_dim"]),
        expected_count=float(exp.get("expected_count", 30.0)),
        epsilon=float(exp.get("epsilon", 1e-4)),
        beta=float(exp.get("beta", 2.0)),
        alpha=exp.get("alpha"),
    )
    true_intensity.metadata.update({"scenario": str(metadata["scenario"])})
    return true_intensity


def _load_estimator_for_plot(path: str) -> Any | None:
    if not path:
        return None
    model_path = Path(path)
    if not model_path.exists():
        return None
    if model_path.suffix == ".pt":
        from .methods.dnn_npmle import DNNNPMLEEstimator

        return DNNNPMLEEstimator.load(model_path)
    from .methods.base import Estimator

    return Estimator.load(model_path)


def _plot_euclidean_1d(true_intensity, estimator, output_dir, z_dim, covariate_value, grid_size):
    plt = _mpl()
    x = np.linspace(0.0, 1.0, grid_size).reshape(-1, 1)
    z = fixed_covariate(z_dim, covariate_value)
    true_vals = true_intensity.evaluate(x, z)
    pred_vals = estimator.predict(x, z)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x[:, 0], true_vals, label="True", linewidth=2)
    ax.plot(x[:, 0], pred_vals, label="Estimated", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True, alpha=0.25)
    path = Path(output_dir) / "intensity_1d.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_euclidean_2d(true_intensity, estimator, output_dir, z_dim, covariate_value, grid_size):
    plt = _mpl()
    m = min(grid_size, 90)
    xs = np.linspace(0.0, 1.0, m)
    ys = np.linspace(0.0, 1.0, m)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([xv.ravel(), yv.ravel()])
    z = fixed_covariate(z_dim, covariate_value)
    true_vals = true_intensity.evaluate(points, z).reshape(m, m)
    pred_vals = estimator.predict(points, z).reshape(m, m)
    err = np.abs(pred_vals - true_vals)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, vals, title in zip(axes, [true_vals, pred_vals, err], ["True", "Estimated", "Absolute error"]):
        im = ax.imshow(vals, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = Path(output_dir) / "intensity_2d_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_circle(true_intensity, estimator, output_dir, z_dim, covariate_value, grid_size):
    plt = _mpl()
    theta = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False).reshape(-1, 1)
    z = fixed_covariate(z_dim, covariate_value)
    true_vals = true_intensity.evaluate(theta, z)
    pred_vals = estimator.predict(theta, z)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(theta[:, 0], true_vals, label="True", linewidth=2)
    axes[0].plot(theta[:, 0], pred_vals, label="Estimated", linewidth=2)
    axes[0].set_xlabel("theta")
    axes[0].set_ylabel("Intensity")
    axes[0].legend()
    xy = circle_to_embedding(theta[:, 0])
    scatter = axes[1].scatter(xy[:, 0], xy[:, 1], c=pred_vals, s=18, cmap="viridis")
    axes[1].set_aspect("equal")
    axes[1].set_title("Estimated on circle")
    axes[1].axis("off")
    fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
    path = Path(output_dir) / "intensity_circle.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_sphere(true_intensity, estimator, output_dir, z_dim, covariate_value):
    plt = _mpl()
    theta = np.linspace(0.0, np.pi, 64)
    phi = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    points = np.column_stack([th.ravel(), ph.ravel()])
    z = fixed_covariate(z_dim, covariate_value)
    true_vals = true_intensity.evaluate(points, z).reshape(th.shape)
    pred_vals = estimator.predict(points, z).reshape(th.shape)
    err = np.abs(pred_vals - true_vals)

    heatmap_path = Path(output_dir) / "intensity_sphere_equirectangular.png"
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    extent = [0.0, 2.0 * np.pi, 0.0, np.pi]
    for ax, vals, title in zip(axes, [true_vals, pred_vals, err], ["True", "Estimated", "Absolute error"]):
        im = ax.imshow(vals, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("phi")
        ax.set_ylabel("theta")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    surface_path = Path(output_dir) / "intensity_sphere_surface.png"
    try:
        from matplotlib import cm, colors

        embedded = sphere_to_embedding(points).reshape(th.shape + (3,))
        fig = plt.figure(figsize=(12, 4), constrained_layout=True)
        for i, (vals, title) in enumerate([(true_vals, "True"), (pred_vals, "Estimated"), (err, "Absolute error")], start=1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            norm = colors.Normalize(vmin=float(np.min(vals)), vmax=float(np.max(vals)))
            facecolors = cm.viridis(norm(vals))
            ax.plot_surface(
                embedded[:, :, 0],
                embedded[:, :, 1],
                embedded[:, :, 2],
                facecolors=facecolors,
                linewidth=0,
                antialiased=False,
                shade=False,
            )
            ax.set_title(title)
            ax.set_axis_off()
            ax.set_box_aspect([1, 1, 1])
        fig.savefig(surface_path, dpi=180)
        plt.close(fig)
    except Exception:
        surface_path = heatmap_path
    return [heatmap_path, surface_path]


def create_summary_tables(results_dir: str | Path = "results") -> None:
    import pandas as pd

    results_dir = Path(results_dir)
    metrics_path = results_dir / "metrics" / "all_metrics.csv"
    if not metrics_path.exists():
        collect_metric_json(results_dir)
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)
    group_cols = ["scenario", "support_type", "event_dim", "manifold_type", "z_dim", "n", "method"]
    summary = (
        df.groupby(group_cols, dropna=False)["squared_hellinger"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    q = df.groupby(group_cols, dropna=False)["squared_hellinger"].quantile([0.25, 0.75]).unstack().reset_index()
    q = q.rename(columns={0.25: "q25", 0.75: "q75"})
    summary = summary.merge(q, on=group_cols, how="left")
    requested_cols = [
        "bandwidth_selection",
        "bandwidth_theory_mode",
        "bandwidth_cv_folds",
        "kernel_cv_max_replicates",
        "kernel_cv_max_events_per_replicate",
        "validation_z_chunk_size",
        "validation_quadrature_method",
        "boundary_correction",
        "bandwidth_x",
        "bandwidth_z",
        "bandwidth_manifold",
        "bandwidth_cv_score",
        "model_saved",
        "best_repeat",
        "best_model_path",
        "save_model_policy",
    ]
    present = [col for col in requested_cols if col in df.columns]
    if present:
        def first_non_null(series):
            non_null = series.dropna()
            return non_null.iloc[0] if len(non_null) else pd.NA

        agg_map = {}
        for col in present:
            if col in {"bandwidth_x", "bandwidth_z", "bandwidth_manifold", "bandwidth_cv_score"}:
                agg_map[col] = "mean"
            else:
                agg_map[col] = first_non_null
        extra = df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
        summary = summary.merge(extra, on=group_cols, how="left")
    summary.to_csv(results_dir / "summary_metrics.csv", index=False)
    summary.to_markdown(results_dir / "summary_table.md", index=False)
    with (results_dir / "summary_table.tex").open("w", encoding="utf-8") as f:
        f.write(summary.to_latex(index=False, float_format="%.4g"))


def collect_metric_json(results_dir: str | Path = "results") -> None:
    collect_metric_json_csv(results_dir)


def create_boxplots(results_dir: str | Path = "results") -> None:
    import pandas as pd

    plt = _mpl()
    results_dir = Path(results_dir)
    metrics_path = results_dir / "metrics" / "all_metrics.csv"
    if not metrics_path.exists():
        collect_metric_json(results_dir)
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)
    out_dir = ensure_dir(results_dir / "plots")
    combo_cols = ["scenario", "support", "z_dim"]
    for combo, sub in df.groupby(combo_cols, dropna=False):
        scenario, support, z_dim = combo
        if sub["n"].nunique() < 1:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = list(sub["method"].dropna().unique())
        n_values = sorted(sub["n"].unique())
        offsets = np.linspace(-0.16, 0.16, max(len(methods), 1))
        for offset, method in zip(offsets, methods):
            method_sub = sub[sub["method"] == method]
            data = [method_sub[method_sub["n"] == n]["squared_hellinger"].dropna().values for n in n_values]
            positions = np.array(n_values, dtype=float) * np.exp(offset)
            ax.boxplot(data, positions=positions, widths=np.array(n_values) * 0.05, manage_ticks=False)
            med = [np.median(d) if len(d) else np.nan for d in data]
            ax.plot(positions, med, marker="o", linestyle="-", label=method)
        ref = sub[sub["method"].str.contains("dnn", na=False)]
        if ref.empty:
            ref = sub
        gamma = float(ref["theory_rate_exponent"].dropna().iloc[0]) if ref["theory_rate_exponent"].notna().any() else 0.5
        first_n = min(n_values)
        first_med = float(ref[ref["n"] == first_n]["squared_hellinger"].median())
        if np.isfinite(first_med) and first_med > 0:
            ns = np.asarray(n_values, dtype=float)
            ax.plot(ns, first_med * (ns / first_n) ** (-gamma), "k--", label=f"n^(-{gamma:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of replicated processes n")
        ax.set_ylabel("squared Hellinger")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        stem = f"boxplot_{scenario}_{support}_zdim_{int(z_dim)}"
        fig.savefig(out_dir / f"{stem}.png", dpi=180)
        fig.savefig(out_dir / f"{stem}.pdf")
        plt.close(fig)
