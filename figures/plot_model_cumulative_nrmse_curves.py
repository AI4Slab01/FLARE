from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

try:
    from IPython import get_ipython

    IN_IPYTHON = get_ipython() is not None
except Exception:
    IN_IPYTHON = False

if not IN_IPYTHON:
    mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter


try:
    OUTPUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUTPUT_DIR = Path.cwd()

PNG_PATH = OUTPUT_DIR / "sheet_robot_agg_model_cumulative_nrmse_curves.png"
PDF_PATH = OUTPUT_DIR / "sheet_robot_agg_model_cumulative_nrmse_curves.pdf"
CSV_PATH = OUTPUT_DIR / "sheet_robot_agg_model_cumulative_nrmse_curves.csv"


# Main switch: evaluate cumulative NRMSE every STEP_STRIDE prediction steps.
STEP_STRIDE = 20

# Very large divergent curves are clipped only for visualization.  Raw values
# written to CSV remain unclipped.
Y_CLIP_PERCENT = 1.0e4

OTHER_MODELS_ROOT = Path(r"D:\FLARE\model test\other models")
FLARE_TEST_ROOT = Path(r"C:\Users\HP\Desktop\test1+test2")
FLARE_EXTRAP_ROOT = Path(r"C:\Users\HP\Desktop\test2+")


DATASETS = {
    "sheet": {
        "title": "Thin-plate heating",
        "flare_test_parent": FLARE_TEST_ROOT / "AAAsheet" / "outputs",
        "flare_extrap_parent": FLARE_EXTRAP_ROOT / "sheet",
    },
    "robot": {
        "title": "Robotic arm",
        "flare_test_parent": FLARE_TEST_ROOT / "AAArobot" / "outputs",
        "flare_extrap_parent": FLARE_EXTRAP_ROOT / "robot",
    },
    "agg": {
        "title": "Two-story frame",
        "flare_test_parent": FLARE_TEST_ROOT / "AAAagg" / "outputs",
        "flare_extrap_parent": FLARE_EXTRAP_ROOT / "agg",
    },
}


# The 9 benchmark folders plus FLARE itself.  Display order is chosen so that
# interpretable / equation-based models are grouped before black-box sequence models.
BASELINE_MODELS = [
    ("AE-SINDY-f", "AE-SINDY-f"),
    ("DMDc", "DMDc"),
    ("POD-SINDYc", "POD-SINDYc"),
    ("Deep Koopman", "Deep Koopman"),
    ("SUBNET", "SUBNET"),
    ("TiDE", "TiDE"),
    ("TFT", "TFT"),
    ("TCN", "TCN"),
    ("GRU", "GRU"),
]

MODEL_ORDER = ["FLARE"] + [name for name, _ in BASELINE_MODELS]


MODEL_STYLE = {
    "FLARE": dict(color="#111111", lw=2.25, ls="-", zorder=20),
    "AE-SINDY-f": dict(color="#B6424A", lw=1.35, ls="-"),
    "DMDc": dict(color="#7B8790", lw=1.35, ls="-"),
    "POD-SINDYc": dict(color="#28607D", lw=1.45, ls="-"),
    "Deep Koopman": dict(color="#7B5EA7", lw=1.35, ls="-"),
    "SUBNET": dict(color="#2F6F6D", lw=1.35, ls="-"),
    "TiDE": dict(color="#C97928", lw=1.35, ls="-"),
    "TFT": dict(color="#3A8FB7", lw=1.35, ls="--"),
    "TCN": dict(color="#8B5A2B", lw=1.35, ls="--"),
    "GRU": dict(color="#D07BA6", lw=1.35, ls="--"),
}


TEXT_COLOR = "#20262B"
SPINE_COLOR = "#7B8790"
GRID_COLOR = "#D5DDE1"
CLIP_COLOR = "#B6424A"


@dataclass(frozen=True)
class Curve:
    steps: np.ndarray
    raw: np.ndarray
    plot: np.ndarray
    clipped: bool
    files: int
    directory: Path


def configure_mpl() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.linewidth": 0.7,
        }
    )


def prediction_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("offline_prediction_*.npz"))
    if len(files) < 1:
        raise FileNotFoundError(f"No offline_prediction_*.npz files found in {directory}")
    return files[:10]


def newest_prediction_dir(parent: Path, name_contains: str | None = None, suffix: str | None = None) -> Path:
    if not parent.exists():
        raise FileNotFoundError(parent)
    candidates: list[Path] = []
    for d in parent.iterdir():
        if not d.is_dir():
            continue
        if name_contains and name_contains not in d.name:
            continue
        if suffix and not d.name.endswith(suffix):
            continue
        if list(d.glob("offline_prediction_*.npz")):
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"No matching prediction directory under {parent}")
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def flare_dir(dataset: str, split: str) -> Path:
    cfg = DATASETS[dataset]
    if split == "test":
        return newest_prediction_dir(cfg["flare_test_parent"], name_contains="offline_encoder_rollout")
    if split == "test10":
        return newest_prediction_dir(cfg["flare_extrap_parent"], name_contains="offline_encoder_rollout")
    raise ValueError(split)


def baseline_dir(model_folder: str, dataset: str, split: str) -> Path:
    parent = OTHER_MODELS_ROOT / model_folder / "outputs" / dataset
    return newest_prediction_dir(parent, suffix=f"_{split}")


def cumulative_nrmse_for_file(path: Path, steps: np.ndarray) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "p_true" not in data.files or "p_pred" not in data.files:
        raise KeyError(f"{path} must contain p_true and p_pred.")

    truth = np.asarray(data["p_true"], dtype=np.float64)
    pred = np.asarray(data["p_pred"], dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 2:
        raise ValueError(f"Expected matching [T,D] arrays in {path}, got {truth.shape}/{pred.shape}")

    values = []
    diff = pred - truth
    for h in steps:
        # Exclude the anchor/initial-condition frame, matching the previous
        # cumulative-horizon convention.
        error_seg = diff[1 : int(h) + 1]
        truth_seg = truth[1 : int(h) + 1]
        if error_seg.size == 0 or truth_seg.size == 0:
            values.append(np.nan)
            continue
        if not np.all(np.isfinite(error_seg)):
            values.append(np.inf)
            continue
        err_norm = float(np.linalg.norm(error_seg.ravel()))
        true_norm = float(np.linalg.norm(truth_seg.ravel()))
        denom = max(true_norm, 1e-12)
        values.append(100.0 * err_norm / denom)
    return np.asarray(values, dtype=np.float64)


def cumulative_curve(directory: Path, stride: int = STEP_STRIDE) -> Curve:
    files = prediction_files(directory)
    lengths = []
    for path in files:
        data = np.load(path, allow_pickle=True)
        lengths.append(int(np.asarray(data["p_true"]).shape[0]) - 1)
    predicted_steps = min(lengths)
    if predicted_steps < 1:
        raise ValueError(f"Need at least one predicted step in {directory}")

    steps = np.arange(stride, predicted_steps + 1, stride, dtype=int)
    if steps.size == 0 or int(steps[-1]) != predicted_steps:
        steps = np.unique(np.append(steps, predicted_steps)).astype(int)

    curves = []
    for path in files:
        curves.append(cumulative_nrmse_for_file(path, steps))
    stack = np.stack(curves, axis=0)
    raw = np.mean(stack, axis=0)
    clipped = bool(np.any(~np.isfinite(raw)) or np.any(raw > Y_CLIP_PERCENT))
    plot = np.asarray(raw, dtype=np.float64).copy()
    plot[~np.isfinite(plot)] = Y_CLIP_PERCENT
    plot = np.clip(plot, 1e-12, Y_CLIP_PERCENT)
    return Curve(steps=steps, raw=raw, plot=plot, clipped=clipped, files=len(files), directory=directory)


def load_all_curves() -> dict[tuple[str, str, str], Curve]:
    curves: dict[tuple[str, str, str], Curve] = {}
    for dataset in DATASETS:
        for split in ("test", "test10"):
            curves[(dataset, split, "FLARE")] = cumulative_curve(flare_dir(dataset, split))
            for model_name, folder in BASELINE_MODELS:
                curves[(dataset, split, model_name)] = cumulative_curve(baseline_dir(folder, dataset, split))
    return curves


def panel_ylim(panel_curves: list[Curve]) -> tuple[float, float]:
    vals = []
    clipped = False
    for curve in panel_curves:
        finite = curve.plot[np.isfinite(curve.plot) & (curve.plot > 0)]
        if finite.size:
            vals.append(finite)
        clipped = clipped or curve.clipped
    if not vals:
        return 1e-3, Y_CLIP_PERCENT
    all_vals = np.concatenate(vals)
    ymin = max(1e-4, 10 ** math.floor(math.log10(max(float(np.nanmin(all_vals)) * 0.75, 1e-12))))
    ymax_data = float(np.nanmax(all_vals)) * 1.35
    ymax = 10 ** math.ceil(math.log10(max(ymax_data, ymin * 10.0)))
    if clipped:
        ymax = Y_CLIP_PERCENT
    ymax = min(max(ymax, ymin * 10.0), Y_CLIP_PERCENT)
    return ymin, ymax


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=7.5, width=0.65, length=3.0)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.55, alpha=0.82)
    ax.grid(True, which="minor", color=GRID_COLOR, linewidth=0.35, alpha=0.34)
    ax.set_axisbelow(True)


def write_csv(curves: dict[tuple[str, str, str], Curve]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "split", "model", "step", "cumulative_nrmse_percent", "plot_value_clipped", "source_dir"])
        for dataset in DATASETS:
            for split in ("test", "test10"):
                for model in MODEL_ORDER:
                    curve = curves[(dataset, split, model)]
                    for step, raw, plot in zip(curve.steps, curve.raw, curve.plot):
                        writer.writerow([dataset, split, model, int(step), raw, plot, str(curve.directory)])


def build_figure(save: bool = True, show: bool = False) -> plt.Figure:
    configure_mpl()
    curves = load_all_curves()
    write_csv(curves)

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.25), constrained_layout=False)
    plt.subplots_adjust(left=0.070, right=0.992, bottom=0.185, top=0.905, wspace=0.255, hspace=0.335)

    split_title = {"test": "test set", "test10": "extrapolation"}
    split_row = {"test": 0, "test10": 1}

    for col, (dataset, cfg) in enumerate(DATASETS.items()):
        for split in ("test", "test10"):
            ax = axes[split_row[split], col]
            panel_curves = [curves[(dataset, split, model)] for model in MODEL_ORDER]
            for model in MODEL_ORDER:
                curve = curves[(dataset, split, model)]
                style = MODEL_STYLE[model]
                ax.plot(curve.steps, curve.plot, label=model, **style)
                if curve.clipped:
                    clipped_mask = (~np.isfinite(curve.raw)) | (curve.raw > Y_CLIP_PERCENT)
                    if np.any(clipped_mask):
                        ax.scatter(
                            curve.steps[clipped_mask],
                            np.full(np.sum(clipped_mask), Y_CLIP_PERCENT),
                            marker="^",
                            s=12,
                            color=style["color"],
                            edgecolors="white",
                            linewidths=0.25,
                            zorder=style.get("zorder", 8) + 1,
                        )

            ax.set_yscale("log")
            ymin, ymax = panel_ylim(panel_curves)
            ax.set_ylim(ymin, ymax)
            max_step = max(int(curve.steps[-1]) for curve in panel_curves)
            ax.set_xlim(0, max_step)
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"{cfg['title']} ({split_title[split]})", fontsize=10.4, color=TEXT_COLOR, pad=6)
            ax.set_xlabel("prediction step", fontsize=8.6, color=TEXT_COLOR, labelpad=2)
            if col == 0:
                ax.set_ylabel("Cumulative NRMSE (%)", fontsize=8.6, color=TEXT_COLOR, labelpad=2)
            style_axis(ax)
            if any(curve.clipped for curve in panel_curves):
                ax.text(
                    0.985,
                    0.955,
                    r"clipped at $10^4$%",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.8,
                    color=CLIP_COLOR,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=5,
        frameon=False,
        fontsize=8.1,
        handlelength=1.9,
        columnspacing=1.2,
        handletextpad=0.42,
    )

    fig.text(
        0.072,
        0.965,
        f"Cumulative rollout error sampled every {STEP_STRIDE} prediction steps",
        ha="left",
        va="top",
        fontsize=10.1,
        color=TEXT_COLOR,
    )

    if save:
        fig.savefig(PNG_PATH, bbox_inches="tight")
        fig.savefig(PDF_PATH, bbox_inches="tight")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    fig = build_figure(save=True, show=False)
    plt.close(fig)
    print(PNG_PATH)
    print(PDF_PATH)
    print(CSV_PATH)
