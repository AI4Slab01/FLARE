from __future__ import annotations

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
from matplotlib.colors import LinearSegmentedColormap


try:
    OUTPUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUTPUT_DIR = Path.cwd()

BASE_DIR = Path(r"C:\Users\HP\Desktop\test3")
VDP_DIR = (
    BASE_DIR
    / "vdp"
    / "outputs"
    / "offline_encoder_rollout_duffing_video_sindy_2026_06_10_16_46_25_best"
)

PNG_PATH = OUTPUT_DIR / "vdp_test_slice_grid_true_pred_10x10.png"
PDF_PATH = OUTPUT_DIR / "vdp_test_slice_grid_true_pred_10x10.pdf"

IMAGE_SHAPE = (32, 32)
STEP_REQUESTS = [0, 125, 250, 375, 500]

VIDEO_CMAP = LinearSegmentedColormap.from_list(
    "paper_dark_teal_video",
    ["#071C25", "#0C3640", "#3E6870", "#DCE4DF", "#FFFFFF"],
)

TEXT_COLOR = "#20262B"
SPINE_COLOR = "#9AA5AB"


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
            "axes.linewidth": 0.55,
        }
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def reshape_video(flat: np.ndarray, shape: tuple[int, int] = IMAGE_SHAPE) -> np.ndarray:
    arr = np.asarray(flat, dtype=np.float64)
    if arr.ndim == 3:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"Expected video as [T,H*W] or [T,H,W], got {arr.shape}")
    return arr.reshape(arr.shape[0], shape[0], shape[1])


def prediction_files() -> list[Path]:
    files = sorted(VDP_DIR.glob("offline_prediction_vdp_*.npz"))[:10]
    if len(files) != 10:
        raise FileNotFoundError(f"Expected 10 vdp prediction files in {VDP_DIR}, got {len(files)}")
    return files


def selected_steps(num_frames: int) -> list[int]:
    return [min(int(step), num_frames - 1) for step in STEP_REQUESTS]


def style_image_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#071C25")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.46)
        spine.set_color(SPINE_COLOR)


def build_figure(save: bool = True, show: bool = False) -> plt.Figure:
    configure_mpl()
    files = prediction_files()
    true_videos = []
    pred_videos = []
    for path in files:
        data = load_npz(path)
        for key in ("p_true", "p_pred"):
            if key not in data:
                raise KeyError(f"{path} does not contain {key!r}")
        true_videos.append(reshape_video(data["p_true"], IMAGE_SHAPE))
        pred_videos.append(reshape_video(data["p_pred"], IMAGE_SHAPE))

    steps = selected_steps(true_videos[0].shape[0])

    fig, axes = plt.subplots(
        2 * len(steps),
        len(true_videos),
        figsize=(11.2, 10.65),
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.062, right=0.995, bottom=0.045, top=0.945, wspace=0.030, hspace=0.035)

    for step_idx, step in enumerate(steps):
        for pair_idx, (label, videos) in enumerate((("true", true_videos), ("pred", pred_videos))):
            row = 2 * step_idx + pair_idx
            for col, video in enumerate(videos):
                ax = axes[row, col]
                ax.imshow(
                    video[step],
                    cmap=VIDEO_CMAP,
                    vmin=0.0,
                    vmax=1.0,
                    origin="lower",
                    interpolation="bicubic",
                )
                style_image_axis(ax)
                if row == 0:
                    ax.set_title(f"test {col + 1}", fontsize=8.2, color=TEXT_COLOR, pad=4)
                if col == 0:
                    row_label = f"step {step}\n{label}" if label == "true" else label
                    ax.text(
                        -0.24,
                        0.50,
                        row_label,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=7.8,
                        color=TEXT_COLOR,
                        linespacing=1.05,
                    )

    fig.text(
        0.5,
        0.985,
        "Forced Van der Pol test-set slices: true / pred",
        ha="center",
        va="top",
        fontsize=11.4,
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
