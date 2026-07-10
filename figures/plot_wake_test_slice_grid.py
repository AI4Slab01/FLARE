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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


try:
    OUTPUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUTPUT_DIR = Path.cwd()

BASE_DIR = Path(r'/Users/zhuyi/Desktop/FLARE data/results')
WAKE_DIR = (
    BASE_DIR
    / "wake"
    / "outputs"
    / "offline_encoder_rollout_cylinder_flow_32x96_window_sindy_2026_06_15_15_14_05_best"
)

PNG_PATH_1_5 = OUTPUT_DIR / "wake_test_slice_grid_true_pred_tests_1_5.png"
PDF_PATH_1_5 = OUTPUT_DIR / "wake_test_slice_grid_true_pred_tests_1_5.pdf"
PNG_PATH_6_10 = OUTPUT_DIR / "wake_test_slice_grid_true_pred_tests_6_10.png"
PDF_PATH_6_10 = OUTPUT_DIR / "wake_test_slice_grid_true_pred_tests_6_10.pdf"

IMAGE_SHAPE = (32, 96)
STEP_REQUESTS = [0, 125, 250, 375, 500]

WAKE_CMAP = LinearSegmentedColormap.from_list(
    "paper_wake_red_blue",
    [(0.00, "#28607D"), (0.50, "#FFFFFF"), (1.00, "#B6424A")],
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
    files = sorted(WAKE_DIR.glob("offline_prediction_osc_cylinder_*.npz"))[:10]
    if len(files) != 10:
        raise FileNotFoundError(f"Expected 10 wake prediction files in {WAKE_DIR}, got {len(files)}")
    return files


def selected_steps(num_frames: int) -> list[int]:
    return [min(int(step), num_frames - 1) for step in STEP_REQUESTS]


def style_image_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.46)
        spine.set_color(SPINE_COLOR)


def global_wake_norm(true_videos: list[np.ndarray], pred_videos: list[np.ndarray]) -> TwoSlopeNorm:
    values = np.concatenate(
        [video.reshape(video.shape[0], -1).ravel() for video in true_videos + pred_videos],
        axis=0,
    )
    vmin, vmax = np.nanpercentile(values, [0.5, 99.5])
    vmin = float(vmin)
    vmax = float(vmax)
    # Keep the same center used in the main-text test3 wake rendering.
    center = 0.5
    if not (vmin < center < vmax):
        pad = max(abs(vmax - vmin), 1e-6) * 0.05
        vmin = min(vmin, center - pad)
        vmax = max(vmax, center + pad)
    return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)


def build_subset_figure(
    *,
    true_videos: list[np.ndarray],
    pred_videos: list[np.ndarray],
    test_numbers: list[int],
    steps: list[int],
    norm: TwoSlopeNorm,
    png_path: Path,
    pdf_path: Path,
    title: str,
    save: bool = True,
    show: bool = False,
) -> plt.Figure:
    fig, axes = plt.subplots(
        2 * len(steps),
        len(test_numbers),
        figsize=(13.0, 8.15),
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.070, right=0.995, bottom=0.050, top=0.925, wspace=0.032, hspace=0.040)

    for step_idx, step in enumerate(steps):
        for pair_idx, (label, videos) in enumerate((("true", true_videos), ("pred", pred_videos))):
            row = 2 * step_idx + pair_idx
            for local_col, test_number in enumerate(test_numbers):
                video = videos[test_number - 1]
                ax = axes[row, local_col]
                ax.imshow(
                    video[step],
                    cmap=WAKE_CMAP,
                    norm=norm,
                    origin="lower",
                    interpolation="bicubic",
                    aspect="auto",
                )
                style_image_axis(ax)
                if row == 0:
                    ax.set_title(f"test {test_number}", fontsize=8.8, color=TEXT_COLOR, pad=4)
                if local_col == 0:
                    row_label = f"step {step}\n{label}" if label == "true" else label
                    ax.text(
                        -0.115,
                        0.50,
                        row_label,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=8.0,
                        color=TEXT_COLOR,
                        linespacing=1.05,
                    )

    fig.text(0.5, 0.982, title, ha="center", va="top", fontsize=11.8, color=TEXT_COLOR)

    if save:
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def build_figures(save: bool = True, show: bool = False) -> list[plt.Figure]:
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
    norm = global_wake_norm(true_videos, pred_videos)

    figs = [
        build_subset_figure(
            true_videos=true_videos,
            pred_videos=pred_videos,
            test_numbers=[1, 2, 3, 4, 5],
            steps=steps,
            norm=norm,
            png_path=PNG_PATH_1_5,
            pdf_path=PDF_PATH_1_5,
            title="Forced cylinder wake test-set slices: true / pred (tests 1-5)",
            save=save,
            show=show,
        ),
        build_subset_figure(
            true_videos=true_videos,
            pred_videos=pred_videos,
            test_numbers=[6, 7, 8, 9, 10],
            steps=steps,
            norm=norm,
            png_path=PNG_PATH_6_10,
            pdf_path=PDF_PATH_6_10,
            title="Forced cylinder wake test-set slices: true / pred (tests 6-10)",
            save=save,
            show=show,
        ),
    ]
    return figs


if __name__ == "__main__":
    figures = build_figures(save=True, show=False)
    for fig in figures:
        plt.close(fig)
    print(PNG_PATH_1_5)
    print(PDF_PATH_1_5)
    print(PNG_PATH_6_10)
    print(PDF_PATH_6_10)
