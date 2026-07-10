from __future__ import annotations

import sys
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
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageFilter


try:
    OUTPUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUTPUT_DIR = Path.cwd()

PNG_PATH = OUTPUT_DIR / "test_system_observation_input_schematic.png"
PDF_PATH = OUTPUT_DIR / "test_system_observation_input_schematic.pdf"

PANDA_REF_DIR = Path(r"D:\FLARE\test files\panda")
ROBOT_SAMPLE = Path(r"D:\FLARE\FLARE\robot\train\panda_train_traj_000.npy")

if str(PANDA_REF_DIR) not in sys.path:
    sys.path.insert(0, str(PANDA_REF_DIR))


# Paper-like palette.
TEXT = "#20262B"
BLUE = "#28607D"
BLUE_LIGHT = "#DDEBE8"
RED = "#B6424A"
ORANGE = "#C97928"
GOLD = "#E6AA2C"
GRID = "#D5DDE1"
SPINE = "#8B969E"
INPUT = ORANGE
FRAME_VERTICAL_STRETCH = 1.75


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


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.055,
        1.045,
        letter,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
        clip_on=False,
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_plate(ax: plt.Axes) -> None:
    ax.set_title("Thin-plate heating", fontsize=11.5, color=TEXT, pad=8)

    lx, ly = 0.20, 0.20
    heater_positions = np.asarray([[0.055, 0.065], [0.145, 0.135]], dtype=float)

    xs = np.linspace(0, lx, 260)
    ys = np.linspace(0, ly, 260)
    xx, yy = np.meshgrid(xs, ys)
    field = np.zeros_like(xx)
    for amp, (hx, hy), sig in zip([1.00, 0.82], heater_positions, [0.040, 0.045]):
        field += amp * np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * sig**2))
    field += 0.12 * (xx / lx) + 0.06 * (yy / ly)

    ax.imshow(
        field,
        extent=(0, lx, 0, ly),
        origin="lower",
        cmap=mpl.cm.YlOrRd,
        alpha=0.80,
        interpolation="bicubic",
        zorder=0,
    )
    ax.add_patch(
        patches.Rectangle((0, 0), lx, ly, facecolor="none", edgecolor=BLUE, lw=1.25, zorder=4)
    )

    sensor_grid = np.linspace(0.02, 0.18, 10)
    sensor_x, sensor_y = np.meshgrid(sensor_grid, sensor_grid)
    ax.scatter(
        sensor_x.ravel(),
        sensor_y.ravel(),
        s=13,
        c=BLUE_LIGHT,
        edgecolors=BLUE,
        linewidths=0.45,
        zorder=8,
    )

    ax.scatter(
        heater_positions[:, 0],
        heater_positions[:, 1],
        marker="*",
        s=285,
        c="white",
        edgecolors=INPUT,
        linewidths=1.35,
        zorder=10,
    )

    ax.set_xlim(-0.01, lx + 0.01)
    ax.set_ylim(-0.01, ly + 0.01)
    ax.set_aspect("equal")
    clean_axis(ax)


def postprocess_robot_image(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.uint8)
    h, w = img.shape[:2]
    gray = img.mean(axis=2)
    yy = np.arange(h)[:, None]
    mask = (gray < 222) & (yy > 0.30 * h)
    ys, xs = np.where(mask)
    if xs.size > 0 and ys.size > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        side = int(max(x1 - x0, y1 - y0) * 1.22)
        side = max(side, int(0.30 * min(h, w)))
        side = min(side, min(h, w))
        left = int(np.clip(cx - 0.5 * side, 0, w - side))
        top = int(np.clip(cy - 0.5 * side, 0, h - side))
        img = img[top : top + side, left : left + side].copy()

    img_float = img.astype(float)
    img_float = 1.02 * (img_float - 128.0) + 128.0
    cleaned = np.clip(img_float, 0, 255).astype(np.uint8)
    pil = Image.fromarray(cleaned)
    pil = pil.filter(ImageFilter.UnsharpMask(radius=0.9, percent=190, threshold=2))
    return np.asarray(pil)


def render_panda_image(frame_idx: int = 155) -> np.ndarray:
    import render_panda_offline_comparison as panda_ref

    sample = np.load(ROBOT_SAMPLE, allow_pickle=True).item()
    q = np.asarray(sample["q"], dtype=float)
    frame_idx = int(np.clip(frame_idx, 0, q.shape[0] - 1))

    scene_xml = panda_ref.write_render_scene_xml()
    xml_text = scene_xml.read_text(encoding="utf-8")
    xml_text = xml_text.replace(
        '<global azimuth="135" elevation="-25"/>',
        '<global azimuth="135" elevation="-25" offwidth="1600" offheight="1600"/>',
    )
    scene_xml.write_text(xml_text, encoding="utf-8")

    model = panda_ref.mujoco.MjModel.from_xml_path(str(scene_xml))
    data = panda_ref.mujoco.MjData(model)
    renderer = panda_ref.mujoco.Renderer(model, height=1100, width=1100)
    camera = panda_ref.make_camera()
    camera.lookat[:] = [0.34, -0.02, 0.52]
    camera.distance = 1.30
    camera.azimuth = 134.0
    camera.elevation = -20.0
    option = panda_ref.make_scene_option()

    image = panda_ref.render_robot(model, data, renderer, camera, option, q[frame_idx])
    if hasattr(renderer, "close"):
        renderer.close()
    return postprocess_robot_image(image)


def draw_robot(ax: plt.Axes) -> None:
    ax.set_title("Robotic arm response", fontsize=11.5, color=TEXT, pad=8)
    img = render_panda_image()
    ax.imshow(img, interpolation="none")

    # Approximate visible locations of the four actuated joints in the rendered
    # Panda image.  They are plotted only as input-location markers.
    input_xy_axes = np.asarray(
        [
            [0.382, 0.275],
            [0.315, 0.505],
            [0.302, 0.704],
            [0.505, 0.872],
        ],
        dtype=float,
    )
    ax.scatter(
        input_xy_axes[:, 0],
        input_xy_axes[:, 1],
        transform=ax.transAxes,
        marker="*",
        s=135,
        c="white",
        edgecolors=INPUT,
        linewidths=1.25,
        zorder=20,
    )
    ax.set_facecolor("#F7F8F5")
    clean_axis(ax)


def frame_geometry() -> tuple[np.ndarray, list[tuple[int, int]]]:
    lx, ly, lz = 7.50, 5.00, 3.20
    coords = np.asarray(
        [
            [0, 0, 0],
            [lx, 0, 0],
            [2 * lx, 0, 0],
            [2 * lx, ly, 0],
            [lx, ly, 0],
            [0, ly, 0],
            [0, 0, lz],
            [lx, 0, lz],
            [2 * lx, 0, lz],
            [2 * lx, ly, lz],
            [lx, ly, lz],
            [0, ly, lz],
            [0, 0, 2 * lz],
            [lx, 0, 2 * lz],
            [2 * lx, 0, 2 * lz],
            [2 * lx, ly, 2 * lz],
            [lx, ly, 2 * lz],
            [0, ly, 2 * lz],
        ],
        dtype=float,
    )
    edges: list[tuple[int, int]] = []
    for base in (0, 6, 12):
        edges += [(base + 0, base + 1), (base + 1, base + 2)]
        edges += [(base + 5, base + 4), (base + 4, base + 3)]
        edges += [(base + 0, base + 5), (base + 1, base + 4), (base + 2, base + 3)]
    for lower in (0, 6):
        for i in range(6):
            edges.append((lower + i, lower + i + 6))
    return coords, edges


def project_frame(coords: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            coords[:, 0] + 0.36 * coords[:, 1],
            FRAME_VERTICAL_STRETCH * (coords[:, 2] + 0.18 * coords[:, 1]),
        ]
    )


def draw_frame(ax: plt.Axes) -> None:
    ax.set_title("Two-story frame response", fontsize=11.5, color=TEXT, pad=8)
    coords, edges = frame_geometry()
    xy = project_frame(coords)

    # Faint floor planes to match the original structural schematic style.
    floor_loops = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
    ]
    for loop in floor_loops:
        ax.add_patch(
            patches.Polygon(
                xy[loop],
                closed=True,
                facecolor=BLUE_LIGHT,
                edgecolor="none",
                alpha=0.22,
                zorder=1,
            )
        )

    for i, j in edges:
        ax.plot(
            [xy[i, 0], xy[j, 0]],
            [xy[i, 1], xy[j, 1]],
            color=BLUE,
            lw=1.55,
            solid_capstyle="round",
            zorder=3,
        )

    # Sampling nodes.
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=35,
        c="white",
        edgecolors=BLUE,
        linewidths=0.85,
        zorder=8,
    )

    # Ground line and base input.
    x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
    ground_y = float(xy[:, 1].min()) - 0.50
    ax.plot([x_min - 0.35, x_max + 0.35], [ground_y, ground_y], color=SPINE, lw=1.0, zorder=2)
    for x in np.linspace(x_min - 0.25, x_max + 0.22, 12):
        ax.plot([x, x + 0.20], [ground_y - 0.16, ground_y], color=GRID, lw=0.75, zorder=2)

    base_mid = np.asarray([0.5 * (x_min + x_max), ground_y], dtype=float)
    ax.scatter([base_mid[0]], [base_mid[1]], s=110, marker="*", c="white", edgecolors=INPUT, linewidths=1.25, zorder=10)
    arrow = patches.FancyArrowPatch(
        (base_mid[0] - 2.2, ground_y - 0.45),
        (base_mid[0] - 0.25, ground_y - 0.45),
        arrowstyle="-|>",
        mutation_scale=15,
        color=INPUT,
        lw=1.5,
        zorder=10,
    )
    ax.add_patch(arrow)

    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xlim(x_min - 0.58, x_max + 0.58)
    ax.set_ylim(ground_y - 0.45, float(xy[:, 1].max()) + 0.14)
    clean_axis(ax)


def add_legend(fig: plt.Figure) -> None:
    observation = plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=6.4,
        markerfacecolor="white",
        markeredgecolor=BLUE,
        markeredgewidth=1.0,
        label="sampled observation",
    )
    input_point = plt.Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markersize=10.5,
        markerfacecolor="white",
        markeredgecolor=INPUT,
        markeredgewidth=1.2,
        label="external input",
    )
    fig.legend(
        [observation, input_point],
        ["sampled observation", "external input"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.016),
        ncol=2,
        frameon=False,
        fontsize=8.7,
        handlelength=1.4,
        columnspacing=1.6,
    )


def build_figure(save: bool = True, show: bool = False) -> plt.Figure:
    configure_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.45), constrained_layout=False)
    plt.subplots_adjust(left=0.038, right=0.992, bottom=0.150, top=0.870, wspace=0.18)

    draw_plate(axes[0])
    draw_robot(axes[1])
    draw_frame(axes[2])

    for letter, ax in zip("abc", axes):
        add_panel_letter(ax, letter)
    add_legend(fig)

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
