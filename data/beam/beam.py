from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import butter, sosfiltfilt

try:
    import imageio.v2 as imageio

    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


# ============================================================
# Global generation settings
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "standard_cantilever_beam_video_dataset_32")
CLEAR_EXISTING = True

NUM_TRAIN = 80
NUM_VAR = 10
NUM_TEST = 10

SEQ_LEN = 400
T_END = 24.0
IMAGE_SIZE = 32

SEED = 42
T_BURN = 10.0


# ============================================================
# Euler-Bernoulli cantilever beam settings
# ============================================================

# Nondimensional beam: L = 1, rho*A = 1, EI = 1. Natural frequencies are
# omega_n = beta_n^2 * sqrt(EI / rhoA / L^4).
BEAM_LENGTH = 1.0
BETA_ROOTS = np.asarray(
    [1.875104068711961, 4.694091132974174, 7.854757438237612],
    dtype=np.float64,
)
NUM_MODES = len(BETA_ROOTS)
OMEGA_SCALE = 1.0
MODAL_ZETA = np.asarray([0.030, 0.040, 0.050], dtype=np.float64)

# Tip-force-to-modal-force gain. Kept explicit so f(t) remains the saved input.
FORCE_GAIN = 1.0


# ============================================================
# Non-harmonic force settings
# ============================================================

FORCE_TYPE_PROB_BANDLIMITED = 0.55
FORCE_PEAK_RANGE = (0.12, 0.24)
BANDLIMIT_CUTOFF_HZ_RANGE = (0.22, 0.42)
BANDLIMIT_FILTER_ORDER = 4
PLATEAU_SEGMENT_RANGE = (5, 8)
PLATEAU_TRANSITION_RANGE = (0.7, 1.4)


# ============================================================
# Rendering settings
# ============================================================

WALL_X = 3.5
CENTER_Y = 16.0
BEAM_LENGTH_PIXELS = 24.0
PIXELS_PER_DISP = 62.0
DISP_RENDER_LIMIT = 0.17

N_CURVE_POINTS = 96
BEAM_SIGMA = 0.52
TIP_SIGMA = 1.05
WALL_STRENGTH = 0.33
BEAM_STRENGTH = 0.94
TIP_STRENGTH = 0.92
BACKGROUND_LEVEL = 0.035


@dataclass
class SampleSpec:
    global_index: int
    split: str
    local_index: int
    file_path: str
    force_type: str
    force_peak: float
    force_cutoff_hz: float | None
    plateau_segments: int | None
    plateau_transition_s: float | None
    seed: int


# ============================================================
# Beam modal utilities
# ============================================================


def raw_cantilever_mode(beta: float, x: np.ndarray) -> np.ndarray:
    """Euler-Bernoulli cantilever eigenfunction before mass normalization."""
    bx = beta * x
    sigma = (np.cosh(beta) + np.cos(beta)) / (np.sinh(beta) + np.sin(beta))
    return np.cosh(bx) - np.cos(bx) - sigma * (np.sinh(bx) - np.sin(bx))


def normalized_cantilever_mode(beta: float, x: np.ndarray) -> np.ndarray:
    dense_x = np.linspace(0.0, 1.0, 4000, dtype=np.float64)
    dense_phi = raw_cantilever_mode(beta, dense_x)
    norm = float(np.sqrt(np.trapz(dense_phi * dense_phi, dense_x)))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Invalid cantilever mode normalization.")

    phi = raw_cantilever_mode(beta, x) / norm
    if phi[-1] < 0.0:
        phi = -phi
    return phi.astype(np.float64)


def build_modal_basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modes = np.vstack([normalized_cantilever_mode(beta, x) for beta in BETA_ROOTS])
    tip_values = modes[:, -1].copy()
    omegas = (BETA_ROOTS**2) * OMEGA_SCALE
    return modes, tip_values, omegas


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ============================================================
# Force generation
# ============================================================


def normalize_to_peak(signal: np.ndarray, target_peak: float) -> np.ndarray:
    out = np.asarray(signal, dtype=np.float64).copy()
    out -= np.mean(out)
    peak = float(np.max(np.abs(out)))
    if peak <= 1e-12 or not np.isfinite(peak):
        raise ValueError("Generated force has invalid peak.")
    return target_peak * out / peak


def build_time_grids(seq_len: int, t_end: float, t_burn: float) -> tuple[np.ndarray, np.ndarray]:
    t_record = np.linspace(0.0, t_end, seq_len, dtype=np.float64)
    dt = t_record[1] - t_record[0]
    n_burn = int(round(t_burn / dt))
    t_full = np.linspace(-n_burn * dt, t_end, n_burn + seq_len, dtype=np.float64)
    return t_full, t_record


def generate_bandlimited_force(
    t_full: np.ndarray,
    rng: np.random.Generator,
    peak: float,
    cutoff_hz: float,
) -> np.ndarray:
    dt = float(t_full[1] - t_full[0])
    fs = 1.0 / dt
    white = rng.standard_normal(t_full.shape[0])
    sos = butter(
        BANDLIMIT_FILTER_ORDER,
        cutoff_hz,
        btype="lowpass",
        fs=fs,
        output="sos",
    )
    filtered = sosfiltfilt(sos, white)
    return normalize_to_peak(filtered, peak)


def generate_plateau_force(
    t_full: np.ndarray,
    rng: np.random.Generator,
    peak: float,
    n_segments: int,
    transition_s: float,
) -> np.ndarray:
    edges = np.linspace(t_full[0], t_full[-1], n_segments + 1, dtype=np.float64)
    levels = rng.uniform(-1.0, 1.0, size=n_segments)
    if np.max(np.abs(levels)) < 0.15:
        levels[0] = 1.0

    force = np.full_like(t_full, levels[0], dtype=np.float64)
    for i in range(1, n_segments):
        center = edges[i]
        weight = smoothstep((t_full - (center - 0.5 * transition_s)) / transition_s)
        force = (1.0 - weight) * force + weight * levels[i]

    return normalize_to_peak(force, peak)


def generate_force(
    t_full: np.ndarray,
    rng: np.random.Generator,
    preferred_force_type: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    peak = float(rng.uniform(*FORCE_PEAK_RANGE))

    if preferred_force_type not in (None, "bandlimited_random", "smooth_plateau"):
        raise ValueError("preferred_force_type must be None, 'bandlimited_random', or 'smooth_plateau'.")

    use_bandlimited = (
        preferred_force_type == "bandlimited_random"
        or (
            preferred_force_type is None
            and rng.uniform() < FORCE_TYPE_PROB_BANDLIMITED
        )
    )

    if use_bandlimited:
        cutoff_hz = float(rng.uniform(*BANDLIMIT_CUTOFF_HZ_RANGE))
        force = generate_bandlimited_force(t_full, rng, peak=peak, cutoff_hz=cutoff_hz)
        info = {
            "force_type": "bandlimited_random",
            "force_peak": peak,
            "force_cutoff_hz": cutoff_hz,
            "plateau_segments": None,
            "plateau_transition_s": None,
        }
    else:
        n_segments = int(rng.integers(PLATEAU_SEGMENT_RANGE[0], PLATEAU_SEGMENT_RANGE[1] + 1))
        transition_s = float(rng.uniform(*PLATEAU_TRANSITION_RANGE))
        force = generate_plateau_force(
            t_full,
            rng,
            peak=peak,
            n_segments=n_segments,
            transition_s=transition_s,
        )
        info = {
            "force_type": "smooth_plateau",
            "force_peak": peak,
            "force_cutoff_hz": None,
            "plateau_segments": n_segments,
            "plateau_transition_s": transition_s,
        }

    return force.astype(np.float64), info


# ============================================================
# Dynamics
# ============================================================


def integrate_one_sequence(
    seq_len: int,
    t_end: float,
    t_burn: float,
    rng: np.random.Generator,
    preferred_force_type: str | None = None,
) -> tuple[Dict[str, np.ndarray], dict[str, object]]:
    """Integrate the forced cantilever modal equations."""
    t_full, t_record = build_time_grids(seq_len=seq_len, t_end=t_end, t_burn=t_burn)
    f_full, force_info = generate_force(
        t_full,
        rng,
        preferred_force_type=preferred_force_type,
    )

    modes_tip_x = np.asarray([1.0], dtype=np.float64)
    _, tip_values, omegas = build_modal_basis(modes_tip_x)
    zeta = MODAL_ZETA[:NUM_MODES]

    def f_at(tt: float) -> float:
        return float(np.interp(tt, t_full, f_full))

    def rhs(tt: float, state: np.ndarray) -> np.ndarray:
        q = state[:NUM_MODES]
        qdot = state[NUM_MODES:]
        ft = f_at(tt)
        qddot = (
            FORCE_GAIN * tip_values * ft
            - 2.0 * zeta * omegas * qdot
            - (omegas**2) * q
        )
        return np.concatenate([qdot, qddot])

    y0 = np.zeros(2 * NUM_MODES, dtype=np.float64)
    sol = solve_ivp(
        rhs,
        t_span=(float(t_full[0]), float(t_full[-1])),
        y0=y0,
        t_eval=t_record,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError("Cantilever modal integration failed.")

    q = sol.y[:NUM_MODES].T
    qdot = sol.y[NUM_MODES:].T
    f_record = np.interp(t_record, t_full, f_full)

    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qdot)):
        raise RuntimeError("Non-finite modal trajectory generated.")

    return (
        {
            "t": t_record.astype(np.float32),
            "f": f_record.astype(np.float32),
            "q": q.astype(np.float32),
            "qdot": qdot.astype(np.float32),
        },
        force_info,
    )


# ============================================================
# Rendering
# ============================================================


def _make_mesh(image_size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(
        np.arange(image_size, dtype=np.float32),
        np.arange(image_size, dtype=np.float32),
        indexing="ij",
    )
    return yy, xx


def render_beam_video(
    q: np.ndarray,
    image_size: int = IMAGE_SIZE,
    wall_x: float = WALL_X,
    center_y: float = CENTER_Y,
    beam_length_pixels: float = BEAM_LENGTH_PIXELS,
    pixels_per_disp: float = PIXELS_PER_DISP,
    disp_render_limit: float = DISP_RENDER_LIMIT,
    beam_sigma: float = BEAM_SIGMA,
    tip_sigma: float = TIP_SIGMA,
    wall_strength: float = WALL_STRENGTH,
    beam_strength: float = BEAM_STRENGTH,
    tip_strength: float = TIP_STRENGTH,
    background_level: float = BACKGROUND_LEVEL,
    n_curve_points: int = N_CURVE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:

    q = np.asarray(q, dtype=np.float32)
    T = q.shape[0]
    H = W = int(image_size)

    yy, xx = _make_mesh(image_size)
    x_norm = np.linspace(0.0, 1.0, n_curve_points, dtype=np.float64)
    mode_matrix, _, _ = build_modal_basis(x_norm)
    x_curve = wall_x + beam_length_pixels * x_norm

    frames = np.full((T, H, W), background_level, dtype=np.float32)
    tip_displacement = np.empty(T, dtype=np.float32)

    wall_col = int(round(wall_x)) - 1
    wall_col = int(np.clip(wall_col, 0, W - 1))

    support = np.zeros((H, W), dtype=np.float32)
    support[:, max(0, wall_col - 1):min(W, wall_col + 1)] = wall_strength
    root_y = int(round(center_y))
    support[max(0, root_y - 2):min(H, root_y + 3), max(0, wall_col):min(W, wall_col + 4)] = max(
        wall_strength,
        0.45,
    )

    for k in range(T):
        w_curve = np.dot(q[k].astype(np.float64), mode_matrix)
        w_curve = np.clip(w_curve, -disp_render_limit, disp_render_limit)
        tip_displacement[k] = float(w_curve[-1])
        y_curve = center_y - pixels_per_disp * w_curve

        beam_img = np.zeros((H, W), dtype=np.float32)
        for xp, yp in zip(x_curve, y_curve):
            beam_img += np.exp(
                -(
                    (xx - float(xp)) ** 2
                    + (yy - float(yp)) ** 2
                )
                / (2.0 * beam_sigma**2)
            )

        beam_img = beam_img / (beam_img.max() + 1e-8)
        tip_blob = np.exp(
            -(
                (xx - float(x_curve[-1])) ** 2
                + (yy - float(y_curve[-1])) ** 2
            )
            / (2.0 * tip_sigma**2)
        )

        frame = frames[k] + support + beam_strength * beam_img + tip_strength * tip_blob
        frames[k] = np.clip(frame, 0.0, 1.0)

    return frames.astype(np.float32), tip_displacement


# ============================================================
# Saving utilities
# ============================================================


def save_sequence_as_model_npy(file_path: str, t: np.ndarray, f: np.ndarray, frames: np.ndarray) -> None:
    frames = np.asarray(frames, dtype=np.float32)
    T, H, W = frames.shape
    X = frames.reshape(T, H * W)

    data = {
        "t": np.asarray(t, dtype=np.float32),
        "f": np.asarray(f, dtype=np.float32),
    }
    for i in range(H * W):
        data[f"x{i + 1}"] = X[:, i].astype(np.float32)

    out_dir = os.path.dirname(file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(file_path, data)


def frames_to_uint8(frames: np.ndarray, scale: int = 1) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    out = (255.0 * np.clip(frames, 0.0, 1.0)).astype(np.uint8)
    if scale > 1:
        out = np.repeat(np.repeat(out, scale, axis=1), scale, axis=2)
    return out


def save_gif(frames: np.ndarray, output_path: str, fps: int = 24, scale: int = 8) -> None:
    if not HAS_IMAGEIO:
        print("imageio is not available; skipping GIF preview.")
        return
    imageio.mimsave(output_path, frames_to_uint8(frames, scale=scale), fps=fps)
    print(f"Saved preview GIF: {output_path}")


def save_force_plot(t: np.ndarray, f: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.plot(t, f, color="#1f4e79", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("tip force f(t)")
    ax.set_title("First cantilever-beam sample: non-harmonic force")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved force plot: {output_path}")


def load_npy_video(npy_path: str, image_size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npy_path, allow_pickle=True).item()
    t = np.asarray(data["t"], dtype=np.float32)
    f = np.asarray(data["f"], dtype=np.float32)
    T = len(t)
    X = np.zeros((T, image_size * image_size), dtype=np.float32)
    for i in range(image_size * image_size):
        X[:, i] = np.asarray(data[f"x{i + 1}"], dtype=np.float32)
    frames = X.reshape(T, image_size, image_size)
    return t, f, frames


# ============================================================
# Dataset generation
# ============================================================


def generate_dataset(
    root_dir: str = ROOT_DIR,
    num_train: int = NUM_TRAIN,
    num_var: int = NUM_VAR,
    num_test: int = NUM_TEST,
    seq_len: int = SEQ_LEN,
    t_end: float = T_END,
    image_size: int = IMAGE_SIZE,
    seed: int = SEED,
    clear_existing: bool = CLEAR_EXISTING,
    make_preview: bool = True,
) -> None:
    rng = np.random.default_rng(seed)

    if clear_existing and os.path.exists(root_dir):
        shutil.rmtree(root_dir)

    train_dir = os.path.join(root_dir, "train")
    var_dir = os.path.join(root_dir, "var")
    test_dir = os.path.join(root_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(var_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    split_counts = {"train": int(num_train), "var": int(num_var), "test": int(num_test)}
    sample_specs: list[dict[str, object]] = []

    global_index = 0
    first_preview_written = False
    for split, count in split_counts.items():
        folder = {"train": train_dir, "var": var_dir, "test": test_dir}[split]
        for local_index in range(1, count + 1):
            global_index += 1
            sample_seed = int(rng.integers(1, 2**31 - 1))
            sample_rng = np.random.default_rng(sample_seed)

            seq, force_info = integrate_one_sequence(
                seq_len=seq_len,
                t_end=t_end,
                t_burn=T_BURN,
                rng=sample_rng,
                preferred_force_type="smooth_plateau" if global_index == 1 else None,
            )
            frames, tip_displacement = render_beam_video(q=seq["q"], image_size=image_size)

            file_path = os.path.join(folder, f"beam_{local_index:03d}.npy")
            save_sequence_as_model_npy(file_path=file_path, t=seq["t"], f=seq["f"], frames=frames)

            spec = SampleSpec(
                global_index=global_index,
                split=split,
                local_index=local_index,
                file_path=file_path,
                force_type=str(force_info["force_type"]),
                force_peak=float(force_info["force_peak"]),
                force_cutoff_hz=(
                    None
                    if force_info["force_cutoff_hz"] is None
                    else float(force_info["force_cutoff_hz"])
                ),
                plateau_segments=(
                    None
                    if force_info["plateau_segments"] is None
                    else int(force_info["plateau_segments"])
                ),
                plateau_transition_s=(
                    None
                    if force_info["plateau_transition_s"] is None
                    else float(force_info["plateau_transition_s"])
                ),
                seed=sample_seed,
            )
            spec_dict = asdict(spec)
            spec_dict.update(
                {
                    "f_rms": float(np.sqrt(np.mean(seq["f"] ** 2))),
                    "tip_disp_abs_max": float(np.max(np.abs(tip_displacement))),
                    "q_abs_max": float(np.max(np.abs(seq["q"]))),
                }
            )
            sample_specs.append(spec_dict)

            print(
                f"[{global_index:03d}/{sum(split_counts.values()):03d}] "
                f"{split}/beam_{local_index:03d}.npy "
                f"type={spec.force_type} peak_f={spec.force_peak:.3f} "
                f"tip_max={spec_dict['tip_disp_abs_max']:.4f}"
            )

            if make_preview and not first_preview_written:
                gif_path = os.path.join(root_dir, "beam_001_preview.gif")
                force_plot_path = os.path.join(root_dir, "beam_001_force.png")
                save_gif(frames, gif_path, fps=24, scale=8)
                save_force_plot(seq["t"], seq["f"], force_plot_path)
                first_preview_written = True

    x_meta = np.linspace(0.0, 1.0, 400, dtype=np.float64)
    mode_matrix, tip_values, omegas = build_modal_basis(x_meta)
    metadata = {
        "system": "standard_forced_euler_bernoulli_cantilever_beam_video_32",
        "format": "dict-style .npy files with keys t, f, x1, ..., x1024",
        "root_dir": root_dir,
        "num_sensors": int(image_size * image_size),
        "image_size": int(image_size),
        "seq_len": int(seq_len),
        "t_end": float(t_end),
        "num_train": int(num_train),
        "num_var": int(num_var),
        "num_test": int(num_test),
        "total_samples": int(num_train + num_var + num_test),
        "beam_model": {
            "type": "Euler-Bernoulli cantilever modal expansion",
            "boundary_conditions": {
                "x=0": "w=0, dw/dx=0",
                "x=L": "d2w/dx2=0, EI*d3w/dx3=f(t)",
            },
            "length": float(BEAM_LENGTH),
            "beta_roots": BETA_ROOTS.tolist(),
            "modal_omegas_rad_s": omegas.tolist(),
            "modal_zeta": MODAL_ZETA.tolist(),
            "tip_participation_values": tip_values.tolist(),
            "num_modes": int(NUM_MODES),
            "force_gain": float(FORCE_GAIN),
            "mode_shape_note": "standard cantilever eigenfunctions normalized by integral phi_n(x)^2 dx = 1",
        },
        "forcing": {
            "types": ["bandlimited_random", "smooth_plateau"],
            "force_peak_range": [float(FORCE_PEAK_RANGE[0]), float(FORCE_PEAK_RANGE[1])],
            "bandlimited_cutoff_hz_range": [
                float(BANDLIMIT_CUTOFF_HZ_RANGE[0]),
                float(BANDLIMIT_CUTOFF_HZ_RANGE[1]),
            ],
            "plateau_segment_range": [int(PLATEAU_SEGMENT_RANGE[0]), int(PLATEAU_SEGMENT_RANGE[1])],
            "plateau_transition_range_s": [
                float(PLATEAU_TRANSITION_RANGE[0]),
                float(PLATEAU_TRANSITION_RANGE[1]),
            ],
            "no_simple_harmonic_forcing": True,
        },
        "rendering": {
            "wall_x": float(WALL_X),
            "center_y": float(CENTER_Y),
            "beam_length_pixels": float(BEAM_LENGTH_PIXELS),
            "pixels_per_disp": float(PIXELS_PER_DISP),
            "disp_render_limit": float(DISP_RENDER_LIMIT),
            "beam_sigma": float(BEAM_SIGMA),
            "tip_sigma": float(TIP_SIGMA),
            "wall_strength": float(WALL_STRENGTH),
            "beam_strength": float(BEAM_STRENGTH),
            "tip_strength": float(TIP_STRENGTH),
            "background_level": float(BACKGROUND_LEVEL),
            "n_curve_points": int(N_CURVE_POINTS),
        },
        "preview_files": {
            "first_sample_gif": os.path.join(root_dir, "beam_001_preview.gif"),
            "first_sample_force_plot": os.path.join(root_dir, "beam_001_force.png"),
        },
        "recommended_model_settings": {
            "num_sensors": int(image_size * image_size),
            "image_size": int(image_size),
            "forcing_key": "f",
            "forcing_dim": 1,
            "window_length": 1,
            "use_mask": False,
            "model_order": 2,
            "normalize_x": True,
            "normalize_forcing": True,
        },
    }

    with open(os.path.join(root_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(root_dir, "sample_specs.json"), "w", encoding="utf-8") as f:
        json.dump(sample_specs, f, indent=2)

    print(f"Done. Dataset saved to: {root_dir}")
    print(f"Metadata saved to: {os.path.join(root_dir, 'metadata.json')}")
    print(f"Sample specs saved to: {os.path.join(root_dir, 'sample_specs.json')}")


if __name__ == "__main__":
    generate_dataset(
        root_dir=ROOT_DIR,
        num_train=NUM_TRAIN,
        num_var=NUM_VAR,
        num_test=NUM_TEST,
        seq_len=SEQ_LEN,
        t_end=T_END,
        image_size=IMAGE_SIZE,
        seed=SEED,
        clear_existing=CLEAR_EXISTING,
        make_preview=True,
    )
