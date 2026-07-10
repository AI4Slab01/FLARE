"""
Standard forced transverse-cylinder wake video dataset generator.

This generator keeps the saved .npy format used by the earlier dataset:

    t, f, f_dot, y_c, x1, x2, ..., x3072

The numerical method is changed to avoid the main artifact source in the
previous moving-mask code.  The flow is solved in a cylinder-attached frame:

    - The cylinder is fixed on the LBM grid.
    - The prescribed lab-frame cylinder motion is still
          y_c(t) = y0 + A sin(omega t + phi).
    - In the cylinder-attached frame, the far field has transverse velocity
          v_far(t) = -dy_c/dt.
    - A uniform non-inertial body acceleration -d2y_c/dt2 is applied with a
      Guo forcing term.

This keeps the physical external input while avoiding hard moving masks and
newly-uncovered-cell refill artifacts.  The simulation domain is larger than
the saved observation.  A high-resolution crop is block-averaged to 32 x 96.

The code is intended as a clean benchmark generator, not a force-coefficient
CFD solver.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ============================================================
# Dataset settings
# ============================================================

ROOT_DIR = "standard_forced_cylinder_flow_video_dataset_32x96"
CLEAR_EXISTING = True

NUM_TRAIN = 80
NUM_VAR = 10
NUM_TEST = 10

RECORD_STEPS = 500
FRAME_STRIDE = 4
BURN_STEPS = 4000
SEED = 42

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 96
NUM_SENSORS = IMAGE_HEIGHT * IMAGE_WIDTH


# ============================================================
# LBM grid and observation window
# ============================================================

# The saved output remains 32 x 96.  The LBM solve is done on a larger grid,
# and a 64 x 192 crop is block-averaged by 2 x 2.
DOWNSAMPLE = 2
CROP_HEIGHT = IMAGE_HEIGHT * DOWNSAMPLE
CROP_WIDTH = IMAGE_WIDTH * DOWNSAMPLE

NX = 288
NY = 144

CYL_RADIUS = 8
CYL_DIAMETER = 2 * CYL_RADIUS
CYL_X = 48
CYL_Y0 = NY // 2

# Observation window: about 1.25D upstream and 10.75D downstream of the cylinder.
# This moves the observation window right, placing the cylinder nearer the
# left side of the saved 32 x 96 video.
CROP_X0 = CYL_X - int(round(1.25 * CYL_DIAMETER))
CROP_X1 = CROP_X0 + CROP_WIDTH
CROP_Y0 = CYL_Y0 - CROP_HEIGHT // 2
CROP_Y1 = CROP_Y0 + CROP_HEIGHT


# ============================================================
# Flow and forcing parameters
# ============================================================

U0 = 0.10
REYNOLDS = 120.0

# Use forced Strouhal number instead of raw omega.  This makes the frequency
# range portable across U0 and D:
#     St_f = omega_f * D / (2*pi*U0).
ST_F_RANGE = (0.15, 0.20)
A_OVER_D_RANGE = (0.12, 0.20)
PHI_RANGE = (0.0, 2.0 * np.pi)

# Avoid aggressive high-frequency/high-amplitude cases that tend to produce
# nonphysical saturated near-cylinder images.
MAX_F_DOT = 0.26

MAX_ABS_VELOCITY_ALLOWED = 0.30
MIN_DENSITY_ALLOWED = 0.2
MAX_DENSITY_ALLOWED = 5.0
MAX_ATTEMPTS_FACTOR = 20


# ============================================================
# Rendering settings
# ============================================================

BASE_GRAY = 0.50
CONTRAST = 0.47
VORTICITY_RENDER_LIMIT = 0.012
CYLINDER_GRAY = 0.06
VORTICITY_MASK_DILATION = 2
RENDER_LAB_FRAME = True


# ============================================================
# D2Q9 constants
# ============================================================

C = np.asarray(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=np.int32,
)

W = np.asarray(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float64,
)

OPP = np.asarray([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
CS2 = 1.0 / 3.0
CS4 = CS2 * CS2


@dataclass(frozen=True)
class SampleSpec:
    a_over_d: float
    st_f: float
    phi: float

    @property
    def omega_f(self) -> float:
        return 2.0 * np.pi * self.st_f * U0 / float(CYL_DIAMETER)

    @property
    def max_f_dot(self) -> float:
        return 2.0 * np.pi * self.a_over_d * self.st_f


def validate_geometry() -> None:
    if CROP_X0 < 1 or CROP_X1 > NX - 2 or CROP_Y0 < 1 or CROP_Y1 > NY - 2:
        raise ValueError(
            "Observation crop must be inside the LBM domain: "
            f"crop y=[{CROP_Y0},{CROP_Y1}), x=[{CROP_X0},{CROP_X1}), "
            f"domain NY x NX = {NY} x {NX}."
        )
    if CROP_HEIGHT % IMAGE_HEIGHT != 0 or CROP_WIDTH % IMAGE_WIDTH != 0:
        raise ValueError("Crop size must be divisible by saved image size.")
    if CROP_X0 >= CYL_X:
        raise ValueError("The crop should include upstream space before the cylinder.")


def viscosity_and_relaxation() -> Tuple[float, float, float]:
    nu = float(U0) * float(CYL_DIAMETER) / float(REYNOLDS)
    tau = 3.0 * nu + 0.5
    if tau <= 0.5:
        raise ValueError("Invalid LBM relaxation time. Increase viscosity or reduce Re.")
    return nu, tau, 1.0 / tau


# ============================================================
# Motion and masks
# ============================================================

def cylinder_motion(
    t_step: np.ndarray | float,
    spec: SampleSpec,
    y0: float = CYL_Y0,
    diameter: float = CYL_DIAMETER,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return y_c, y_dot, y_ddot, and f=(y_c-y0)/D in lattice units."""
    t_step = np.asarray(t_step, dtype=np.float64)
    amp = float(spec.a_over_d) * float(diameter)
    omega = float(spec.omega_f)
    phase = omega * t_step + float(spec.phi)

    y_c = float(y0) + amp * np.sin(phase)
    y_dot = amp * omega * np.cos(phase)
    y_ddot = -amp * omega * omega * np.sin(phase)
    f_input = (y_c - float(y0)) / float(diameter)
    return y_c, y_dot, y_ddot, f_input


def make_cylinder_mask() -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    return ((xx - CYL_X) ** 2 + (yy - CYL_Y0) ** 2) <= CYL_RADIUS ** 2


OBSTACLE = make_cylinder_mask()
FLUID = ~OBSTACLE


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(iterations)):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                expanded |= p[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
        out = expanded
    return out


VORTICITY_EXCLUSION_MASK = dilate_mask(OBSTACLE, VORTICITY_MASK_DILATION)


# ============================================================
# LBM utilities
# ============================================================

def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    u2 = ux * ux + uy * uy
    feq = []
    for i in range(9):
        cx, cy = C[i]
        cu = 3.0 * (cx * ux + cy * uy)
        feq.append(W[i] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2))
    return np.stack(feq, axis=0)


def macroscopic(
    f_lbm: np.ndarray,
    accel_x: float = 0.0,
    accel_y: float = 0.0,
    obstacle: Optional[np.ndarray] = OBSTACLE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.sum(f_lbm, axis=0)
    rho_safe = np.maximum(rho, 1e-12)

    jx = np.zeros_like(rho)
    jy = np.zeros_like(rho)
    for i in range(9):
        jx += C[i, 0] * f_lbm[i]
        jy += C[i, 1] * f_lbm[i]

    fx = rho * float(accel_x)
    fy = rho * float(accel_y)
    ux = (jx + 0.5 * fx) / rho_safe
    uy = (jy + 0.5 * fy) / rho_safe

    if obstacle is not None:
        rho = rho.copy()
        ux = ux.copy()
        uy = uy.copy()
        rho[obstacle] = 1.0
        ux[obstacle] = 0.0
        uy[obstacle] = 0.0
    return rho, ux, uy


def guo_force_term(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    accel_x: float,
    accel_y: float,
    omega_relax: float,
) -> np.ndarray:
    fx = rho * float(accel_x)
    fy = rho * float(accel_y)
    terms = []
    prefactor = 1.0 - 0.5 * float(omega_relax)
    for i in range(9):
        cx, cy = C[i]
        ci_dot_u = cx * ux + cy * uy
        ci_dot_f = cx * fx + cy * fy
        first = ((cx - ux) * fx + (cy - uy) * fy) / CS2
        second = ci_dot_u * ci_dot_f / CS4
        terms.append(prefactor * W[i] * (first + second))
    return np.stack(terms, axis=0)


def initialize_lbm(far_uy: float) -> np.ndarray:
    rho = np.ones((NY, NX), dtype=np.float64)
    ux = np.full((NY, NX), float(U0), dtype=np.float64)
    uy = np.full((NY, NX), float(far_uy), dtype=np.float64)

    # A tiny deterministic perturbation breaks the perfectly symmetric wake.
    # Without this, short 1000-step windows can remain too laminar-looking.
    yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    perturb = (
        1.5e-3
        * float(U0)
        * np.sin(2.0 * np.pi * yy / float(NY))
        * np.exp(-((xx - float(CYL_X)) / (6.0 * float(CYL_DIAMETER))) ** 2)
    )
    uy += perturb

    ux[OBSTACLE] = 0.0
    uy[OBSTACLE] = 0.0
    return equilibrium(rho, ux, uy)


def _stream_slices(cy: int, cx: int) -> Tuple[slice, slice, slice, slice]:
    if cy >= 0:
        sy = slice(0, NY - cy)
        dy = slice(cy, NY)
    else:
        sy = slice(-cy, NY)
        dy = slice(0, NY + cy)

    if cx >= 0:
        sx = slice(0, NX - cx)
        dx = slice(cx, NX)
    else:
        sx = slice(-cx, NX)
        dx = slice(0, NX + cx)
    return sy, sx, dy, dx


def stream_with_halfway_bounce_back(f_post: np.ndarray) -> np.ndarray:
    """Stream only from fluid cells and bounce populations hitting the cylinder."""
    f_next = np.zeros_like(f_post)
    f_next[0, FLUID] = f_post[0, FLUID]

    for i in range(1, 9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        sy, sx, dy, dx = _stream_slices(cy=cy, cx=cx)

        src_fluid = FLUID[sy, sx]
        dst_solid = OBSTACLE[dy, dx]
        src_pop = f_post[i, sy, sx]

        to_fluid = src_fluid & (~dst_solid)
        dst_view = f_next[i, dy, dx]
        dst_view[to_fluid] = src_pop[to_fluid]

        to_solid = src_fluid & dst_solid
        bounce_view = f_next[OPP[i], sy, sx]
        bounce_view[to_solid] = src_pop[to_solid]

    f_next[:, OBSTACLE] = equilibrium(
        np.ones((NY, NX), dtype=np.float64),
        np.zeros((NY, NX), dtype=np.float64),
        np.zeros((NY, NX), dtype=np.float64),
    )[:, OBSTACLE]
    return f_next


def apply_farfield_boundaries(f_lbm: np.ndarray, far_uy: float) -> np.ndarray:
    rho_y = np.ones((NY,), dtype=np.float64)
    ux_y = np.full((NY,), float(U0), dtype=np.float64)
    uy_y = np.full((NY,), float(far_uy), dtype=np.float64)
    f_lbm[:, :, 0] = equilibrium(rho_y, ux_y, uy_y)

    # Convective/zero-gradient outlet.
    f_lbm[:, :, -1] = f_lbm[:, :, -2]

    rho_x = np.ones((NX,), dtype=np.float64)
    ux_x = np.full((NX,), float(U0), dtype=np.float64)
    uy_x = np.full((NX,), float(far_uy), dtype=np.float64)
    f_lbm[:, 0, :] = equilibrium(rho_x, ux_x, uy_x)
    f_lbm[:, -1, :] = equilibrium(rho_x, ux_x, uy_x)
    return f_lbm


def lbm_step(
    f_lbm: np.ndarray,
    omega_relax: float,
    far_uy: float,
    accel_y: float,
) -> np.ndarray:
    rho, ux, uy = macroscopic(f_lbm, accel_x=0.0, accel_y=accel_y, obstacle=OBSTACLE)
    feq = equilibrium(rho, ux, uy)
    force = guo_force_term(
        rho=rho,
        ux=ux,
        uy=uy,
        accel_x=0.0,
        accel_y=accel_y,
        omega_relax=omega_relax,
    )
    force[:, OBSTACLE] = 0.0

    f_post = f_lbm - float(omega_relax) * (f_lbm - feq) + force
    f_next = stream_with_halfway_bounce_back(f_post)
    f_next = apply_farfield_boundaries(f_next, far_uy=far_uy)
    return f_next


def compute_vorticity(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    duy_dx = np.gradient(uy, axis=1)
    dux_dy = np.gradient(ux, axis=0)
    vort = (duy_dx - dux_dy).astype(np.float32)
    vort[VORTICITY_EXCLUSION_MASK] = 0.0
    return vort


# ============================================================
# Rendering
# ============================================================

def shift_field_y(field: np.ndarray, displacement: float, fill_value: float = 0.0) -> np.ndarray:
    """Return output[y, x] = field[y - displacement, x] with linear interpolation."""
    if abs(float(displacement)) < 1e-12:
        return field.astype(np.float32, copy=False)

    field = np.asarray(field, dtype=np.float32)
    y_src = np.arange(NY, dtype=np.float32) - float(displacement)
    y0 = np.floor(y_src).astype(np.int32)
    y1 = y0 + 1
    w = y_src - y0
    valid = (y0 >= 0) & (y1 < NY)

    out = np.full_like(field, float(fill_value), dtype=np.float32)
    if np.any(valid):
        out[valid, :] = (1.0 - w[valid, None]) * field[y0[valid], :] + w[valid, None] * field[y1[valid], :]
    return out


def block_average_crop(field: np.ndarray) -> np.ndarray:
    crop = field[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    img = crop.reshape(IMAGE_HEIGHT, DOWNSAMPLE, IMAGE_WIDTH, DOWNSAMPLE).mean(axis=(1, 3))
    return img.astype(np.float32)


def downsample_obstacle_mask(mask: np.ndarray = OBSTACLE) -> np.ndarray:
    crop = mask[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1].astype(np.float32)
    img = crop.reshape(IMAGE_HEIGHT, DOWNSAMPLE, IMAGE_WIDTH, DOWNSAMPLE).mean(axis=(1, 3))
    return img.astype(np.float32)


OBSTACLE_IMG = downsample_obstacle_mask(OBSTACLE)


def render_vorticity_frame(vorticity: np.ndarray, y_c: float = CYL_Y0) -> np.ndarray:
    displacement = float(y_c) - float(CYL_Y0) if RENDER_LAB_FRAME else 0.0
    if abs(displacement) > 1e-12:
        vorticity = shift_field_y(vorticity, displacement=displacement, fill_value=0.0)
        obstacle_img = block_average_crop(shift_field_y(OBSTACLE.astype(np.float32), displacement=displacement, fill_value=0.0))
    else:
        obstacle_img = OBSTACLE_IMG

    vort_img = block_average_crop(vorticity)
    frame = BASE_GRAY + CONTRAST * np.tanh(vort_img / float(VORTICITY_RENDER_LIMIT))
    frame = frame * (1.0 - obstacle_img) + CYLINDER_GRAY * obstacle_img
    return np.clip(frame, 0.0, 1.0).astype(np.float32)


# ============================================================
# Simulation and saving
# ============================================================

def is_stable(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> bool:
    if not (np.all(np.isfinite(rho)) and np.all(np.isfinite(ux)) and np.all(np.isfinite(uy))):
        return False
    fluid_rho = rho[FLUID]
    if fluid_rho.min() < MIN_DENSITY_ALLOWED or fluid_rho.max() > MAX_DENSITY_ALLOWED:
        return False
    max_speed = max(float(np.max(np.abs(ux[FLUID]))), float(np.max(np.abs(uy[FLUID]))))
    return max_speed <= MAX_ABS_VELOCITY_ALLOWED


def simulate_one_sequence(
    spec: SampleSpec,
    record_steps: int = RECORD_STEPS,
    frame_stride: int = FRAME_STRIDE,
    burn_steps: int = BURN_STEPS,
    omega_relax: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    if omega_relax is None:
        _, _, omega_relax = viscosity_and_relaxation()

    y_c0, y_dot0, y_ddot0, _ = cylinder_motion(0.0, spec)
    f_lbm = initialize_lbm(far_uy=-float(y_dot0))

    total_steps = int(burn_steps) + int(record_steps) * int(frame_stride)
    record_start = int(burn_steps)

    frames: List[np.ndarray] = []
    t_values: List[float] = []
    f_values: List[float] = []
    f_dot_values: List[float] = []
    y_c_values: List[float] = []

    for step in range(total_steps):
        y_c, y_dot, y_ddot, f_input = cylinder_motion(float(step), spec)
        far_uy = -float(y_dot)
        accel_y = -float(y_ddot)

        f_lbm = lbm_step(
            f_lbm=f_lbm,
            omega_relax=float(omega_relax),
            far_uy=far_uy,
            accel_y=accel_y,
        )

        should_record = step >= record_start and ((step - record_start) % int(frame_stride) == 0)
        if should_record:
            rho, ux, uy = macroscopic(f_lbm, accel_x=0.0, accel_y=accel_y, obstacle=OBSTACLE)
            if not is_stable(rho, ux, uy):
                raise FloatingPointError("LBM became unstable or exceeded velocity/density limits.")

            vort = compute_vorticity(ux=ux, uy=uy)
            frames.append(render_vorticity_frame(vort, y_c=float(y_c)))
            t_values.append(float(step))
            f_values.append(float(f_input))
            f_dot_values.append(float(y_dot / U0))
            y_c_values.append(float(y_c))

    expected = int(record_steps)
    if len(frames) != expected:
        raise RuntimeError(f"Expected {expected} recorded frames, got {len(frames)}.")

    return {
        "t": np.asarray(t_values, dtype=np.float32),
        "f": np.asarray(f_values, dtype=np.float32),
        "f_dot": np.asarray(f_dot_values, dtype=np.float32),
        "y_c": np.asarray(y_c_values, dtype=np.float32),
        "frames": np.stack(frames, axis=0).astype(np.float32),
    }


def save_sequence_as_model_npy(
    file_path: str | os.PathLike[str],
    t: np.ndarray,
    f: np.ndarray,
    f_dot: np.ndarray,
    y_c: np.ndarray,
    frames: np.ndarray,
) -> None:
    frames = np.asarray(frames, dtype=np.float32)
    T, H, W_img = frames.shape
    if (H, W_img) != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Expected frames [T,{IMAGE_HEIGHT},{IMAGE_WIDTH}], got {frames.shape}.")

    X = frames.reshape(T, H * W_img)
    data: Dict[str, np.ndarray] = {
        "t": np.asarray(t, dtype=np.float32),
        "f": np.asarray(f, dtype=np.float32),
        "f_dot": np.asarray(f_dot, dtype=np.float32),
        "y_c": np.asarray(y_c, dtype=np.float32),
    }
    for i in range(H * W_img):
        data[f"x{i + 1}"] = X[:, i].astype(np.float32)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, data)


def load_npy_video(
    npy_path: str | os.PathLike[str],
    image_height: int = IMAGE_HEIGHT,
    image_width: int = IMAGE_WIDTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npy_path, allow_pickle=True).item()
    t = np.asarray(data["t"], dtype=np.float32)
    f = np.asarray(data["f"], dtype=np.float32)
    f_dot = np.asarray(data["f_dot"], dtype=np.float32)
    y_c = np.asarray(data["y_c"], dtype=np.float32)
    T = len(t)
    X = np.zeros((T, image_height * image_width), dtype=np.float32)
    for i in range(image_height * image_width):
        X[:, i] = np.asarray(data[f"x{i + 1}"], dtype=np.float32)
    frames = X.reshape(T, image_height, image_width)
    return t, f, f_dot, y_c, frames


def frames_to_uint8(frames: np.ndarray) -> np.ndarray:
    out = np.clip(np.asarray(frames, dtype=np.float32), 0.0, 1.0)
    return (255.0 * out).astype(np.uint8)


def sample_spec(rng: np.random.Generator) -> SampleSpec:
    for _ in range(1000):
        a_over_d = float(rng.uniform(*A_OVER_D_RANGE))
        st_f = float(rng.uniform(*ST_F_RANGE))
        phi = float(rng.uniform(*PHI_RANGE))
        spec = SampleSpec(a_over_d=a_over_d, st_f=st_f, phi=phi)
        if spec.max_f_dot <= MAX_F_DOT:
            return spec
    raise RuntimeError("Could not sample a stable forcing specification.")


def split_for_index(sample_id: int, num_train: int, num_var: int) -> Tuple[str, int]:
    if sample_id <= num_train:
        return "train", sample_id
    if sample_id <= num_train + num_var:
        return "var", sample_id - num_train
    return "test", sample_id - num_train - num_var


def generate_dataset(
    root_dir: str | os.PathLike[str] = ROOT_DIR,
    num_train: int = NUM_TRAIN,
    num_var: int = NUM_VAR,
    num_test: int = NUM_TEST,
    record_steps: int = RECORD_STEPS,
    frame_stride: int = FRAME_STRIDE,
    burn_steps: int = BURN_STEPS,
    seed: int = SEED,
    clear_existing: bool = CLEAR_EXISTING,
) -> None:
    validate_geometry()
    nu, tau, omega_relax = viscosity_and_relaxation()
    root = Path(root_dir)
    if clear_existing and root.exists():
        shutil.rmtree(root)

    for split in ("train", "var", "test"):
        (root / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    total = int(num_train) + int(num_var) + int(num_test)
    max_attempts = max(total, total * int(MAX_ATTEMPTS_FACTOR))
    sample_specs = []

    sample_id = 1
    attempt = 0
    while sample_id <= total and attempt < max_attempts:
        attempt += 1
        split, local_id = split_for_index(sample_id, int(num_train), int(num_var))
        spec = sample_spec(rng)

        try:
            seq = simulate_one_sequence(
                spec=spec,
                record_steps=record_steps,
                frame_stride=frame_stride,
                burn_steps=burn_steps,
                omega_relax=omega_relax,
            )
        except Exception as exc:
            print(
                f"Attempt {attempt}: skipped unstable sample "
                f"(A/D={spec.a_over_d:.3f}, St_f={spec.st_f:.3f}, phi={spec.phi:.3f}): {exc}"
            )
            continue

        file_path = root / split / f"osc_cylinder_{local_id:03d}.npy"
        save_sequence_as_model_npy(
            file_path=file_path,
            t=seq["t"],
            f=seq["f"],
            f_dot=seq["f_dot"],
            y_c=seq["y_c"],
            frames=seq["frames"],
        )

        sample_specs.append(
            {
                "global_id": sample_id,
                "split": split,
                "local_id": local_id,
                "file_path": str(file_path),
                **asdict(spec),
                "omega_f": spec.omega_f,
                "max_f_dot": spec.max_f_dot,
                "f_range": [float(seq["f"].min()), float(seq["f"].max())],
                "f_dot_range": [float(seq["f_dot"].min()), float(seq["f_dot"].max())],
            }
        )

        print(
            f"Saved {file_path} | A/D={spec.a_over_d:.3f}, St_f={spec.st_f:.3f}, "
            f"omega={spec.omega_f:.6f}, f_dot_max={spec.max_f_dot:.3f}"
        )
        sample_id += 1

    if sample_id <= total:
        raise RuntimeError(f"Only generated {sample_id - 1}/{total} samples after {attempt} attempts.")

    metadata = {
        "system": "standard_transverse_forced_cylinder_wake_lbm_body_fixed_video_32x96",
        "description": (
            "D2Q9 LBM wake past a transversely forced cylinder, solved in the "
            "cylinder-attached frame. The saved input f is the lab-frame "
            "dimensionless cylinder displacement."
        ),
        "npy_format": ["t", "f", "f_dot", "y_c", "x1", "...", f"x{NUM_SENSORS}"],
        "splits": {"train": int(num_train), "var": int(num_var), "test": int(num_test)},
        "time": {
            "burn_steps": int(burn_steps),
            "record_steps": int(record_steps),
            "frame_stride": int(frame_stride),
            "saved_t": "absolute LBM step index; default records steps 4000..5999",
        },
        "image": {
            "height": int(IMAGE_HEIGHT),
            "width": int(IMAGE_WIDTH),
            "num_sensors": int(NUM_SENSORS),
            "downsample": int(DOWNSAMPLE),
        },
        "lbm_grid": {
            "NX": int(NX),
            "NY": int(NY),
            "crop": [int(CROP_Y0), int(CROP_Y1), int(CROP_X0), int(CROP_X1)],
            "cylinder_center": [int(CYL_X), int(CYL_Y0)],
            "cylinder_radius": int(CYL_RADIUS),
            "cylinder_diameter": int(CYL_DIAMETER),
        },
        "flow_parameters": {
            "U0": float(U0),
            "Reynolds": float(REYNOLDS),
            "nu": float(nu),
            "tau": float(tau),
            "omega_relax": float(omega_relax),
        },
        "forcing": {
            "A_over_D_range": [float(A_OVER_D_RANGE[0]), float(A_OVER_D_RANGE[1])],
            "St_f_range": [float(ST_F_RANGE[0]), float(ST_F_RANGE[1])],
            "max_f_dot": float(MAX_F_DOT),
            "f": "(y_c-y0)/D",
            "f_dot": "dy_c/dt/U0",
            "body_fixed_farfield": "u_far=(U0, -dy_c/dt)",
            "body_force": "a_y=-d2y_c/dt2 with Guo forcing",
            "rendering_frame": "lab-frame observation reconstructed by shifting the body-fixed field by y_c-y0",
        },
        "boundary_conditions": {
            "cylinder": "fixed no-slip cylinder with halfway bounce-back",
            "left": "time-dependent far-field equilibrium velocity",
            "top_bottom": "time-dependent far-field equilibrium velocity, far from observation crop",
            "right": "zero-gradient copy from penultimate column",
        },
        "rendering": {
            "vorticity_render_limit": float(VORTICITY_RENDER_LIMIT),
            "vorticity_mask_dilation": int(VORTICITY_MASK_DILATION),
            "render_lab_frame": bool(RENDER_LAB_FRAME),
            "base_gray": float(BASE_GRAY),
            "contrast": float(CONTRAST),
            "cylinder_gray": float(CYLINDER_GRAY),
        },
    }

    with open(root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(root / "sample_specs.json", "w", encoding="utf-8") as f:
        json.dump(sample_specs, f, indent=2)

    print(f"Done. Dataset saved to: {root}")
    print(f"Metadata saved to: {root / 'metadata.json'}")
    print(f"Sample specs saved to: {root / 'sample_specs.json'}")


if __name__ == "__main__":
    generate_dataset()
