"""
Generate a controlled 2D thin-plate heat dataset for FLARE.

The internal simulator solves a 2D heat-conduction model with environmental
cooling and two local heaters. The saved FLARE data contains only sparse
temperature-sensor observations, not the full simulation grid.

Default output:
    train/ 80 dict-style .npy files
    val/   10 dict-style .npy files
    test/  10 dict-style .npy files

Each .npy file contains:
    t, u1, u2, T_00, ..., T_99

The T_* channels are temperature rise over ambient temperature, in degC.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# Dataset settings
# =========================
SPLIT_COUNTS = {
    "train": 80,
    "val": 10,
    "test": 10,
}

SAVE_LENGTH = 500
DT_SAVE = 0.4             # 2.5 Hz
DT_INT = 0.1              # internal integration step, seconds
SAVE_EVERY = int(round(DT_SAVE / DT_INT))
T_TOTAL = (SAVE_LENGTH - 1) * DT_SAVE


# =========================
# Plate / heat model settings
# =========================
L_X = 0.20                # m
L_Y = 0.20                # m
THICKNESS = 0.002         # m

# Aluminum-like effective parameters. The thermal conductivity is deliberately
# conservative so that the two local heaters remain distinguishable.
RHO = 2700.0              # kg / m^3
CP = 900.0                # J / (kg K)
K_THERMAL = 60.0          # W / (m K)
H_EFF = 12.0              # W / (m^2 K), effective environmental cooling
T_AMBIENT = 25.0          # degC, saved outputs are T - T_AMBIENT

N_X = 41
N_Y = 41
HEATER_SIGMA = 0.018      # m
HEATER_1_XY = (0.055, 0.065)
HEATER_2_XY = (0.145, 0.135)


# =========================
# Input settings
# =========================
# Low-frequency continuous heater-power modulation. Frequencies are in Hz.
U1_FREQ_RANGE = (0.006, 0.024)
U2_FREQ_RANGE = (0.012, 0.040)
MIN_FREQ_SEPARATION = 0.004

U1_POWER_RANGE = (4.0, 9.0)   # W
U2_POWER_RANGE = (3.5, 8.0)   # W
MIN_POWER_FRACTION = 0.06
INPUT_RAMP_SECONDS = 18.0


# =========================
# Sensors
# =========================
SENSOR_GRID_NX = 10
SENSOR_GRID_NY = 10
SENSOR_MARGIN = 0.02      # m, avoid placing sensors directly on boundaries


@dataclass(frozen=True)
class SampleSpec:
    split: str
    index: int
    seed: int
    output_path: str
    u1_main_freq: float
    u2_main_freq: float
    u1_power_max: float
    u2_power_max: float


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def make_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    x = np.linspace(0.0, L_X, N_X)
    y = np.linspace(0.0, L_Y, N_Y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return x, y, xx, yy, dx, dy


def make_heater_weight(
    xx: np.ndarray,
    yy: np.ndarray,
    center_xy: tuple[float, float],
    dx: float,
    dy: float,
) -> np.ndarray:
    cx, cy = center_xy
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    weight = np.exp(-0.5 * r2 / (HEATER_SIGMA ** 2))
    integral = np.sum(weight) * dx * dy
    if integral <= 0.0:
        raise ValueError("Invalid heater weight integral.")
    return weight / integral


def make_sensor_points() -> tuple[np.ndarray, np.ndarray, list[str]]:
    sensor_x = np.linspace(SENSOR_MARGIN, L_X - SENSOR_MARGIN, SENSOR_GRID_NX)
    sensor_y = np.linspace(SENSOR_MARGIN, L_Y - SENSOR_MARGIN, SENSOR_GRID_NY)
    xs, ys = np.meshgrid(sensor_x, sensor_y, indexing="xy")
    xs = xs.ravel()
    ys = ys.ravel()
    names = [f"T_{i:02d}" for i in range(xs.size)]
    return xs, ys, names


def bilinear_sample(
    field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]

    fx = np.clip((xs - grid_x[0]) / dx, 0.0, len(grid_x) - 1.000001)
    fy = np.clip((ys - grid_y[0]) / dy, 0.0, len(grid_y) - 1.000001)

    ix0 = np.floor(fx).astype(int)
    iy0 = np.floor(fy).astype(int)
    ix1 = np.minimum(ix0 + 1, len(grid_x) - 1)
    iy1 = np.minimum(iy0 + 1, len(grid_y) - 1)

    wx = fx - ix0
    wy = fy - iy0

    v00 = field[iy0, ix0]
    v10 = field[iy0, ix1]
    v01 = field[iy1, ix0]
    v11 = field[iy1, ix1]

    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )


def normalized_low_frequency_power(
    t: np.ndarray,
    rng: np.random.Generator,
    main_freq: float,
    freq_range: tuple[float, float],
    power_max: float,
) -> np.ndarray:
    f_min, f_max = freq_range
    phase1 = rng.uniform(0.0, 2.0 * math.pi)
    phase2 = rng.uniform(0.0, 2.0 * math.pi)
    phase3 = rng.uniform(0.0, 2.0 * math.pi)

    f2 = rng.uniform(f_min, f_max)
    f3 = rng.uniform(f_min, f_max)
    a2 = rng.uniform(0.12, 0.22)
    a3 = rng.uniform(0.06, 0.14)

    raw = (
        0.50
        + 0.34 * np.sin(2.0 * math.pi * main_freq * t + phase1)
        + a2 * np.sin(2.0 * math.pi * f2 * t + phase2)
        + a3 * np.sin(2.0 * math.pi * f3 * t + phase3)
    )

    raw_min = float(np.min(raw))
    raw_max = float(np.max(raw))
    if raw_max - raw_min < 1e-12:
        scaled = np.full_like(raw, 0.5)
    else:
        scaled = (raw - raw_min) / (raw_max - raw_min)

    scaled = MIN_POWER_FRACTION + (1.0 - MIN_POWER_FRACTION) * scaled
    ramp = smoothstep(t / INPUT_RAMP_SECONDS)
    return power_max * scaled * ramp


def make_input_timeseries(
    t_int: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    f1 = rng.uniform(*U1_FREQ_RANGE)
    for _ in range(100):
        f2 = rng.uniform(*U2_FREQ_RANGE)
        if abs(f2 - f1) >= MIN_FREQ_SEPARATION:
            break

    p1 = rng.uniform(*U1_POWER_RANGE)
    p2 = rng.uniform(*U2_POWER_RANGE)

    u1 = normalized_low_frequency_power(t_int, rng, f1, U1_FREQ_RANGE, p1)
    u2 = normalized_low_frequency_power(t_int, rng, f2, U2_FREQ_RANGE, p2)
    return u1, u2, f1, f2, p1, p2


def laplacian_neumann(theta: np.ndarray, dx: float, dy: float) -> np.ndarray:
    padded = np.pad(theta, ((1, 1), (1, 1)), mode="edge")
    d2x = (padded[1:-1, 2:] - 2.0 * theta + padded[1:-1, :-2]) / (dx * dx)
    d2y = (padded[2:, 1:-1] - 2.0 * theta + padded[:-2, 1:-1]) / (dy * dy)
    return d2x + d2y


def simulate_sample(spec: SampleSpec) -> dict[str, object]:
    rng = np.random.default_rng(spec.seed)
    grid_x, grid_y, xx, yy, dx, dy = make_grid()

    alpha = K_THERMAL / (RHO * CP)
    dt_limit = 1.0 / (2.0 * alpha * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    if DT_INT > 0.95 * dt_limit:
        raise ValueError(
            f"DT_INT={DT_INT} is too large for explicit integration; "
            f"stability limit is about {dt_limit:.4g} s."
        )

    heater1 = make_heater_weight(xx, yy, HEATER_1_XY, dx, dy)
    heater2 = make_heater_weight(xx, yy, HEATER_2_XY, dx, dy)
    sensor_xs, sensor_ys, sensor_names = make_sensor_points()

    n_int = (SAVE_LENGTH - 1) * SAVE_EVERY + 1
    t_int = np.arange(n_int, dtype=np.float64) * DT_INT

    u1, u2, _, _, _, _ = make_input_timeseries(t_int, rng)

    theta = np.zeros((N_Y, N_X), dtype=np.float64)
    saved = np.empty((SAVE_LENGTH, len(sensor_names)), dtype=np.float64)
    save_i = 0

    cooling_rate = H_EFF / (RHO * CP * THICKNESS)
    heat_capacity_area = RHO * CP * THICKNESS

    for step in range(n_int):
        if step % SAVE_EVERY == 0:
            saved[save_i, :] = bilinear_sample(theta, sensor_xs, sensor_ys, grid_x, grid_y)
            save_i += 1

        if step == n_int - 1:
            break

        source = (u1[step] * heater1 + u2[step] * heater2) / heat_capacity_area
        dtheta_dt = alpha * laplacian_neumann(theta, dx, dy) - cooling_rate * theta + source
        theta = theta + DT_INT * dtheta_dt

    t_save = t_int[::SAVE_EVERY]
    u1_save = u1[::SAVE_EVERY]
    u2_save = u2[::SAVE_EVERY]

    record: dict[str, np.ndarray] = {
        "t": t_save.astype(np.float64),
        "u1": u1_save.astype(np.float64),
        "u2": u2_save.astype(np.float64),
    }
    for j, name in enumerate(sensor_names):
        record[name] = saved[:, j].astype(np.float64)

    output_path = Path(spec.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, record)

    return {
        "split": spec.split,
        "index": spec.index,
        "path": str(output_path),
        "seed": spec.seed,
        "u1_main_freq": spec.u1_main_freq,
        "u2_main_freq": spec.u2_main_freq,
        "u1_power_max": spec.u1_power_max,
        "u2_power_max": spec.u2_power_max,
        "max_temp_rise_C": float(np.max(saved)),
        "mean_temp_rise_C": float(np.mean(saved)),
    }


def build_specs(output_root: Path, seed: int) -> list[SampleSpec]:
    rng = np.random.default_rng(seed)
    specs: list[SampleSpec] = []

    for split, count in SPLIT_COUNTS.items():
        for idx in range(count):
            sample_seed = int(rng.integers(0, 2**31 - 1))
            local_rng = np.random.default_rng(sample_seed)
            dummy_t = np.arange((SAVE_LENGTH - 1) * SAVE_EVERY + 1) * DT_INT
            _, _, f1, f2, p1, p2 = make_input_timeseries(dummy_t, local_rng)
            output_path = output_root / split / f"sheet_{split}_{idx:03d}.npy"
            specs.append(
                SampleSpec(
                    split=split,
                    index=idx,
                    seed=sample_seed,
                    output_path=str(output_path),
                    u1_main_freq=float(f1),
                    u2_main_freq=float(f2),
                    u1_power_max=float(p1),
                    u2_power_max=float(p2),
                )
            )

    return specs


def check_output_dirs(output_root: Path, overwrite: bool) -> None:
    for split in SPLIT_COUNTS:
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(split_dir.glob("*.npy"))
        if existing and not overwrite:
            raise FileExistsError(
                f"{split_dir} already contains .npy files. "
                "Use --overwrite to overwrite files with matching names."
            )


def write_dataset_info(output_root: Path) -> None:
    _, _, sensor_names = make_sensor_points()
    info = {
        "system": "2D thin aluminum-like plate with two local heaters and environmental cooling",
        "saved_temperature_channels": "temperature rise over ambient, degC",
        "split_counts": SPLIT_COUNTS,
        "save_length": SAVE_LENGTH,
        "dt_save_s": DT_SAVE,
        "fs_save_hz": 1.0 / DT_SAVE,
        "dt_int_s": DT_INT,
        "t_total_s": T_TOTAL,
        "plate": {
            "Lx_m": L_X,
            "Ly_m": L_Y,
            "thickness_m": THICKNESS,
            "rho_kg_m3": RHO,
            "cp_J_kgK": CP,
            "k_W_mK": K_THERMAL,
            "h_eff_W_m2K": H_EFF,
            "ambient_C": T_AMBIENT,
            "internal_grid": [N_Y, N_X],
        },
        "heaters": {
            "names": ["u1", "u2"],
            "units": "W",
            "positions_xy_m": [HEATER_1_XY, HEATER_2_XY],
            "sigma_m": HEATER_SIGMA,
            "u1_freq_range_hz": U1_FREQ_RANGE,
            "u2_freq_range_hz": U2_FREQ_RANGE,
            "u1_power_range_W": U1_POWER_RANGE,
            "u2_power_range_W": U2_POWER_RANGE,
        },
        "sensors": {
            "grid": [SENSOR_GRID_NY, SENSOR_GRID_NX],
            "count": len(sensor_names),
            "names": sensor_names,
            "margin_m": SENSOR_MARGIN,
        },
    }
    with (output_root / "sheet_dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def plot_preview(output_root: Path, sample_path: Path | None = None) -> Path:
    if sample_path is None:
        candidates = sorted((output_root / "train").glob("*.npy"))
        if not candidates:
            raise FileNotFoundError("No train .npy files found for preview plotting.")
        sample_path = candidates[0]

    data = np.load(sample_path, allow_pickle=True).item()
    t = np.asarray(data["t"])

    sensor_names = [f"T_{i:02d}" for i in range(SENSOR_GRID_NX * SENSOR_GRID_NY)]
    selected = ["T_00", "T_09", "T_22", "T_45", "T_54", "T_77", "T_90", "T_99"]
    selected = [name for name in selected if name in sensor_names]

    fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=True)
    axes = axes.ravel()

    axes[0].plot(t, data["u1"], color="tab:red", linewidth=1.6)
    axes[0].set_title("u1 heater power")
    axes[0].set_ylabel("W")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, data["u2"], color="tab:blue", linewidth=1.6)
    axes[1].set_title("u2 heater power")
    axes[1].set_ylabel("W")
    axes[1].grid(True, alpha=0.3)

    for ax, name in zip(axes[2:], selected):
        ax.plot(t, data[name], linewidth=1.3)
        ax.set_title(f"{name} temperature rise")
        ax.set_ylabel("degC")
        ax.grid(True, alpha=0.3)

    for ax in axes[-2:]:
        ax.set_xlabel("time (s)")

    fig.suptitle(f"Sheet heat dataset preview: {sample_path.name}", y=0.995)
    fig.tight_layout()
    out_path = output_root / "preview_sheet_timeseries.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FLARE dict-style .npy data for a controlled 2D heat sheet."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where train/val/test folders will be created.",
    )
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting .npy files with matching names in split folders.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    check_output_dirs(output_root, overwrite=args.overwrite)

    specs = build_specs(output_root, args.seed)
    write_dataset_info(output_root)

    print(f"Output root: {output_root}")
    print(f"Generating {len(specs)} samples: {SPLIT_COUNTS}")
    print(f"Save length={SAVE_LENGTH}, dt_save={DT_SAVE}s, dt_int={DT_INT}s")
    print(f"Internal grid={N_Y}x{N_X}; saved sensors={SENSOR_GRID_NY}x{SENSOR_GRID_NX}")

    results: list[dict[str, object]] = []
    if args.workers <= 1:
        for i, spec in enumerate(specs, start=1):
            result = simulate_sample(spec)
            results.append(result)
            print(
                f"[{i:03d}/{len(specs):03d}] {Path(spec.output_path).name} "
                f"max_dT={result['max_temp_rise_C']:.3f} C"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_spec = {executor.submit(simulate_sample, spec): spec for spec in specs}
            for i, future in enumerate(as_completed(future_to_spec), start=1):
                spec = future_to_spec[future]
                result = future.result()
                results.append(result)
                print(
                    f"[{i:03d}/{len(specs):03d}] {Path(spec.output_path).name} "
                    f"max_dT={result['max_temp_rise_C']:.3f} C"
                )

    results = sorted(results, key=lambda r: (str(r["split"]), int(r["index"])))
    with (output_root / "sheet_generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    spec_dump = [asdict(spec) for spec in specs]
    with (output_root / "sheet_sample_specs.json").open("w", encoding="utf-8") as f:
        json.dump(spec_dump, f, indent=2)

    preview_path = plot_preview(output_root)
    print(f"Preview saved to: {preview_path}")
    print("Done.")


if __name__ == "__main__":
    main()
