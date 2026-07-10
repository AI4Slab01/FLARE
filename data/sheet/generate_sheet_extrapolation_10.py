from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import generate_sheet_dataset as base


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "extrapolation_10"
OUTPUT_SUMMARY = OUTPUT_DIR / "sheet_extrapolation_10_summary.json"

N_CASES = 10


def raised_cosine_pulse(t: np.ndarray, center: float, width: float) -> np.ndarray:
    x = np.abs(t - center) / (0.5 * width)
    out = np.zeros_like(t, dtype=np.float64)
    mask = x <= 1.0
    out[mask] = 0.5 * (1.0 + np.cos(np.pi * x[mask]))
    return out


def chirp_sin(t: np.ndarray, f0: float, f1: float, phase: float = 0.0) -> np.ndarray:
    duration = float(t[-1] - t[0]) if len(t) > 1 else base.T_TOTAL
    k = (f1 - f0) / max(duration, 1e-12)
    angle = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t) + phase
    return np.sin(angle)


def smooth_plateaus(
    t: np.ndarray,
    breakpoints: list[float],
    levels: list[float],
    transition_s: float = 6.0,
) -> np.ndarray:
    if len(levels) != len(breakpoints) + 1:
        raise ValueError("levels must have exactly one more entry than breakpoints.")
    out = np.full_like(t, float(levels[0]), dtype=np.float64)
    for i, boundary in enumerate(breakpoints):
        s = base.smoothstep((t - (boundary - 0.5 * transition_s)) / transition_s)
        out = (1.0 - s) * out + s * float(levels[i + 1])
    return out


def asym_triangle(t: np.ndarray, period: float, rise_fraction: float) -> np.ndarray:
    phase = (t / period) % 1.0
    rise_fraction = float(np.clip(rise_fraction, 0.1, 0.9))
    return np.where(
        phase < rise_fraction,
        phase / rise_fraction,
        1.0 - (phase - rise_fraction) / (1.0 - rise_fraction),
    )


def bounded_power(u: np.ndarray, max_power: float) -> np.ndarray:
    return np.clip(u, 0.0, max_power).astype(np.float64)


def build_case_inputs(case_index: int, t: np.ndarray) -> tuple[str, str, np.ndarray, np.ndarray]:
    ramp = base.smoothstep(t / base.INPUT_RAMP_SECONDS)

    if case_index == 0:
        name = "ultra_low_sync"
        description = "Both heaters follow mildly slower smooth cycles with conservative power."
        u1 = 3.8 + 0.8 * np.sin(2.0 * np.pi * 0.0080 * t + 0.2)
        u2 = 3.4 + 0.7 * np.sin(2.0 * np.pi * 0.0140 * t + 0.2)
    elif case_index == 1:
        name = "mild_high_frequency"
        description = "Slightly faster smooth heater modulation, with low amplitudes."
        u1 = 3.8 + 0.6 * np.sin(2.0 * np.pi * 0.026 * t + 0.4) + 0.25 * np.sin(
            2.0 * np.pi * 0.036 * t
        )
        u2 = 3.4 + 0.55 * np.sin(2.0 * np.pi * 0.038 * t + 1.1) + 0.22 * np.sin(
            2.0 * np.pi * 0.030 * t
        )
    elif case_index == 2:
        name = "up_chirp"
        description = "Heater frequencies sweep gently across the training band."
        u1 = 3.9 + 0.9 * chirp_sin(t, 0.008, 0.026, 0.5)
        u2 = 3.5 + 0.75 * chirp_sin(t, 0.014, 0.038, 1.3)
    elif case_index == 3:
        name = "down_chirp"
        description = "Heater frequencies sweep gently downward across the training band."
        u1 = 3.9 + 0.9 * chirp_sin(t, 0.026, 0.008, 1.0)
        u2 = 3.5 + 0.75 * chirp_sin(t, 0.038, 0.014, 2.0)
    elif case_index == 4:
        name = "alternating_sources"
        description = "The two fixed heaters alternate dominance with mild smooth cross-fades."
        switch = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.011 * t - 0.7)
        u1 = 2.8 + 2.4 * switch
        u2 = 2.6 + 2.2 * (1.0 - switch)
    elif case_index == 5:
        name = "delayed_pulses"
        description = "u1 pulse train is followed by a delayed u2 pulse train."
        centers = [45.0, 95.0, 145.0]
        u1 = 2.2 + sum(4.4 * raised_cosine_pulse(t, c, 24.0) for c in centers)
        u2 = 2.0 + sum(3.9 * raised_cosine_pulse(t, c + 16.0, 24.0) for c in centers)
    elif case_index == 6:
        name = "synchronized_pulses"
        description = "Both heaters receive mild synchronized heating windows."
        centers = [35.0, 84.0, 132.0, 176.0]
        pulse = sum(raised_cosine_pulse(t, c, 18.0) for c in centers)
        u1 = 2.4 + 4.7 * pulse
        u2 = 2.2 + 4.0 * pulse
    elif case_index == 7:
        name = "piecewise_plateaus"
        description = "Smooth low-contrast plateaus instead of continuous sinusoids."
        u1 = smooth_plateaus(t, [38.0, 78.0, 124.0, 164.0], [2.5, 5.1, 3.4, 5.4, 3.0])
        u2 = smooth_plateaus(t, [32.0, 86.0, 118.0, 172.0], [4.7, 2.8, 5.2, 3.3, 4.4])
    elif case_index == 8:
        name = "asymmetric_triangles"
        description = "Mild asymmetric rise and fall patterns absent from the training generator."
        u1 = 2.5 + 2.9 * asym_triangle(t + 8.0, 62.0, 0.72)
        u2 = 2.3 + 2.6 * asym_triangle(t + 19.0, 48.0, 0.34)
    elif case_index == 9:
        name = "source_shift"
        description = "Dominant heating shifts from u1 to u2 with moderate power levels."
        shift = base.smoothstep((t - 86.0) / 30.0)
        u1 = (1.0 - shift) * (5.1 + 0.35 * np.sin(2.0 * np.pi * 0.014 * t)) + shift * (
            2.8 + 0.30 * np.sin(2.0 * np.pi * 0.018 * t + 0.4)
        )
        u2 = (1.0 - shift) * (2.6 + 0.28 * np.sin(2.0 * np.pi * 0.017 * t + 1.0)) + shift * (
            4.9 + 0.36 * np.sin(2.0 * np.pi * 0.015 * t + 1.6)
        )
    else:
        raise ValueError(f"Unknown case index: {case_index}")

    if case_index in {0, 1, 2, 3, 4, 7, 8, 9}:
        u1 *= 0.86
        u2 *= 0.86

    u1 = bounded_power(u1 * ramp, 8.8)
    u2 = bounded_power(u2 * ramp, 7.8)
    return name, description, u1, u2


def simulate_with_inputs(u1: np.ndarray, u2: np.ndarray) -> dict[str, np.ndarray]:
    grid_x, grid_y, xx, yy, dx, dy = base.make_grid()

    alpha = base.K_THERMAL / (base.RHO * base.CP)
    dt_limit = 1.0 / (2.0 * alpha * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    if base.DT_INT > 0.95 * dt_limit:
        raise ValueError(
            f"DT_INT={base.DT_INT} is too large for explicit integration; "
            f"stability limit is about {dt_limit:.4g} s."
        )

    heater1 = base.make_heater_weight(xx, yy, base.HEATER_1_XY, dx, dy)
    heater2 = base.make_heater_weight(xx, yy, base.HEATER_2_XY, dx, dy)
    sensor_xs, sensor_ys, sensor_names = base.make_sensor_points()

    n_int = (base.SAVE_LENGTH - 1) * base.SAVE_EVERY + 1
    if len(u1) != n_int or len(u2) != n_int:
        raise ValueError("Input lengths do not match the internal simulation grid.")

    theta = np.zeros((base.N_Y, base.N_X), dtype=np.float64)
    saved = np.empty((base.SAVE_LENGTH, len(sensor_names)), dtype=np.float64)
    save_i = 0

    cooling_rate = base.H_EFF / (base.RHO * base.CP * base.THICKNESS)
    heat_capacity_area = base.RHO * base.CP * base.THICKNESS

    for step in range(n_int):
        if step % base.SAVE_EVERY == 0:
            saved[save_i, :] = base.bilinear_sample(
                theta, sensor_xs, sensor_ys, grid_x, grid_y
            )
            save_i += 1

        if step == n_int - 1:
            break

        source = (u1[step] * heater1 + u2[step] * heater2) / heat_capacity_area
        dtheta_dt = (
            alpha * base.laplacian_neumann(theta, dx, dy)
            - cooling_rate * theta
            + source
        )
        theta = theta + base.DT_INT * dtheta_dt

    t_int = np.arange(n_int, dtype=np.float64) * base.DT_INT
    record: dict[str, np.ndarray] = {
        "t": t_int[:: base.SAVE_EVERY].astype(np.float64),
        "u1": u1[:: base.SAVE_EVERY].astype(np.float64),
        "u2": u2[:: base.SAVE_EVERY].astype(np.float64),
    }
    for j, name in enumerate(sensor_names):
        record[name] = saved[:, j].astype(np.float64)
    return record


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_int = (base.SAVE_LENGTH - 1) * base.SAVE_EVERY + 1
    t_int = np.arange(n_int, dtype=np.float64) * base.DT_INT

    sample_metadata = []
    for case_index in range(N_CASES):
        case_name, description, u1, u2 = build_case_inputs(case_index, t_int)
        sample = simulate_with_inputs(u1, u2)
        out_file = OUTPUT_DIR / f"sheet_extrapolation_{case_index:03d}.npy"
        np.save(out_file, sample, allow_pickle=True)

        sensor_names = [name for name in sample if name.startswith("T_")]
        max_temp = max(float(np.max(sample[name])) for name in sensor_names)
        sample_metadata.append(
            {
                "case_index": case_index,
                "case_name": case_name,
                "description": description,
                "u1_peak_W": float(np.max(sample["u1"])),
                "u2_peak_W": float(np.max(sample["u2"])),
                "u1_rms_W": float(np.sqrt(np.mean(sample["u1"] ** 2))),
                "u2_rms_W": float(np.sqrt(np.mean(sample["u2"] ** 2))),
                "max_temp_rise_C": max_temp,
                "file": str(out_file),
            }
        )
        print(
            f"[sheet {case_index + 1:02d}/{N_CASES}] "
            f"{out_file.name}: {case_name}, max_dT={max_temp:.3f} C"
        )

    summary = {
        "system": "2D thin plate heat extrapolation inputs",
        "format": "10 separate dict-style FLARE .npy files",
        "source_generator": Path(__file__).name,
        "base_generator": "generate_sheet_dataset.py",
        "n_samples": N_CASES,
        "output_dir": str(OUTPUT_DIR),
        "dt_save_s": base.DT_SAVE,
        "save_length": base.SAVE_LENGTH,
        "input_names": ["u1", "u2"],
        "sensor_names": [f"T_{i:02d}" for i in range(base.SENSOR_GRID_NX * base.SENSOR_GRID_NY)],
        "sample_metadata": sample_metadata,
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {N_CASES} files to: {OUTPUT_DIR}")
    print(f"Summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
