from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

import generate_boucwen_random_dataset as base


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "extrapolation_10"
OUTPUT_SUMMARY = OUTPUT_DIR / "boucwen_extrapolation_10_summary.json"

N_CASES = 10
SEED = 20260610

# Keep the extrapolation moderate: no high-frequency earthquake inputs.
TARGET_PEAKS_M = [0.0038, 0.0040, 0.0044, 0.0042, 0.0038, 0.0074, 0.0074, 0.0078, 0.0070, 0.0076]
INITIAL_RMS_MPS2 = [0.26, 0.28, 0.34, 0.30, 0.26, 0.48, 0.48, 0.50, 0.44, 0.50]
ACCEPT_PEAK_RANGES_M = [
    (0.0032, 0.0048),
    (0.0032, 0.0048),
    (0.0032, 0.0048),
    (0.0032, 0.0048),
    (0.0032, 0.0048),
    (0.0050, 0.0090),
    (0.0050, 0.0090),
    (0.0050, 0.0090),
    (0.0050, 0.0090),
    (0.0050, 0.0090),
]


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def normalize_unit(signal: np.ndarray) -> np.ndarray:
    out = np.asarray(signal, dtype=np.float64).copy()
    out -= np.mean(out)
    rms = float(np.sqrt(np.mean(out**2)))
    if rms <= 0.0 or not np.isfinite(rms):
        raise ValueError("Generated input has invalid RMS.")
    return out / rms


def time_grid() -> np.ndarray:
    n = int(round(base.T_TOTAL * base.FS_INT))
    return np.arange(n, dtype=np.float64) / base.FS_INT


def unit_bandlimited(
    rng: np.random.Generator,
    freq_min: float,
    freq_max: float,
    order: int = 4,
) -> np.ndarray:
    n_keep = int(round(base.T_TOTAL * base.FS_INT))
    n_buffer = int(round(base.FILTER_BUFFER_SECONDS * base.FS_INT))
    n_long = n_keep + 2 * n_buffer

    white = rng.standard_normal(n_long)
    sos = butter(order, [freq_min, freq_max], btype="bandpass", fs=base.FS_INT, output="sos")
    filtered = sosfiltfilt(sos, white)
    return normalize_unit(filtered[n_buffer : n_buffer + n_keep])


def chirp_sin(t: np.ndarray, f0: float, f1: float, phase: float = 0.0) -> np.ndarray:
    duration = float(t[-1] - t[0]) if len(t) > 1 else base.T_TOTAL
    k = (f1 - f0) / max(duration, 1e-12)
    angle = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t) + phase
    return np.sin(angle)


def quake_envelope(t: np.ndarray) -> np.ndarray:
    attack = smoothstep(t / 2.0)
    decay = np.exp(-np.maximum(t - 5.0, 0.0) / 3.2)
    return attack * decay


def burst_envelope(t: np.ndarray, center: float, width: float) -> np.ndarray:
    x = np.abs(t - center) / (0.5 * width)
    out = np.zeros_like(t, dtype=np.float64)
    mask = x <= 1.0
    out[mask] = 0.5 * (1.0 + np.cos(np.pi * x[mask]))
    return out


def ricker(t: np.ndarray, center: float, freq: float) -> np.ndarray:
    a = np.pi * freq * (t - center)
    return (1.0 - 2.0 * a * a) * np.exp(-a * a)


def case_description(case_index: int) -> tuple[str, str]:
    descriptions = [
        ("low_mid_band_noise", "Gentle band-limited random acceleration in 2.0-3.4 Hz."),
        ("mid_band_noise", "Gentle band-limited random acceleration in 2.2-3.8 Hz."),
        ("moderate_band_noise", "Moderate 2.0-4.0 Hz band-limited random acceleration."),
        ("soft_quake_enveloped_noise", "2.2-3.8 Hz random input with a soft earthquake-like envelope."),
        ("soft_midband_enveloped_noise", "Soft 2.2-3.8 Hz random input with a low earthquake-like envelope."),
        ("up_chirp_low_mid", "Sine acceleration sweeps from 1.2 to 4.2 Hz."),
        ("down_chirp_mid_low", "Sine acceleration sweeps from 4.2 to 1.2 Hz."),
        ("two_moderate_bursts", "Two moderate 1.8-4.0 Hz random excitation bursts."),
        ("narrowband_three_hz", "Mostly narrowband 3 Hz acceleration with small filtered noise."),
        ("low_mid_mixture", "Low-frequency component plus a smaller mid-frequency component."),
    ]
    return descriptions[case_index]


def build_unit_input(case_index: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + 1009 * case_index)
    t = time_grid()

    if case_index == 0:
        ag = unit_bandlimited(rng, 2.0, 3.5) * (0.60 + 0.40 * quake_envelope(t))
    elif case_index == 1:
        ag = unit_bandlimited(rng, 2.2, 3.8) * (0.55 + 0.45 * quake_envelope(t))
    elif case_index == 2:
        ag = unit_bandlimited(rng, 2.0, 4.0)
    elif case_index == 3:
        ag = unit_bandlimited(rng, 2.2, 3.8) * (0.45 + 0.55 * quake_envelope(t))
    elif case_index == 4:
        ag = unit_bandlimited(rng, 2.2, 3.8) * (0.45 + 0.55 * quake_envelope(t))
    elif case_index == 5:
        ag = chirp_sin(t, 1.2, 4.2, 0.3) * (0.65 + 0.35 * quake_envelope(t))
    elif case_index == 6:
        ag = chirp_sin(t, 4.2, 1.2, 1.1) * (0.65 + 0.35 * quake_envelope(t))
    elif case_index == 7:
        base_noise = unit_bandlimited(rng, 1.8, 4.0)
        ag = base_noise * (burst_envelope(t, 3.0, 2.0) + 0.85 * burst_envelope(t, 7.0, 2.2))
    elif case_index == 8:
        small_noise = 0.18 * unit_bandlimited(rng, 2.0, 4.0)
        ag = np.sin(2.0 * np.pi * 3.0 * t + 0.5) * quake_envelope(t) + small_noise
    elif case_index == 9:
        ag = (
            0.70 * np.sin(2.0 * np.pi * 1.35 * t + 0.2)
            + 0.38 * np.sin(2.0 * np.pi * 3.35 * t + 1.1)
            + 0.14 * unit_bandlimited(rng, 2.0, 4.2)
        ) * (0.55 + 0.45 * quake_envelope(t))
    else:
        raise ValueError(f"Unknown case index: {case_index}")

    return normalize_unit(ag)


def simulate_case(case_index: int) -> dict:
    t_full = time_grid()
    ag_unit = build_unit_input(case_index)
    target_peak = TARGET_PEAKS_M[case_index]
    current_rms = INITIAL_RMS_MPS2[case_index]
    rms_history = []
    retry_history = []
    sim = None

    for _ in range(4):
        sim, used_rms, retries = base.simulate_with_backoff(t_full, ag_unit, current_rms)
        current_rms = float(used_rms)
        rms_history.append(current_rms)
        retry_history.append(int(retries))

        response_peak = float(sim["diagnostics"]["horizontal_abs_max_m"])
        peak_low, peak_high = ACCEPT_PEAK_RANGES_M[case_index]
        if peak_low <= response_peak <= peak_high:
            break
        if response_peak > 0.0 and np.isfinite(response_peak):
            correction = float(np.clip(target_peak / response_peak, 0.45, 2.2))
            current_rms *= correction

    if sim is None:
        raise RuntimeError("Simulation did not produce a result.")

    data = base.build_file_dict(sim)
    case_name, description = case_description(case_index)
    diagnostics = dict(sim["diagnostics"])
    diagnostics.update(
        {
            "case_index": case_index,
            "case_name": case_name,
            "description": description,
            "target_horizontal_peak_m": float(target_peak),
            "final_rms_accel_mps2": float(rms_history[-1]),
            "rms_history_mps2": [float(v) for v in rms_history],
            "convergence_retry_history": retry_history,
            "angle_rad": float(base.ANGLE_RAD),
        }
    )
    return {"case_index": case_index, "sample": data, "metadata": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers))
    if workers == 1:
        results = [simulate_case(case_index) for case_index in range(N_CASES)]
    else:
        results = []
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_case = {
                executor.submit(simulate_case, case_index): case_index
                for case_index in range(N_CASES)
            }
            for done, future in enumerate(futures.as_completed(future_to_case), start=1):
                case_index = future_to_case[future]
                result = future.result()
                results.append(result)
                print(
                    f"[boucwen {done:02d}/{N_CASES}] "
                    f"{result['metadata']['case_name']}: "
                    f"peak={result['metadata']['horizontal_abs_max_m']:.6f} m"
                )

    results = sorted(results, key=lambda item: item["case_index"])
    sample_metadata = []
    for item in results:
        case_index = item["case_index"]
        out_file = OUTPUT_DIR / f"boucwen_extrapolation_{case_index:03d}.npy"
        np.save(out_file, item["sample"], allow_pickle=True)
        metadata = dict(item["metadata"])
        metadata["file"] = str(out_file)
        sample_metadata.append(metadata)

    summary = {
        "system": "Nonlinear Bouc-Wen frame extrapolation inputs",
        "format": "10 separate dict-style FLARE .npy files",
        "source_generator": Path(__file__).name,
        "base_generator": "generate_boucwen_random_dataset.py",
        "n_samples": N_CASES,
        "output_dir": str(OUTPUT_DIR),
        "dt_save_s": float(1.0 / base.FS_SAVE),
        "save_length": base.SAVE_LENGTH,
        "input_name": base.INPUT_KEY,
        "angle_rad": float(base.ANGLE_RAD),
        "sensor_names": list(base.SENSOR_NAMES),
        "sample_metadata": sample_metadata,
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {N_CASES} files to: {OUTPUT_DIR}")
    print(f"Summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
