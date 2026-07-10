from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt


SCRIPT_DIR = Path(__file__).resolve().parent


def find_python_implementation_dir() -> Path:
    candidates = sorted(SCRIPT_DIR.glob("**/PythonImplementation"))
    for path in candidates:
        has_input = (path / "InputFiles" / "InputFile.py").exists()
        has_newmark = (path / "core" / "Newmark.py").exists()
        if has_input and has_newmark:
            return path
    raise FileNotFoundError(
        "Could not find the benchmark PythonImplementation folder under "
        f"{SCRIPT_DIR}. Keep the downloaded NonlinearBoucWenFrameBenchmark-main "
        "folder inside this directory."
    )


PY_IMPL_DIR = find_python_implementation_dir()
sys.path.insert(0, str(PY_IMPL_DIR))

from InputFiles.InputFile import Material  # noqa: E402
from core.Assembly import ModelAssembly  # noqa: E402
from core.BoucWenModel import BoucWenModel  # noqa: E402
from core.Newmark import Newmark  # noqa: E402


@dataclass
class CustomExcitation:
    time: np.ndarray
    SynthesizedAccelerogram: np.ndarray
    dt: float
    angle: float
    Amp: float = 1.0


# =========================
# Dataset settings
# =========================
SEED = 20260602
OUTPUT_ROOT = SCRIPT_DIR

N_TOTAL = 100
SPLIT_INTENSITY_COUNTS = {
    "train": {"weak": 40, "medium": 40},
    "val": {"weak": 5, "medium": 5},
    "test": {"weak": 5, "medium": 5},
}

# This matches run.py:
#   params["forcing_names"] = "u"
#   params["forcing_key"] = "u"
INPUT_KEY = "u"

# These values intentionally mirror the current ramped-sine script.
FS_INT = 500.0
FS_SAVE = 50.0
SAVE_LENGTH = 500
T_TOTAL = SAVE_LENGTH / FS_SAVE

# Non-ramped random case: stationary band-limited white-noise ground acceleration.
FREQ_MIN = 2.0
FREQ_MAX = 5.0
FILTER_ORDER = 6
FILTER_BUFFER_SECONDS = 5.0
ANGLE_RAD = math.pi / 4.0
INPUT_FILE = "FloorsExample"

# Initial RMS values are starting guesses. Each sample is then calibrated so
# its maximum horizontal displacement response is in the requested range.
INITIAL_RMS_RANGES_MPS2 = {
    "weak": (0.20, 0.70),
    "medium": (0.35, 1.10),
}
TARGET_HORIZONTAL_PEAK_RANGES_M = {
    "weak": (0.0050, 0.0075),
    "medium": (0.0075, 0.0100),
}
RESPONSE_CALIBRATION_ATTEMPTS = 2
CALIBRATION_CLAMP = (0.15, 6.0)

# If a too-strong trajectory fails to converge, reduce the input and retry.
CONVERGENCE_RETRY_SCALE = 0.65
MAX_CONVERGENCE_RETRIES = 2


def sensor_names_108() -> list[str]:
    dof_names = ["ux", "uy", "uz", "rx", "ry", "rz"]
    return [f"node{node:02d}_{dof}" for node in range(1, 19) for dof in dof_names]


def horizontal_sensor_names_36() -> list[str]:
    return [
        f"node{node:02d}_{dof}"
        for node in range(1, 19)
        for dof in ["ux", "uy"]
    ]


SENSOR_NAMES = sensor_names_108()
PREVIEW_SENSOR_NAMES = horizontal_sensor_names_36()


def build_sample_specs() -> list[dict]:
    rng = np.random.default_rng(SEED)
    specs: list[dict] = []
    global_id = 0

    for split in ("train", "val", "test"):
        split_specs = []
        for intensity, count in SPLIT_INTENSITY_COUNTS[split].items():
            rms_low, rms_high = INITIAL_RMS_RANGES_MPS2[intensity]
            peak_low, peak_high = TARGET_HORIZONTAL_PEAK_RANGES_M[intensity]
            for _ in range(count):
                split_specs.append(
                    {
                        "global_id": global_id,
                        "split": split,
                        "intensity": intensity,
                        "initial_rms_accel_mps2": float(rng.uniform(rms_low, rms_high)),
                        "target_horizontal_peak_m": float(rng.uniform(peak_low, peak_high)),
                        "seed": int(rng.integers(1, 2**31 - 1)),
                    }
                )
                global_id += 1

        rng.shuffle(split_specs)
        for split_index, spec in enumerate(split_specs):
            spec["split_index"] = split_index
            spec["file_name"] = f"boucwen_{split}_{split_index:03d}.npy"
        specs.extend(split_specs)

    if len(specs) != N_TOTAL:
        raise RuntimeError(f"Expected {N_TOTAL} samples, got {len(specs)}.")
    return specs


def output_path_for_spec(spec: dict) -> Path:
    return OUTPUT_ROOT / spec["split"] / spec["file_name"]


def make_unit_random_bandlimited(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_keep = int(round(T_TOTAL * FS_INT))
    n_buffer = int(round(FILTER_BUFFER_SECONDS * FS_INT))
    n_long = n_keep + 2 * n_buffer

    white = rng.standard_normal(n_long)
    sos = butter(
        FILTER_ORDER,
        [FREQ_MIN, FREQ_MAX],
        btype="bandpass",
        fs=FS_INT,
        output="sos",
    )
    filtered = sosfiltfilt(sos, white)

    # Crop the middle after filtering. This removes filter edge artifacts, but
    # deliberately keeps the resulting segment non-ramped and stationary.
    ag = filtered[n_buffer : n_buffer + n_keep].copy()
    ag -= np.mean(ag)
    rms = float(np.sqrt(np.mean(ag**2)))
    if rms <= 0.0 or not np.isfinite(rms):
        raise ValueError("Generated random input has invalid RMS.")
    ag /= rms

    t = np.arange(n_keep, dtype=np.float64) / FS_INT
    return t, ag


def simulate_response(t_full: np.ndarray, ag_full: np.ndarray) -> dict:
    excitation = CustomExcitation(
        time=t_full,
        SynthesizedAccelerogram=ag_full.reshape(-1, 1),
        dt=1.0 / FS_INT,
        angle=ANGLE_RAD,
    )

    material = Material([210e9, 210e9, 210e9, 210e9])
    bw_model = BoucWenModel(
        [2.0e8, 2.0e8, 2.0e8, 2.0e8],
        bwa=0.25,
        Beta=3.0,
        Gamma=2.0,
    )
    model = ModelAssembly(material, excitation, bw_model, np.ones((1, 1)), INPUT_FILE)

    solver = Newmark(model)
    with contextlib.redirect_stdout(io.StringIO()):
        results = solver.simulation()

    decimation = int(round(FS_INT / FS_SAVE))
    save_idx = np.arange(SAVE_LENGTH) * decimation

    t_save = t_full[save_idx]
    u_save = ag_full[save_idx]
    original_disps = results["OriginalDisps"].T[save_idx]
    original_vels = results["Velocities"][:108, :].T[save_idx]
    original_accs = results["Accelerations"][:108, :].T[save_idx]

    horizontal_cols = np.array(
        [6 * node + dof for node in range(18) for dof in (0, 1)],
        dtype=int,
    )
    translation_cols = np.array(
        [6 * node + dof for node in range(18) for dof in (0, 1, 2)],
        dtype=int,
    )
    rotation_cols = np.array(
        [6 * node + dof for node in range(18) for dof in (3, 4, 5)],
        dtype=int,
    )

    diagnostics = {
        "u_rms_mps2": float(np.sqrt(np.mean(u_save**2))),
        "u_peak_mps2": float(np.max(np.abs(u_save))),
        "u_first_abs_mps2": float(abs(u_save[0])),
        "horizontal_abs_max_m": float(np.max(np.abs(original_disps[:, horizontal_cols]))),
        "translation_abs_max_m": float(np.max(np.abs(original_disps[:, translation_cols]))),
        "rotation_abs_max_rad": float(np.max(np.abs(original_disps[:, rotation_cols]))),
        "x_abs_max": float(np.max(np.abs(original_disps))),
        "velocity_abs_max": float(np.max(np.abs(original_vels))),
        "acceleration_abs_max": float(np.max(np.abs(original_accs))),
    }

    return {
        "t": t_save.astype(np.float64),
        "u": u_save.astype(np.float64),
        "original_disps": original_disps.astype(np.float64),
        "diagnostics": diagnostics,
    }


def simulate_with_backoff(
    t_full: np.ndarray,
    ag_unit: np.ndarray,
    rms_accel: float,
) -> tuple[dict, float, int]:
    current_rms = float(rms_accel)
    last_error = None
    for retry in range(MAX_CONVERGENCE_RETRIES + 1):
        try:
            return simulate_response(t_full, ag_unit * current_rms), current_rms, retry
        except Exception as exc:
            last_error = exc
            current_rms *= CONVERGENCE_RETRY_SCALE
    raise RuntimeError(
        f"Simulation failed after {MAX_CONVERGENCE_RETRIES + 1} attempts. "
        f"Last error: {last_error}"
    )


def build_file_dict(sim: dict) -> dict:
    data = {
        "t": sim["t"],
        INPUT_KEY: sim["u"],
    }
    x = sim["original_disps"]
    for i, name in enumerate(SENSOR_NAMES):
        data[name] = x[:, i].astype(np.float64)
    return data


def load_sensor_matrix(data: dict, names: list[str]) -> np.ndarray:
    return np.column_stack([np.asarray(data[name], dtype=np.float64) for name in names])


def diagnostics_from_file_data(data: dict) -> dict:
    u = np.asarray(data[INPUT_KEY], dtype=np.float64)
    x = load_sensor_matrix(data, SENSOR_NAMES)
    horizontal = load_sensor_matrix(data, PREVIEW_SENSOR_NAMES)
    translation_cols = np.array(
        [6 * node + dof for node in range(18) for dof in (0, 1, 2)],
        dtype=int,
    )
    rotation_cols = np.array(
        [6 * node + dof for node in range(18) for dof in (3, 4, 5)],
        dtype=int,
    )
    return {
        "u_rms_mps2": float(np.sqrt(np.mean(u**2))),
        "u_peak_mps2": float(np.max(np.abs(u))),
        "u_first_abs_mps2": float(abs(u[0])),
        "horizontal_abs_max_m": float(np.max(np.abs(horizontal))),
        "translation_abs_max_m": float(np.max(np.abs(x[:, translation_cols]))),
        "rotation_abs_max_rad": float(np.max(np.abs(x[:, rotation_cols]))),
        "x_abs_max": float(np.max(np.abs(x))),
    }


def run_one_sample(spec: dict, overwrite: bool = True) -> dict:
    out_path = output_path_for_spec(spec)
    if out_path.exists() and not overwrite:
        data = np.load(out_path, allow_pickle=True).item()
        return {
            "status": "existing",
            "path": str(out_path),
            "split": spec["split"],
            "split_index": spec["split_index"],
            "intensity": spec["intensity"],
            "diagnostics": diagnostics_from_file_data(data),
        }

    rng = np.random.default_rng(spec["seed"])
    t_full, ag_unit = make_unit_random_bandlimited(rng)

    current_rms = float(spec["initial_rms_accel_mps2"])
    rms_history: list[float] = []
    retry_history: list[int] = []
    sim = None

    for _ in range(RESPONSE_CALIBRATION_ATTEMPTS + 1):
        sim, used_rms, retries = simulate_with_backoff(t_full, ag_unit, current_rms)
        current_rms = used_rms
        rms_history.append(float(current_rms))
        retry_history.append(int(retries))

        response_peak = float(sim["diagnostics"]["horizontal_abs_max_m"])
        low, high = TARGET_HORIZONTAL_PEAK_RANGES_M[spec["intensity"]]
        if low <= response_peak <= high:
            break

        if response_peak > 0.0 and np.isfinite(response_peak):
            correction = float(spec["target_horizontal_peak_m"]) / response_peak
            correction = float(np.clip(correction, CALIBRATION_CLAMP[0], CALIBRATION_CLAMP[1]))
            current_rms *= correction

    if sim is None:
        raise RuntimeError("Internal error: simulation did not produce a result.")

    diagnostics = dict(sim["diagnostics"])
    diagnostics.update(
        {
            "calibration_steps": len(rms_history) - 1,
            "initial_rms_accel_mps2": float(spec["initial_rms_accel_mps2"]),
            "final_rms_accel_mps2": float(rms_history[-1]),
            "target_horizontal_peak_m": float(spec["target_horizontal_peak_m"]),
            "rms_history_mps2": [float(v) for v in rms_history],
            "convergence_retry_history": [int(v) for v in retry_history],
        }
    )

    data = build_file_dict(sim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, data, allow_pickle=True)

    return {
        "status": "generated",
        "path": str(out_path),
        "split": spec["split"],
        "split_index": spec["split_index"],
        "intensity": spec["intensity"],
        "diagnostics": diagnostics,
    }


def validate_one_file(path: Path) -> dict:
    data = np.load(path, allow_pickle=True).item()
    missing = [name for name in SENSOR_NAMES if name not in data]
    allowed = set(["t", INPUT_KEY, *SENSOR_NAMES])
    extra = sorted(set(data.keys()) - allowed)
    return {
        "path": str(path),
        "has_t": "t" in data,
        "t_shape": list(np.asarray(data["t"]).shape) if "t" in data else None,
        "has_u": INPUT_KEY in data,
        "u_shape": list(np.asarray(data[INPUT_KEY]).shape) if INPUT_KEY in data else None,
        "missing_sensor_count": len(missing),
        "first_missing_sensors": missing[:5],
        "extra_keys": extra,
    }


def collect_existing_results(specs: list[dict]) -> tuple[list[dict], list[dict]]:
    existing = []
    missing = []
    for spec in specs:
        path = output_path_for_spec(spec)
        if path.exists():
            data = np.load(path, allow_pickle=True).item()
            existing.append(
                {
                    "path": str(path),
                    "split": spec["split"],
                    "split_index": spec["split_index"],
                    "intensity": spec["intensity"],
                    "diagnostics": diagnostics_from_file_data(data),
                }
            )
        else:
            missing.append(spec)
    return existing, missing


def summarize(existing: list[dict]) -> dict:
    summary = {
        "n_existing": len(existing),
        "expected_total": N_TOTAL,
        "settings": {
            "input_type": "non-ramped 2-5 Hz band-limited white-noise ground acceleration",
            "input_key": INPUT_KEY,
            "input_file": INPUT_FILE,
            "angle_rad": ANGLE_RAD,
            "fs_integration_hz": FS_INT,
            "fs_save_hz": FS_SAVE,
            "save_length": SAVE_LENGTH,
            "duration_s": T_TOTAL,
            "freq_min_hz": FREQ_MIN,
            "freq_max_hz": FREQ_MAX,
            "filter_order": FILTER_ORDER,
            "filter_buffer_seconds_each_side": FILTER_BUFFER_SECONDS,
            "num_saved_sensor_keys": len(SENSOR_NAMES),
            "recommended_training_sensor_count": len(PREVIEW_SENSOR_NAMES),
            "recommended_training_observation": "node01-node18 ux, uy",
            "target_horizontal_peak_ranges_m": TARGET_HORIZONTAL_PEAK_RANGES_M,
            "split_intensity_counts": SPLIT_INTENSITY_COUNTS,
        },
        "by_split": {},
        "by_intensity": {},
    }

    if not existing:
        return summary

    for key, target in (("split", "by_split"), ("intensity", "by_intensity")):
        for value in sorted({item[key] for item in existing}):
            group = [item for item in existing if item[key] == value]
            summary[target][value] = {"count": len(group)}
            for diag_key in (
                "u_rms_mps2",
                "u_peak_mps2",
                "horizontal_abs_max_m",
                "translation_abs_max_m",
                "rotation_abs_max_rad",
            ):
                values = np.asarray([item["diagnostics"][diag_key] for item in group], dtype=float)
                summary[target][value][diag_key] = {
                    "min": float(values.min()),
                    "mean": float(values.mean()),
                    "max": float(values.max()),
                }
    return summary


def plot_preview(sample_path: Path, out_path: Path) -> None:
    data = np.load(sample_path, allow_pickle=True).item()
    t = np.asarray(data["t"], dtype=np.float64)
    u = np.asarray(data[INPUT_KEY], dtype=np.float64)

    rng = np.random.default_rng(SEED + 999)
    selected_names = list(rng.choice(PREVIEW_SENSOR_NAMES, size=12, replace=False))

    fig, axes = plt.subplots(4, 4, figsize=(17, 10), sharex=True)
    axes = axes.ravel()
    axes[0].plot(t, u, color="#1f77b4", linewidth=1.1)
    axes[0].set_title("input: u")
    axes[0].set_ylabel("m/s^2")
    axes[0].grid(True, alpha=0.25)

    for ax in axes[1:]:
        ax.axis("off")

    for ax, name in zip(axes[1:], selected_names):
        ax.axis("on")
        ax.plot(t, np.asarray(data[name], dtype=np.float64), color="#2a9d8f", linewidth=1.0)
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.25)

    for ax in axes[-4:]:
        ax.set_xlabel("time [s]")

    fig.suptitle("Bouc-Wen non-ramped random generated sample preview", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing npy files. By default this script regenerates the dataset.",
    )
    args = parser.parse_args()
    overwrite = not bool(args.no_overwrite)

    start = time.time()
    for split in ("train", "val", "test"):
        (OUTPUT_ROOT / split).mkdir(parents=True, exist_ok=True)

    specs = build_sample_specs()
    (OUTPUT_ROOT / "random_sample_specs.json").write_text(
        json.dumps(specs, indent=2),
        encoding="utf-8",
    )

    existing_before, missing_before = collect_existing_results(specs)
    selected = specs if overwrite else missing_before
    if args.max_new is not None:
        selected = selected[: max(0, int(args.max_new))]

    print(
        json.dumps(
            {
                "event": "start",
                "input_type": "non_ramped_random_bandlimited",
                "overwrite": overwrite,
                "existing_before": len(existing_before),
                "missing_before": len(missing_before),
                "selected_this_run": len(selected),
                "workers": args.workers,
                "output_root": str(OUTPUT_ROOT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if selected:
        with futures.ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            future_to_spec = {
                executor.submit(run_one_sample, spec, overwrite): spec for spec in selected
            }
            for done, future in enumerate(futures.as_completed(future_to_spec), start=1):
                spec = future_to_spec[future]
                result = future.result()
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "done": done,
                            "total_this_run": len(selected),
                            "split": spec["split"],
                            "index": spec["split_index"],
                            "intensity": spec["intensity"],
                            "status": result["status"],
                            "u_peak_mps2": round(result["diagnostics"]["u_peak_mps2"], 5),
                            "horizontal_abs_max_m": round(
                                result["diagnostics"]["horizontal_abs_max_m"], 8
                            ),
                            "elapsed_s": round(time.time() - start, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    existing_after, missing_after = collect_existing_results(specs)
    summary = summarize(existing_after)
    summary["missing_after"] = len(missing_after)
    summary["elapsed_s"] = round(time.time() - start, 1)

    if existing_after:
        preview_rng = np.random.default_rng(SEED + len(existing_after))
        preview_item = existing_after[int(preview_rng.integers(0, len(existing_after)))]
        preview_path = OUTPUT_ROOT / "preview_random_bandlimited_timeseries.png"
        plot_preview(Path(preview_item["path"]), preview_path)
        summary["preview_path"] = str(preview_path)
        summary["validation_example"] = validate_one_file(Path(preview_item["path"]))

    (OUTPUT_ROOT / "random_generation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "event": "complete",
                "existing_after": len(existing_after),
                "missing_after": len(missing_after),
                "summary": str(OUTPUT_ROOT / "random_generation_summary.json"),
                "preview": summary.get("preview_path"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
