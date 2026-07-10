from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import generate_panda_marker_4input_dataset as base


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "panda_extrapolation_10"
OUTPUT_DIR = OUTPUT_ROOT / "test"
OUTPUT_SUMMARY = OUTPUT_ROOT / "summary.json"

N_CASES = 10


def smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(x, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def box_envelope(t: float, start: float, end: float, edge: float = 0.35) -> float:
    start_on = float(smoothstep((t - start) / edge))
    end_off = float(smoothstep((t - end) / edge))
    return start_on * (1.0 - end_off)


def chirp_sin(t: float, f0: float, f1: float, duration: float, phase: float = 0.0) -> float:
    k = (f1 - f0) / max(duration, 1e-12)
    angle = 2.0 * math.pi * (f0 * t + 0.5 * k * t * t) + phase
    return math.sin(angle)


def active_limits() -> np.ndarray:
    return np.asarray(
        [base.ACTIVE_AMPLITUDE_LIMITS[joint_id] for joint_id in base.ACTIVE_JOINT_IDS],
        dtype=np.float64,
    )


def case_description(case_index: int) -> tuple[str, str]:
    descriptions = [
        ("slow_quasi_static", "Slow desired-angle motion with reduced offset amplitude."),
        ("mild_high_multisine", "Slightly faster smooth multisine with conservative amplitude."),
        ("in_phase", "All four active joints move in a shared phase relationship."),
        ("opposed_pairs", "joint1/joint4 and joint2/joint6 move in opposed phases."),
        ("sequential_joint_sweeps", "The dominant motion shifts from one active joint to the next."),
        ("up_chirp", "Joint trajectories sweep gently from low to higher frequency."),
        ("down_chirp", "Joint trajectories sweep gently from higher to low frequency."),
        ("smooth_step_plateaus", "Smooth desired-angle plateaus replace sinusoidal training inputs."),
        ("mid_trajectory_burst", "Most excitation is concentrated in the middle of the trajectory."),
        ("offset_hold", "Joints spend more time away from home while staying inside safe limits."),
    ]
    return descriptions[case_index]


def desired_active_offsets(case_index: int, t: float) -> np.ndarray:
    limits = active_limits()
    duration = base.TRAJECTORY_SECONDS
    ramp = float(base.smooth_ramp(np.asarray([t], dtype=np.float64))[0])

    if case_index == 0:
        phases = np.asarray([0.0, 0.7, 1.4, 2.1])
        offsets = 0.42 * limits * np.sin(2.0 * np.pi * 0.10 * t + phases)
    elif case_index == 1:
        phases = np.asarray([0.2, 1.0, 1.8, 2.5])
        offsets = 0.22 * limits * (
            np.sin(2.0 * np.pi * 0.78 * t + phases)
            + 0.35 * np.sin(2.0 * np.pi * 0.92 * t + phases[::-1])
        )
    elif case_index == 2:
        shared = math.sin(2.0 * math.pi * 0.34 * t + 0.4)
        offsets = 0.58 * limits * shared
    elif case_index == 3:
        shared = math.sin(2.0 * math.pi * 0.42 * t + 0.2)
        signs = np.asarray([1.0, -1.0, 1.0, -1.0])
        offsets = 0.58 * limits * signs * shared
    elif case_index == 4:
        offsets = np.zeros(4, dtype=np.float64)
        for i in range(4):
            start = 0.35 + i * 2.25
            end = min(start + 2.05, duration - 0.1)
            env = box_envelope(t, start, end, edge=0.35)
            offsets[i] = 0.62 * limits[i] * env * math.sin(2.0 * math.pi * 0.45 * t + 0.5 * i)
    elif case_index == 5:
        phases = np.asarray([0.0, 0.8, 1.6, 2.4])
        offsets = np.asarray(
            [
                0.54 * limits[i] * chirp_sin(t, 0.12 + 0.02 * i, 0.92 - 0.03 * i, duration, phases[i])
                for i in range(4)
            ]
        )
    elif case_index == 6:
        phases = np.asarray([0.4, 1.2, 2.0, 2.8])
        offsets = np.asarray(
            [
                0.54 * limits[i] * chirp_sin(t, 0.92 - 0.03 * i, 0.12 + 0.02 * i, duration, phases[i])
                for i in range(4)
            ]
        )
    elif case_index == 7:
        levels = np.asarray(
            [
                [0.00, 0.00, 0.00, 0.00],
                [0.48, -0.28, 0.36, -0.22],
                [-0.34, -0.18, 0.18, 0.32],
                [0.20, 0.30, -0.38, 0.24],
                [0.00, 0.00, 0.00, 0.00],
            ],
            dtype=np.float64,
        )
        breaks = [1.2, 3.4, 5.9, 8.0]
        weights = levels[0].copy()
        for j, boundary in enumerate(breaks):
            s = float(smoothstep((t - boundary) / 0.55))
            weights = (1.0 - s) * weights + s * levels[j + 1]
        offsets = limits * weights
    elif case_index == 8:
        env = box_envelope(t, 2.8, 7.2, edge=0.55)
        phases = np.asarray([0.1, 1.2, 2.0, 2.8])
        offsets = 0.64 * limits * env * np.sin(2.0 * np.pi * 0.62 * t + phases)
    elif case_index == 9:
        shift = float(smoothstep((t - 4.7) / 1.2))
        early = np.asarray([0.55, -0.34, 0.42, -0.30])
        late = np.asarray([-0.38, 0.40, -0.28, 0.34])
        weights = (1.0 - shift) * early + shift * late
        small_motion = 0.10 * np.sin(2.0 * np.pi * 0.28 * t + np.asarray([0.0, 1.0, 2.0, 3.0]))
        offsets = limits * (weights + small_motion)
    else:
        raise ValueError(f"Unknown case index: {case_index}")

    return ramp * offsets


def desired_ctrl_at_time(
    t: float,
    home_ctrl: np.ndarray,
    ctrl_range: np.ndarray,
    case_index: int,
) -> np.ndarray:
    ctrl = home_ctrl.copy()
    offsets = desired_active_offsets(case_index, t)
    for local_index, joint_id in enumerate(base.ACTIVE_JOINT_IDS):
        ctrl[joint_id] = home_ctrl[joint_id] + offsets[local_index]
        low, high = ctrl_range[joint_id]
        ctrl[joint_id] = np.clip(ctrl[joint_id], low + 0.05, high - 0.05)
    return ctrl


def generate_one_case(model: mujoco.MjModel, case_index: int) -> dict:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise ValueError("The model does not contain a 'home' keyframe.")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    home_qpos = model.key_qpos[key_id].copy()
    home_ctrl = model.key_ctrl[key_id].copy()
    ctrl_range = model.actuator_ctrlrange.copy()

    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in base.LINK_NAMES
    ]
    if any(body_id < 0 for body_id in body_ids):
        missing = [name for name, body_id in zip(base.LINK_NAMES, body_ids) if body_id < 0]
        raise ValueError(f"Missing link bodies in model: {missing}")

    model_dt = float(model.opt.timestep)
    save_interval = int(round((1.0 / base.SAVE_HZ) / model_dt))
    save_dt = save_interval * model_dt
    n_samples = int(round(base.TRAJECTORY_SECONDS * base.SAVE_HZ))
    total_steps = n_samples * save_interval

    t = np.arange(n_samples, dtype=np.float64) * save_dt
    q = np.zeros((n_samples, model.nq), dtype=np.float32)
    dq = np.zeros((n_samples, model.nv), dtype=np.float32)
    u = np.zeros((n_samples, len(base.ACTIVE_JOINT_IDS)), dtype=np.float32)
    ctrl_full = np.zeros((n_samples, model.nu), dtype=np.float32)
    x = np.zeros(
        (n_samples, len(base.LINK_NAMES) * base.LOCAL_MARKERS.shape[0] * 3),
        dtype=np.float32,
    )

    sample_index = 0
    for step in range(total_steps):
        sim_t = step * model_dt
        ctrl = desired_ctrl_at_time(sim_t, home_ctrl, ctrl_range, case_index)
        data.ctrl[:] = ctrl

        if step % save_interval == 0:
            q[sample_index] = data.qpos.astype(np.float32)
            dq[sample_index] = data.qvel.astype(np.float32)
            u[sample_index] = ctrl[base.ACTIVE_JOINT_IDS].astype(np.float32)
            ctrl_full[sample_index] = ctrl.astype(np.float32)
            x[sample_index] = base.marker_world_coordinates(data, body_ids)
            sample_index += 1

        mujoco.mj_step(model, data)

    if sample_index != n_samples:
        raise RuntimeError(f"Internal sample count mismatch: {sample_index} != {n_samples}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
        raise FloatingPointError(f"Non-finite state encountered in case {case_index}.")

    sensor_names = base.build_sensor_names()
    case_name, description = case_description(case_index)
    sample = {
        "t": t.astype(np.float32),
        "dt": np.float32(save_dt),
        "u": u,
        "x": x,
        "q": q,
        "dq": dq,
        "ctrl_full": ctrl_full,
        "q_home": home_qpos.astype(np.float32),
        "ctrl_home": home_ctrl.astype(np.float32),
        "forcing_names": list(base.FORCING_NAMES),
        "input_names": list(base.FORCING_NAMES),
        "sensor_names": sensor_names,
        "joint_names": [f"joint{i}" for i in range(1, model.njnt + 1)],
        "active_joint_names": list(base.ACTIVE_JOINT_NAMES),
        "link_names": list(base.LINK_NAMES),
        "local_markers": base.LOCAL_MARKERS.astype(np.float32),
        "column_names": ["t", *base.FORCING_NAMES, *sensor_names],
        "model_xml": str(base.MODEL_XML),
        "split": "extrapolation",
        "trajectory_id": int(case_index),
        "random_seed": 20260610,
        "extrapolation_case_name": case_name,
        "input_description": description,
        "simulation_dt": np.float32(model_dt),
        "save_hz": np.float32(base.SAVE_HZ),
        "save_interval_steps": int(save_interval),
        "trajectory_seconds": np.float32(base.TRAJECTORY_SECONDS),
    }

    for index, name in enumerate(base.FORCING_NAMES):
        sample[name] = u[:, index].astype(np.float32)
    for index, name in enumerate(sensor_names):
        sample[name] = x[:, index].astype(np.float32)
    for index, name in enumerate(sample["joint_names"]):
        sample[f"q_{name}"] = q[:, index].astype(np.float32)
        sample[f"dq_{name}"] = dq[:, index].astype(np.float32)

    return sample


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not base.MODEL_XML.exists():
        raise FileNotFoundError(base.MODEL_XML)

    model = mujoco.MjModel.from_xml_path(str(base.MODEL_XML))
    if model.nq != 7 or model.nv != 7 or model.nu != 7:
        raise ValueError(
            f"Expected panda_nohand with nq=nv=nu=7, got nq={model.nq}, nv={model.nv}, nu={model.nu}."
        )

    sample_metadata = []
    for case_index in range(N_CASES):
        sample = generate_one_case(model, case_index)
        case_name, description = case_description(case_index)
        out_file = OUTPUT_DIR / f"panda_extrapolation_traj_{case_index:03d}.npy"
        np.save(out_file, sample, allow_pickle=True)
        u = sample["u"]
        x = sample["x"]
        sample_metadata.append(
            {
                "case_index": case_index,
                "case_name": case_name,
                "description": description,
                "file": str(out_file),
                "u_min": np.min(u, axis=0).astype(float).tolist(),
                "u_max": np.max(u, axis=0).astype(float).tolist(),
                "marker_abs_max_m": float(np.max(np.abs(x))),
                "q_abs_max_rad": float(np.max(np.abs(sample["q"]))),
                "dq_abs_max_rad_s": float(np.max(np.abs(sample["dq"]))),
            }
        )
        print(f"[panda {case_index + 1:02d}/{N_CASES}] {out_file.name}: {case_name}")

    summary = {
        "system": "Franka Panda marker extrapolation inputs",
        "format": "10 separate dict-style FLARE .npy files",
        "source_generator": Path(__file__).name,
        "base_generator": "generate_panda_marker_4input_dataset.py",
        "n_samples": N_CASES,
        "output_root": str(OUTPUT_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "dt_save_s": float(1.0 / base.SAVE_HZ),
        "save_length": int(round(base.TRAJECTORY_SECONDS * base.SAVE_HZ)),
        "input_names": list(base.FORCING_NAMES),
        "active_joint_names": list(base.ACTIVE_JOINT_NAMES),
        "sensor_names": base.build_sensor_names(),
        "sample_metadata": sample_metadata,
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {N_CASES} files to: {OUTPUT_DIR}")
    print(f"Summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
