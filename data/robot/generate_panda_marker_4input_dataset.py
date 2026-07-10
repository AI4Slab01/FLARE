from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_XML = SCRIPT_DIR / "franka_emika_panda" / "panda_nohand.xml"
OUTPUT_ROOT = SCRIPT_DIR / "panda_marker_4input_world_500"

RANDOM_SEED = 0

TRAIN_TRAJECTORIES = 80
VAL_TRAJECTORIES = 10
TEST_TRAJECTORIES = 10

SAVE_HZ = 50.0
TRAJECTORY_SECONDS = 10.0

ACTIVE_JOINT_IDS = [0, 1, 3, 5]
ACTIVE_JOINT_NAMES = ["joint1", "joint2", "joint4", "joint6"]
FORCING_NAMES = [f"qdes_{name}" for name in ACTIVE_JOINT_NAMES]

LINK_NAMES = ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]

# Five virtual marker points rigidly attached to every link body, expressed in
# each body's local coordinate frame. The coordinates are small offsets around
# the body frame origin; they do not need to be MuJoCo sites or mesh vertices.
LOCAL_MARKERS = np.asarray(
    [
        [0.045, 0.015, 0.000],
        [-0.035, 0.025, 0.040],
        [0.025, -0.045, 0.030],
        [-0.025, -0.035, 0.070],
        [0.050, 0.040, 0.080],
    ],
    dtype=np.float64,
)

# Keep commands well inside joint limits. The actuators are position servos in
# this MJCF model, so ctrl is a desired joint angle, not raw torque.
ACTIVE_AMPLITUDE_LIMITS = {
    0: 0.45,
    1: 0.35,
    3: 0.45,
    5: 0.40,
}


def build_sensor_names() -> list[str]:
    names = []
    for link_name in LINK_NAMES:
        for marker_index in range(LOCAL_MARKERS.shape[0]):
            for axis in ["x", "y", "z"]:
                names.append(f"{link_name}_m{marker_index + 1}_{axis}")
    return names


def smooth_ramp(t: np.ndarray, ramp_seconds: float = 1.0) -> np.ndarray:
    """Half-cosine ramp from zero to one, then hold."""
    out = np.ones_like(t, dtype=np.float64)
    idx = t < ramp_seconds
    out[idx] = 0.5 - 0.5 * np.cos(np.pi * t[idx] / ramp_seconds)
    return out


def make_multisine_params(rng: np.random.Generator) -> dict[int, dict[str, np.ndarray]]:
    params = {}
    for joint_id in ACTIVE_JOINT_IDS:
        n_terms = 3
        freqs = np.sort(rng.uniform(0.12, 0.85, size=n_terms))
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_terms)
        weights = rng.uniform(0.4, 1.0, size=n_terms)
        weights = weights / np.sum(np.abs(weights))
        total_amp = rng.uniform(0.55, 1.0) * ACTIVE_AMPLITUDE_LIMITS[joint_id]
        amps = total_amp * weights
        params[joint_id] = {"freqs": freqs, "phases": phases, "amps": amps}
    return params


def desired_ctrl_at_time(
    t: float,
    home_ctrl: np.ndarray,
    ctrl_range: np.ndarray,
    multisine_params: dict[int, dict[str, np.ndarray]],
) -> np.ndarray:
    ctrl = home_ctrl.copy()
    ramp = smooth_ramp(np.asarray([t], dtype=np.float64))[0]
    for joint_id in ACTIVE_JOINT_IDS:
        p = multisine_params[joint_id]
        signal = np.sum(p["amps"] * np.sin(2.0 * np.pi * p["freqs"] * t + p["phases"]))
        ctrl[joint_id] = home_ctrl[joint_id] + ramp * signal
        low, high = ctrl_range[joint_id]
        ctrl[joint_id] = np.clip(ctrl[joint_id], low + 0.05, high - 0.05)
    return ctrl


def marker_world_coordinates(data: mujoco.MjData, body_ids: list[int]) -> np.ndarray:
    coords = []
    for body_id in body_ids:
        pos = data.xpos[body_id].copy()
        rot = data.xmat[body_id].reshape(3, 3).copy()
        world = pos[None, :] + LOCAL_MARKERS @ rot.T
        coords.append(world.reshape(-1))
    return np.concatenate(coords, axis=0).astype(np.float32)


def generate_one_trajectory(
    model: mujoco.MjModel,
    traj_id: int,
    split: str,
    rng: np.random.Generator,
) -> dict:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise ValueError("The model does not contain a 'home' keyframe.")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    home_qpos = model.key_qpos[key_id].copy()
    home_ctrl = model.key_ctrl[key_id].copy()
    ctrl_range = model.actuator_ctrlrange.copy()

    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in LINK_NAMES]
    if any(body_id < 0 for body_id in body_ids):
        missing = [name for name, body_id in zip(LINK_NAMES, body_ids) if body_id < 0]
        raise ValueError(f"Missing link bodies in model: {missing}")

    model_dt = float(model.opt.timestep)
    save_interval = int(round((1.0 / SAVE_HZ) / model_dt))
    if save_interval < 1:
        raise ValueError("SAVE_HZ is higher than the MuJoCo simulation frequency.")
    save_dt = save_interval * model_dt
    n_samples = int(round(TRAJECTORY_SECONDS * SAVE_HZ))
    total_steps = n_samples * save_interval

    t = np.arange(n_samples, dtype=np.float64) * save_dt
    q = np.zeros((n_samples, model.nq), dtype=np.float32)
    dq = np.zeros((n_samples, model.nv), dtype=np.float32)
    u = np.zeros((n_samples, len(ACTIVE_JOINT_IDS)), dtype=np.float32)
    ctrl_full = np.zeros((n_samples, model.nu), dtype=np.float32)
    x = np.zeros((n_samples, len(LINK_NAMES) * LOCAL_MARKERS.shape[0] * 3), dtype=np.float32)

    multisine_params = make_multisine_params(rng)

    sample_index = 0
    for step in range(total_steps):
        sim_t = step * model_dt
        ctrl = desired_ctrl_at_time(sim_t, home_ctrl, ctrl_range, multisine_params)
        data.ctrl[:] = ctrl

        if step % save_interval == 0:
            q[sample_index] = data.qpos.astype(np.float32)
            dq[sample_index] = data.qvel.astype(np.float32)
            u[sample_index] = ctrl[ACTIVE_JOINT_IDS].astype(np.float32)
            ctrl_full[sample_index] = ctrl.astype(np.float32)
            x[sample_index] = marker_world_coordinates(data, body_ids)
            sample_index += 1

        mujoco.mj_step(model, data)

    if sample_index != n_samples:
        raise RuntimeError(f"Internal sample count mismatch: {sample_index} != {n_samples}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
        raise FloatingPointError(f"Non-finite state encountered in trajectory {traj_id}.")

    sensor_names = build_sensor_names()
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
        "forcing_names": list(FORCING_NAMES),
        "input_names": list(FORCING_NAMES),
        "sensor_names": sensor_names,
        "joint_names": [f"joint{i}" for i in range(1, model.njnt + 1)],
        "active_joint_names": list(ACTIVE_JOINT_NAMES),
        "link_names": list(LINK_NAMES),
        "local_markers": LOCAL_MARKERS.astype(np.float32),
        "column_names": ["t", *FORCING_NAMES, *sensor_names],
        "model_xml": str(MODEL_XML),
        "split": split,
        "trajectory_id": int(traj_id),
        "random_seed": int(RANDOM_SEED),
        "simulation_dt": np.float32(model_dt),
        "save_hz": np.float32(SAVE_HZ),
        "save_interval_steps": int(save_interval),
        "trajectory_seconds": np.float32(TRAJECTORY_SECONDS),
    }

    for index, name in enumerate(FORCING_NAMES):
        sample[name] = u[:, index].astype(np.float32)
    for index, name in enumerate(sensor_names):
        sample[name] = x[:, index].astype(np.float32)
    for index, name in enumerate(sample["joint_names"]):
        sample[f"q_{name}"] = q[:, index].astype(np.float32)
        sample[f"dq_{name}"] = dq[:, index].astype(np.float32)

    return sample


def write_dataset() -> None:
    if not MODEL_XML.exists():
        raise FileNotFoundError(MODEL_XML)

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    if model.nq != 7 or model.nv != 7 or model.nu != 7:
        raise ValueError(
            f"Expected panda_nohand with nq=nv=nu=7, got nq={model.nq}, nv={model.nv}, nu={model.nu}."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    split_counts = {
        "train": TRAIN_TRAJECTORIES,
        "val": VAL_TRAJECTORIES,
        "test": TEST_TRAJECTORIES,
    }
    for split in split_counts:
        (OUTPUT_ROOT / split).mkdir(parents=True, exist_ok=True)

    summary = {
        "model_xml": str(MODEL_XML),
        "output_root": str(OUTPUT_ROOT),
        "random_seed": RANDOM_SEED,
        "simulation_dt": float(model.opt.timestep),
        "save_hz": SAVE_HZ,
        "trajectory_seconds": TRAJECTORY_SECONDS,
        "samples_per_trajectory": int(round(TRAJECTORY_SECONDS * SAVE_HZ)),
        "train_trajectories": TRAIN_TRAJECTORIES,
        "val_trajectories": VAL_TRAJECTORIES,
        "test_trajectories": TEST_TRAJECTORIES,
        "forcing_dim": len(FORCING_NAMES),
        "sensor_dim": len(build_sensor_names()),
        "state_q_dim": int(model.nq),
        "state_dq_dim": int(model.nv),
        "forcing_names": list(FORCING_NAMES),
        "sensor_names": build_sensor_names(),
        "joint_names": [f"joint{i}" for i in range(1, model.njnt + 1)],
        "active_joint_names": list(ACTIVE_JOINT_NAMES),
        "link_names": list(LINK_NAMES),
        "local_markers": LOCAL_MARKERS.tolist(),
        "splits": {"train": [], "val": [], "test": []},
    }

    traj_id = 0
    for split, count in split_counts.items():
        for split_index in range(count):
            sample = generate_one_trajectory(model, traj_id=traj_id, split=split, rng=rng)
            out_file = OUTPUT_ROOT / split / f"panda_{split}_traj_{split_index:03d}.npy"
            np.save(out_file, sample, allow_pickle=True)
            summary["splits"][split].append(
                {
                    "trajectory_id": traj_id,
                    "file": str(out_file),
                    "shape_t": list(sample["t"].shape),
                    "shape_u": list(sample["u"].shape),
                    "shape_x": list(sample["x"].shape),
                    "shape_q": list(sample["q"].shape),
                    "shape_dq": list(sample["dq"].shape),
                }
            )
            traj_id += 1

    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote dataset to: {OUTPUT_ROOT}")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {len(summary['splits'][split])} files")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    write_dataset()
