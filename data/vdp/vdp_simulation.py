import json
import os
import shutil

import numpy as np
from scipy.integrate import solve_ivp

try:
    import imageio.v2 as imageio

    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


def forcing_signal(t, F0, omega, phi, kind="cos"):
    if kind == "cos":
        return F0 * np.cos(omega * t + phi)
    if kind == "sin":
        return F0 * np.sin(omega * t + phi)
    raise ValueError(f"Unknown forcing kind: {kind}")


def normalize_to_uint8(frames):
    frames = np.asarray(frames, dtype=np.float32)
    frames = np.clip(frames, 0.0, 1.0)
    return (255.0 * frames).astype(np.uint8)


def save_video(frames, output_path, fps=30):
    if not HAS_IMAGEIO:
        print("imageio is not available. Skipping video saving.")
        return

    frames_uint8 = normalize_to_uint8(frames)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        imageio.mimsave(output_path, frames_uint8, fps=fps, macro_block_size=1)
        print(f"Saved video: {output_path}")
    except Exception as e:
        print(f"MP4 saving failed: {e}")
        gif_path = os.path.splitext(output_path)[0] + ".gif"
        imageio.mimsave(gif_path, frames_uint8, fps=fps)
        print(f"Saved GIF instead: {gif_path}")


def save_sequence_as_model_npy(file_path, t, f, frames):
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


def _make_mesh(image_size):
    yy, xx = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing="ij")
    return yy.astype(np.float32), xx.astype(np.float32)


def vdp_rhs(t, state, mu, F0, omega, phi):
    q, v = state
    f = forcing_signal(t, F0, omega, phi, kind="cos")
    dqdt = v
    dvdt = mu * (1.0 - q**2) * v - q + f
    return [dqdt, dvdt]


def render_vdp_phase_video(
    q,
    v,
    image_size=32,
    q_lim=2.35,
    v_lim=3.8,
    sigma=3.2,
    blob_strength=1.0,
):
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    T = len(q)
    H = W = image_size
    yy, xx = _make_mesh(image_size)

    frames = np.zeros((T, H, W), dtype=np.float32)

    for k in range(T):
        qk = np.clip(q[k], -q_lim, q_lim)
        vk = np.clip(v[k], -v_lim, v_lim)

        col = (qk + q_lim) / (2.0 * q_lim) * (W - 1)
        row = H - 1 - (vk + v_lim) / (2.0 * v_lim) * (H - 1)

        blob = np.exp(-((xx - col) ** 2 + (yy - row) ** 2) / (2.0 * sigma**2))
        frames[k] = np.clip(blob_strength * blob, 0.0, 1.0)

    return frames


def render_vdp_position_video(
    q,
    image_size=32,
    q_lim=2.35,
    sigma=3.2,
    blob_strength=1.0,
):
    q = np.asarray(q, dtype=np.float32)
    T = len(q)
    H = W = image_size
    yy, xx = _make_mesh(image_size)
    row = 0.5 * (H - 1)

    frames = np.zeros((T, H, W), dtype=np.float32)

    for k in range(T):
        qk = np.clip(q[k], -q_lim, q_lim)
        col = (qk + q_lim) / (2.0 * q_lim) * (W - 1)
        blob = np.exp(-((xx - col) ** 2 + (yy - row) ** 2) / (2.0 * sigma**2))
        frames[k] = np.clip(blob_strength * blob, 0.0, 1.0)

    return frames


def generate_vdp_sequence(
    seq_len=400,
    t_end=40.0,
    burn_in=True,
    T_burn=3.0,
    F0=0.5,
    omega=1.2,
    phi=0.0,
    mu=1.0,
    y_burn0=(0.5, 0.0),
    y0_direct=(0.5, 0.0),
    image_size=32,
    blob_sigma=3.0,
    render_mode="phase",
):
    if burn_in:
        burn_len = max(50, int(seq_len * T_burn / t_end))
        t_burn = np.linspace(-T_burn, 0.0, burn_len)
        sol_burn = solve_ivp(
            fun=lambda tt, yy: vdp_rhs(tt, yy, mu, F0, omega, phi),
            t_span=(t_burn[0], t_burn[-1]),
            y0=list(y_burn0),
            t_eval=t_burn,
            method="DOP853",
            rtol=1e-9,
            atol=1e-9,
        )
        if not sol_burn.success:
            raise RuntimeError("Van der Pol burn-in integration failed.")
        y0 = sol_burn.y[:, -1]
    else:
        y0 = np.asarray(y0_direct, dtype=np.float64)

    t = np.linspace(0.0, t_end, seq_len)
    f = forcing_signal(t, F0, omega, phi, kind="cos")

    sol = solve_ivp(
        fun=lambda tt, yy: vdp_rhs(tt, yy, mu, F0, omega, phi),
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        method="DOP853",
        rtol=1e-9,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError("Van der Pol integration failed.")

    q = sol.y[0]
    v = sol.y[1]

    if render_mode == "phase":
        frames = render_vdp_phase_video(q=q, v=v, image_size=image_size, sigma=blob_sigma)
    elif render_mode == "position":
        frames = render_vdp_position_video(q=q, image_size=image_size, sigma=blob_sigma)
    else:
        raise ValueError("render_mode must be 'phase' or 'position'.")

    return {
        "t": t.astype(np.float32),
        "f": f.astype(np.float32),
        "q": q.astype(np.float32),
        "v": v.astype(np.float32),
        "frames": frames.astype(np.float32),
        "params": {
            "system": "vdp",
            "F0": float(F0),
            "omega": float(omega),
            "phi": float(phi),
            "mu": float(mu),
            "burn_in": bool(burn_in),
            "T_burn": float(T_burn),
            "render_mode": render_mode,
            "image_size": int(image_size),
            "blob_sigma": float(blob_sigma),
            "background": "none",
        },
    }


def generate_demo_videos(out_dir="video_demo_32_fast_large"):
    os.makedirs(out_dir, exist_ok=True)

    vdp = generate_vdp_sequence(
        seq_len=500,
        t_end=50.0,
        burn_in=True,
        T_burn=3.0,
        F0=0.85,
        omega=2.0,
        phi=0.5,
        mu=1.3,
        image_size=32,
        blob_sigma=3.2,
        render_mode="phase",
    )
    save_video(vdp["frames"], os.path.join(out_dir, "forced_vdp_phase_32_clean_demo.mp4"), fps=30)
    save_sequence_as_model_npy(
        os.path.join(out_dir, "forced_vdp_phase_32_clean_demo.npy"),
        vdp["t"],
        vdp["f"],
        vdp["frames"],
    )
    with open(os.path.join(out_dir, "forced_vdp_phase_32_clean_demo_params.json"), "w") as f:
        json.dump(vdp["params"], f, indent=2)

    print(f"Demo files saved to: {out_dir}")


def generate_vdp_video_dataset(
    root_dir="vdp_phase_video_dataset_32_fast_large",
    num_train=80,
    num_var=10,
    num_test=10,
    seq_len=400,
    t_end=40.0,
    image_size=32,
    seed=42,
    clear_existing=True,
    render_mode="phase",
):
    rng = np.random.default_rng(seed)

    if clear_existing and os.path.exists(root_dir):
        shutil.rmtree(root_dir)

    train_dir = os.path.join(root_dir, "train")
    var_dir = os.path.join(root_dir, "var")
    test_dir = os.path.join(root_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(var_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    mu = 1.3
    F0_range = (0.45, 0.95)
    omega_range = (1.4, 2.4)
    phi_range = (0.0, 2.0 * np.pi)
    T_burn = 2.0
    blob_sigma = 3.2

    total = num_train + num_var + num_test
    for idx in range(1, total + 1):
        if idx <= num_train:
            folder = train_dir
            local_id = idx
        elif idx <= num_train + num_var:
            folder = var_dir
            local_id = idx - num_train
        else:
            folder = test_dir
            local_id = idx - num_train - num_var

        F0 = rng.uniform(*F0_range)
        omega = rng.uniform(*omega_range)
        phi = rng.uniform(*phi_range)

        seq = generate_vdp_sequence(
            seq_len=seq_len,
            t_end=t_end,
            burn_in=True,
            T_burn=T_burn,
            F0=F0,
            omega=omega,
            phi=phi,
            mu=mu,
            image_size=image_size,
            blob_sigma=blob_sigma,
            render_mode=render_mode,
        )

        file_path = os.path.join(folder, f"vdp_{local_id:03d}.npy")
        save_sequence_as_model_npy(file_path, seq["t"], seq["f"], seq["frames"])
        print(f"Saved {file_path}")

    metadata = {
        "system": "forced_vdp_video_32_clean",
        "render_mode": render_mode,
        "num_sensors": image_size * image_size,
        "image_size": image_size,
        "seq_len": seq_len,
        "t_end": t_end,
        "mu": mu,
        "F0_range": F0_range,
        "omega_range": omega_range,
        "phi_range": phi_range,
        "T_burn": T_burn,
        "blob_sigma": blob_sigma,
        "background": "none",
        "saved_keys": ["t", "f", "x1", "...", f"x{image_size * image_size}"],
        "recommended_run_settings": {
            "num_sensors": image_size * image_size,
            "normalize_x": False,
            "normalize_forcing": True,
            "model_order": 1 if render_mode == "phase" else 2,
            "poly_order": 3,
            "include_sine": False,
            "latent_dim_expected": 2 if render_mode == "phase" else 1,
        },
        "note": "Burn-in is used only to determine initial condition and is not saved.",
    }
    with open(os.path.join(root_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Van der Pol dataset saved to: {root_dir}")


if __name__ == "__main__":
    generate_demo_videos(out_dir="video_demo_32_fast_largevdp")

    generate_vdp_video_dataset(
        root_dir="vdp_phase_video_dataset_32_fast_largeAAA",
        num_train=80,
        num_var=10,
        num_test=10,
        seq_len=400,
        t_end=40.0,
        image_size=32,
        seed=42,
        clear_existing=True,
        render_mode="position",
    )
