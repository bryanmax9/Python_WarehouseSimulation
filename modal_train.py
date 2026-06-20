"""
modal_train.py
==============
The HEAVY training run on Modal's cloud (CPU), using the hackathon sponsor's
free credits. Same WarehouseEnv + PPO as train.py, but 1,000,000 timesteps with
more parallel worlds, saved to a Modal Volume for download.

Setup + run:
    pip install modal
    modal token new                  # one-time browser auth

    modal run modal_train.py                                   # 1M steps
    modal run modal_train.py --timesteps 2000000 --num-worlds 16

    # download the trained policy into ./models:
    modal volume get warehouse-models ppo_warehouse_cloud.zip models/ppo_warehouse_cloud.zip

Then visualize:  python visualize.py
"""

import modal

# Cloud image: identical CPU-only training stack + our source modules.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.12.1", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gymnasium==1.0.0", "stable-baselines3==2.6.0", "numpy==2.4.6")
    .add_local_python_source("warehouse_core", "warehouse_env")
)

volume = modal.Volume.from_name("warehouse-models", create_if_missing=True)
app = modal.App("warehouse-ppo", image=image)
MOUNT = "/models"


@app.function(volumes={MOUNT: volume}, cpu=8.0, memory=8192, timeout=24 * 3600)
def train_cloud(timesteps: int, num_worlds: int) -> str:
    import os
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3.common.callbacks import CheckpointCallback

    from warehouse_env import make_vec_env

    print(f"[cloud] {num_worlds} worlds, {timesteps:,} timesteps, CPU")
    venv = VecMonitor(make_vec_env(num_worlds=num_worlds, seed=0))

    model = PPO(
        "MlpPolicy", venv, device="cpu", verbose=1,
        learning_rate=3e-4, n_steps=512, batch_size=512, n_epochs=4,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01, vf_coef=0.5, clip_range=0.2,
    )

    # periodic checkpoints so a long run is never fully lost
    ckpt = CheckpointCallback(
        save_freq=max(100_000 // venv.num_envs, 1),
        save_path=MOUNT, name_prefix="ppo_warehouse_ckpt",
    )
    model.learn(total_timesteps=timesteps, callback=ckpt, progress_bar=False)

    out = os.path.join(MOUNT, "ppo_warehouse_cloud.zip")
    model.save(out)
    volume.commit()
    venv.close()
    print(f"[cloud] saved -> {out}")
    return out


@app.local_entrypoint()
def main(timesteps: int = 1_000_000, num_worlds: int = 8):
    print(f"Dispatching cloud training: {timesteps:,} timesteps, {num_worlds} worlds")
    saved = train_cloud.remote(timesteps, num_worlds)
    print(f"\nDone. Saved in volume 'warehouse-models' at: {saved}")
    print("Download:  modal volume get warehouse-models "
          "ppo_warehouse_cloud.zip models/ppo_warehouse_cloud.zip")
