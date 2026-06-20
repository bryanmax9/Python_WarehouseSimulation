"""
train.py
========
Local PPO training (CPU-only) for the warehouse robots. Trains a SMALL number of
timesteps to verify the whole pipeline works on your laptop, logs reward over
time, plots a learning curve, and saves a checkpoint.

The heavy 1,000,000-step run goes to Modal cloud -> see modal_train.py. Both use
the exact same env, so checkpoints are interchangeable.

Usage:
    python train.py                                  # 50k steps
    python train.py --timesteps 200000 --num-worlds 6
Outputs:
    models/ppo_warehouse.zip       (the trained shared policy)
    models/reward_curve.png        (learning curve)
    models/reward_log.csv          (timesteps, mean episode reward)
"""

import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from warehouse_env import make_vec_env

MODELS = Path(__file__).resolve().parent / "models"
MODELS.mkdir(exist_ok=True)
MODEL_PATH = MODELS / "ppo_warehouse.zip"

PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=512,
    batch_size=512,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,     # exploration helps with the collision penalties
    vf_coef=0.5,
    clip_range=0.2,
)


class RewardLogger(BaseCallback):
    """Records mean episode reward at the end of every rollout."""

    def __init__(self):
        super().__init__()
        self.history = []  # list[(timesteps, mean_reward)]

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        if buf:
            mean_r = sum(e["r"] for e in buf) / len(buf)
            self.history.append((self.num_timesteps, mean_r))
            print(f"  [{self.num_timesteps:>8} steps]  mean_ep_reward = {mean_r:8.1f}")


def main():
    p = argparse.ArgumentParser(description="Local PPO training for WarehouseEnv")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--num-worlds", type=int, default=4)
    p.add_argument("--out", default=str(MODEL_PATH))
    args = p.parse_args()

    print(f"Building env: {args.num_worlds} worlds x robots "
          f"(parameter-sharing). Device: CPU.")
    venv = VecMonitor(make_vec_env(num_worlds=args.num_worlds, seed=0))

    model = PPO("MlpPolicy", venv, device="cpu", verbose=0, **PPO_KWARGS)
    logger = RewardLogger()

    print(f"Training for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=logger, progress_bar=True)

    model.save(args.out)
    venv.close()
    print(f"\nSaved model -> {args.out}")

    # write reward log + plot
    if logger.history:
        with open(MODELS / "reward_log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "mean_episode_reward"])
            w.writerows(logger.history)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs, ys = zip(*logger.history)
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys, marker="o", ms=3)
            plt.xlabel("timesteps")
            plt.ylabel("mean episode reward")
            plt.title("Warehouse PPO learning curve")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(MODELS / "reward_curve.png", dpi=110)
            print(f"Saved learning curve -> {MODELS / 'reward_curve.png'}")
        except Exception as e:
            print(f"(plot skipped: {e})")

    print("\nVisualize the trained robots:  python visualize.py")


if __name__ == "__main__":
    main()
