"""
rft_dataset.py
==============
RFT step 1/3 -- build the post-training dataset.

We roll out episodes of the warehouse with a strong dispatch TEACHER (the
nearest-unreserved-rack heuristic, which is near-optimal) while PPO drives
motion. Every dispatch decision becomes a supervised example:

    system  : the dispatch instructions (same prompt used at eval time)
    user    : the warehouse state (pending orders, idle robots, racks)
    assistant: the teacher's optimal assignment, as JSON

Optionally keep only decisions from HIGH-REWARD episodes (reward-filtered /
rejection-sampling RFT) so the model imitates verifiably-good behavior.

Output: data/warehouse_dispatch_sft.jsonl  (Fireworks chat fine-tuning format)

Usage:
    python rft_dataset.py                      # 40 episodes
    python rft_dataset.py --episodes 80 --min-reward -2200
"""

import json
import argparse
from pathlib import Path

from stable_baselines3 import PPO

from warehouse_env import WarehouseEnv
from llm_agent import MockCoordinator, SYSTEM_PROMPT, _build_user_prompt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
MODEL_PATH = ROOT / "models" / "ppo_warehouse.zip"
OUT_PATH = DATA / "warehouse_dispatch_sft.jsonl"


def generate(episodes, out_path, min_reward=None):
    if not MODEL_PATH.exists():
        print("Need a trained policy. Run: python train.py --timesteps 1000000")
        return
    policy = PPO.load(str(MODEL_PATH), device="cpu")
    teacher = MockCoordinator()          # near-optimal nearest-rack dispatcher

    samples = []
    kept_eps = 0
    for ep in range(episodes):
        seed = 1000 + ep
        env = WarehouseEnv(seed=seed)
        obs, _ = env.reset(seed=seed)
        records = []  # (user_prompt, plan) collected this episode

        def logging_dispatch(state, _rec=records):
            plan = teacher(state)
            if plan:
                _rec.append((_build_user_prompt(state), plan))
            return plan

        env.dispatcher = logging_dispatch
        while True:
            a, _ = policy.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break

        keep = (min_reward is None) or (env.episode_reward >= min_reward)
        flag = "keep" if keep else "drop"
        print(f"ep {ep:>3}: orders={env.orders_fulfilled:>3} "
              f"reward={env.episode_reward:>8.0f} decisions={len(records):>3} [{flag}]")
        if keep:
            kept_eps += 1
            for user_prompt, plan in records:
                samples.append({"messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": json.dumps({"assignments": plan})},
                ]})

    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"\nKept {kept_eps}/{episodes} episodes -> {len(samples)} training examples")
    print(f"Wrote {out_path}")
    if samples:
        print("\nSample example (assistant target):")
        print("  " + samples[0]["messages"][2]["content"])


def main():
    p = argparse.ArgumentParser(description="Build the RFT dataset")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--min-reward", type=float, default=None,
                   help="only keep decisions from episodes with reward >= this")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()
    generate(args.episodes, args.out, args.min_reward)


if __name__ == "__main__":
    main()
