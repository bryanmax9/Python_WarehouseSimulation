"""
eval_agents.py
==============
The verifiable EVAL for the hackathon: run the warehouse with different
high-level COORDINATORS and score them on the same reward
(orders fulfilled up, missing-item alerts down). This is exactly the
"improve & verify frontier models" loop the hackathon is about.

Low-level motion is the trained PPO policy in every case; only the DISPATCH
brain changes, so the comparison is apples-to-apples.

Usage:
    python eval_agents.py                      # heuristic + mock (no API needed)
    python eval_agents.py --fireworks          # also eval the Fireworks LLM
    python eval_agents.py --fireworks --episodes 5
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from warehouse_env import WarehouseEnv
from llm_agent import MockCoordinator, FireworksCoordinator

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ppo_warehouse.zip"


def run_episode(policy, coordinator, seed):
    env = WarehouseEnv(seed=seed)
    env.dispatcher = coordinator          # None -> built-in heuristic
    obs, _ = env.reset(seed=seed)
    env.dispatcher = coordinator          # reset() preserves it, but be explicit
    missing = collisions = 0
    while True:
        a, _ = policy.predict(obs, deterministic=True)
        before = len(env.floor_items)
        obs, _, term, trunc, info = env.step(a)
        missing += max(0, len(env.floor_items) - before)
        for r in env.robots:
            collisions += r.last_collision == "robot"
        if trunc or term:
            break
    return {
        "orders": env.orders_fulfilled,
        "missing": missing,
        "collisions": collisions,
        "reward": round(env.episode_reward, 1),
    }


def evaluate(name, policy, coordinator, episodes, seeds):
    rows = [run_episode(policy, coordinator, s) for s in seeds[:episodes]]
    n = len(rows)
    avg = {k: sum(r[k] for r in rows) / n for k in rows[0]}
    return name, avg


def main():
    p = argparse.ArgumentParser(description="Evaluate warehouse coordinators")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fireworks", action="store_true", help="also eval the Fireworks LLM")
    p.add_argument("--llm-every", type=int, default=3,
                   help="call the LLM every Nth dispatch (raise it if you hit 429s)")
    p.add_argument("--ft-model", default=None,
                   help="fine-tuned model id -> adds a 'Fine-tuned' row for comparison")
    args = p.parse_args()

    if not MODEL_PATH.exists():
        print("No trained policy found. Run:  python train.py --timesteps 1000000")
        return
    policy = PPO.load(str(MODEL_PATH), device="cpu")
    seeds = [11, 22, 33, 44, 55]

    coordinators = [
        ("Heuristic (built-in)", None),
        ("Mock greedy LLM", MockCoordinator()),
    ]
    llm_coords = []
    if args.fireworks:
        # throttle so we stay under the serverless rate limit (heuristic fills gaps)
        fc = FireworksCoordinator(every=args.llm_every)
        coordinators.append((f"Base LLM: {fc.model.split('/')[-1]}", fc))
        llm_coords.append(("base", fc))
    if args.ft_model:
        # fine-tuned model was trained to emit the JSON directly -> no json_mode
        ftc = FireworksCoordinator(model=args.ft_model, every=args.llm_every,
                                   json_mode=False)
        coordinators.append((f"Fine-tuned: {args.ft_model.split('/')[-1]}", ftc))
        llm_coords.append(("fine-tuned", ftc))

    print(f"\nEvaluating coordinators over {args.episodes} episode(s) "
          f"(PPO drives motion in all)\n")
    results = [evaluate(n, policy, c, args.episodes, seeds) for n, c in coordinators]

    print(f"{'COORDINATOR':<28}{'orders':>8}{'missing':>9}{'collis.':>9}{'reward':>10}")
    print("-" * 64)
    for name, avg in sorted(results, key=lambda r: -r[1]["orders"]):
        print(f"{name:<28}{avg['orders']:>8.1f}{avg['missing']:>9.1f}"
              f"{avg['collisions']:>9.1f}{avg['reward']:>10.1f}")
    print("-" * 64)
    for tag, c in llm_coords:
        print(f"{tag:10} LLM: {c.calls} API calls, {c.skipped} throttled, {c.failures} failed.")
    print("Higher orders + lower missing = better dispatch. This is the verifiable")
    print("signal you fine-tune (RFT) a frontier model against.")


if __name__ == "__main__":
    main()
