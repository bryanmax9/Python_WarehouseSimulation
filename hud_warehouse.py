"""
hud_warehouse.py
================
HUD (v6) integration for the warehouse RL environment.

This exposes the warehouse's high-level DISPATCH decision as a HUD task with a
*verifiable* reward, so a frontier model can be evaluated (and later RL-trained)
on hud.ai. It is the same loop the hackathon is about:

    environment  ->  run a frontier model as the coordinator  ->  verifiable
    reward (orders fulfilled up)  ->  improve & compare models.

How it maps onto our stack
--------------------------
  * PPO drives low-level robot MOTION (loaded from models/ppo_warehouse.zip).
  * The HUD agent (Claude / a Fireworks model / our fine-tuned model) is the
    DISPATCH BRAIN: each decision point we hand it the warehouse state and it
    returns JSON assignments {robot, sku, rack}. We apply them, roll the world
    forward, and repeat.
  * Reward = orders_fulfilled / target_orders, clamped to [0, 1] -- counted by
    the environment, not by a model. That is the whole point: verifiable.

We deliberately reuse the SAME prompt + parser as llm_agent.py / the RFT
dataset, so the HUD eval, the live eval (eval_agents.py), and the fine-tune all
speak the identical protocol.

------------------------------------------------------------------------------
CREDITS  (you have a small HUD balance -- protect it)
------------------------------------------------------------------------------
HUD credits are consumed only by "Task Runs" whose model calls go through the
HUD *gateway*. To develop/test for ~free, route the model straight to a provider
(Fireworks: we have credits there) and run the env locally:

  FREE offline wiring check (no HUD, no model at all):
      python hud_warehouse.py

  ~FREE eval (model billed to FIREWORKS, NOT HUD; env runs locally):
      export OPENAI_API_KEY=$FIREWORKS_API_KEY                 # provider key present
      hud eval hud_warehouse.py openai_compatible \
          --model accounts/fireworks/models/minimax-m3 \
          -c base_url=https://api.fireworks.ai/inference/v1 \
          --runtime local --max-steps 10

  SPENDS HUD CREDITS (do this once, for the dashboard trace at demo time):
      hud eval hud_warehouse.py claude --max-steps 4
  Rule of thumb: this task is single-turn -> ONE model call per task, and
  `hud eval <file>` runs only the FIRST task unless you pass --all (3 tasks).
"""

from pathlib import Path

import numpy as np

try:
    from hud import Environment
except ModuleNotFoundError:
    # Lets the FREE offline wiring check (and `import`) work before hud-python
    # is installed. `uv tool install hud-python` to enable real `hud eval`.
    class Environment:  # minimal stand-in: just records the template
        def __init__(self, *a, **k):
            pass

        def template(self, *a, **k):
            def deco(fn):
                return fn
            return deco

from warehouse_env import WarehouseEnv, IDLE
from llm_agent import SYSTEM_PROMPT, _build_user_prompt, _extract_assignments

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ppo_warehouse.zip"

env = Environment(name="warehouse-dispatch", version="0.1.0")


# --------------------------------------------------------------------------- #
# PPO motion policy (optional): loaded once, falls back to the scripted greedy
# motion if torch / the trained model is unavailable. Either way the DISPATCH
# brain (the thing under eval) is the HUD agent, not this.
# --------------------------------------------------------------------------- #
_POLICY = None
_POLICY_TRIED = False


def _load_policy():
    global _POLICY, _POLICY_TRIED
    if _POLICY_TRIED:
        return _POLICY
    _POLICY_TRIED = True
    try:
        from stable_baselines3 import PPO
        if MODEL_PATH.exists():
            _POLICY = PPO.load(str(MODEL_PATH), device="cpu")
    except Exception:
        _POLICY = None
    return _POLICY


def _motion_actions(world, obs):
    """One action per robot for this tick (trained PPO, else scripted greedy)."""
    policy = _load_policy()
    if policy is not None:
        actions, _ = policy.predict(obs, deterministic=True)
        return actions
    return np.array([world.greedy_action(r) for r in world.robots])


def _roll(world, obs, n):
    """Advance the world up to n ticks. Returns (obs, done)."""
    for _ in range(n):
        obs, _, term, trunc, _ = world.step(_motion_actions(world, obs))
        if term or trunc:
            return obs, True
    return obs, False


def _dispatch_prompt(state):
    """Same instructions + state encoding the model sees everywhere else."""
    return f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(state)}"


def _make_world(seed):
    """A world where the AGENT is the sole dispatcher (heuristic auto-fill off),
    so its decisions alone determine throughput -- the honest HUD signal. The
    agent's plan is applied exactly once per decision."""
    world = WarehouseEnv(seed=seed)
    holder = {"plan": []}

    def _dispatcher(_state):
        plan = holder["plan"]
        holder["plan"] = []
        return plan

    world.dispatcher = _dispatcher
    world.auto_heuristic_dispatch = False
    obs, _ = world.reset(seed=seed)
    world.dispatcher = _dispatcher          # reset() preserves these, but be
    world.auto_heuristic_dispatch = False   # explicit
    return world, holder, obs


# --------------------------------------------------------------------------- #
# Single-turn dispatch task.
#
# HUD's task protocol (this SDK version) is: the template yields ONE prompt, the
# agent answers, and the template's NEXT yield is the numeric grade. So we frame
# one crisp, verifiable decision:
#
#   1. Warm the warehouse to a BUSY state (orders piled up, robots idle) by
#      running with no dispatch.
#   2. Hand that state to the agent; it returns dispatch assignments (JSON).
#   3. Apply them and let the assigned robots run; the GRADE is how many of the
#      orders the agent could have dispatched actually get fulfilled -- counted
#      by the environment, not a model. Verifiable, and one model call per task
#      (cheap on credits).
#
# This is exactly the (state -> assignment) decision the RFT dataset teaches, so
# the HUD eval, eval_agents.py, and the fine-tune all score the same skill.
# --------------------------------------------------------------------------- #
def _warmup(seed, warmup_steps):
    """Run with no dispatch so orders accumulate and robots go idle; return the
    busy decision state plus how many orders are dispatchable right now."""
    world, holder, obs = _make_world(seed)
    obs, _ = _roll(world, obs, warmup_steps)
    idle = [i for i, r in enumerate(world.robots) if r.phase == IDLE]
    state = world._dispatch_state(idle)
    assignable = min(len(idle), len(set(state["pending_orders"])))
    return world, holder, obs, state, assignable


def _settle_score(world, holder, obs, plan, target_orders, settle_steps):
    """Apply the agent's plan once, let those robots run, and score by the orders
    fulfilled vs a fixed target (a robot dispatched to the trending item hovers
    and re-delivers, so smart dispatch snowballs). Returns (score, gained)."""
    holder["plan"] = plan
    before = world.orders_fulfilled
    _roll(world, obs, settle_steps)
    gained = world.orders_fulfilled - before
    return max(0.0, min(1.0, gained / float(target_orders))), gained


@env.template(
    description="Given a busy warehouse state, dispatch idle robots to pending "
                "orders/racks as JSON; rewarded for orders the dispatch fulfills.",
)
async def warehouse_dispatch(seed: int = 11, warmup_steps: int = 72,
                             settle_steps: int = 160, target_orders: int = 15):
    world, holder, obs, state, assignable = _warmup(seed, warmup_steps)
    answer = yield _dispatch_prompt(state)
    score, _ = _settle_score(world, holder, obs,
                             _extract_assignments(answer or ""),
                             target_orders, settle_steps)
    yield score


def evaluate_local(coordinator, seed: int = 11, warmup_steps: int = 72,
                   settle_steps: int = 160, target_orders: int = 15):
    """Score any sync coordinator (state-dict -> plan) through the SAME task the
    HUD eval uses -- locally, no HUD, no model, no credits. Handy for offline
    checks and for sanity-comparing against the live eval."""
    world, holder, obs, state, assignable = _warmup(seed, warmup_steps)
    plan = coordinator(state) or []
    score, gained = _settle_score(world, holder, obs, plan,
                                  target_orders, settle_steps)
    return {"orders_from_dispatch": gained, "assignable": assignable,
            "reward": round(score, 3)}


# `hud eval hud_warehouse.py <agent>` runs the FIRST task by default; pass --all
# for the whole set. A few fixed seeds give a small, reproducible task suite.
tasks = [warehouse_dispatch(seed=s) for s in (11, 22, 33)]


# --------------------------------------------------------------------------- #
# FREE offline check -- no HUD, no model, no credits. Scores the same task two
# ways to prove the agent's dispatch (not the env) drives the reward.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from llm_agent import MockCoordinator

    print("FREE offline check (no HUD, no model, no credits)\n")

    # 1) A no-op agent dispatches nothing -> reward ~0 (proves the agent is the
    #    sole dispatcher; the env isn't quietly doing the work).
    noop = evaluate_local(lambda s: [])
    print(f"  no-op agent     : {noop}")

    # 2) The greedy MockCoordinator dispatches well -> reward > 0 (proves valid
    #    JSON plans flow through and produce a real, verifiable reward).
    mock = evaluate_local(MockCoordinator())
    print(f"  mock greedy LLM : {mock}")

    ok = mock["reward"] > noop["reward"] and mock["orders_from_dispatch"] > 0
    print(f"\n{'OK' if ok else 'PROBLEM'}: the agent's dispatch drives the "
          f"verifiable reward (mock {mock['reward']} vs no-op {noop['reward']}).")
    print("Next: `hud eval hud_warehouse.py ...` runs a real model as the agent "
          "(see the CREDITS notes at the top of this file).")
