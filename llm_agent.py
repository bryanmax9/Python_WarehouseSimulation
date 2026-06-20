"""
llm_agent.py
============
Frontier-model COORDINATOR for the warehouse. This is the "teach a frontier
model" piece for the HUD/YC RL-environments hackathon:

  * The PPO policy handles low-level motion (fast, learned).
  * A frontier LLM handles the high-level DISPATCH decision -- given the pending
    orders, the trending item, the idle robots, and which racks stock each
    barcoded SKU, it assigns idle robots to (order, rack) pairs.
  * The environment's verifiable reward (orders up, missing-items down) SCORES
    the LLM -- so you can compare models and fine-tune (RFT) against it.

A coordinator is just a callable: state(dict) -> list of assignments, each
{"robot": <id>, "sku": <int>, "rack": [col,row]}. The env validates everything,
so a bad/empty answer just falls back to the built-in heuristic.

FireworksCoordinator talks to Fireworks' OpenAI-compatible endpoint, so it works
with any model you have credits for (GLM, Qwen, Kimi, MiniMax, ...). Set:
    FIREWORKS_API_KEY   (in .env)
    FIREWORKS_MODEL     (optional; e.g. accounts/fireworks/models/qwen3p7-plus)
"""

import os
import json
import re
import time

from warehouse_core import SKUS
from env_setup import load_keys

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_MODEL = "accounts/fireworks/models/minimax-m3"  # fast (~1s) + cheap; good fit

SYSTEM_PROMPT = (
    "You are the dispatch coordinator for an autonomous warehouse of robots that "
    "carry storage pods (racks) to human pickers. Each rack stores barcoded items "
    "(SKUs). Assign each idle robot to fulfill a pending order by sending it to a "
    "nearby rack that stocks that SKU. Goals, in order: (1) fulfill as many orders "
    "as possible, (2) prefer racks closest to the assigned robot to cut travel, "
    "(3) don't double-book a rack. Do NOT explain or think out loud. Output ONLY a "
    'JSON object of this exact form:\n'
    '{"assignments": [{"robot": <id>, "sku": <int>, "rack": [col,row]}]}'
)


def _sku_label(i):
    return f"{i}={SKUS[i][0]}({SKUS[i][1]})"


def _build_user_prompt(state):
    pend = ", ".join(_sku_label(s) for s in state["pending_orders"]) or "(none)"
    idle = "; ".join(f'R{r["id"]}@{tuple(r["cell"])}' for r in state["idle_robots"]) or "(none)"
    racks = "\n".join(
        f'  rack {tuple(r["cell"])} stocks SKUs {r["skus"]}' for r in state["racks"]
    ) or "  (none)"
    return (
        f"grid={state['grid']}  picker={state['picker']}  "
        f"trending_sku={state['trending_sku']}\n"
        f"pending orders (SKUs): {pend}\n"
        f"idle robots: {idle}\n"
        f"available racks (unreserved, stocking a pending SKU):\n{racks}\n"
        "Assign idle robots now. JSON array only."
    )


def _extract_assignments(text):
    """Pull the assignment list out of the model reply, tolerating reasoning
    tags, markdown fences, an {"assignments": [...]} object, or a bare array."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip().strip("`").strip()
    # whole reply is JSON?
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("assignments"), list):
            return obj["assignments"]
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # find an object with "assignments"
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("assignments"), list):
                return obj["assignments"]
        except Exception:
            pass
    # last resort: a bare array
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


class FireworksCoordinator:
    """Callable coordinator backed by a Fireworks-hosted model."""

    def __init__(self, model=None, temperature=0.0, timeout=60, every=1,
                 json_mode=True):
        load_keys()
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        self.model = model or os.environ.get("FIREWORKS_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        # Force JSON output via response_format. Good for base chat models (stops
        # rambling); turn OFF for our fine-tuned model, which was trained to emit
        # the exact JSON directly and instead emits a 'thought' channel when the
        # json_object constraint is applied.
        self.json_mode = json_mode
        self.timeout = timeout
        self.every = max(1, every)   # only hit the API every Nth dispatch chance
        self.calls = 0
        self.skipped = 0
        self.failures = 0
        self.min_interval = 0.4      # client-side spacing (s) to ease rate limits
        self._last = 0.0
        self._since = 0

    def __call__(self, state):
        if not self.api_key:
            raise RuntimeError("FIREWORKS_API_KEY not set (see .env / env_setup.py)")
        self._since += 1
        if (self._since - 1) % self.every != 0:   # throttled -> heuristic handles it
            self.skipped += 1
            return []
        import requests
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(state)},
            ],
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}  # stop base models rambling
        self.calls += 1
        for attempt in range(2):                 # one light retry on rate limit
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = requests.post(
                    FIREWORKS_URL, json=body, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                self._last = time.time()
                if resp.status_code == 429 and attempt == 0:
                    time.sleep(1.0)
                    continue
                resp.raise_for_status()
                return _extract_assignments(resp.json()["choices"][0]["message"]["content"])
            except Exception as e:
                self._last = time.time()
                self.failures += 1
                if attempt == 1:
                    print(f"[FireworksCoordinator] call failed ({e}); using heuristic")
                break
        return []


class MockCoordinator:
    """No-API stand-in: greedily assigns each idle robot to the nearest rack that
    stocks a pending SKU. Lets you test the dispatch pipeline offline."""

    def __call__(self, state):
        plan, taken_racks, used_sku = [], set(), []
        pend = list(state["pending_orders"])
        for r in state["idle_robots"]:
            best = None
            for rk in state["racks"]:
                cell = tuple(rk["cell"])
                if cell in taken_racks:
                    continue
                hit = next((s for s in rk["skus"] if s in pend), None)
                if hit is None:
                    continue
                d = abs(cell[0] - r["cell"][0]) + abs(cell[1] - r["cell"][1])
                if best is None or d < best[0]:
                    best = (d, cell, hit)
            if best:
                _, cell, sku = best
                plan.append({"robot": r["id"], "sku": sku, "rack": list(cell)})
                taken_racks.add(cell)
                pend.remove(sku)
        return plan


# Run `python llm_agent.py` to verify your Fireworks key + model work end-to-end.
if __name__ == "__main__":
    fc = FireworksCoordinator()
    print(f"Model:   {fc.model}")
    print(f"API key: {'set' if fc.api_key else 'MISSING - edit .env'}")
    if not fc.api_key:
        raise SystemExit("Set FIREWORKS_API_KEY in .env (see env_setup.py).")
    sample = {
        "grid": [13, 11], "picker": [6, 10], "trending_sku": 3,
        "pending_orders": [3, 7], "idle_robots": [{"id": 0, "cell": [2, 0]},
                                                  {"id": 1, "cell": [10, 0]}],
        "racks": [{"cell": [1, 2], "skus": [3, 11, 5]},
                  {"cell": [8, 4], "skus": [7, 2, 9]}],
    }
    print("\nCalling Fireworks with a sample warehouse state...")
    plan = fc(sample)
    print(f"Assignments returned: {plan}")
    print(f"calls={fc.calls} failures={fc.failures}")
    print("OK -- the model is reachable." if plan or fc.failures == 0
          else "Reachable but returned no valid plan (try another model).")
