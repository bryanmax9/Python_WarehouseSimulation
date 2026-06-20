# PROJECT STATUS — what's done, what's left, and context to resume

> This doc is the memory/handoff for Claude (and the team). It captures the
> architecture, the decisions already made (and WHY — so we don't repeat dead
> ends), current state, and the prioritized work left to win the
> **HUD/YC RSI RL-Environments hackathon**.

---

## 1. The one-sentence thesis (keep everything pointed at this)
**A verifiable warehouse RL environment that improves a *frontier model* at fleet
coordination via RFT** — train fast PPO "muscle," let an LLM be the "brain"
(dispatch), and use the env's reward to fine-tune the brain and prove it got better.
Judges reward *environments that improve frontier models*, NOT a good PPO controller.

## 2. Architecture (hierarchical)
- **PPO policy** = low-level robot motion (reflexes). Small net, NOT a frontier
  model. Already trained (~41 orders/episode, missing ~1–4). Shown in the pygame demo.
- **LLM coordinator** = high-level dispatch (brain): assigns idle robots →
  (order, rack). This is the frontier model we RFT. Plugged in via `env.dispatcher`.
- **Environment** = `WarehouseEnv` (Gymnasium). Verifiable reward = orders ↑,
  missing-item alerts ↓, collisions ↓. This is the product.

## 3. File map
| File | Role |
|---|---|
| `warehouse_core.py` | grid, pods/racks (+stock qty), 20 SKUs, OrderQueue+trending, AlertSystem, BFS |
| `warehouse_env.py` | `WarehouseEnv` (6 robots, 25-dim obs, 7 actions), rewards, deadlock-breaker, `dispatcher` hook, `ParameterSharingVecEnv`, `gym.register("Warehouse-v0")`, `# HUD_HOOK`s |
| `train.py` / `modal_train.py` | local / Modal-cloud PPO training |
| `visualize.py` | pygame demo (pods, robots carry tower, monitor, pickers, fallen-item alerts, click-to-inspect robot & rack, AI-reasoning panel w/ action probs, live RL log) |
| `llm_agent.py` | `FireworksCoordinator` (Minimax M3) + `MockCoordinator` + prompts |
| `eval_agents.py` | verifiable leaderboard: heuristic vs base-LLM vs `--ft-model` |
| `hud_warehouse.py` | **HUD (v6) integration**: dispatch decision as a HUD task with a verifiable reward; `evaluate_local()` for free offline scoring |
| `rft_dataset.py` | builds SFT dataset from teacher rollouts → `data/*.jsonl` |
| `rft_finetune.py` | validates dataset + prints Fireworks fine-tune steps |
| `env_setup.py` | loads `.env`, reports sponsor keys |

## 4. Key decisions & GOTCHAS (do NOT relitigate)
- **Open floor (robots drive UNDER pods).** `is_walkable` = in-bounds only.
  ❌ Do NOT make carrying robots collide with pods — tested, it turns delivery into
  a maze the reactive PPO can't learn: **orders crashed 32 → 9**. Reverted. Humans
  (tall) DO respect pods via `is_walkable_human`.
- **Collision penalties moderate: robot −5, human −8** (NOT the spec's −100/−50 —
  those make PPO freeze to avoid moving). Human penalized more than robot on purpose.
- **Pickup/drop auto-fire on arrival**; delivery triggers within manhattan ≤ 2 of
  picker (avoids single-cell gridlock). Dense shaping: progress +1.0/cell, pickup +2.0.
- **Deadlock-breaker** (`_break_deadlocks`) nudges stuck robots → no permanent gridlock.
- **6 robots**; policy is per-agent (obs fixed 25-dim) so it works for any N without retrain.
- **Item drops scale with collisions** (jostle) → smoother policy = fewer missing alerts.
- **LLM models:** Minimax M3 ≈ 1s/call (good). Qwen3.7 = slow reasoning model (~60s, avoid).
  Serverless **rate limit is low → 429s**; throttle with `every=N` (eval default 3).
- **RFT dataset prompt MUST match eval prompt** (`SYSTEM_PROMPT` + `_build_user_prompt`).
- **Fine-tune base must be a "Tunable (LoRA)" model** (Qwen3.7 Plus is not tunable).
- **Modal auth** = `~/.modal.toml` (token set), not `.env`.
- `WarehouseEnv.reset()` preserves `dispatcher`, `human_random`, and
  `auto_heuristic_dispatch` (all set in `__init__`).
- **HUD = `hud_warehouse.py`** (NOT the old invented `hud.wrap`/`hud.run` hooks).
  - HUD v6 task protocol in the installed SDK is **single-turn**: template yields
    ONE prompt, agent answers, the template's NEXT yield is the numeric grade.
    Yielding a 2nd prompt (string) before the grade -> error "task graded with
    str". So the task is one decision: warm to a busy state -> agent dispatches
    -> grade = orders fulfilled / target (a trending-item assign hovers & snowballs).
  - `WarehouseEnv.auto_heuristic_dispatch=False` makes the AGENT the sole
    dispatcher (heuristic no longer auto-fills idle robots) -> the agent's call
    actually moves the reward (no-op 0.0 vs greedy ~0.87). Default True elsewhere.
  - **CREDITS:** HUD bills only model calls through its *gateway*
    (`inference.beta.hud.ai`). Provider-direct (Fireworks `base_url`+`api_key`,
    `--runtime local`, no `--gateway`) = **0 HUD credits**; only telemetry traces
    hit the platform. We have ~10 credits -> dev/test on Fireworks, spend HUD
    credits only on the one `claude`/gateway demo trace. No CLI balance command;
    check the hud.ai dashboard.

## 5. Sponsor status
| Sponsor | Status | Notes |
|---|---|---|
| **Fireworks** | ✅ working | Minimax M3 coordinator; RFT dataset ready (1,438 ex). $506 credits. |
| **MiniMax** | ✅ (indirect) | We use **Minimax M3** as the agent — counts toward MiniMax too. |
| **Modal** | ✅ working | authed; cloud train + volume download validated. $251 credits. |
| **Antim Labs (GIZMO)** | 🟡 in progress | generating 3D warehouse assets (AMR, box, picker, light); storage-pod gen FAILED — retry. $200. For demo video / sim-to-real story. |
| **HUD** | ✅ working | `hud_warehouse.py`: dispatch as a HUD v6 task, verifiable reward. Validated end-to-end (reward 0.867, 100% success) via Fireworks-direct local eval = **0 HUD credits**. Only ~10 HUD credits available -- protect them (see gotchas). |
| **Anthropic** | ⏭️ skipped | no credits; Fireworks is the agent. |
| Hillclimb / Protege | 💡 optional angle | "training data for RSI" / "real-world datasets" — our RFT dataset story aligns; mention if asked. |
| Daytona / Exa / SixtyFour / DeepMind | — | not used; not core. |

## 6. WHAT'S LEFT TO WIN (prioritized)
1. **Run the actual RFT fine-tune** (Fireworks dashboard) → get ft model id →
   `eval_agents.py --fireworks --ft-model <id>` → capture the base-vs-FT leaderboard.
   *This is the headline result; everything else supports it.*
2. ~~HUD integration~~ ✅ DONE: `hud_warehouse.py` (HUD v6 task, verifiable
   reward, validated via Fireworks-direct = 0 HUD credits). Remaining: spend ONE
   HUD credit on a `claude`/gateway run to get the dashboard trace for the demo;
   optionally `hud eval ... --ft-model <id>` to score the fine-tuned model on HUD.
3. **Demo video + slides** (use GIZMO 3D assets): show the sim, then the leaderboard
   improvement. ~2–3 min.
4. **Full 1M Modal run** for a stronger cloud model + visible Modal usage.
5. **Strengthen the result** (optional but high-value):
   - reward-filtered dataset (`--min-reward`) so RFT = rejection sampling, not imitation.
   - more eval episodes/seeds + report mean ± spread (credibility).
   - a HARD task variant where heuristics are weak (e.g. **rack-handoff**: stage hot
     pods near pickers) so the fine-tuned model can clearly BEAT the heuristic.
6. **README polish** tying env + eval + RFT + sponsors together for the repo.

## 7. Honest framing (don't oversell)
The nearest-rack heuristic is strong, so a base LLM ties it. The win is **the loop**:
verifiable env → reward-filtered post-training data → fine-tune → measured gain.
If you want "model beats heuristic," build the hard task (#5) where planning matters.

## 8. Quick resume for Claude
Read this file + `RUNBOOK.md` + `PITCH.md`. Current model: `models/ppo_warehouse.zip`
(local 1M, ~41 orders). Dataset: `data/warehouse_dispatch_sft.jsonl`. Next action:
help the user run/interpret the Fireworks fine-tune, then build the HUD wrapper.
