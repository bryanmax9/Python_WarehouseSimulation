# RUNBOOK — every command, in order

Project: **RL warehouse environment that improves a frontier model (RFT)** for the
HUD/YC RSI hackathon.

> Two "brains": **PPO** = how robots *move* (the pygame demo). **LLM coordinator**
> = how orders are *assigned to robots* (the part you fine-tune). The env's reward
> (orders ↑, missing-items ↓) verifies both.

---

## 0. One-time setup
```bash
cd /home/bryanmax9/Desktop/Python_WarehouseSimulation
python3.12 -m venv .venv            # Python 3.10–3.12 ONLY (not 3.13/3.14)
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
The `.venv` is already built — normally just:
```bash
cd /home/bryanmax9/Desktop/Python_WarehouseSimulation && source .venv/bin/activate
```

## 0b. Sponsor credentials
```bash
python env_setup.py                 # shows which keys are set
```
- **Fireworks:** put `FIREWORKS_API_KEY` + `FIREWORKS_MODEL` in `.env` (done; model = minimax-m3).
- **Modal:** `python -m modal setup` (browser) OR `modal token set --token-id ... --token-secret ...` (done).
- **HUD / Anthropic:** add keys to `.env` when you have them (HUD pending).

---

## 1. Train the robots (PPO motion policy)
```bash
python train.py --timesteps 1000000 --num-worlds 8
```
→ `models/ppo_warehouse.zip` (+ `models/reward_curve.png`, `reward_log.csv`).
Already trained (~41 orders/episode). Quick check run: `python train.py` (50k).

## 2. Show the simulation (visual demo)
```bash
python visualize.py
```
Controls: **click a robot** = inspect its AI reasoning · **click a rack** = see its
items · **TAB** cycle robot · **SPACE** pause · **R** reset · **ESC** quit.

## 3. (Optional) Heavy training on Modal cloud (uses $251 credits)
```bash
modal run modal_train.py
modal volume get warehouse-models ppo_warehouse_cloud.zip models/ppo_warehouse_cloud.zip
```
(Validated with a 200k run; run the full thing for a stronger cloud model.)

---

## 4. Baseline eval — heuristic vs base LLM (no fine-tune yet)
```bash
python eval_agents.py --fireworks --episodes 2
```
Prints a leaderboard: Heuristic vs Base LLM (minimax-m3) on orders / missing / reward.

## 5. RFT step 1 — build the post-training dataset
```bash
python rft_dataset.py --episodes 40
# reward-filtered version (rejection sampling): only keep good episodes
python rft_dataset.py --episodes 80 --min-reward -2200
```
→ `data/warehouse_dispatch_sft.jsonl` (done: 1,438 examples).

## 6. RFT step 2 — fine-tune on Fireworks (uses $506 credits)
```bash
python rft_finetune.py              # validates dataset + prints exact steps
```
Then on **app.fireworks.ai** (website, not terminal):
1. **Datasets → Upload** → `data/warehouse_dispatch_sft.jsonl`
2. **Fine Tuning → Supervised Fine Tuning** → base = a **Tunable (LoRA)** model
   (e.g. Kimi K2.7 Code) → **Start**
3. Copy the fine-tuned model id, e.g. `accounts/<you>/models/warehouse-dispatch-ft`

## 7. RFT step 3 — prove it improved (base vs fine-tuned, one table)
```bash
python eval_agents.py --fireworks \
    --ft-model accounts/<you>/models/warehouse-dispatch-ft \
    --episodes 3
```
Compare the **Base LLM** row vs the **Fine-tuned** row (orders ↑ / missing ↓).
This table is your headline result.

---

## 8. HUD integration (the required sponsor platform)  -- PROTECT YOUR CREDITS

`hud_warehouse.py` exposes the dispatch decision as a HUD (v6) task with a
verifiable reward (orders the agent's dispatch fulfills, 0..1). It is single
turn: ONE prompt -> agent's JSON assignments -> one numeric grade, so **one
model call per task**.

**What costs HUD credits:** only model calls through the HUD *gateway*
(`--gateway`, `--remote`, or having ONLY `HUD_API_KEY` set). Routing the model
to a provider you have keys for (Fireworks) does NOT spend HUD credits.

Install (once; already done in `.venv`):
```bash
pip install hud-python            # or: uv tool install hud-python --python 3.12
```

```bash
# (a) FREE -- no HUD, no model, no credits. Sanity-check the task wiring.
python hud_warehouse.py
#   -> no-op agent reward 0.0 ; mock greedy reward ~0.87  (agent drives reward)

# (b) ~FREE on HUD -- model billed to FIREWORKS, not HUD; env runs locally.
FW=$(python -c "from env_setup import load_keys; load_keys(); import os; print(os.environ['FIREWORKS_API_KEY'])")
hud eval hud_warehouse.py openai_compatible \
    --model accounts/fireworks/models/minimax-m3 \
    -c base_url=https://api.fireworks.ai/inference/v1 -c api_key=$FW \
    --runtime local --max-steps 4 -y
#   -> validated: reward 0.867, success 100%. Add --all for all 3 seeds.
#   To eval the FINE-TUNED model, swap --model accounts/<you>/models/<ft-id>.

# (c) SPENDS HUD CREDITS -- the dashboard trace for the demo. Do this ONCE.
hud eval hud_warehouse.py claude --max-steps 4        # needs ANTHROPIC_API_KEY
#   View the replayable trace at the printed https://hud.ai/jobs/<id> URL.
```
Tip: `hud jobs` lists runs; `hud trace <id>` inspects one. Credit balance is on
the hud.ai dashboard (no CLI command for it).

---

## TL;DR demo flow for judges
```bash
python visualize.py                                   # the world, live
python eval_agents.py --fireworks --ft-model <id> --episodes 3   # the result
hud eval hud_warehouse.py claude --max-steps 4        # the same task, on HUD
```
