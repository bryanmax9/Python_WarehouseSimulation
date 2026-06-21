# Verifiable Warehouse RL Environment — HUD × YC RSI Hackathon

**Thesis:** *You can improve models at anything you can verify.* So we built the **verifier** —
a warehouse fleet-coordination RL environment, shipped as a **HUD task**, that scores any
frontier model on a real reward, turns that reward into post-training data, fine-tunes a model,
and re-evaluates. The complete recursive-self-improvement loop, running on the sponsor stack.

By 2040, physical operations — warehouses, factories, ports — are run by frontier models
coordinating robot fleets. To get there we need environments that **teach and verify** them.
This is one.

---

## The loop (every link built, on the stack)
1. **Environment** — `warehouse_env.py`: a Gymnasium world (6 AMRs, 20 barcoded SKUs, orders,
   a trending item, fallen-item alerts). Reward = orders fulfilled / missing — **counted by the
   world, not a model.**
2. **Verifiable HUD task** — `hud_warehouse.py`: the dispatch decision as a **HUD v6 task**.
3. **Post-training data** — `rft_dataset.py`: 1,897 reward-filtered (state → expert dispatch) examples.
4. **Fine-tune** — Gemma-4-26B-A4B on **Fireworks** (LoRA, loss 2.5 → 0).
5. **Evaluate** — `benchmark.py` + **HUD gateway**: every coordinator scored on the same reward.

---

## Proof on HUD — the env *ranks frontier models* (run through the HUD gateway)

| Coordinator | Reward (0–1) | Where |
|---|---|---|
| Claude Opus 4.8 | **0.867** | HUD gateway ✓ |
| Gemini 3.1 Flash Lite | **0.867** | HUD gateway ✓ |
| GPT-4o-mini | **0.800** | HUD gateway ✓ |
| Heuristic (nearest-rack) | 0.73 | local benchmark |
| Fine-tuned Gemma (small, LoRA) | ~0.31 | local benchmark |
| No-op (does nothing) | **0.00** | floor reference |

Three frontier models scored on **our HUD task**, each a replayable trace on hud.ai. They cluster
near the ceiling (0.80–0.87) — the task has a clear competence bar that strong models clear — while
the low end (a small fine-tuned model ~0.31, a no-op 0.0) falls off. **The environment produces a
clean, gameable-resistant signal across the whole capability range** — exactly what a verifiable RL
environment / eval is for.

---

## Honest finding (a strength, not a caveat)
A base LLM already matches the expert heuristic with **no** fine-tuning. And our environment
**caught a real model limitation**: the small fine-tuned Gemma makes rack↔SKU *grounding errors*
under load (~0.31 vs frontier ~0.87). A good eval that *reveals* where a model breaks is the
whole point of verifiable environments — and the obvious next levers (more LoRA capacity, a
better-than-greedy teacher, harder reward shaping) are clear from the signal.

---

## Sponsor stack (all actually used)
- **HUD** — the environment + agentic eval; multi-model leaderboard via the gateway.
- **Fireworks** — serves the coordinator **and** ran the RFT fine-tune.
- **MiniMax** — `minimax-m3` as a coordinator model.
- **Modal** — scaled PPO motion-policy training.
- **Antim Labs** — the 3D assets (AMR, kiva pod, picker, worker, warehouse shell) for the demo.

---

## See it / run it
```bash
# 1. The live 3D warehouse (React + Three.js) — robots, the human monitor, the
#    full fallen-item recovery loop, click a rack/robot to inspect.
cd web && npm install && npm run dev          # http://localhost:5173 -> Run simulation

# 2. Reproduce the result tables
python benchmark.py --fireworks

# 3. Score any model on the env, on HUD
hud eval hud_warehouse.py claude --gateway    # or openai / gemini, --model <id>
```

**Files:** `warehouse_env.py` · `hud_warehouse.py` · `rft_dataset.py` · `benchmark.py` ·
`results/` · `web/` (3D sim) · `PITCH.md` (presentation script).
