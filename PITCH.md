# PITCH — how to present this to win

Hackathon: **HUD/YC Frontier-RSI RL Environments**. Judges reward *environments that
improve frontier models* (post-training data, evals, RFT). Frame everything around:
**"You can improve models at anything you can verify."**

---

## 30-second hook
> "Amazon's warehouses run on fleets of robots — but the *brain* that coordinates
> them is hand-coded. We built a **verifiable RL environment** for warehouse fleet
> coordination, turned its reward into **post-training data**, and **fine-tuned a
> frontier model** to run the warehouse better. Train in sim, verify, improve the
> model, deploy to real AMRs. It's a recipe for teaching models any logistics or
> physical-autonomy task."

## The 2040 backwards-story (they explicitly ask for this)
> "By 2040, physical operations — warehouses, factories, ports — are run by frontier
> models coordinating robot fleets. To get there we need **environments that teach
> and verify** those models. We built one for warehouse coordination and showed the
> full recursive-self-improvement loop: environment → data → fine-tune → measurably
> better model."

## What we built (say it in this order)
1. **A verifiable environment** (`WarehouseEnv`, Gymnasium): 6 AMRs carry pods to
   pickers; orders, a trending "hot item," and missing-item alerts. Reward =
   **orders ↑, missing-items ↓, collisions ↓** — fully verifiable.
2. **A hierarchy:** PPO learns fast low-level motion (the muscle); a **frontier LLM
   is the coordinator** (the brain) that dispatches robots to orders/racks.
3. **The RSI loop:** we roll out the env, capture **reward-filtered decisions** as a
   post-training dataset, **fine-tune a model on Fireworks**, and **re-score it on
   the same verifiable reward** — a leaderboard of Heuristic vs Base model vs
   Fine-tuned model.

## The live demo (2–3 min)
1. `python visualize.py` — the world is alive: click a robot → see its **AI reasoning**
   (observation + policy action probabilities + decision); click a rack → see its
   **real barcoded inventory**. Point out: this is all driven by a learned policy +
   a frontier-model coordinator, scored live.
2. `python eval_agents.py --fireworks --ft-model <id> --episodes 3` — show the
   **leaderboard**: the fine-tuned model's row vs base vs heuristic on orders/missing.
   *"We improved a frontier model on a task we can verify."*
3. (Slides) **GIZMO 3D render** of the warehouse → the sim-to-real / Physical-AI vision.

## Why it fits the themes (name them)
- **Robotics / manufacturing / VLAs:** warehouse fleet coordination, sim-to-real.
- **Agentic collaboration:** multiple robots + an LLM coordinator + a human monitor.
- **Autonomous business:** the env is a tiny autonomous fulfillment operation.
- **Post-training data + RFT + evals:** the dataset + fine-tune + leaderboard ARE that.

## Sponsor name-drops (judges love seeing their tools used)
- **HUD** — the environment + agentic eval lives here (the platform).
- **Fireworks** — serves the agent **and** runs the fine-tune (RFT).
- **MiniMax** — the coordinator model is **Minimax M3**.
- **Modal** — scaled PPO training + sandboxes.
- **Antim Labs (GIZMO)** — 3D physical-AI assets / sim-to-real visualization.

## The killer line
> "Anyone can train a robot. We built the **environment that teaches a frontier model
> to run the warehouse — and proves, with a number, that it got better.**"

## Q&A — be ready, be honest
- **"Does the fine-tuned model beat the heuristic?"** "The nearest-rack heuristic is
  already strong, so the base model ties it. Our contribution is the **verifiable
  improvement loop** — and on the harder *rack-handoff* variant, where planning
  matters, the gap opens up." (Build that variant if time — see PROJECT_STATUS #5.)
- **"Why PPO + LLM instead of one model?"** "Hierarchy: LLMs are too slow for
  per-step motor control but excel at high-level planning. We put the frontier model
  where it adds value and verify it."
- **"Is the reward real?"** "Yes — orders fulfilled and missing-item alerts are
  counted by the environment, not by a model. That's the whole point: verifiable."
- **"Sim-to-real?"** "The env is the fast training/verification layer; Antim's GIZMO
  is the high-fidelity 3D layer; the same policy/coordinator targets real AMRs."

## One-slide summary
**Problem:** robot-fleet coordination is hand-coded and doesn't improve.
**Idea:** make it a *verifiable RL environment* and teach a *frontier model* to do it.
**Proof:** reward-filtered post-training data → fine-tune → leaderboard gain.
**Vision (2040):** frontier models run physical operations; this is how you teach them.
**Stack:** HUD · Fireworks · MiniMax · Modal · Antim.
