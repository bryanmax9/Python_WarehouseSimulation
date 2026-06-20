# 🤖 RL Warehouse — robots that *learn* to run a warehouse

A Reinforcement Learning environment where Amazon-Kiva-style robots **learn**
(via PPO) to coordinate, fulfill orders, avoid each other and a human, and
intelligently **hover near the exit for trending items** — all learned behavior,
not hardcoded rules. Train in simulation → deploy to real AMRs.

**CPU-only. Cross-platform. Python 3.10–3.12.**

> The RL environment + training pipeline is the core. The pygame window is just
> a viewer that shows what the trained policy is doing.

## Architecture (in order of importance)

| File | Role |
|------|------|
| `warehouse_env.py` | **The RL environment.** Gymnasium-compatible, multi-robot, custom obs/action/reward. Plus `ParameterSharingVecEnv` so one shared PPO policy controls every robot. |
| `train.py` | Local PPO training (fast verification) + learning-curve plot. |
| `modal_train.py` | The same training on **Modal** cloud for 1M+ steps (sponsor credits). |
| `visualize.py` | pygame viewer of the **trained** robots; WASD human they avoid. |
| `warehouse_core.py` | Shared world model: grid, shelves, 20 SKUs, order queue + trending, alerts. No ML/pygame deps. |

### The RL formulation
- **Action (Discrete 7):** up / down / left / right / pickup / drop / hover.
- **Observation (25 floats, normalized):** own position, carrying + SKU, phase,
  vector to current target, 2 nearest robots, human position/distance/zone,
  trending SKU, order-queue signals.
- **Rewards:** `+10` order fulfilled · `+5` trending item delivered straight
  from hover (no return trip) · `+20` team reward when the human restocks a
  fallen item · `-1`/step (speed) · collision penalties for hitting a robot /
  human.
- **Parameter sharing:** N robots × K worlds are presented to Stable-Baselines3
  as one flat batch; PPO trains a single shared policy. No
  supersuit/pettingzoo, CPU-only.

### Engineering notes (what we actually learned tuning it)
- **Robots drive *under* the pods** (real Kiva behavior) → the floor is an open
  grid, which makes navigation learnable from a `(dx,dy)`-to-target signal.
- **Pickup/drop fire on arrival** — lifting a pod is mechanical; PPO spends its
  capacity on the hard part (navigation + collision avoidance + the hover
  decision).
- **Congestion ≠ crash:** being blocked by a stationary robot is a no-penalty
  wait; only same-cell entries and head-on swaps are penalized. This is what
  stops PPO collapsing into a "never move to avoid penalties" optimum.
- **Collision magnitudes were tuned down** from the brief's -100/-50: robots
  physically can't overlap (the move is blocked), and huge penalties freeze
  learning. Hitting the **human is penalized more than a robot** (people first).

Result: from a standing start PPO goes **0 → ~32 orders/episode** (matches a
BFS planner baseline) in ~1M CPU steps (~75s on an 8-core laptop).

### HUD.ai
`WarehouseEnv` is a standard `gymnasium.Env` registered as `Warehouse-v0`. See
the `# HUD_HOOK` comments for the wrap/register points.

---

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate        # (Win: .venv\Scripts\activate)
pip install --upgrade pip
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
> The `.venv` here is already set up — just `source .venv/bin/activate`.

## Run it — in order
```bash
# 1. Quick local training (verify the pipeline) -> models/ppo_warehouse.zip + learning curve
python train.py                          # 50k steps (fast)
python train.py --timesteps 1000000 --num-worlds 8   # full run (~75s on CPU, ~32 orders/ep)

# 2. Heavy cloud training on Modal (sponsor) — 1,000,000 steps
modal token new
modal run modal_train.py
modal volume get warehouse-models ppo_warehouse_cloud.zip models/ppo_warehouse_cloud.zip

# 3. Watch the trained robots (pygame)
python visualize.py
```
**Visualizer controls:** `WASD`/arrows move the human · `E` interact (pick up
fallen item → BOX → STOCK, +20 team reward) · `R` reset · `ESC` quit. The viewer
auto-loads the cloud model, then the local model, else a scripted baseline.

## Outputs
- `models/ppo_warehouse.zip` — trained shared policy
- `models/reward_curve.png` — learning curve · `models/reward_log.csv`
