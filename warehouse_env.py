"""
warehouse_env.py
================
THE PRIMARY DELIVERABLE: a Gymnasium-compatible multi-agent RL environment in
which warehouse robots LEARN (via PPO) to coordinate, fulfill orders, avoid
collisions with each other and with a human, and intelligently hover near the
exit for trending items.

Two things live here:

  1. WarehouseEnv          - the Gymnasium environment (multi-robot).
                             observation_space / action_space describe ONE robot
                             (all robots are identical -> parameter sharing).
                             reset() / step() operate on all N robots at once.

  2. ParameterSharingVecEnv- adapts WarehouseEnv to Stable-Baselines3's VecEnv
                             so a SINGLE shared PPO policy controls every robot.
                             This is how a single-agent algorithm (PPO) trains a
                             cooperative multi-agent system, with no
                             supersuit/pettingzoo dependency. CPU-only.

ACTION SPACE  (Discrete 7):  0=up 1=down 2=left 3=right 4=pickup 5=drop 6=hover
OBSERVATION   (Box, 25 floats, normalized): own pos, carry/SKU, phase, vector to
                             current target, 2 nearest robots, human pos/dist/
                             zone, trending SKU, order-queue signals.
REWARDS (exactly the hackathon spec, see REWARD_* constants below).

--------------------------------------------------------------------------
HUD.AI INTEGRATION  (https://hud.ai)
--------------------------------------------------------------------------
WarehouseEnv is a standard gymnasium.Env, which is exactly what HUD wraps and
scores. See the `# HUD_HOOK` comments for the registration / wrap points.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import warehouse_core as core
from warehouse_core import (
    COLS, ROWS, NUM_ROBOTS, NUM_SKUS,
    WarehouseLayout, OrderQueue, AlertSystem, FallenItem,
    ITEM_FALL_PROB,
)

# ---------------------------------------------------------------------------
# Reward function  (the hackathon judging criteria, in one place)
# ---------------------------------------------------------------------------
REWARD_STEP = -1.0            # per timestep -> encourages speed
REWARD_ORDER = +10.0          # order fulfilled
REWARD_HOVER_BONUS = +5.0     # delivered trending item straight from hover
REWARD_ALERT_RESOLVED = +20.0  # human restocked a fallen item (team reward)

# Collision penalties. The hackathon brief suggested -100 (robot) / -50 (human),
# but two things made us tune these down:
#   1. Robots physically CANNOT overlap -- the env blocks the move -- so a
#      "collision" here is really a penalized *near-miss / attempt*.
#   2. With penalties that large, PPO collapses into a "never move" local
#      optimum (verified empirically) -- it freezes to dodge the huge negative
#      instead of learning to deliver. Moderate magnitudes keep the gradient
#      toward "deliver, but coordinate".
# We also penalize hitting the HUMAN more than bumping a robot (people first),
# which both performs best in training and is the safer real-world objective.
REWARD_ROBOT_COLLISION = -5.0
REWARD_HUMAN_COLLISION = -8.0

# Optional dense shaping to make 1M steps actually converge on a CPU. Without
# this, PPO finds the lazy-safe optimum (don't move -> never eat a -100 collision
# -> never deliver). These are NOT the spec's headline rewards; set to 0.0 for
# "pure spec". The progress term is potential-based (telescopes to start-minus-
# end distance), so it cannot be farmed by oscillating.
SHAPING_PROGRESS = 1.0        # reward for getting closer to the current target
SHAPING_PICKUP = 2.0          # reward for a correct pickup
SHAPING_DELIVER_NEAR = 0.0    # (reserved) extra pull near the picker

MAX_EPISODE_STEPS = 512

# Phases a robot moves through (also one-hot encoded into the observation).
IDLE, FETCHING, DELIVERING, RETURNING, HOVERING = range(5)
PHASE_NAMES = {IDLE: "IDLE", FETCHING: "FETCHING", DELIVERING: "DELIVERING",
               RETURNING: "RETURNING", HOVERING: "HOVERING"}

# Actions
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_PICKUP, A_DROP, A_HOVER = range(7)
_MOVE = {A_UP: (0, -1), A_DOWN: (0, 1), A_LEFT: (-1, 0), A_RIGHT: (1, 0)}

OBS_DIM = 25


class Robot:
    """Per-robot mutable state (the policy is shared and external)."""

    def __init__(self, cell):
        self.cell = cell
        self.phase = IDLE
        self.carrying = False
        self.carry_sku = -1
        self.shelf = None          # the Shelf object it owns/carries
        self.target = None         # current goal cell (or None)
        self.from_hover = False    # next delivery is a hover re-delivery (+5)
        self.last_collision = ""   # "", "robot", or "human" (for viz coloring)

    def target_vec(self):
        if self.target is None:
            return 0.0, 0.0, 1.0  # dx, dy, at_target(=1 when no target)
        dx = (self.target[0] - self.cell[0]) / (COLS - 1)
        dy = (self.target[1] - self.cell[1]) / (ROWS - 1)
        at = 1.0 if self.cell == self.target else 0.0
        return dx, dy, at


class WarehouseEnv(gym.Env):
    """Multi-robot warehouse. Per-agent spaces (parameter sharing)."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.n_agents = NUM_ROBOTS

        # Per-ROBOT spaces (one shared policy is applied to each robot).
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(7)

        self._np_random = np.random.default_rng(seed)
        self._py_seed = seed
        # True -> the human random-walks (training). The viz sets this False and
        # drives the worker with goal-directed behavior. Set before reset so
        # reset() preserves it.
        self.human_random = True
        # Optional high-level COORDINATOR (e.g., an LLM). If set, it assigns idle
        # robots to (order, rack) pairs; otherwise the built-in heuristic does.
        # This is the hook a frontier model plugs into. See llm_agent.py.
        self.dispatcher = None
        # When False, the built-in heuristic no longer auto-assigns idle robots;
        # the dispatcher (e.g. a HUD-evaluated frontier model) becomes the sole
        # source of dispatch, so its decisions alone determine throughput. Used
        # by the HUD eval (hud_warehouse.py); training/viz/eval leave it True.
        self.auto_heuristic_dispatch = True
        # Manhattan distance from the picker at which a delivery counts. Widened
        # to 2 for training (stops robots gridlocking on the single picker tile).
        # The visualizer sets this to 1 so a robot visibly pulls up to the picker.
        self.deliver_radius = 2
        self.reset(seed=seed)

    # ----------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None):
        import random as _random
        if seed is not None:
            self._py_seed = seed
            self._np_random = np.random.default_rng(seed)
        rng = _random.Random(self._py_seed)

        self.layout = WarehouseLayout(rng=rng)
        self.orders = OrderQueue(rng=rng)
        self.alerts = AlertSystem()
        self.floor_items: list[FallenItem] = []

        self.robots = [Robot(c) for c in self.layout.robot_spawns]
        self.human_cell = self.layout.human_spawn
        self.human_carry: FallenItem | None = None
        # external control hooks (used by the pygame viz)
        self.human_action = (0, 0)     # (dx, dy) movement requested this step
        self.human_interact = False    # E key pressed this step

        self.step_count = 0
        self.orders_fulfilled = 0
        self.episode_reward = 0.0
        self._team_reward = 0.0        # pending team reward (alert resolved)
        self._stuck = [0] * self.n_agents       # per-robot stuck counter (liveness)
        self._move_from = [r.cell for r in self.robots]

        obs = np.stack([self._observe(r) for r in self.robots]).astype(np.float32)
        return obs, {}

    # ------------------------------------------------------------------ step
    def step(self, actions):
        """actions: array-like of length n_agents (Discrete action per robot)."""
        actions = [int(a) for a in actions]
        rewards = np.full(self.n_agents, REWARD_STEP, dtype=np.float32)
        for r in self.robots:
            r.last_collision = ""

        # goal + distance-to-goal BEFORE moving (for potential-based shaping).
        # We remember the *target object* so we can skip shaping on steps where
        # an interaction switches the goal (pickup/deliver) -- otherwise the
        # sudden jump in target distance would wrongly punish those actions.
        prev_target = [r.target for r in self.robots]
        prev_dist = [None if t is None else abs(t[0] - r.cell[0]) + abs(t[1] - r.cell[1])
                     for r, t in zip(self.robots, prev_target)]

        # 1) move the human (random during training; WASD-driven in the viz)
        self._update_human()

        # 2) resolve robot movement with collision detection
        self._move_from = [r.cell for r in self.robots]
        self._resolve_moves(actions, rewards)
        # 2b) liveness layer: nudge robots that have been stuck too long so the
        # fleet can never permanently gridlock (a real AMR motion-safety feature).
        self._break_deadlocks()

        # 3) pickups / drops / deliveries
        self._handle_interactions(actions, rewards)

        # 4) progress shaping (only when the goal didn't change this step)
        if SHAPING_PROGRESS:
            for i, r in enumerate(self.robots):
                t = prev_target[i]
                if t is not None and r.target == t:
                    new_d = abs(t[0] - r.cell[0]) + abs(t[1] - r.cell[1])
                    rewards[i] += SHAPING_PROGRESS * (prev_dist[i] - new_d)

        # 5) world tick: orders, trending, item falls, alerts
        self.orders.update(self.step_count)
        self._maybe_drop_items()
        self.alerts.update()

        # 6) human restock interaction (mainly exercised in the viz)
        self._handle_human_interact()

        # 7) task allocation: hand idle/hover robots their next job
        self._assign_orders()

        # 8) team reward (alert resolved) shared across all robots
        if self._team_reward:
            rewards += self._team_reward
            self._team_reward = 0.0

        self.step_count += 1
        self.episode_reward += float(rewards.sum())
        truncated = self.step_count >= MAX_EPISODE_STEPS
        terminated = False  # continuing task; episodes end by time limit

        obs = np.stack([self._observe(r) for r in self.robots]).astype(np.float32)
        info = {
            "orders_fulfilled": self.orders_fulfilled,
            "trending_sku": self.orders.trending_sku,
            "active_alerts": len(self.alerts.alerts),
        }
        return obs, rewards, terminated, truncated, info

    # --------------------------------------------------------- human movement
    def _update_human(self):
        dx, dy = self.human_action
        if dx == 0 and dy == 0 and self.human_random:
            # random walk during training so robots learn to avoid a human
            moves = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
            dx, dy = moves[int(self._np_random.integers(len(moves)))]
        nc = (self.human_cell[0] + int(dx), self.human_cell[1] + int(dy))
        # The human is blocked by pods (can't pass under racks) AND by robots
        # (no walking through them). Robots separately avoid the human's cell,
        # so the two never overlap.
        robot_cells = {r.cell for r in self.robots}
        if self.layout.is_walkable_human(nc) and nc not in robot_cells:
            self.human_cell = nc
        self.human_action = (0, 0)  # consume

    # ---------------------------------------------------- movement resolution
    def _resolve_moves(self, actions, rewards):
        cur = [r.cell for r in self.robots]
        want = list(cur)
        for i, (r, a) in enumerate(zip(self.robots, actions)):
            if a in _MOVE:
                d = _MOVE[a]
                cand = (cur[i][0] + d[0], cur[i][1] + d[1])
                if not self.layout.is_walkable(cand):
                    continue            # wall / shelf -> just blocked, no penalty
                if cand == self.human_cell:
                    rewards[i] += REWARD_HUMAN_COLLISION
                    r.last_collision = "human"
                    continue            # blocked
                want[i] = cand

        # Resolve robot-robot interactions. We distinguish two cases:
        #   * COLLISION (penalized): two robots try to enter the SAME cell, or
        #     two robots SWAP cells (head-on). These are real crashes.
        #   * BLOCKED (no penalty): a robot tries to move into a cell a
        #     stationary robot already occupies -> it just can't, like a wall.
        # Treating ordinary congestion as a no-penalty block (instead of a
        # crash) is what stops PPO collapsing into the "never move" optimum.
        final = list(want)
        for _ in range(2):  # a couple of passes to settle chained reverts
            for i in range(self.n_agents):
                if final[i] == cur[i]:
                    continue
                crash = blocked = False
                for j in range(self.n_agents):
                    if i == j:
                        continue
                    same_cell = final[i] == final[j] and final[j] != cur[j]
                    swap = final[i] == cur[j] and final[j] == cur[i]
                    if same_cell or swap:
                        crash = True
                        break
                    if final[i] == cur[j] and final[j] == cur[j]:
                        blocked = True  # stationary robot in the way
                if crash:
                    rewards[i] += REWARD_ROBOT_COLLISION
                    self.robots[i].last_collision = "robot"
                    final[i] = cur[i]
                elif blocked:
                    final[i] = cur[i]   # just wait, no penalty

        for r, c in zip(self.robots, final):
            r.cell = c

    def _break_deadlocks(self):
        """Liveness guarantee: a robot stuck (has a goal, but hasn't moved) for
        several steps is nudged to the best free neighbor. Prevents permanent
        gridlock without changing what the policy observes or is rewarded for."""
        occupied = {r.cell for r in self.robots}
        for i, r in enumerate(self.robots):
            moved = r.cell != self._move_from[i]
            if r.target is not None and r.cell != r.target and not moved:
                self._stuck[i] += 1
            else:
                self._stuck[i] = 0
            if self._stuck[i] >= 4:
                best, bd = None, 1e9
                for d in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    nb = (r.cell[0] + d[0], r.cell[1] + d[1])
                    if (self.layout.is_walkable(nb) and nb not in occupied
                            and nb != self.human_cell):
                        dist = abs(nb[0] - r.target[0]) + abs(nb[1] - r.target[1])
                        if dist < bd:
                            bd, best = dist, nb
                if best is not None:
                    occupied.discard(r.cell)
                    r.cell = best
                    occupied.add(best)
                    self._stuck[i] = 0

    # --------------------------------------------- pickup / drop / deliver
    def _handle_interactions(self, actions, rewards):
        """
        Pickup/drop fire automatically on ARRIVAL at the right cell. Lifting a
        pod and setting it on the picker is a mechanical act; the hard, learnable
        part we leave to PPO is *navigation + collision avoidance + the hover
        decision for trending items*. The 7-action space still includes
        pickup/drop/hover for HUD compatibility and future hand-control.
        """
        L = self.layout
        for i, (r, a) in enumerate(zip(self.robots, actions)):
            # PICKUP: arrived at the fetch target, grab the shelf
            if r.phase == FETCHING and not r.carrying \
                    and r.shelf is not None and r.cell == r.shelf.access:
                r.carrying = True
                r.carry_sku = r.assigned_sku
                r.phase = DELIVERING
                r.target = L.picker_station
                rewards[i] += SHAPING_PICKUP

            # DROP at picker = deliver the order. Triggering on ARRIVAL within 1
            # cell (not the exact cell) lets several robots unload around the
            # station at once instead of fighting over one tile.
            elif r.phase == DELIVERING and r.carrying and (
                    abs(r.cell[0] - L.picker_station[0])
                    + abs(r.cell[1] - L.picker_station[1]) <= self.deliver_radius):
                self.orders_fulfilled += 1
                rewards[i] += REWARD_ORDER
                if r.from_hover:
                    rewards[i] += REWARD_HOVER_BONUS  # didn't have to travel back
                    r.from_hover = False
                if r.carry_sku == self.orders.trending_sku:
                    r.phase = HOVERING                # keep shelf, hover near exit
                    r.target = self._free_hover_cell()
                else:
                    r.phase = RETURNING               # take the shelf back
                    r.target = r.shelf.access

            # arrived home = finish returning the shelf
            elif r.phase == RETURNING and r.carrying \
                    and r.shelf is not None and r.cell == r.shelf.access:
                r.shelf.reserved = False
                r.carrying = False
                r.carry_sku = -1
                r.shelf = None
                r.phase = IDLE
                r.target = None

    # --------------------------------------------------------- item falling
    def _maybe_drop_items(self):
        # Items fall off a carried pod mainly when the robot moves roughly -- a
        # near-miss/jostle. So a smoother, better-coordinated policy causes FEWER
        # missing-item alerts: minimizing alerts is an emergent RL outcome.
        for r in self.robots:
            if r.carrying and r.phase in (DELIVERING, RETURNING):
                p = ITEM_FALL_PROB * (4.0 if r.last_collision else 1.0)
                if self._np_random.random() < p:
                    item = FallenItem(r.carry_sku, r.cell, f"Row {r.cell[1]}")
                    self.floor_items.append(item)
                    self.alerts.add(item)

    # ----------------------------------------- human restock (+20 team reward)
    def _handle_human_interact(self):
        if not self.human_interact:
            return
        self.human_interact = False
        L = self.layout
        h = self.human_cell
        if self.human_carry is None:
            # pick up an adjacent fallen item
            for item in self.floor_items:
                if abs(item.cell[0] - h[0]) + abs(item.cell[1] - h[1]) <= 1:
                    self.human_carry = item
                    item.on_floor = False
                    self.floor_items.remove(item)
                    break
        else:
            item = self.human_carry
            near = lambda s: abs(s[0] - h[0]) + abs(s[1] - h[1]) <= 1
            if not item.boxed and near(L.boxing_station):
                item.boxed = True
            elif item.boxed and near(L.restock_station):
                self.alerts.resolve(item)
                self.human_carry = None
                self._team_reward += REWARD_ALERT_RESOLVED  # +20 shared

    # ----------------------------------------------------- task allocation
    def _dispatch_state(self, idle_idx):
        """Compact, JSON-serializable snapshot for an external coordinator."""
        pend = list(self.orders.pending)
        pset = set(pend)
        racks = []
        for sh in self.layout.shelves.values():
            if sh.reserved:
                continue
            if any(s in pset for s in sh.slots):
                racks.append({"cell": list(sh.cell), "skus": list(sh.slots)})
        return {
            "grid": [COLS, ROWS],
            "picker": list(self.layout.picker_station),
            "trending_sku": int(self.orders.trending_sku),
            "pending_orders": pend,
            "idle_robots": [{"id": i, "cell": list(self.robots[i].cell)} for i in idle_idx],
            "racks": racks,
        }

    def _apply_dispatch(self, disp):
        """Let the coordinator assign idle robots to (sku, rack). Invalid or
        missing assignments simply fall through to the heuristic afterwards."""
        idle_idx = [i for i, r in enumerate(self.robots) if r.phase == IDLE]
        if not idle_idx or not self.orders.pending:
            return
        try:
            plan = disp(self._dispatch_state(idle_idx))
        except Exception:
            return
        if not plan:
            return
        pending = list(self.orders.pending)
        idle_set = set(idle_idx)
        def _as_int(v):
            if isinstance(v, bool):
                raise ValueError
            if isinstance(v, (int, float)):
                return int(v)
            return int("".join(c for c in str(v) if c.isdigit()))  # "R0"->0
        for a in plan:
            try:
                ri, sku, cell = _as_int(a["robot"]), _as_int(a["sku"]), tuple(a["rack"])
            except Exception:
                continue
            if ri not in idle_set or sku not in pending:
                continue
            sh = self.layout.shelves.get(cell)
            if sh is None or sh.reserved or sku not in sh.slots:
                continue
            r = self.robots[ri]
            sh.reserved = True
            r.shelf, r.assigned_sku = sh, sku
            r.phase, r.target = FETCHING, sh.access
            idle_set.discard(ri)
            pending.remove(sku)
        self.orders.pending = deque(pending)

    def _assign_orders(self):
        if not self.orders.pending:
            return
        # an external coordinator (LLM) gets first pick at the idle robots
        if self.dispatcher is not None:
            self._apply_dispatch(self.dispatcher)
        remaining = []
        idle = [r for r in self.robots if r.phase == IDLE]
        while self.orders.pending:
            sku = self.orders.pending.popleft()
            assigned = False
            # (a) a hovering robot already carrying this SKU re-delivers instantly
            for r in self.robots:
                if r.phase == HOVERING and r.carry_sku == sku:
                    r.phase = DELIVERING
                    r.from_hover = True
                    r.target = self.layout.picker_station
                    assigned = True
                    break
            if assigned:
                continue
            # (b) send an idle robot to the nearest shelf holding this SKU
            if self.auto_heuristic_dispatch and idle:
                r = idle[0]
                shelf = self.layout.nearest_shelf_with_sku(r.cell, sku)
                if shelf is not None:
                    shelf.reserved = True
                    r.shelf = shelf
                    r.assigned_sku = sku
                    r.phase = FETCHING
                    r.target = shelf.access
                    idle.pop(0)
                    assigned = True
            if not assigned:
                remaining.append(sku)
        for sku in remaining:
            self.orders.pending.append(sku)

        # hovering robots whose SKU is no longer trending should head home
        for r in self.robots:
            if r.phase == HOVERING and r.carry_sku != self.orders.trending_sku:
                r.phase = RETURNING
                r.target = r.shelf.access
                r.from_hover = False

    def _free_hover_cell(self):
        taken = {r.target for r in self.robots if r.phase == HOVERING}
        for c in self.layout.hover_cells:
            if c not in taken:
                return c
        return self.layout.hover_cells[0]

    # -------------------------------------------------------- observation
    def _target_dist(self, r):
        if r.target is None:
            return None
        return abs(r.target[0] - r.cell[0]) + abs(r.target[1] - r.cell[1])

    def _observe(self, robot):
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[0] = robot.cell[0] / (COLS - 1)
        o[1] = robot.cell[1] / (ROWS - 1)
        o[2] = 1.0 if robot.carrying else 0.0
        o[3] = (robot.carry_sku + 1) / NUM_SKUS if robot.carrying else 0.0
        # phase one-hot (4..8)
        o[4 + robot.phase] = 1.0
        dx, dy, at = robot.target_vec()
        o[9], o[10], o[11] = dx, dy, at
        # two nearest robots (12..15)
        others = sorted(
            ((abs(x.cell[0] - robot.cell[0]) + abs(x.cell[1] - robot.cell[1]), x)
             for x in self.robots if x is not robot),
            key=lambda t: t[0],
        )
        for k in range(2):
            if k < len(others):
                ox = (others[k][1].cell[0] - robot.cell[0]) / (COLS - 1)
                oy = (others[k][1].cell[1] - robot.cell[1]) / (ROWS - 1)
                o[12 + k * 2], o[13 + k * 2] = ox, oy
        # human (16..19)
        o[16] = (self.human_cell[0] - robot.cell[0]) / (COLS - 1)
        o[17] = (self.human_cell[1] - robot.cell[1]) / (ROWS - 1)
        hd = abs(self.human_cell[0] - robot.cell[0]) + abs(self.human_cell[1] - robot.cell[1])
        o[18] = hd / (COLS + ROWS)
        o[19] = 1.0 if self.layout.in_robot_zone(self.human_cell) else 0.0
        # trending + order signals (20..24)
        o[20] = (self.orders.trending_sku + 1) / NUM_SKUS
        o[21] = 1.0 if (robot.carrying and robot.carry_sku == self.orders.trending_sku) else 0.0
        o[22] = min(len(self.orders.pending) / core.MAX_PENDING_ORDERS, 1.0)
        o[23] = 1.0 if (robot.carrying and self.orders.has_order_for(robot.carry_sku)) else 0.0
        o[24] = self.step_count / MAX_EPISODE_STEPS
        return o

    # --------------------------------------------- RL_HOOK / convenience
    def greedy_action(self, robot):
        """
        Reference scripted policy used ONLY to smoke-test mechanics and as a
        baseline. The trained PPO policy replaces this entirely.
        # RL_HOOK: PPO's policy(obs) -> action is what drives robots in training.
        """
        if robot.target is None:
            return A_HOVER
        if robot.cell == robot.target:
            if robot.phase == FETCHING:
                return A_PICKUP
            if robot.phase in (DELIVERING, RETURNING):
                return A_DROP
            return A_HOVER
        # BFS toward the target, routing around shelves, other robots, the human
        blocked = {r.cell for r in self.robots if r is not robot}
        blocked.add(self.human_cell)
        path = self.layout.bfs_path(robot.cell, robot.target, frozenset(blocked))
        if not path:
            return A_HOVER  # wait this turn rather than crash into someone
        nxt = path[0]
        dx, dy = nxt[0] - robot.cell[0], nxt[1] - robot.cell[1]
        if dx > 0:
            return A_RIGHT
        if dx < 0:
            return A_LEFT
        return A_DOWN if dy > 0 else A_UP


# Register with Gymnasium (handy for `gym.make("Warehouse-v0")`).
gym.register(id="Warehouse-v0", entry_point="warehouse_env:WarehouseEnv")
# HUD integration lives in hud_warehouse.py: it exposes the dispatch decision as
# a HUD (v6) task with a verifiable reward -> `hud eval hud_warehouse.py <agent>`.


# ===========================================================================
# Stable-Baselines3 adapter: one shared policy across all robots & worlds
# ===========================================================================
from stable_baselines3.common.vec_env.base_vec_env import VecEnv  # noqa: E402


class ParameterSharingVecEnv(VecEnv):
    """
    Presents `num_worlds` WarehouseEnvs (each with N robots) as a flat batch of
    num_worlds*N single-agent envs to SB3. PPO learns ONE policy applied to every
    robot. Auto-reset contract is respected (terminal_observation in info).
    """

    def __init__(self, num_worlds: int = 1, seed: int | None = None):
        self.worlds = [WarehouseEnv(seed=(None if seed is None else seed + i))
                       for i in range(num_worlds)]
        self.num_worlds = num_worlds
        self.n_agents = self.worlds[0].n_agents
        super().__init__(
            num_envs=num_worlds * self.n_agents,
            observation_space=self.worlds[0].observation_space,
            action_space=self.worlds[0].action_space,
        )
        self._actions = None
        self._ep_r = np.zeros(self.num_envs)
        self._ep_l = np.zeros(self.num_envs, dtype=int)

    def reset(self):
        obs = [w.reset()[0] for w in self.worlds]
        self._ep_r[:] = 0
        self._ep_l[:] = 0
        return np.concatenate(obs, axis=0).astype(np.float32)

    def step_async(self, actions):
        self._actions = np.asarray(actions).reshape(self.num_worlds, self.n_agents)

    def step_wait(self):
        obs_all, rew_all, done_all, info_all = [], [], [], []
        for wi, w in enumerate(self.worlds):
            obs, rew, term, trunc, _info = w.step(self._actions[wi])
            done = bool(term or trunc)
            infos = [{} for _ in range(self.n_agents)]
            if done:
                for ai in range(self.n_agents):
                    infos[ai]["terminal_observation"] = obs[ai]
                    if trunc and not term:
                        infos[ai]["TimeLimit.truncated"] = True
                obs = w.reset()[0]
            obs_all.append(obs)
            rew_all.extend(float(x) for x in rew)
            done_all.extend([done] * self.n_agents)
            info_all.extend(infos)

        rew_arr = np.asarray(rew_all, dtype=np.float32)
        done_arr = np.asarray(done_all, dtype=bool)
        self._ep_r += rew_arr
        self._ep_l += 1
        for i in range(self.num_envs):
            if done_arr[i]:
                info_all[i]["episode"] = {"r": float(self._ep_r[i]), "l": int(self._ep_l[i])}
                self._ep_r[i] = 0
                self._ep_l[i] = 0
        return (np.concatenate(obs_all, axis=0).astype(np.float32),
                rew_arr, done_arr, info_all)

    def close(self):
        pass

    # minimal VecEnv plumbing
    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * self._n(indices)

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *a, indices=None, **k):
        return [None] * self._n(indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self._n(indices)

    def _n(self, indices):
        if indices is None:
            return self.num_envs
        if isinstance(indices, int):
            return 1
        return len(indices)


def make_vec_env(num_worlds: int, seed: int | None = None) -> VecEnv:
    """Build the SB3 training env (parameter sharing across robots & worlds)."""
    return ParameterSharingVecEnv(num_worlds=num_worlds, seed=seed)
