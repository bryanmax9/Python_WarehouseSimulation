"""
visualize.py
============
A professional viewer for the TRAINED warehouse robots. It is a *window onto the
RL policy*, laid out in three zones:

  +-----------------------------+---------------------------+
  |                             |  AI REASONING             |
  |   ROBOTIC ACTION SIM        |  step - action - decision |
  |   (warehouse, robots carry  |  (obs + policy action     |
  |    racks to the picker)     |   probabilities)          |
  |                             +---------------------------+
  |                             |  REINFORCEMENT LEARNING   |
  |                             |  live event log           |
  +-----------------------------+---------------------------+

Design goals from the demo feedback:
  * Looks professional and readable (clear states, legend, labels).
  * Realistic speed: robots move deliberately, SLOWER when carrying a rack, and
    pause to LIFT / PLACE racks (grab is slow, like real AMRs).
  * No manual controls needed -- everything is autonomous. A human worker walks
    the floor on its own and the robots avoid it (that's in their observation).
  * Surfaces the AI's reasoning (policy action distribution + decision) and a
    real-time reinforcement-learning event log.

Run:  python visualize.py
Keys: SPACE pause | TAB cycle the focused robot | R reset | ESC quit
"""

import random
from pathlib import Path
from collections import deque

import numpy as np

from warehouse_env import (
    WarehouseEnv, PHASE_NAMES, IDLE, FETCHING, DELIVERING, RETURNING, HOVERING,
    A_UP, A_DOWN, A_LEFT, A_RIGHT,
)
import warehouse_core as core
from warehouse_core import SKUS, sku_color

MODELS = Path(__file__).resolve().parent / "models"

# --- pacing (seconds) : realistic, deliberate robot motion -------------------
STEP_DUR = 0.62      # base time to traverse one cell (continuous, no stalls)
EMPTY_FAST = 0.55    # empty robots finish the cell in this fraction (snappier)
LIFT_FX = 0.5        # duration of the lift/place flourish (visual only, no stall)

# --- canvas (scaled to the window) -------------------------------------------
BASE_W, BASE_H = 1320, 760
M, TOP = 12, 58
ACT_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "DROP", "HOVER"]

# palette
BG = (17, 19, 25)
PANEL = (27, 30, 39)
HEAD = (37, 41, 54)
BORDER = (52, 58, 74)
ACCENT = (88, 166, 255)
TXT = (222, 226, 234)
MUTE = (140, 149, 166)
GOOD = (96, 214, 146)
WARN = (240, 184, 84)
BAD = (242, 96, 96)
INFO = (122, 182, 255)
GRID = (33, 37, 48)

PHASE_COLOR = {
    IDLE: (120, 128, 144),
    FETCHING: (90, 170, 244),
    DELIVERING: (96, 214, 146),
    RETURNING: (240, 184, 84),
    HOVERING: (226, 120, 214),
}
PHASE_HELP = {
    IDLE: "waiting for an order",
    FETCHING: "driving to a pod",
    DELIVERING: "carrying rack to picker",
    RETURNING: "returning rack to storage",
    HOVERING: "holding trending rack near exit",
}


def _load_policy():
    for name in ("ppo_warehouse_cloud.zip", "ppo_warehouse.zip"):
        p = MODELS / name
        if p.exists():
            from stable_baselines3 import PPO
            return PPO.load(str(p), device="cpu"), name
    return None, None


def _action_probs(model, obs):
    """Policy's probability over the 7 actions for each robot -> (n, 7)."""
    try:
        import torch
        obs_t, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)
        return dist.distribution.probs.cpu().numpy()
    except Exception:
        return None


class Visualizer:
    def __init__(self):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
        pygame.display.set_caption("Robotic Action Simulation - PPO Warehouse")
        self.base = pygame.Surface((BASE_W, BASE_H))
        self.clock = pygame.time.Clock()
        self._fonts = {}

        # layout rects
        right_w = 470
        self.sim_rect = (M, TOP + M, BASE_W - right_w - 3 * M, BASE_H - TOP - 2 * M)
        rx = self.sim_rect[0] + self.sim_rect[2] + M
        rw = BASE_W - rx - M
        rh = self.sim_rect[3]
        reason_h = int(rh * 0.66)
        self.reason_rect = (rx, TOP + M, rw, reason_h)
        self.rl_rect = (rx, TOP + M + reason_h + M, rw, rh - reason_h - M)

        # grid geometry inside the sim panel
        pad, hdr = 12, 36
        ix = self.sim_rect[0] + pad
        iy = self.sim_rect[1] + hdr + pad
        iw = self.sim_rect[2] - 2 * pad
        ih = self.sim_rect[3] - hdr - 2 * pad - 50   # reserve a 2-row legend/help strip
        self.tile = min(iw // core.COLS, ih // core.ROWS)
        self.ox = ix + (iw - self.tile * core.COLS) // 2
        self.oy = iy + (ih - self.tile * core.ROWS) // 2

        self.model, self.model_name = _load_policy()
        self.env = WarehouseEnv(seed=0)
        self.obs, _ = self.env.reset(seed=0)
        self.env.human_random = False   # viz drives the worker with purpose
        self.env.deliver_radius = 1     # robot must pull up TO the picker (realism)

        self.logs = deque(maxlen=200)
        self.probs = None
        self.acts = [6] * self.env.n_agents
        self.focus = 0
        self.inspect_shelf = None     # rack the user clicked to inspect
        self.picker_pod_sku = -1      # SKU of the pod currently being picked
        self.paused = False
        self.session_orders = 0
        self.near_miss = 0
        self.brake_count = 0
        self.total_missing = 0   # total missing-item alerts raised (RL minimizes this)
        self.step_timer = 0.0
        self.lift_anim = {}   # robot_index -> [remaining, max, kind] (visual only)
        self.wait_anim = {}   # robot_index -> [remaining, max, kind] blocked/brake cue
        self.fx = []          # floating "+10"/"LIFT" effects that tie logs to the sim
        self.picker_flash = 0.0
        self.t = 1.0
        self._snapshot()
        self.start_ticks = pygame.time.get_ticks()
        self.log("system", "policy loaded: " + (self.model_name or "scripted baseline"))

    # ---- helpers ----------------------------------------------------------
    def font(self, sz, bold=False, mono=False):
        key = (sz, bold, mono)
        if key not in self._fonts:
            fam = "dejavusansmono,consolas,monospace" if mono else "dejavusans,arial"
            self._fonts[key] = self.pg.font.SysFont(fam, sz, bold=bold)
        return self._fonts[key]

    def text(self, s, x, y, color=TXT, sz=15, bold=False, mono=False, center=False, right=False):
        surf = self.font(sz, bold, mono).render(s, True, color)
        r = surf.get_rect()
        if center:
            r.center = (x, y)
        elif right:
            r.topright = (x, y)
        else:
            r.topleft = (x, y)
        self.base.blit(surf, r)
        return r

    def panel(self, rect, title, accent=ACCENT):
        pg = self.pg
        x, y, w, h = rect
        pg.draw.rect(self.base, PANEL, rect, border_radius=10)
        pg.draw.rect(self.base, HEAD, (x, y, w, 30), border_top_left_radius=10,
                     border_top_right_radius=10)
        pg.draw.rect(self.base, BORDER, rect, 1, border_radius=10)
        pg.draw.rect(self.base, accent, (x, y + 6, 4, 18))
        self.text(title, x + 14, y + 7, TXT, 15, bold=True)
        return (x + 12, y + 38, w - 24, h - 48)

    def cc(self, cell):
        return (self.ox + cell[0] * self.tile + self.tile // 2,
                self.oy + cell[1] * self.tile + self.tile // 2)

    def person(self, x, y, color):
        """Draw a little human figure (head + body) -- monitor & pickers."""
        pg = self.pg
        x, y = int(x), int(y)
        pg.draw.rect(self.base, color, (x - 5, y - 1, 10, 12), border_radius=4)
        pg.draw.circle(self.base, color, (x, y - 6), 5)
        pg.draw.rect(self.base, (18, 20, 26), (x - 5, y - 1, 10, 12), 1, border_radius=4)
        pg.draw.circle(self.base, (18, 20, 26), (x, y - 6), 5, 1)

    def _snapshot(self):
        self.prev_r = [r.cell for r in self.env.robots]
        self.cur_r = list(self.prev_r)
        self.prev_h = self.cur_h = self.env.human_cell

    def log(self, kind, msg):
        t = self.env.step_count
        self.logs.append((t, kind, msg))

    def spawn_fx(self, cell, text, color):
        """A floating label that pops off a cell -- makes a log line *visible*."""
        cx, cy = self.cc(cell)
        self.fx.append({"x": cx, "y": cy, "text": text, "color": color,
                        "age": 0.0, "life": 1.2})

    # ---- autonomous worker (purposeful, so the stations mean something) ----
    def _approach(self, cell):
        """Nearest corridor cell the (tall) human can stand on next to `cell`."""
        L = self.env.layout
        if L.is_walkable_human(cell):
            return cell
        for d in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nb = (cell[0] + d[0], cell[1] + d[1])
            if L.is_walkable_human(nb):
                return nb
        return cell

    def control_worker(self):
        """Drive the human: fetch a fallen item -> BOXING -> RESTOCK (+20)."""
        env, L = self.env, self.env.layout
        h = env.human_cell
        env.human_action = (0, 0)
        env.human_interact = False
        near = lambda c: abs(c[0] - h[0]) + abs(c[1] - h[1]) <= 1

        if env.human_carry is None:
            if not env.floor_items:               # nothing to do -> patrol a bit
                if random.random() < 0.5:
                    env.human_action = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
                return
            item = min(env.floor_items,
                       key=lambda it: abs(it.cell[0] - h[0]) + abs(it.cell[1] - h[1]))
            if near(item.cell):                   # close enough -> pick it up
                env.human_interact = True
                return
            goal = self._approach(item.cell)
        else:                                     # carrying -> box, then restock
            goal = L.boxing_station if not env.human_carry.boxed else L.restock_station
            if near(goal):
                env.human_interact = True
                return

        path = L.bfs_path(h, goal, human=True)
        if path:
            nx = path[0]
            env.human_action = (nx[0] - h[0], nx[1] - h[1])

    # ---- simulation -------------------------------------------------------
    def _carry_route_action(self, i):
        """VISUAL realism only: a robot HOLDING a pod is tall, so it must travel
        the aisles AROUND other pods, not under them (empty robots still drive
        Kiva-style underneath). Returns a movement action that steps one cell
        toward the target along the pod-avoiding ("human"/tall) graph, or None to
        let the trained policy handle it (e.g. final placement into a slot).

        Cosmetic: does NOT affect training, the reward, or the fine-tuned
        dispatch -- it only changes how a carrying robot is routed on screen.
        """
        env, L = self.env, self.env.layout
        r = env.robots[i]
        if not r.carrying or r.target is None or r.cell == r.target:
            return None
        goal = r.target
        # If the target is a pod slot (not walkable for a tall load), aim for the
        # nearest aisle cell beside it, then let the policy place the pod.
        if not L.is_walkable_human(goal):
            aisle = [(goal[0] + dx, goal[1] + dy) for dx, dy in
                     ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if L.is_walkable_human((goal[0] + dx, goal[1] + dy))]
            if not aisle:
                return None
            goal = min(aisle, key=lambda c: abs(c[0] - r.cell[0]) + abs(c[1] - r.cell[1]))
            if r.cell == goal:
                return None
        blocked = {rb.cell for j, rb in enumerate(env.robots) if j != i}
        blocked.add(env.human_cell)
        path = (L.bfs_path(r.cell, goal, frozenset(blocked), human=True)
                or L.bfs_path(r.cell, goal, human=True))
        if not path:
            return None
        nx = path[0]
        dx, dy = nx[0] - r.cell[0], nx[1] - r.cell[1]
        if dx > 0:
            return A_RIGHT
        if dx < 0:
            return A_LEFT
        if dy > 0:
            return A_DOWN
        if dy < 0:
            return A_UP
        return None

    def sim_step(self):
        env = self.env
        self.control_worker()
        b_hcarry = env.human_carry
        b_hsku = b_hcarry.sku_idx if b_hcarry else None
        b_hboxed = b_hcarry.boxed if b_hcarry else False
        b_orders = env.orders_fulfilled
        b_phase = [r.phase for r in env.robots]
        b_carry = [r.carrying for r in env.robots]
        b_hover = [r.from_hover for r in env.robots]
        b_trend = env.orders.trending_sku
        b_floor = len(env.floor_items)

        if self.model is not None:
            # RL_HOOK: trained shared policy decides every robot's action
            self.probs = _action_probs(self.model, self.obs)
            acts, _ = self.model.predict(self.obs, deterministic=True)
            self.acts = [int(a) for a in acts]
        else:
            self.acts = [env.greedy_action(r) for r in env.robots]
            self.probs = None

        # VISUAL realism: route pod-carrying robots AROUND other pods (a loaded
        # robot is tall). Empty robots keep their learned under-pod motion.
        for i in range(env.n_agents):
            a = self._carry_route_action(i)
            if a is not None:
                self.acts[i] = a

        self.prev_r = [r.cell for r in env.robots]
        self.prev_h = env.human_cell
        self.obs, _, term, trunc, _ = env.step(self.acts)
        self.cur_r = [r.cell for r in env.robots]
        self.cur_h = env.human_cell

        # ---- decision narration: WHY each robot does what it does ----
        grabbed = placed = -1
        for i, r in enumerate(env.robots):
            sku = SKUS[r.carry_sku][0] if r.carrying else (
                SKUS[r.assigned_sku][0] if hasattr(r, "assigned_sku") else "?")
            # an idle robot just received an order -> goes to fetch
            if b_phase[i] == IDLE and r.phase == FETCHING:
                self.log("assign", f"R{i}: order {SKUS[r.assigned_sku][0]} -> drive to nearest pod")
            # arrived at the pod and lifted the rack
            if not b_carry[i] and r.carrying:
                self.log("pickup", f"R{i}: reached pod -> LIFT rack {sku}, haul to picker")
                self.spawn_fx(self.cur_r[i], "LIFT", INFO)
                grabbed = i
            # delivered a TRENDING rack -> keeps it and hovers (skips return trip)
            if b_phase[i] == DELIVERING and r.phase == HOVERING:
                bonus = " +5" if b_hover[i] else ""
                self.log("deliver", f"R{i}: delivered {sku} +10{bonus} -> it's HOT, hover near exit")
                self.spawn_fx(self.cur_r[i], f"+10{bonus}", GOOD)
                self.picker_flash = 0.6
                self.picker_pod_sku = r.carry_sku
                placed = i
            # delivered a normal rack -> returns it to storage
            if b_phase[i] == DELIVERING and r.phase == RETURNING:
                self.log("deliver", f"R{i}: delivered {sku} +10 -> return rack to storage")
                self.spawn_fx(self.cur_r[i], "+10", GOOD)
                self.picker_flash = 0.6
                self.picker_pod_sku = r.carry_sku
                placed = i
            # a hovering robot is re-tasked for the same hot SKU (fast path)
            if b_phase[i] == HOVERING and r.phase == DELIVERING:
                self.log("redeploy", f"R{i}: {sku} ordered again -> re-deploy from hover (fast, +5)")
            # the hot SKU cooled down -> stop hovering, return the rack
            if b_phase[i] == HOVERING and r.phase == RETURNING:
                self.log("cool", f"R{i}: {sku} no longer trending -> return rack")
            # finished returning -> ready for the next order
            if b_phase[i] == RETURNING and r.phase == IDLE:
                self.log("store", f"R{i}: rack stored -> idle, awaiting order")
            # collision avoidance (the policy chose to wait / detour)
            if r.last_collision == "robot":
                self.near_miss += 1
                self.wait_anim[i] = [0.5, 0.5, "robot"]
                if self.near_miss % 15 == 0:
                    self.log("yield", f"R{i}: path blocked by another robot -> wait & re-route")
            if r.last_collision == "human":
                self.brake_count += 1
                self.wait_anim[i] = [0.5, 0.5, "human"]
                self.spawn_fx(self.cur_r[i], "BRAKE", BAD)
                if self.brake_count % 12 == 0:
                    self.log("safety", f"R{i}: human in safety range -> brake & yield")
        if env.orders.trending_sku != b_trend:
            c, n = SKUS[env.orders.trending_sku]
            self.log("trend", f"DEMAND SPIKE: {c} {n} is now trending (robots will hover it)")
        if len(env.floor_items) > b_floor:
            it = env.floor_items[-1]
            self.total_missing += 1
            self.log("alert", f"MISSING ITEM: {SKUS[it.sku_idx][0]} {it.row_label} -> monitor alerted")
            self.spawn_fx(it.cell, "FELL", BAD)
        self.session_orders += env.orders_fulfilled - b_orders

        # worker progress narration (pick -> box -> restock), driven by control_worker
        hc = env.human_carry
        if b_hcarry is None and hc is not None:
            self.log("restock", f"MONITOR: recovered missing {SKUS[hc.sku_idx][0]} -> carry to BOXING")
            self.spawn_fx(env.human_cell, "PICK UP", INFO)
        elif b_hcarry is not None and hc is not None and not b_hboxed and hc.boxed:
            self.log("restock", f"MONITOR: boxed {SKUS[hc.sku_idx][0]} -> take to RESTOCK")
            self.spawn_fx(env.human_cell, "BOXED", INFO)
        elif b_hcarry is not None and hc is None:
            self.log("restock", f"MONITOR: restocked {SKUS[b_hsku][0]} +20 -> alert cleared")
            self.spawn_fx(env.layout.restock_station, "+20", GOOD)

        if grabbed >= 0:
            self.lift_anim[grabbed] = [LIFT_FX, LIFT_FX, "LIFTING"]
        if placed >= 0:
            self.lift_anim[placed] = [LIFT_FX, LIFT_FX, "PLACING"]

        if trunc or term:
            self.log("episode", f"episode complete - {env.orders_fulfilled} orders")
            self.obs, _ = env.reset()
            env.human_random = False
            self._snapshot()

    def progress(self, i):
        """Smooth, continuous glide across the full step for every robot, so
        motion never finishes early and 'pauses' (which read as stutter/jumps)."""
        return self.t

    def lerp(self, a, b, p):
        ca, cb = self.cc(a), self.cc(b)
        return (ca[0] + (cb[0] - ca[0]) * p, ca[1] + (cb[1] - ca[1]) * p)

    # ---- drawing ----------------------------------------------------------
    def draw(self):
        self.base.fill(BG)
        self.draw_topbar()
        self.draw_sim()
        self.draw_inspector()
        self.draw_reasoning()
        self.draw_rl_log()

    def draw_inspector(self):
        """Popup showing a clicked rack's real contents -- proves the data is live."""
        sh = self.inspect_shelf
        if sh is None:
            return
        pg = self.pg
        T = self.tile
        rx, ry = self.ox + sh.cell[0] * T, self.oy + sh.cell[1] * T
        W, H = 248, 70 + 18 * len(sh.slots)
        sx0, sy0, sw, sh_ = self.sim_rect
        bx = rx + T + 8
        if bx + W > sx0 + sw:
            bx = rx - W - 8
        bx = max(sx0 + 6, bx)
        by = max(sy0 + 34, min(ry - 10, sy0 + sh_ - H - 6))
        pg.draw.rect(self.base, ACCENT, (rx - 2, ry - 2, T + 4, T + 4), 2, border_radius=5)
        pg.draw.rect(self.base, (20, 23, 31), (bx, by, W, H), border_radius=8)
        pg.draw.rect(self.base, ACCENT, (bx, by, W, H), 1, border_radius=8)
        x, y = bx + 10, by + 8
        self.text(f"RACK @ ({sh.cell[0]},{sh.cell[1]})", x, y, ACCENT, 14, bold=True); y += 20
        user = next((i for i, rb in enumerate(self.env.robots) if rb.shelf is sh), None)
        if user is not None:
            self.text(f"IN USE by R{user} ({PHASE_NAMES[self.env.robots[user].phase]})",
                      x, y, WARN, 12)
        else:
            self.text("RESERVED" if sh.reserved else "available",
                      x, y, WARN if sh.reserved else GOOD, 12)
        y += 20
        self.text("Stored items (barcoded):", x, y, MUTE, 12); y += 18
        total = 0
        for sku, q in zip(sh.slots, sh.qty):
            pg.draw.rect(self.base, sku_color(sku), (x, y + 2, 10, 10), border_radius=2)
            self.text(SKUS[sku][0], x + 16, y, TXT, 12, mono=True)
            self.text(SKUS[sku][1][:13], x + 96, y, MUTE, 11)
            self.text(f"x{q}", bx + W - 12, y, TXT, 12, mono=True, right=True)
            total += q; y += 18
        self.text(f"Total units: {total}", x, y + 2, GOOD, 12, bold=True)
        self.text("(click rack again to close)", bx + 10, by + H - 15, MUTE, 10)

    def draw_topbar(self):
        pg = self.pg
        pg.draw.rect(self.base, (13, 15, 20), (0, 0, BASE_W, TOP))
        pg.draw.line(self.base, BORDER, (0, TOP), (BASE_W, TOP), 1)
        self.text("ROBOTIC ACTION SIMULATION", 16, 8, ACCENT, 22, bold=True)
        self.text("Robots carry pods to human pickers; a monitor clears missing items  -  "
                  "RL goal: MAX orders, MIN missing-item alerts", 16, 34, MUTE, 12)
        env = self.env
        mins = max((self.pg.time.get_ticks() - self.start_ticks) / 60000, 1e-6)
        opm = self.session_orders / mins
        kpis = [("ORDERS", str(self.session_orders), GOOD),
                ("ORDERS/MIN", f"{opm:.1f}", INFO),
                ("MISSING (total)", str(self.total_missing), WARN),
                ("ALERTS NOW", str(len(env.alerts.alerts)), BAD if env.alerts.alerts else MUTE)]
        x = BASE_W - 16
        for label, val, col in reversed(kpis):
            w = 118
            x -= w
            self.text(val, x + w, 10, col, 22, bold=True, right=True)
            self.text(label, x + w, 36, MUTE, 11, right=True)
        if self.paused:
            self.text("PAUSED", BASE_W // 2, 20, WARN, 20, bold=True, center=True)

    def draw_sim(self):
        pg = self.pg
        inner = self.panel(self.sim_rect, "WAREHOUSE FLOOR  -  live", ACCENT)
        T = self.tile
        L = self.env.layout

        # floor + robot-zone tint + grid
        pg.draw.rect(self.base, (23, 26, 34),
                     (self.ox, self.oy, T * core.COLS, T * core.ROWS), border_radius=6)
        x0, y0, x1, y1 = L.robot_zone
        pg.draw.rect(self.base, (30, 35, 47),
                     (self.ox + x0 * T, self.oy + y0 * T, (x1 - x0 + 1) * T, (y1 - y0 + 1) * T))
        for c in range(core.COLS + 1):
            pg.draw.line(self.base, GRID, (self.ox + c * T, self.oy),
                         (self.ox + c * T, self.oy + core.ROWS * T))
        for r in range(core.ROWS + 1):
            pg.draw.line(self.base, GRID, (self.ox, self.oy + r * T),
                         (self.ox + core.COLS * T, self.oy + r * T))

        # pods on the floor. A pod a robot has lifted is NOT drawn here (it rides
        # the robot); we leave a dashed footprint so you can see it was taken.
        carried = {id(r.shelf) for r in self.env.robots if r.carrying and r.shelf}
        for sh in L.shelves.values():
            rx, ry = self.ox + sh.cell[0] * T, self.oy + sh.cell[1] * T
            if id(sh) in carried:
                pg.draw.rect(self.base, (62, 60, 54),
                             (rx + int(T * 0.12), ry + int(T * 0.12),
                              int(T * 0.76), int(T * 0.76)), 1, border_radius=3)
            else:
                self._draw_rack(rx, ry, T, sh.slots)

        # stations
        self.station(L.picker_station, GOOD, "PICKER")
        self.station(L.boxing_station, WARN, "BOXING")
        self.station(L.restock_station, (170, 120, 220), "RESTOCK")
        for hc in L.hover_cells:
            cx, cy = self.cc(hc)
            pg.draw.circle(self.base, (110, 86, 120), (cx, cy), T // 2 - 5, 1)
        if self.picker_flash > 0:   # picker lights up when an order is delivered
            rx, ry = self.ox + L.picker_station[0] * T, self.oy + L.picker_station[1] * T
            pg.draw.rect(self.base, (190, 255, 205), (rx - 3, ry - 3, T + 6, T + 6),
                         3, border_radius=7)
        # human PICKERS standing at the picking station (they pull items from the
        # pod a robot brings, then pack/ship)
        pcx, pcy = self.cc(L.picker_station)
        self.person(pcx - T // 2 - 4, pcy, (90, 215, 200))
        self.person(pcx + T // 2 + 4, pcy, (90, 215, 200))
        if self.picker_flash > 0:
            # the just-delivered pod sits at the station while pickers pull the item
            if self.picker_pod_sku >= 0:
                self._draw_rack(self.ox + L.picker_station[0] * T, self.oy + L.picker_station[1] * T,
                                T, [self.picker_pod_sku] * 3)
            self.text("PICK!", pcx, pcy - int(T * 0.5), GOOD, 13, bold=True, center=True)

        # fallen items (pulsing red alert so it's obvious something dropped)
        pulse = int(3 + 3 * (1 + np.sin(self.pg.time.get_ticks() * 0.006)))
        for it in self.env.floor_items:
            cx, cy = self.cc(it.cell)
            pg.draw.circle(self.base, BAD, (cx, cy), 11 + pulse, 1)
            pts = [(cx, cy - 8), (cx + 8, cy), (cx, cy + 8), (cx - 8, cy)]
            pg.draw.polygon(self.base, sku_color(it.sku_idx), pts)
            pg.draw.polygon(self.base, BAD, pts, 2)
            self.text("FALLEN", cx, cy - 19, BAD, 9, bold=True, center=True)

        # roaming MONITOR -- the person who handles missing-item alerts. White
        # figure + cyan safety ring when inside the robot zone (robots avoid it).
        hx, hy = self.lerp(self.prev_h, self.cur_h, self.t)
        hx, hy = int(hx), int(hy)
        if L.in_robot_zone(self.env.human_cell):
            pg.draw.circle(self.base, (90, 210, 240), (hx, hy), int(T * 2.4), 1)
        self.person(hx, hy, (245, 246, 252))
        if self.env.human_carry is not None:   # carrying a recovered item
            pg.draw.rect(self.base, sku_color(self.env.human_carry.sku_idx),
                         (hx - 4, hy - 15, 8, 8), border_radius=2)
        self.text("MONITOR", hx, hy - int(T * 0.62), (210, 240, 255), 10, bold=True, center=True)

        # robots: an AMR chassis; when loaded, the rack TOWER sits on top of it
        # (the robot drives underneath the pod, Kiva-style).
        for i, r in enumerate(self.env.robots):
            fx, fy = self.lerp(self.prev_r[i], self.cur_r[i], self.progress(i))
            px, py = int(fx), int(fy)
            col = PHASE_COLOR[r.phase]
            s = int(T * 0.56)
            if i == self.focus:
                pg.draw.rect(self.base, ACCENT,
                             (px - T // 2, py - T // 2, T, T), 2, border_radius=8)
            # chassis (always visible as the colored base)
            rect = (px - s // 2, py - s // 2, s, s)
            pg.draw.rect(self.base, col, rect, border_radius=6)
            pg.draw.rect(self.base, (12, 14, 18), rect, 2, border_radius=6)

            if r.carrying:
                lift = 0
                if i in self.lift_anim:
                    rem, mx, _ = self.lift_anim[i]
                    lift = int(6 * np.sin((1 - rem / mx) * np.pi))
                slots = r.shelf.slots if (r.shelf and r.shelf.slots) else [r.carry_sku]
                # the actual rack the robot lifted, sitting on top (and lifting)
                self._draw_rack(px - T // 2, py - T // 2 - 4 - lift, T, slots, carry=r.carry_sku)
                # id badge so you still know which robot it is
                pg.draw.circle(self.base, col, (px - s // 2 + 2, py - s // 2 + 2), 8)
                pg.draw.circle(self.base, (12, 14, 18), (px - s // 2 + 2, py - s // 2 + 2), 8, 1)
                self.text(str(i), px - s // 2 + 2, py - s // 2 + 1, (12, 14, 18), 11,
                          bold=True, center=True)
            else:
                # heading arrow + id for an empty robot
                dxn = self.cur_r[i][0] - self.prev_r[i][0]
                dyn = self.cur_r[i][1] - self.prev_r[i][1]
                if dxn or dyn:
                    a = s // 2 - 2
                    tip = (px + dxn * a, py + dyn * a)
                    pp = (dyn, -dxn)
                    b1 = (px + dxn * (a - 5) + pp[0] * 4, py + dyn * (a - 5) + pp[1] * 4)
                    b2 = (px + dxn * (a - 5) - pp[0] * 4, py + dyn * (a - 5) - pp[1] * 4)
                    pg.draw.polygon(self.base, (12, 14, 18), [tip, b1, b2])
                self.text(str(i), px, py, (12, 14, 18), 14, bold=True, center=True)
            # blocked/brake cue: pulsing ring when the robot is stuck this moment
            if i in self.wait_anim:
                rem, mx, kind = self.wait_anim[i]
                pr = int(2 + 3 * np.sin((1 - rem / mx) * np.pi))
                pg.draw.circle(self.base, BAD if kind == "human" else WARN,
                               (px, py), int(T * 0.5) + pr, 2)
            # label: full state only for the focused robot (keeps clusters clean);
            # transient cues (LIFTING/PLACING/WAIT/BRAKE) show for any robot.
            label = None
            if i in self.lift_anim:
                label, lc = self.lift_anim[i][2], col
            elif i in self.wait_anim:
                label = "BRAKE" if self.wait_anim[i][2] == "human" else "WAIT"
                lc = BAD if self.wait_anim[i][2] == "human" else WARN
            elif i == self.focus:
                label, lc = PHASE_NAMES[r.phase], col
            if label:
                self.text(label, px, py - int(T * 0.66), lc, 10, bold=True, center=True)

        # focused robot's intent line -> target (and outline the target rack when
        # fetching, so it's clear the robot picks THAT rack for its barcode)
        fr = self.env.robots[self.focus]
        if fr.target is not None:
            frx, fry = self.lerp(self.prev_r[self.focus], self.cur_r[self.focus],
                                 self.progress(self.focus))
            tcx, tcy = self.cc(fr.target)
            pg.draw.line(self.base, ACCENT, (int(frx), int(fry)), (tcx, tcy), 1)
            pg.draw.circle(self.base, ACCENT, (tcx, tcy), 5, 1)
            if fr.phase == FETCHING and fr.shelf is not None:
                gx, gy = self.ox + fr.shelf.cell[0] * T, self.oy + fr.shelf.cell[1] * T
                pg.draw.rect(self.base, ACCENT, (gx + 1, gy + 1, T - 2, T - 2), 2, border_radius=4)

        # floating effects: pop a label off the cell so the live log is VISIBLE
        for e in self.fx:
            f = e["age"] / e["life"]
            surf = self.font(16, bold=True).render(e["text"], True, e["color"])
            surf.set_alpha(int(255 * max(0.0, 1 - f)))
            self.base.blit(surf, surf.get_rect(center=(e["x"], int(e["y"] - 30 * f))))

        # ---- legend / help footer (2 rows) ----
        bottom = self.sim_rect[1] + self.sim_rect[3]
        # row 1: what the objects are
        y1 = bottom - 42
        lx = self.sim_rect[0] + 14
        pg.draw.rect(self.base, (120, 128, 144), (lx, y1, 14, 14), border_radius=3)
        pg.draw.rect(self.base, (12, 14, 18), (lx, y1, 14, 14), 1, border_radius=3)
        r = self.text("Robot", lx + 19, y1, TXT, 12); lx = r.right + 14
        self.person(lx + 5, y1 + 9, (245, 246, 252))
        r = self.text("Monitor", lx + 14, y1, TXT, 12); lx = r.right + 14
        self.person(lx + 5, y1 + 9, (90, 215, 200))
        r = self.text("Picker", lx + 14, y1, TXT, 12); lx = r.right + 14
        self._draw_rack(lx, y1 - 2, 18, [3, 7, 11])
        r = self.text("Pod (rack of items)", lx + 21, y1, TXT, 12); lx = r.right + 14
        cx, cy = lx + 7, y1 + 7
        pg.draw.polygon(self.base, WARN, [(cx, cy - 6), (cx + 6, cy), (cx, cy + 6), (cx - 6, cy)])
        pg.draw.polygon(self.base, BAD, [(cx, cy - 6), (cx + 6, cy), (cx, cy + 6), (cx - 6, cy)], 2)
        r = self.text("Missing item", lx + 19, y1, TXT, 12); lx = r.right + 16
        self.text("PICKER=order out  .  BOXING/RESTOCK=monitor fixes drops", lx, y1, MUTE, 11)
        # row 2: states + controls
        y2 = bottom - 20
        lx = self.sim_rect[0] + 14
        r = self.text("States:", lx, y2, MUTE, 12, bold=True); lx = r.right + 8
        for ph in (FETCHING, DELIVERING, RETURNING, HOVERING):
            pg.draw.rect(self.base, PHASE_COLOR[ph], (lx, y2 + 1, 12, 12), border_radius=3)
            r = self.text(PHASE_NAMES[ph], lx + 15, y2, TXT, 11); lx = r.right + 10
        lx += 12
        self.text("Click ROBOT = inspect AI  .  Click RACK = see contents  .  TAB  .  SPACE  .  R",
                  lx, y2, (150, 175, 220), 12, bold=True)

    def _draw_rack(self, rx, ry, T, slots, carry=None):
        """A pod = a roughly SQUARE rack holding a 2x2 grid of barcoded item
        boxes, centered in a T x T cell. `carry` highlights the ordered SKU."""
        pg = self.pg
        s = max(12, int(T * 0.80))
        fx = rx + (T - s) // 2
        fy = ry + (T - s) // 2
        pg.draw.rect(self.base, (120, 96, 60), (fx, fy, s, s), border_radius=3)
        pg.draw.rect(self.base, (56, 42, 26), (fx, fy, s, s), 1, border_radius=3)
        b = (s - 6) / 2
        for ri in range(2):
            for ci in range(2):
                sku = slots[(ri * 2 + ci) % len(slots)]
                bx = fx + 3 + ci * b
                by = fy + 3 + ri * b
                pg.draw.rect(self.base, sku_color(sku), (bx, by, b - 2, b - 2), border_radius=2)
                edge = (255, 255, 255) if (carry is not None and sku == carry) else (38, 30, 20)
                pg.draw.rect(self.base, edge, (bx, by, b - 2, b - 2), 1, border_radius=2)

    def station(self, cell, color, label):
        pg = self.pg
        rx, ry = self.ox + cell[0] * self.tile, self.oy + cell[1] * self.tile
        pg.draw.rect(self.base, color, (rx + 2, ry + 2, self.tile - 4, self.tile - 4),
                     border_radius=5)
        self.text(label, rx + self.tile // 2, ry + self.tile // 2, (14, 16, 20), 10,
                  bold=True, center=True)

    def draw_reasoning(self):
        env = self.env
        inner = self.panel(self.reason_rect, "AI REASONING  -  step . action . decision", (180, 140, 255))
        x, y, w, h = inner
        r = env.robots[self.focus]

        self.text(f"FOCUS  ROBOT R{self.focus}", x, y, ACCENT, 16, bold=True)
        self.text("click robot / [TAB]", x + w, y + 2, MUTE, 11, right=True)
        y += 26
        pg = self.pg
        pg.draw.circle(self.base, PHASE_COLOR[r.phase], (x + 7, y + 8), 7)
        self.text(f"{PHASE_NAMES[r.phase]} - {PHASE_HELP[r.phase]}", x + 20, y, TXT, 14)
        y += 26

        # OBSERVATION (key features the policy sees)
        self.text("OBSERVATION  (what the robot senses)", x, y, MUTE, 12, bold=True)
        y += 20
        tgt = r.target
        if tgt is not None:
            dx, dy = tgt[0] - r.cell[0], tgt[1] - r.cell[1]
            tgt_s = f"dx={dx:+d}, dy={dy:+d}  (dist {abs(dx)+abs(dy)})"
        else:
            tgt_s = "none"
        nearest = min((abs(o.cell[0]-r.cell[0])+abs(o.cell[1]-r.cell[1])
                       for o in env.robots if o is not r), default=0)
        hd = abs(env.human_cell[0]-r.cell[0]) + abs(env.human_cell[1]-r.cell[1])
        carry = SKUS[r.carry_sku][1] if r.carrying else "(empty)"
        rows = [
            ("target vector", tgt_s),
            ("nearest robot", f"{nearest} cells"),
            ("human worker", f"{hd} cells" + ("  IN ZONE" if env.layout.in_robot_zone(env.human_cell) else "")),
            ("carrying", carry),
            ("trending now", SKUS[env.orders.trending_sku][0]),
        ]
        for k, v in rows:
            self.text(k, x + 6, y, MUTE, 13, mono=True)
            self.text(v, x + 150, y, TXT, 13, mono=True)
            y += 18
        y += 4

        # POLICY action distribution
        self.text("POLICY ACTION DISTRIBUTION", x, y, MUTE, 12, bold=True)
        y += 19
        probs = self.probs[self.focus] if self.probs is not None else None
        chosen = self.acts[self.focus]
        bar_w = w - 110
        for ai, name in enumerate(ACT_NAMES):
            p = float(probs[ai]) if probs is not None else (1.0 if ai == chosen else 0.0)
            is_ch = (ai == chosen)
            col = ACCENT if is_ch else (70, 78, 96)
            self.text(name, x, y, TXT if is_ch else MUTE, 11, mono=True, bold=is_ch)
            pg.draw.rect(self.base, (40, 44, 56), (x + 70, y + 1, bar_w, 11), border_radius=3)
            pg.draw.rect(self.base, col, (x + 70, y + 1, int(bar_w * p), 11), border_radius=3)
            self.text(f"{p*100:4.0f}%", x + 70 + bar_w + 6, y, TXT if is_ch else MUTE, 11, mono=True)
            y += 16
        y += 4
        decision = f"{ACT_NAMES[chosen]}  ->  {PHASE_HELP[r.phase]}"
        pg.draw.rect(self.base, (30, 34, 46), (x, y, w, 24), border_radius=6)
        self.text("DECISION:", x + 8, y + 4, MUTE, 12, bold=True)
        self.text(decision, x + 84, y + 4, GOOD, 12, bold=True)
        y += 30

        # ---- the RACK this robot is working: proves barcode-driven (not random) ----
        deliver_sku = (r.carry_sku if r.carrying else
                       (r.assigned_sku if (r.phase == FETCHING and hasattr(r, "assigned_sku")) else -1))
        if r.shelf is not None:
            verb = "CARRYING RACK" if r.carrying else "TARGET RACK"
            self.text(f"{verb}  @ ({r.shelf.cell[0]},{r.shelf.cell[1]})  - chosen for its barcode",
                      x, y, MUTE, 12, bold=True)
            y += 18
            for sku, q in zip(r.shelf.slots, r.shelf.qty):
                hit = (sku == deliver_sku)
                if hit:
                    pg.draw.rect(self.base, (38, 66, 48), (x - 2, y - 1, w + 2, 16), border_radius=3)
                pg.draw.rect(self.base, sku_color(sku), (x + 3, y + 2, 10, 10), border_radius=2)
                self.text(SKUS[sku][0], x + 18, y, TXT if hit else MUTE, 12, mono=True, bold=hit)
                self.text(SKUS[sku][1][:12], x + 100, y, MUTE, 11)
                if hit:
                    self.text("<- DELIVER", x + w, y, GOOD, 12, bold=True, right=True)
                y += 16
        else:
            self.text("RACK: none (robot is idle / awaiting an order)", x, y, MUTE, 12)

    def draw_rl_log(self):
        inner = self.panel(self.rl_rect, "REINFORCEMENT LEARNING  -  live log", GOOD)
        x, y, w, h = inner
        kindcol = {"deliver": GOOD, "restock": GOOD, "pickup": INFO, "trend": WARN,
                   "alert": BAD, "yield": (200, 160, 90), "safety": BAD,
                   "store": MUTE, "episode": ACCENT, "system": MUTE,
                   "assign": INFO, "hover": (226, 120, 214), "redeploy": (226, 120, 214),
                   "cool": MUTE}
        n = max(int(h // 17), 4)
        for (t, kind, msg) in list(self.logs)[-n:]:
            col = kindcol.get(kind, TXT)
            self.text(f"{t:>4}", x, y, MUTE, 12, mono=True)
            self.text(msg, x + 42, y, col, 12, mono=True)
            y += 17

    # ---- main loop --------------------------------------------------------
    def run(self, max_frames=None):
        pg = self.pg
        running = True
        frames = 0
        while running:
            dt = self.clock.tick(60) / 1000.0
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    running = False
                elif e.type == pg.KEYDOWN:
                    if e.key == pg.K_ESCAPE:
                        running = False
                    elif e.key == pg.K_SPACE:
                        self.paused = not self.paused
                    elif e.key == pg.K_TAB:
                        self.focus = (self.focus + 1) % self.env.n_agents
                    elif e.key == pg.K_r:
                        self.obs, _ = self.env.reset()
                        self.env.human_random = False
                        self._snapshot()
                elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                    # map window click -> base canvas coords
                    mx, my = e.pos
                    sw, sh = self.screen.get_size()
                    bx, by = mx * BASE_W / sw, my * BASE_H / sh
                    T = self.tile
                    cell = (int((bx - self.ox) // T), int((by - self.oy) // T))
                    # clicked a rack? -> open/close the contents inspector
                    if cell in self.env.layout.shelves:
                        shf = self.env.layout.shelves[cell]
                        self.inspect_shelf = None if self.inspect_shelf is shf else shf
                    else:
                        # otherwise focus the nearest robot (and close inspector)
                        best, bd = None, 1e9
                        for i in range(self.env.n_agents):
                            px, py = self.lerp(self.prev_r[i], self.cur_r[i], self.progress(i))
                            d = (px - bx) ** 2 + (py - by) ** 2
                            if d < bd:
                                bd, best = d, i
                        if best is not None and bd <= (self.tile * 0.9) ** 2:
                            self.focus = best
                        self.inspect_shelf = None

            if not self.paused:
                # continuous pacing: advance one env step per STEP_DUR, carrying
                # the remainder so motion never stalls or jumps backward.
                self.step_timer += dt
                if self.step_timer >= STEP_DUR:
                    self.step_timer = min(self.step_timer - STEP_DUR, STEP_DUR)
                    self.sim_step()
                self.t = min(1.0, self.step_timer / STEP_DUR)
                for i in list(self.lift_anim):
                    self.lift_anim[i][0] -= dt
                    if self.lift_anim[i][0] <= 0:
                        del self.lift_anim[i]
                for i in list(self.wait_anim):
                    self.wait_anim[i][0] -= dt
                    if self.wait_anim[i][0] <= 0:
                        del self.wait_anim[i]
                for e in self.fx:
                    e["age"] += dt
                self.fx = [e for e in self.fx if e["age"] < e["life"]]
                self.picker_flash = max(0.0, self.picker_flash - dt)

            self.draw()
            pg.transform.smoothscale(self.base, self.screen.get_size(), self.screen)
            pg.display.flip()

            frames += 1
            if max_frames and frames >= max_frames:
                running = False
        pg.quit()


def main():
    try:
        import pygame  # noqa: F401
    except ImportError:
        print("pygame is required for the visualization:  pip install pygame")
        return
    Visualizer().run()


if __name__ == "__main__":
    main()
