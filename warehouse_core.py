"""
warehouse_core.py
=================
Shared, dependency-light warehouse logic used by BOTH the RL environment
(warehouse_env.py) and the pygame visualization (visualize.py).

Contains NO gymnasium / stable-baselines3 / pygame imports on purpose, so it is
trivially portable and fast to import. Just the warehouse "world model":

    * Grid layout + shelves (Kiva-style lattice with corridors)
    * 20 SKUs (barcoded items) distributed randomly across all shelves,
      deliberately redundant (same SKU on many shelves -> robots never far)
    * OrderQueue with trending-item detection
    * AlertSystem for fallen / missing items

The RL-specific stuff (observation/action/reward) lives in warehouse_env.py.
The robots themselves live in the env, because how a robot *decides* differs
between training (PPO policy) and viz (loaded policy); only the static world is
shared here.
"""

from __future__ import annotations

import random
import colorsys
from collections import deque

# ---------------------------------------------------------------------------
# Grid configuration  (small + CPU-friendly so PPO learns fast on a laptop)
# ---------------------------------------------------------------------------
COLS = 13
ROWS = 11

NUM_ROBOTS = 6          # configurable; the shared policy is per-agent so any N works
NUM_SKUS = 20
SLOTS_PER_SHELF = 3

# Rows that hold shelves; the rows between them are corridors. Columns that are
# a multiple of 3 stay open as vertical corridors -> a fully connected lattice.
SHELF_ROWS = [2, 4, 6, 8]

# World timing, measured in environment STEPS (not wall-clock, so training is
# deterministic and reproducible).
ORDER_INTERVAL_STEPS = 12     # a new order arrives every N steps
TREND_INTERVAL_STEPS = 150    # a new SKU starts trending every N steps
ALERT_CRIT_STEPS = 600        # unhandled alert escalates to CRITICAL
ITEM_FALL_PROB = 0.0035       # chance/step a carried item falls while moving
MAX_PENDING_ORDERS = 12       # cap so the observation stays bounded

# ---------------------------------------------------------------------------
# SKU catalog (barcode + human name) and a stable color per SKU (for viz)
# ---------------------------------------------------------------------------
_SKU_NAMES = [
    "Wireless Headphones", "Phone Case", "USB Cable", "Bluetooth Speaker",
    "Laptop Stand", "Mechanical Keyboard", "Wireless Mouse", "Webcam HD",
    "Power Bank", "HDMI Adapter", "Screen Protector", "Gaming Headset",
    "Smart Watch", "Earbuds Pro", "Tablet Sleeve", "Desk Lamp",
    "Cable Organizer", "Microphone", "Monitor Arm", "Charging Dock",
]

# SKUS[i] = ("SKU-0001", "Wireless Headphones")
SKUS = [(f"SKU-{i + 1:04d}", name) for i, name in enumerate(_SKU_NAMES)]


def sku_color(idx: int):
    """Deterministic distinct RGB color for a SKU index (used by the viz)."""
    r, g, b = colorsys.hsv_to_rgb((idx % NUM_SKUS) / NUM_SKUS, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


# ---------------------------------------------------------------------------
# Shelf
# ---------------------------------------------------------------------------
class Shelf:
    """One shelf cell holding several barcoded item slots."""

    def __init__(self, cell, access, slots):
        self.cell = cell          # (col, row) the shelf occupies
        self.access = access      # adjacent walkable cell a robot stands on
        self.slots = slots        # list[int] SKU indices on this shelf
        self.qty = []             # units in stock per slot (set by the layout)
        self.reserved = False     # a robot currently owns/carries this shelf

    def has_sku(self, sku_idx) -> bool:
        return sku_idx in self.slots


# ---------------------------------------------------------------------------
# Warehouse layout (grid, shelves, stations, pathing helpers)
# ---------------------------------------------------------------------------
class WarehouseLayout:
    """Static warehouse geometry. Built once, shared by env and viz."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.cols, self.rows = COLS, ROWS

        # --- carve shelves into a corridor lattice ------------------------
        self.shelves: dict[tuple, Shelf] = {}
        self.shelves_by_sku: dict[int, list] = {i: [] for i in range(NUM_SKUS)}

        for row in SHELF_ROWS:
            for col in range(1, COLS - 1):
                if col % 3 == 0:
                    continue  # vertical corridor column -> keep walkable
                slots = [self.rng.randrange(NUM_SKUS) for _ in range(SLOTS_PER_SHELF)]
                # Kiva-style: robots drive UNDER the pod to lift it, so the
                # access cell is the pod cell itself and the floor is open.
                access = (col, row)
                shelf = Shelf((col, row), access, slots)
                shelf.qty = [self.rng.randrange(3, 30) for _ in slots]  # units in stock
                self.shelves[(col, row)] = shelf
                for s in set(slots):
                    self.shelves_by_sku[s].append(shelf)

        # Guarantee every SKU exists on at least one shelf (redundancy).
        all_shelves = list(self.shelves.values())
        for sku in range(NUM_SKUS):
            if not self.shelves_by_sku[sku]:
                sh = self.rng.choice(all_shelves)
                sh.slots[0] = sku
                self.shelves_by_sku[sku].append(sh)

        # --- designated stations (all on the bottom corridor row) ---------
        bottom = ROWS - 1
        self.picker_station = (COLS // 2, bottom)   # robots DELIVER shelves here
        self.boxing_station = (COLS - 1, bottom)     # human BOXES fallen items
        self.restock_station = (0, bottom)           # human RESTOCKS items here
        self.hover_cells = [(1, bottom), (2, bottom), (COLS - 3, bottom), (COLS - 2, bottom)]

        # robot + human spawn points (top corridor row, all walkable)
        self.robot_spawns = [(2, 0), (4, 0), (8, 0), (10, 0),
                             (5, 0), (7, 0)][:NUM_ROBOTS]
        self.human_spawn = (COLS // 2, 0)

        # The "robot zone" is the bounding box of the shelf lattice.
        self.robot_zone = (1, SHELF_ROWS[0], COLS - 2, SHELF_ROWS[-1])  # x0,y0,x1,y1

    # -- queries ------------------------------------------------------------
    def in_bounds(self, cell):
        c, r = cell
        return 0 <= c < self.cols and 0 <= r < self.rows

    def is_walkable(self, cell):
        # Open floor: robots drive under pods (Kiva-style). Shelves are pod
        # markers, not walls. Only the grid boundary constrains movement.
        return self.in_bounds(cell)

    def is_shelf(self, cell):
        return cell in self.shelves

    def is_walkable_human(self, cell):
        # The human worker is TALL: it cannot pass under/through pods (only the
        # low robots can). So the human is confined to the corridors.
        return self.in_bounds(cell) and cell not in self.shelves

    def in_robot_zone(self, cell):
        x0, y0, x1, y1 = self.robot_zone
        return x0 <= cell[0] <= x1 and y0 <= cell[1] <= y1

    def bfs_path(self, start, goal, blocked=frozenset(), human=False):
        """Shortest path start->goal (excludes start, includes goal).

        `blocked` adds dynamic obstacles. `human=True` uses the human walkability
        (corridors only, blocked by pods). Returns [] if start==goal, None if
        unreachable.
        """
        if start == goal:
            return []
        walk = self.is_walkable_human if human else self.is_walkable
        from collections import deque as _dq
        frontier = _dq([start])
        came = {start: None}
        while frontier:
            cur = frontier.popleft()
            for d in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nb = (cur[0] + d[0], cur[1] + d[1])
                if nb in came or not walk(nb):
                    continue
                if nb in blocked and nb != goal:
                    continue
                came[nb] = cur
                if nb == goal:
                    path = [nb]
                    while came[path[-1]] != start:
                        path.append(came[path[-1]])
                    path.reverse()
                    return path
                frontier.append(nb)
        return None

    def nearest_shelf_with_sku(self, from_cell, sku_idx):
        """Closest UNRESERVED shelf containing sku_idx (Manhattan), or None."""
        best, best_d = None, 1e9
        for sh in self.shelves_by_sku[sku_idx]:
            if sh.reserved:
                continue
            d = abs(sh.access[0] - from_cell[0]) + abs(sh.access[1] - from_cell[1])
            if d < best_d:
                best, best_d = sh, d
        return best


# ---------------------------------------------------------------------------
# Order queue with trending-item detection
# ---------------------------------------------------------------------------
class OrderQueue:
    """Generates orders over time and tracks which SKU is currently 'hot'."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.pending: deque[int] = deque()
        self.total_generated = 0
        self.trending_sku = self.rng.randrange(NUM_SKUS)
        self.trend_demand = 0

    def update(self, step: int):
        # rotate the trending item
        if step > 0 and step % TREND_INTERVAL_STEPS == 0:
            self.trending_sku = self.rng.randrange(NUM_SKUS)
            self.trend_demand = 0
        # generate a new order (trending SKU is twice as likely to be ordered)
        if step > 0 and step % ORDER_INTERVAL_STEPS == 0:
            if self.rng.random() < 0.4:
                sku = self.trending_sku
            else:
                sku = self.rng.randrange(NUM_SKUS)
            if len(self.pending) < MAX_PENDING_ORDERS:
                self.pending.append(sku)
                self.total_generated += 1
                if sku == self.trending_sku:
                    self.trend_demand += 1

    def has_order_for(self, sku_idx) -> bool:
        return sku_idx in self.pending


# ---------------------------------------------------------------------------
# Missing-item alert system
# ---------------------------------------------------------------------------
class FallenItem:
    """An item that fell off a carried shelf and now sits on the floor."""

    def __init__(self, sku_idx, cell, row_label):
        self.sku_idx = sku_idx
        self.cell = cell
        self.row_label = row_label
        self.on_floor = True   # False once a human is carrying it
        self.boxed = False     # True after the human boxes it


class Alert:
    def __init__(self, item: FallenItem):
        self.item = item
        self.age = 0
        self.critical = False

    @property
    def sku_code(self):
        return SKUS[self.item.sku_idx][0]


class AlertSystem:
    def __init__(self):
        self.alerts: list[Alert] = []

    def add(self, item: FallenItem) -> Alert:
        a = Alert(item)
        self.alerts.append(a)
        return a

    def update(self):
        for a in self.alerts:
            a.age += 1
            if a.age >= ALERT_CRIT_STEPS:
                a.critical = True

    def resolve(self, item: FallenItem):
        self.alerts = [a for a in self.alerts if a.item is not item]
