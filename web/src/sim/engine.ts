// Faithful JS port of the Python warehouse sim (warehouse_core.py + the
// heuristic parts of warehouse_env.py + the human monitor from visualize.py),
// extended with the full fallen-item recovery loop:
//   item falls off a carried rack  ->  MONITOR picks it up -> BOXING -> RESTOCK
//   ->  an idle ROBOT collects it from restock and returns it to the rack.

export type Cell = [number, number]
export const COLS = 13
export const ROWS = 11
export const NUM_ROBOTS = 6
export const NUM_SKUS = 20
const SLOTS_PER_SHELF = 3
const SHELF_ROWS = [2, 4, 6, 8]
const ORDER_INTERVAL = 12
const TREND_INTERVAL = 150
const ITEM_FALL_PROB = 0.012
const MAX_PENDING = 12
const DELIVER_RADIUS = 1

const SKU_NAMES = [
  'Wireless Headphones', 'Phone Case', 'USB Cable', 'Bluetooth Speaker',
  'Laptop Stand', 'Mechanical Keyboard', 'Wireless Mouse', 'Webcam HD',
  'Power Bank', 'HDMI Adapter', 'Screen Protector', 'Gaming Headset',
  'Smart Watch', 'Earbuds Pro', 'Tablet Sleeve', 'Desk Lamp',
  'Cable Organizer', 'Microphone', 'Monitor Arm', 'Charging Dock',
]

export const IDLE = 0, FETCHING = 1, DELIVERING = 2, RETURNING = 3, HOVERING = 4
export const RESTOCK_FETCH = 5, RESTOCK_RETURN = 6
export const PHASE_NAME: Record<number, string> = {
  [IDLE]: 'idle — awaiting order', [FETCHING]: 'fetching pod', [DELIVERING]: 'delivering to picker',
  [RETURNING]: 'returning rack to storage', [HOVERING]: 'hovering (hot item)',
  [RESTOCK_FETCH]: 'to restock station', [RESTOCK_RETURN]: 'restocking rack',
}

const key = (c: Cell) => c[0] + ',' + c[1]
const eq = (a: Cell, b: Cell) => a[0] === b[0] && a[1] === b[1]
const manh = (a: Cell, b: Cell) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1])

function rng(seed: number) {
  let s = seed >>> 0
  return () => {
    s |= 0; s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function skuColor(idx: number): string {
  const h = (idx % NUM_SKUS) / NUM_SKUS, s = 0.65, v = 0.95
  const i = Math.floor(h * 6), f = h * 6 - i
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s)
  let r = 0, g = 0, b = 0
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break
    case 1: r = q; g = v; b = p; break
    case 2: r = p; g = v; b = t; break
    case 3: r = p; g = q; b = v; break
    case 4: r = t; g = p; b = v; break
    case 5: r = v; g = p; b = q; break
  }
  const hx = (n: number) => Math.round(n * 255).toString(16).padStart(2, '0')
  return '#' + hx(r) + hx(g) + hx(b)
}
export const skuName = (i: number) => SKU_NAMES[i % NUM_SKUS]
export const skuCode = (i: number) => 'SKU-' + String((i % NUM_SKUS) + 1).padStart(4, '0')

export interface Shelf { cell: Cell; slots: number[]; qty: number[]; reserved: boolean }
interface RestockJob { sku: number; rack: Cell }
interface Robot {
  cell: Cell; phase: number; carrying: boolean; carrySku: number
  shelf: Shelf | null; target: Cell | null; fromHover: boolean
  assignedSku: number; lastCollision: string; stuck: number; moveFrom: Cell
  restock: RestockJob | null
}
export interface Fallen { sku: number; cell: Cell; rack: Cell }
export interface LogLine { kind: string; msg: string }

export class WarehouseSim {
  cols = COLS; rows = ROWS
  shelves: Shelf[] = []
  private shelfMap = new Map<string, Shelf>()
  private bySku: Shelf[][] = []
  picker: Cell; boxing: Cell; restock: Cell
  private stationKeys: Set<string>
  robots: Robot[] = []
  human: Cell; humanCarry: Fallen | null = null
  humanCarryBoxed = false
  fallen: Fallen[] = []
  restockJobs: RestockJob[] = []
  orders = 0
  pending: number[] = []
  trending: number
  step = 0
  logs: LogLine[] = []
  private rand: () => number

  constructor(seed = 1) {
    this.rand = rng(seed)
    const bottom = ROWS - 1
    this.picker = [Math.floor(COLS / 2), bottom]
    this.boxing = [COLS - 1, bottom]
    this.restock = [0, bottom]
    this.stationKeys = new Set([key(this.picker), key(this.boxing), key(this.restock)])
    this.trending = Math.floor(this.rand() * NUM_SKUS)
    for (let i = 0; i < NUM_SKUS; i++) this.bySku.push([])

    for (const row of SHELF_ROWS) {
      for (let col = 1; col < COLS - 1; col++) {
        if (col % 3 === 0) continue
        const slots = Array.from({ length: SLOTS_PER_SHELF }, () => Math.floor(this.rand() * NUM_SKUS))
        const qty = slots.map(() => 3 + Math.floor(this.rand() * 27))
        const sh: Shelf = { cell: [col, row], slots, qty, reserved: false }
        this.shelves.push(sh); this.shelfMap.set(key([col, row]), sh)
        new Set(slots).forEach((s) => this.bySku[s].push(sh))
      }
    }
    for (let sku = 0; sku < NUM_SKUS; sku++) {
      if (this.bySku[sku].length === 0) {
        const sh = this.shelves[Math.floor(this.rand() * this.shelves.length)]
        sh.slots[0] = sku; this.bySku[sku].push(sh)
      }
    }

    const spawns: Cell[] = [[2, 0], [4, 0], [8, 0], [10, 0], [5, 0], [7, 0]]
    this.robots = spawns.slice(0, NUM_ROBOTS).map((c) => ({
      cell: c, phase: IDLE, carrying: false, carrySku: -1, shelf: null, target: null,
      fromHover: false, assignedSku: -1, lastCollision: '', stuck: 0, moveFrom: c, restock: null,
    }))
    this.human = [Math.floor(COLS / 2), 0]
    this.log('system', 'heuristic coordinator online · ' + this.shelves.length + ' pods')
  }

  private inBounds = (c: Cell) => c[0] >= 0 && c[0] < COLS && c[1] >= 0 && c[1] < ROWS
  isShelf = (c: Cell) => this.shelfMap.has(key(c))
  shelfAt = (c: Cell) => this.shelfMap.get(key(c))
  liftedCells(): Set<string> {
    const s = new Set<string>()
    for (const r of this.robots) if (r.carrying && r.shelf) s.add(key(r.shelf.cell))
    return s
  }

  private bfs(start: Cell, goal: Cell, blocked: Set<string>, human: boolean): Cell[] | null {
    if (eq(start, goal)) return []
    const gk = key(goal)
    // the human is a tall person: blocked by pods AND by the station structures
    // (picker / boxing / restock). Only the destination cell is ever exempt.
    const walk = (c: Cell) =>
      this.inBounds(c) && (!human || (!this.isShelf(c) && !this.stationKeys.has(key(c))))
    const q: Cell[] = [start]
    const came = new Map<string, Cell | null>([[key(start), null]])
    while (q.length) {
      const cur = q.shift()!
      for (const d of [[0, -1], [0, 1], [-1, 0], [1, 0]] as Cell[]) {
        const nb: Cell = [cur[0] + d[0], cur[1] + d[1]]
        const nk = key(nb)
        if (came.has(nk)) continue
        if (nk !== gk && !walk(nb)) continue
        if (nk !== gk && blocked.has(nk)) continue
        came.set(nk, cur)
        if (nk === gk) {
          const path: Cell[] = [nb]
          while (!eq(came.get(key(path[path.length - 1]))!, start)) path.push(came.get(key(path[path.length - 1]))!)
          path.reverse(); return path
        }
        q.push(nb)
      }
    }
    return null
  }

  private nearestShelfWithSku(from: Cell, sku: number): Shelf | null {
    let best: Shelf | null = null, bd = 1e9
    for (const sh of this.bySku[sku]) {
      if (sh.reserved) continue
      const d = manh(sh.cell, from)
      if (d < bd) { bd = d; best = sh }
    }
    return best
  }
  private approach(cell: Cell): Cell {
    if (!this.isShelf(cell) && this.inBounds(cell)) return cell
    let best: Cell = cell, bd = 1e9
    for (const d of [[1, 0], [-1, 0], [0, 1], [0, -1]] as Cell[]) {
      const nb: Cell = [cell[0] + d[0], cell[1] + d[1]]
      if (this.inBounds(nb) && !this.isShelf(nb)) { const dist = manh(nb, this.human); if (dist < bd) { bd = dist; best = nb } }
    }
    return best
  }

  stepSim() {
    for (const r of this.robots) r.lastCollision = ''
    const pp = this.robots.map((r) => r.phase)
    const pc = this.robots.map((r) => r.carrying)
    const pt = this.trending
    this.controlHuman()
    this.updateHuman()
    this.robots.forEach((r) => (r.moveFrom = r.cell))
    this.resolveMoves()
    this.breakDeadlocks()
    this.interactions()
    this.worldTick()
    this.humanInteract()
    this.assignOrders()
    this.narrate(pp, pc, pt)
    this.step++
  }

  private planMove(r: Robot, others: Set<string>): Cell {
    if (!r.target || eq(r.cell, r.target)) return r.cell
    const blocked = new Set(others)
    blocked.add(key(this.human))
    // robots never drive onto the stations (workstations occupy them)
    this.stationKeys.forEach((k) => { if (k !== key(r.target!)) blocked.add(k) })
    const path = this.bfs(r.cell, r.target, blocked, r.carrying)
    if (!path || path.length === 0) return r.cell
    return path[0]
  }

  private resolveMoves() {
    const cur = this.robots.map((r) => r.cell)
    const want = this.robots.map((r, i) => {
      const others = new Set<string>()
      this.robots.forEach((o, j) => { if (j !== i) others.add(key(o.cell)) })
      const nb = this.planMove(r, others)
      if (eq(nb, this.human)) { r.lastCollision = 'human'; return cur[i] }
      return nb
    })
    const final = want.slice()
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i < this.robots.length; i++) {
        if (eq(final[i], cur[i])) continue
        let crash = false, blk = false
        for (let j = 0; j < this.robots.length; j++) {
          if (i === j) continue
          const sameCell = eq(final[i], final[j]) && !eq(final[j], cur[j])
          const swap = eq(final[i], cur[j]) && eq(final[j], cur[i])
          if (sameCell || swap) { crash = true; break }
          if (eq(final[i], cur[j]) && eq(final[j], cur[j])) blk = true
        }
        if (crash) { this.robots[i].lastCollision = 'robot'; final[i] = cur[i] }
        else if (blk) final[i] = cur[i]
      }
    }
    this.robots.forEach((r, i) => (r.cell = final[i]))
  }

  private breakDeadlocks() {
    const occ = new Set(this.robots.map((r) => key(r.cell)))
    for (const r of this.robots) {
      const moved = !eq(r.cell, r.moveFrom)
      if (r.target && !eq(r.cell, r.target) && !moved) r.stuck++; else r.stuck = 0
      if (r.stuck >= 3) {
        let best: Cell | null = null, bd = 1e9
        for (const d of [[0, -1], [0, 1], [-1, 0], [1, 0]] as Cell[]) {
          const nb: Cell = [r.cell[0] + d[0], r.cell[1] + d[1]]
          if (this.inBounds(nb) && !occ.has(key(nb)) && !eq(nb, this.human) && !this.stationKeys.has(key(nb))) {
            const dist = manh(nb, r.target!); if (dist < bd) { bd = dist; best = nb }
          }
        }
        if (best) { occ.delete(key(r.cell)); r.cell = best; occ.add(key(best)); r.stuck = 0 }
      }
    }
  }

  private interactions() {
    for (const r of this.robots) {
      if (r.phase === FETCHING && !r.carrying && r.shelf && eq(r.cell, r.shelf.cell)) {
        r.carrying = true; r.carrySku = r.assignedSku; r.phase = DELIVERING; r.target = this.picker
      } else if (r.phase === DELIVERING && r.carrying && manh(r.cell, this.picker) <= DELIVER_RADIUS) {
        this.orders++
        r.fromHover = false
        if (r.carrySku === this.trending) { r.phase = HOVERING; r.target = this.freeHover() }
        else { r.phase = RETURNING; r.target = r.shelf!.cell }
      } else if (r.phase === RETURNING && r.carrying && r.shelf && eq(r.cell, r.shelf.cell)) {
        r.shelf.reserved = false; r.carrying = false; r.carrySku = -1; r.shelf = null; r.phase = IDLE; r.target = null
      } else if (r.phase === RESTOCK_FETCH && !r.carrying && manh(r.cell, this.restock) <= DELIVER_RADIUS) {
        r.carrying = true; r.carrySku = r.restock!.sku; r.phase = RESTOCK_RETURN; r.target = r.restock!.rack
      } else if (r.phase === RESTOCK_RETURN && r.carrying && r.restock && eq(r.cell, r.restock.rack)) {
        const sh = this.shelfAt(r.restock.rack)
        if (sh) { const slot = sh.slots.indexOf(r.restock.sku); if (slot >= 0) sh.qty[slot] += 1 }
        this.log('restock', `R${this.robots.indexOf(r)}: restocked ${skuCode(r.restock.sku)} into rack (${r.restock.rack}) — recovered`)
        r.carrying = false; r.carrySku = -1; r.restock = null; r.phase = IDLE; r.target = null
      }
    }
  }

  private freeHover(): Cell {
    const taken = new Set(this.robots.filter((r) => r.phase === HOVERING).map((r) => key(r.target!)))
    const cells: Cell[] = [[1, ROWS - 1], [2, ROWS - 1], [COLS - 3, ROWS - 1], [COLS - 2, ROWS - 1]]
    return cells.find((c) => !taken.has(key(c))) || cells[0]
  }

  private worldTick() {
    if (this.step > 0 && this.step % TREND_INTERVAL === 0) this.trending = Math.floor(this.rand() * NUM_SKUS)
    if (this.step > 0 && this.step % ORDER_INTERVAL === 0 && this.pending.length < MAX_PENDING) {
      const sku = this.rand() < 0.4 ? this.trending : Math.floor(this.rand() * NUM_SKUS)
      this.pending.push(sku)
    }
    for (const r of this.robots) {
      if (r.carrying && (r.phase === DELIVERING || r.phase === RETURNING) && r.shelf) {
        const p = ITEM_FALL_PROB * (r.lastCollision ? 4 : 1)
        if (this.rand() < p) {
          const f: Fallen = { sku: r.carrySku, cell: [...r.cell] as Cell, rack: [...r.shelf.cell] as Cell }
          this.fallen.push(f)
          this.log('alert', `ALERT: ${skuCode(f.sku)} fell off R${this.robots.indexOf(r)} at row ${r.cell[1]} — monitor dispatched`)
        }
      }
    }
  }

  private humanAction: Cell = [0, 0]
  private humanWantInteract = false
  private controlHuman() {
    this.humanAction = [0, 0]; this.humanWantInteract = false
    const near = (c: Cell) => manh(c, this.human) <= 1
    let goal: Cell
    if (!this.humanCarry) {
      if (this.fallen.length === 0) {
        if (this.rand() < 0.5) { const d: Cell[] = [[0, -1], [0, 1], [-1, 0], [1, 0]]; this.humanAction = d[Math.floor(this.rand() * 4)] }
        return
      }
      const item = this.fallen.reduce((a, b) => (manh(a.cell, this.human) <= manh(b.cell, this.human) ? a : b))
      if (near(item.cell)) { this.humanWantInteract = true; return }
      goal = this.approach(item.cell)
    } else {
      goal = !this.humanCarryBoxed ? this.boxing : this.restock
      if (near(goal)) { this.humanWantInteract = true; return }
    }
    // route AROUND robots (the person yields to AMRs)
    const robotCells = new Set(this.robots.map((r) => key(r.cell)))
    const path = this.bfs(this.human, goal, robotCells, true)
    if (path && path.length) this.humanAction = [path[0][0] - this.human[0], path[0][1] - this.human[1]]
  }
  private updateHuman() {
    const [dx, dy] = this.humanAction
    const nc: Cell = [this.human[0] + dx, this.human[1] + dy]
    const robotCells = new Set(this.robots.map((r) => key(r.cell)))
    // never step onto a pod, a station, or a robot
    if (this.inBounds(nc) && !this.isShelf(nc) && !this.stationKeys.has(key(nc)) && !robotCells.has(key(nc)))
      this.human = nc
  }
  private humanInteract() {
    if (!this.humanWantInteract) return
    const near = (c: Cell) => manh(c, this.human) <= 1
    if (!this.humanCarry) {
      const idx = this.fallen.findIndex((it) => near(it.cell))
      if (idx >= 0) {
        this.humanCarry = this.fallen[idx]; this.humanCarryBoxed = false; this.fallen.splice(idx, 1)
        this.log('monitor', `monitor picked up ${skuCode(this.humanCarry.sku)} → carry to BOXING`)
      }
    } else if (!this.humanCarryBoxed && near(this.boxing)) {
      this.humanCarryBoxed = true
      this.log('monitor', `monitor BOXED ${skuCode(this.humanCarry.sku)} → carry to RESTOCK`)
    } else if (this.humanCarryBoxed && near(this.restock)) {
      this.restockJobs.push({ sku: this.humanCarry.sku, rack: this.humanCarry.rack })
      this.log('resolve', `monitor staged ${skuCode(this.humanCarry.sku)} at RESTOCK (+20) → robot will return it`)
      this.humanCarry = null; this.humanCarryBoxed = false
    }
  }

  private assignOrders() {
    // restock jobs first: send an idle robot to recover a staged item
    if (this.restockJobs.length) {
      const idleR = this.robots.find((r) => r.phase === IDLE)
      if (idleR) { idleR.restock = this.restockJobs.shift()!; idleR.phase = RESTOCK_FETCH; idleR.target = this.restock }
    }
    if (this.pending.length === 0) return
    const remaining: number[] = []
    const idle = this.robots.filter((r) => r.phase === IDLE)
    while (this.pending.length) {
      const sku = this.pending.shift()!
      let assigned = false
      for (const r of this.robots) {
        if (r.phase === HOVERING && r.carrySku === sku) { r.phase = DELIVERING; r.fromHover = true; r.target = this.picker; assigned = true; break }
      }
      if (assigned) continue
      if (idle.length) {
        const r = idle[0]
        const sh = this.nearestShelfWithSku(r.cell, sku)
        if (sh) { sh.reserved = true; r.shelf = sh; r.assignedSku = sku; r.phase = FETCHING; r.target = sh.cell; idle.shift(); assigned = true }
      }
      if (!assigned) remaining.push(sku)
    }
    this.pending.push(...remaining)
    for (const r of this.robots) if (r.phase === HOVERING && r.carrySku !== this.trending) { r.phase = RETURNING; r.target = r.shelf!.cell }
  }

  private narrate(pp: number[], pc: boolean[], pt: number) {
    let near = 0
    this.robots.forEach((r, i) => {
      const sku = r.carrying ? r.carrySku : r.assignedSku
      if (pp[i] === IDLE && r.phase === FETCHING) this.log('assign', `R${i}: order ${skuCode(r.assignedSku)} → drive to nearest pod`)
      if (!pc[i] && r.carrying && r.phase === DELIVERING) this.log('pickup', `R${i}: reached pod → LIFT rack ${skuCode(sku)}, haul to picker`)
      if (pp[i] === DELIVERING && r.phase === HOVERING) this.log('deliver', `R${i}: delivered ${skuCode(sku)} +10 → it's HOT, hover near exit`)
      if (pp[i] === DELIVERING && r.phase === RETURNING) this.log('deliver', `R${i}: delivered ${skuCode(sku)} +10 → return rack to storage`)
      if (pp[i] === HOVERING && r.phase === DELIVERING) this.log('redeploy', `R${i}: ${skuCode(sku)} ordered again → re-deploy from hover (+5)`)
      if (pp[i] === HOVERING && r.phase === RETURNING) this.log('cool', `R${i}: ${skuCode(r.carrySku)} cooled off → return rack`)
      if (pp[i] === RETURNING && r.phase === IDLE) this.log('store', `R${i}: rack stored → idle, awaiting order`)
      if (pp[i] === IDLE && r.phase === RESTOCK_FETCH) this.log('assign', `R${i}: recover job → collect ${skuCode(r.restock!.sku)} from RESTOCK`)
      if (r.lastCollision === 'robot') near++
    })
    if (this.trending !== pt) this.log('trend', `DEMAND SPIKE: ${skuName(this.trending)} is now trending`)
    if (near && this.step % 5 === 0) this.log('yield', `path conflict → robots brake & re-route`)
  }

  private log(kind: string, msg: string) {
    this.logs.push({ kind, msg: `${String(this.step).padStart(3, ' ')}  ${msg}` })
    if (this.logs.length > 240) this.logs.shift()
  }

  // ---- inspector helpers --------------------------------------------------
  // which robots are involved with a given shelf (fetching/carrying from it)
  robotsForShelf(idx: number): { i: number; status: string; sku: number }[] {
    const sh = this.shelves[idx]
    const out: { i: number; status: string; sku: number }[] = []
    this.robots.forEach((r, i) => {
      if (r.shelf === sh) out.push({ i, status: PHASE_NAME[r.phase], sku: r.carrying ? r.carrySku : r.assignedSku })
    })
    return out
  }
  // pending orders this shelf could fulfil right now
  pendingForShelf(idx: number): number[] {
    const sh = this.shelves[idx]
    return this.pending.filter((sku) => sh.slots.includes(sku))
  }
  robotLog(i: number): LogLine[] {
    return this.logs.filter((l) => l.msg.includes(`R${i}:`))
  }
  // plain-language description of what the monitor is doing right now
  monitorActivity(): string {
    if (this.humanCarry) return `carrying ${skuCode(this.humanCarry.sku)} to ${this.humanCarryBoxed ? 'RESTOCK' : 'BOXING'}`
    if (this.fallen.length) return 'walking over to a fallen item'
    return 'patrolling the aisles'
  }
  monitorLog(): LogLine[] {
    return this.logs.filter((l) => ['monitor', 'alert', 'resolve', 'restock'].includes(l.kind))
  }
}
