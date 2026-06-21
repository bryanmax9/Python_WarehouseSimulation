import { Suspense, useEffect, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Link } from 'react-router-dom'
import { motion } from 'motion/react'
import Warehouse3D, { Sel } from '../sim/Warehouse3D'
import { WarehouseSim, skuName, skuCode, skuColor, PHASE_NAME } from '../sim/engine'

const LOG_COLOR: Record<string, string> = {
  assign: '#5ab0f4', pickup: '#c08bff', deliver: '#46d68a', redeploy: '#9fff00',
  cool: '#f0b85c', store: '#9aa7bd', trend: '#ffd75c', alert: '#ef6b6b',
  resolve: '#46d68a', monitor: '#5cb0f4', restock: '#9fff00', yield: '#f0b85c', system: '#7c8597',
}

function useTick(ms: number) {
  const [, f] = useState(0)
  useEffect(() => { const id = setInterval(() => f((x) => x + 1), ms); return () => clearInterval(id) }, [ms])
}

function Dot({ c }: { c: string }) {
  return <span style={{ background: c }} className="inline-block w-2.5 h-2.5 rounded-sm shrink-0" />
}

function Inspector({ sim, sel, onClose }: { sim: WarehouseSim; sel: Sel; onClose: () => void }) {
  if (!sel) return null
  const Header = ({ title }: { title: string }) => (
    <div className="flex items-center justify-between mb-3">
      <div className="font-display font-bold text-white">{title}</div>
      <button onClick={onClose} className="text-white/50 hover:text-white text-sm">✕</button>
    </div>
  )

  if (sel.type === 'monitor') {
    const log = sim.monitorLog().slice(-9).reverse()
    return (
      <Panel>
        <Header title="Monitor — human worker" />
        <div className="space-y-1.5 text-sm mb-4">
          <Row k="doing" v={sim.monitorActivity()} />
          <Row k="carrying" v={sim.humanCarry ? skuCode(sim.humanCarry.sku) + ' · ' + skuName(sim.humanCarry.sku) : '— (empty-handed)'} />
        </div>
        <div className="text-white/50 text-xs uppercase tracking-wide mb-2">Activity log</div>
        <div className="font-mono text-[11px] leading-relaxed max-h-40 overflow-hidden">
          {log.length ? log.map((l, i) => (
            <div key={i} style={{ color: LOG_COLOR[l.kind] || '#cdd6e6' }} className="truncate">{l.msg}</div>
          )) : <div className="text-white/40">nothing to recover yet — patrolling</div>}
        </div>
      </Panel>
    )
  }

  if (sel.type === 'shelf') {
    const sh = sim.shelves[sel.i]
    const robotsHere = sim.robotsForShelf(sel.i)
    const ordered = sim.pendingForShelf(sel.i)
    return (
      <Panel>
        <Header title={`Rack (${sh.cell[0]}, ${sh.cell[1]})`} />
        <div className="text-white/50 text-xs uppercase tracking-wide mb-2">Barcoded inventory</div>
        <div className="space-y-1.5 mb-4">
          {sh.slots.map((sku, j) => (
            <div key={j} className="flex items-center gap-2 text-sm">
              <Dot c={skuColor(sku)} />
              <span className="font-mono text-white/90">{skuCode(sku)}</span>
              <span className="text-white/60 truncate">{skuName(sku)}</span>
              <span className="text-white/40 ml-auto">×{sh.qty[j]}</span>
              {sku === sim.trending && <span className="text-[#ffd75c]" title="trending">🔥</span>}
              {ordered.includes(sku) && <span className="text-[#46d68a] text-xs">ordered</span>}
            </div>
          ))}
        </div>
        <div className="text-white/50 text-xs uppercase tracking-wide mb-2">Delivery status</div>
        {robotsHere.length ? robotsHere.map((r) => (
          <div key={r.i} className="text-sm text-white/85">
            <b className="text-[#9fff00]">R{r.i}</b> — {r.status} <span className="text-white/50">({skuCode(r.sku)})</span>
          </div>
        )) : <div className="text-sm text-white/45">no robot assigned to this rack right now</div>}
      </Panel>
    )
  }

  const r = sim.robots[sel.i]
  const log = sim.robotLog(sel.i).slice(-9).reverse()
  return (
    <Panel>
      <Header title={`Robot R${sel.i}`} />
      <div className="space-y-1.5 text-sm mb-4">
        <Row k="state" v={PHASE_NAME[r.phase]} />
        <Row k="carrying" v={r.carrying ? skuCode(r.carrySku) + ' · ' + skuName(r.carrySku) : '—'} />
        <Row k="target cell" v={r.target ? `(${r.target[0]}, ${r.target[1]})` : '—'} />
        <Row k="at cell" v={`(${r.cell[0]}, ${r.cell[1]})`} />
      </div>
      <div className="text-white/50 text-xs uppercase tracking-wide mb-2">Decision log (thinking)</div>
      <div className="font-mono text-[11px] leading-relaxed max-h-40 overflow-hidden">
        {log.length ? log.map((l, i) => (
          <div key={i} style={{ color: LOG_COLOR[l.kind] || '#cdd6e6' }} className="truncate">{l.msg}</div>
        )) : <div className="text-white/40">no decisions logged yet</div>}
      </div>
    </Panel>
  )
}

function Row({ k, v }: { k: string; v: string }) {
  return <div className="flex gap-2"><span className="text-white/45 w-20 shrink-0">{k}</span><span className="text-white/90">{v}</span></div>
}
function Panel({ children }: { children: React.ReactNode }) {
  return (
    <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
      className="absolute top-3 right-3 sm:top-5 sm:right-5 w-[min(18rem,calc(100vw-1.5rem))] bg-black/55 backdrop-blur border border-white/15 rounded-2xl p-3 sm:p-4 z-20">
      {children}
    </motion.div>
  )
}

function HUD({ sim, sel, setSel }: { sim: WarehouseSim; sel: Sel; setSel: (s: Sel) => void }) {
  useTick(220)
  const logs = sim.logs.slice(-16).reverse()
  return (
    <>
      <div className="absolute top-3 left-3 sm:top-5 sm:left-5 flex flex-wrap items-center gap-2 max-w-[calc(100vw-1.5rem)]">
        <Link to="/" className="inline-flex items-center gap-1.5 bg-white/90 backdrop-blur text-zinc-900 text-xs sm:text-sm font-medium px-3 py-1.5 sm:px-4 sm:py-2 rounded-full hover:bg-white transition-colors">← back</Link>
        <div className="bg-white/10 backdrop-blur border border-white/15 rounded-full px-3 py-1.5 sm:px-4 sm:py-2 text-white/90 text-xs sm:text-sm flex flex-wrap gap-2 sm:gap-4">
          <span><b className="text-[#46d68a]">{sim.orders}</b> orders</span>
          <span><b className="text-[#ffd75c]">{skuName(sim.trending)}</b> trending</span>
          <span><b className="text-[#ef6b6b]">{sim.fallen.length}</b> alerts</span>
        </div>
      </div>

      <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }}
        className="absolute bottom-3 left-3 sm:bottom-5 sm:left-5 w-[min(420px,calc(100vw-1.5rem))] bg-black/45 backdrop-blur border border-white/12 rounded-2xl p-3 sm:p-4 z-10">
        <div className="text-white/90 text-xs font-semibold tracking-widest uppercase mb-2">Coordinator log · live</div>
        <div className="font-mono text-[11px] sm:text-[12px] leading-relaxed h-28 sm:h-44 overflow-hidden">
          {logs.map((l, i) => (
            <div key={i} style={{ color: LOG_COLOR[l.kind] || '#cdd6e6', opacity: 1 - i * 0.045 }} className="truncate">{l.msg}</div>
          ))}
        </div>
      </motion.div>

      {sel
        ? <Inspector sim={sim} sel={sel} onClose={() => setSel(null)} />
        : (
          <div className="hidden sm:block absolute top-5 right-5 bg-white/10 backdrop-blur border border-white/15 rounded-2xl p-4 text-white/85 text-sm w-64 z-20">
            <div className="font-display font-semibold mb-2 text-white">Live 3D warehouse <span className="text-[#9fff00]">· WebGL</span></div>
            <div className="flex items-center gap-2 mb-1.5"><Dot c="#5ab0f4" /> robots — <b>click to inspect</b></div>
            <div className="flex items-center gap-2 mb-1.5"><Dot c="#cfd5e6" /> racks — <b>click to inspect</b></div>
            <div className="flex items-center gap-2 mb-1.5"><Dot c="#46d68a" /> picker · boxing · restock</div>
            <div className="flex items-center gap-2 mb-1.5"><Dot c="#eef2fb" /> monitor (clears fallen items)</div>
            <div className="flex items-center gap-2"><Dot c="#ef6b6b" /> fallen item (alert)</div>
            <div className="mt-3 pt-3 border-t border-white/10 text-white/55 text-xs leading-relaxed">
              Click a rack to see its barcodes &amp; who needs them; click a robot to read its thinking.
            </div>
          </div>
        )}

      <div className="hidden sm:block absolute bottom-5 right-5 text-white/45 text-xs">drag to orbit · scroll to zoom · click empty space to deselect</div>
    </>
  )
}

export default function Simulation() {
  const sim = useMemo(() => new WarehouseSim(7), [])
  const [sel, setSel] = useState<Sel>(null)
  return (
    <div className="fixed inset-0 bg-[#0d1018]">
      <Canvas shadows camera={{ position: [14, 13, 16], fov: 42 }} onPointerMissed={() => setSel(null)}>
        <Suspense fallback={null}>
          <Warehouse3D sim={sim} selected={sel} onSelect={setSel} />
        </Suspense>
      </Canvas>
      <HUD sim={sim} sel={sel} setSel={setSel} />
    </div>
  )
}
