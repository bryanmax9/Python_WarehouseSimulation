import { Link } from 'react-router-dom'
import { motion } from 'motion/react'
import Navbar from '../components/Navbar'

const fadeUp = { initial: { opacity: 0, y: 16 }, whileInView: { opacity: 1, y: 0 }, viewport: { once: true } }

const FULL = [
  ['Heuristic (built-in)', '41.3', '0.0', '3.0'],
  ['Base LLM (minimax-m3) — no fine-tune', '40.7', '1.0', '4.0'],
  ['Greedy LLM (mock)', '40.3', '1.0', '21.3'],
]
const HARD: [string, string, string][] = [
  ['Greedy dispatcher', '0.733', 'win'],
  ['Base LLM (minimax-m3) — no fine-tune', '0.733', 'win'],
  ['Fine-tuned Gemma (small)', '0.31', ''],
  ['Base Gemma (small)', '0.29', ''],
  ['No-op (does nothing)', '0.00', 'zero'],
]
const LOOP = [
  ['Environment', 'WarehouseEnv — a Gymnasium world of robots, orders & a trending item'],
  ['Verifiable reward', 'orders fulfilled / missing — counted by the world, not a model'],
  ['Post-training data', '1,897 reward-filtered (state → expert dispatch) examples'],
  ['Fine-tune', 'Gemma-4-26B-A4B on Fireworks (LoRA, loss 2.5 → 0)'],
  ['Evaluate on HUD', 're-score on the same verifiable reward — a real HUD v6 task'],
]
const STACK = [
  ['HUD', 'The verifiable environment + agentic eval live here (HUD v6 task).'],
  ['Fireworks', 'Serves the LLM coordinator and runs the RFT fine-tune.'],
  ['MiniMax', 'minimax-m3 is the dispatch coordinator model.'],
  ['Modal', 'Scaled PPO motion-policy training on cloud GPUs.'],
  ['Antim · GIZMO', 'Physical-AI assets / the sim-to-real vision.'],
]

function Heading({ kicker, title, sub }: { kicker: string; title: string; sub?: string }) {
  return (
    <motion.div {...fadeUp} transition={{ duration: 0.6 }} className="mb-8">
      <div className="text-xs font-semibold tracking-widest uppercase text-zinc-400 mb-2">{kicker}</div>
      <h2 className="font-display text-2xl md:text-4xl font-bold tracking-tight">{title}</h2>
      {sub && <p className="text-zinc-500 mt-2 max-w-2xl">{sub}</p>}
    </motion.div>
  )
}

export default function Landing() {
  return (
    <>
      <Navbar />

      {/* HERO */}
      <section className="relative w-full pt-36 md:pt-48 pb-20 overflow-hidden">
        <div className="absolute inset-x-0 top-0 h-[520px] -z-10 pointer-events-none"
          style={{ background: 'radial-gradient(900px 380px at 70% -8%, rgba(159,255,0,0.22), transparent), radial-gradient(700px 360px at 12% 4%, rgba(124,92,255,0.10), transparent)' }} />
        <div className="max-w-7xl mx-auto px-6 md:px-12 grid grid-cols-12">
          <div className="col-span-12 md:col-span-11">
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}
              className="inline-flex items-center gap-2 text-xs font-semibold tracking-widest uppercase text-zinc-500 mb-6">
              <span className="w-2 h-2 rounded-full bg-brand-green" /> HUD × YC · Frontier RSI RL Environments
            </motion.div>
            <motion.h1 initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}
              className="font-display font-extrabold leading-[1.05] text-4xl sm:text-6xl md:text-7xl tracking-tight">
              <span className="text-ink">A verifiable warehouse</span><br />
              <span className="text-zinc-400">environment that teaches a</span><br />
              <span className="text-zinc-400">model to run the </span>
              <span className="inline-flex items-center justify-center align-middle w-[42px] md:w-[64px] h-[26px] md:h-[40px] border-2 border-ink rounded-full mx-1">
                <span className="w-2 h-2 rounded-full bg-ink" />
              </span>
              <span className="text-ink"> floor.</span>
            </motion.h1>
            <motion.p initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.1 }}
              className="mt-7 max-w-2xl text-lg text-zinc-500">
              A frontier model is the dispatch <b className="text-zinc-700">brain</b>; PPO is the motion
              <b className="text-zinc-700"> muscle</b>; the environment's reward — orders fulfilled, counted by the
              world, not a model — is the <b className="text-zinc-700">verifier</b>. The whole post-training loop, wired end to end.
            </motion.p>
            <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.18 }}
              className="mt-9 flex flex-wrap items-center gap-3">
              <Link to="/simulation" className="group inline-flex items-center gap-2 bg-ink text-white font-medium pl-6 pr-2 py-2 rounded-full hover:bg-black transition-colors">
                Run simulation
                <span className="bg-brand-green text-black w-9 h-9 rounded-full inline-flex items-center justify-center group-hover:translate-x-0.5 transition-transform">→</span>
              </Link>
              <a href="#hud" className="inline-flex items-center gap-2 bg-white border border-black/10 text-zinc-700 font-medium px-5 py-2.5 rounded-full hover:border-black/20 transition-colors">
                See it on HUD
              </a>
            </motion.div>
          </div>
        </div>
      </section>

      {/* THESIS */}
      <section className="border-y border-black/5 bg-white/40">
        <div className="max-w-7xl mx-auto px-6 md:px-12 py-14">
          <motion.p {...fadeUp} transition={{ duration: 0.6 }} className="font-display text-2xl md:text-4xl font-semibold leading-snug max-w-4xl">
            “You can improve models at anything you can <span className="bg-brand-green/60 px-1 rounded">verify</span>.”
          </motion.p>
          <motion.p {...fadeUp} transition={{ duration: 0.6, delay: 0.1 }} className="text-zinc-500 mt-4 max-w-2xl">
            By 2040, physical operations — warehouses, factories, ports — are run by frontier models coordinating
            robot fleets. To get there we need environments that <b>teach and verify</b> them. So we built one,
            and ran the full recursive-self-improvement loop on it.
          </motion.p>
        </div>
      </section>

      {/* THE LOOP */}
      <section id="overview" className="max-w-7xl mx-auto px-6 md:px-12 py-16">
        <Heading kicker="How it works" title="The recursive self-improvement loop"
          sub="Every link is built and runs on the sponsor stack — environment to measured improvement." />
        <div className="grid md:grid-cols-5 gap-4">
          {LOOP.map(([t, d], i) => (
            <motion.div key={t} {...fadeUp} transition={{ duration: 0.5, delay: i * 0.06 }}
              className="relative bg-white rounded-2xl border border-black/5 shadow-sm p-5">
              <div className="w-7 h-7 rounded-full bg-ink text-white text-sm font-bold flex items-center justify-center mb-3">{i + 1}</div>
              <div className="font-display font-bold mb-1">{t}</div>
              <p className="text-sm text-zinc-500">{d}</p>
              {i < LOOP.length - 1 && <div className="hidden md:block absolute -right-3 top-1/2 text-brand-green text-xl">→</div>}
            </motion.div>
          ))}
        </div>
        <div className="text-sm text-zinc-400 mt-4">↺ the measured reward feeds back into the next round of data &amp; fine-tuning.</div>
      </section>

      {/* PROVEN ON HUD */}
      <section id="hud" className="bg-white/40 border-y border-black/5">
        <div className="max-w-7xl mx-auto px-6 md:px-12 py-16 grid md:grid-cols-2 gap-10 items-center">
          <div>
            <Heading kicker="Proven on the platform" title="Running as a real HUD task" />
            <p className="text-zinc-500 max-w-xl">
              The dispatch decision is shipped as a HUD v6 task with a verifiable reward, evaluated through the HUD
              gateway. Not a slide — a completed run with a replayable trace on hud.ai.
            </p>
            <div className="flex flex-wrap gap-2 mt-5 text-sm">
              <span className="inline-flex items-center gap-1.5 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full px-3 py-1">✓ COMPLETED</span>
              <span className="bg-white border border-black/10 rounded-full px-3 py-1 text-zinc-600">task: warehouse_dispatch</span>
              <span className="bg-white border border-black/10 rounded-full px-3 py-1 text-zinc-600">scored via HUD gateway</span>
            </div>
          </div>
          <motion.div {...fadeUp} transition={{ duration: 0.6 }} className="bg-white rounded-2xl border border-black/5 shadow-sm p-7">
            <div className="text-sm text-zinc-400 mb-1">Average reward on HUD</div>
            <div className="font-display text-6xl font-extrabold text-emerald-500">87%</div>
            <div className="text-zinc-500 text-sm mt-2 mb-5">score distribution (traces by reward bucket)</div>
            <div className="flex items-end gap-2 h-28">
              {[0, 0, 0, 0, 100].map((h, i) => (
                <div key={i} className="flex-1 flex flex-col items-center justify-end">
                  <div className="w-full rounded-t" style={{ height: `${Math.max(h, 3)}%`, background: h ? '#46d68a' : '#e3e6ef' }} />
                  <div className="text-[10px] text-zinc-400 mt-1">{i * 20}%</div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* RESULTS */}
      <section id="results" className="max-w-7xl mx-auto px-6 md:px-12 py-16">
        <Heading kicker="Results" title="Reproducible & verifiable"
          sub="One command — python benchmark.py --fireworks — scores every coordinator on two metrics." />
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white rounded-2xl border border-black/5 shadow-sm p-6">
            <h3 className="font-display font-bold mb-1">Full-episode throughput</h3>
            <p className="text-sm text-zinc-500 mb-4">The warehouse running a whole shift (orders fulfilled).</p>
            <table className="w-full text-sm">
              <thead><tr className="text-zinc-400 text-xs uppercase tracking-wide">
                <th className="text-left font-medium pb-2">Coordinator</th><th className="text-right font-medium pb-2">Orders</th>
                <th className="text-right font-medium pb-2">Miss</th><th className="text-right font-medium pb-2">Coll.</th></tr></thead>
              <tbody>{FULL.map((r) => (
                <tr key={r[0]} className="border-t border-black/5">
                  <td className="py-2 pr-2">{r[0]}</td><td className="py-2 text-right tabular-nums">{r[1]}</td>
                  <td className="py-2 text-right tabular-nums text-zinc-400">{r[2]}</td><td className="py-2 text-right tabular-nums text-zinc-400">{r[3]}</td>
                </tr>))}</tbody>
            </table>
          </div>
          <div className="bg-white rounded-2xl border border-black/5 shadow-sm p-6">
            <h3 className="font-display font-bold mb-1">Hard dispatch task — the HUD reward</h3>
            <p className="text-sm text-zinc-500 mb-4">One busy-state decision; agent is the sole dispatcher (0–1).</p>
            <table className="w-full text-sm">
              <thead><tr className="text-zinc-400 text-xs uppercase tracking-wide">
                <th className="text-left font-medium pb-2">Coordinator</th><th className="text-right font-medium pb-2">Reward</th></tr></thead>
              <tbody>{HARD.map((r) => (
                <tr key={r[0]} className="border-t border-black/5">
                  <td className="py-2 pr-2">{r[0]}</td>
                  <td className={'py-2 text-right tabular-nums font-semibold ' + (r[2] === 'win' ? 'text-lime-600' : r[2] === 'zero' ? 'text-red-500' : 'text-zinc-700')}>{r[1]}</td>
                </tr>))}</tbody>
            </table>
          </div>
        </div>
        <div className="mt-6 bg-gradient-to-r from-brand-green/20 to-transparent border-l-2 border-brand-green rounded-xl px-5 py-4 text-sm text-zinc-700">
          <b>Runs without fine-tuning.</b> A base LLM brain matches the expert heuristic (0.73) with zero fine-tuning.
          The environment is a real eval: it ranks everything from a do-nothing agent (0.0) to a strong dispatcher (0.73),
          and even caught that a small fine-tuned model makes grounding errors under load.
        </div>
      </section>

      {/* STACK */}
      <section id="architecture" className="bg-white/40 border-t border-black/5">
        <div className="max-w-7xl mx-auto px-6 md:px-12 py-16">
          <Heading kicker="Built on the stack" title="Five sponsors, actually used" />
          <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-4">
            {STACK.map(([n, d], i) => (
              <motion.div key={n} {...fadeUp} transition={{ duration: 0.5, delay: i * 0.05 }}
                className="bg-white rounded-2xl border border-black/5 shadow-sm p-5">
                <div className="font-display font-bold mb-1">{n}</div>
                <p className="text-sm text-zinc-500">{d}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-7xl mx-auto px-6 md:px-12 py-20 text-center">
        <h2 className="font-display text-3xl md:text-5xl font-extrabold tracking-tight">See the world it learns in.</h2>
        <p className="text-zinc-500 mt-3 mb-7">A live 3D warehouse — robots, pods, a human monitor, and the decision log, in your browser.</p>
        <Link to="/simulation" className="group inline-flex items-center gap-2 bg-ink text-white font-medium pl-6 pr-2 py-2.5 rounded-full hover:bg-black transition-colors">
          Run simulation
          <span className="bg-brand-green text-black w-9 h-9 rounded-full inline-flex items-center justify-center group-hover:translate-x-0.5 transition-transform">→</span>
        </Link>
      </section>

      <footer className="border-t border-black/5">
        <div className="max-w-7xl mx-auto px-6 md:px-12 py-8 flex flex-wrap gap-x-6 gap-y-2 text-sm text-zinc-400">
          <span>HUD</span><span>Fireworks</span><span>MiniMax</span><span>Modal</span><span>Antim · GIZMO</span>
          <span className="ml-auto">2026 · verifiable warehouse RL · HUD × YC</span>
        </div>
      </footer>
    </>
  )
}
