import { useState } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'motion/react'

const LINKS = ['overview', 'architecture', 'results', 'github']

function Clover() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="#1a1a1a" aria-hidden>
      <path d="M12 2c1.7 0 3 1.3 3 3 0 .5-.1 1-.4 1.4 .4-.3 .9-.4 1.4-.4 1.7 0 3 1.3 3 3s-1.3 3-3 3c-.5 0-1-.1-1.4-.4 .3 .4 .4 .9 .4 1.4 0 1.7-1.3 3-3 3s-3-1.3-3-3c0-.5 .1-1 .4-1.4-.4 .3-.9 .4-1.4 .4-1.7 0-3-1.3-3-3s1.3-3 3-3c.5 0 1 .1 1.4 .4C9.1 6 9 5.5 9 5c0-1.7 1.3-3 3-3z" />
    </svg>
  )
}

export default function Navbar() {
  const [open, setOpen] = useState(false)
  return (
    <header className="fixed top-0 left-0 w-full z-50 py-5 md:py-7 bg-gradient-to-b from-[#f1f1f1]/80 to-transparent backdrop-blur-[3px]">
      <div className="grid grid-cols-12 items-center max-w-7xl mx-auto px-6 md:px-10">
        {/* brand */}
        <div className="col-span-6 md:col-span-3 flex items-center gap-2">
          <Clover />
          <Link to="/" className="font-display font-extrabold text-lg tracking-tight">
            warehouse<span className="text-zinc-400">·</span>rl
          </Link>
        </div>

        {/* center links */}
        <nav className="hidden md:flex col-span-6 justify-center gap-8">
          {LINKS.map((l) => (
            <a key={l} href={`#${l}`} className="text-sm lowercase text-zinc-500 hover:text-zinc-900 transition-colors">
              {l}
            </a>
          ))}
        </nav>

        {/* right */}
        <div className="col-span-6 md:col-span-3 flex items-center justify-end gap-3">
          <Link
            to="/simulation"
            className="hidden sm:inline-flex items-center gap-1.5 bg-ink text-white text-sm font-medium px-4 py-2 rounded-full hover:bg-black transition-colors"
          >
            run simulation <span aria-hidden>→</span>
          </Link>
          <button
            onClick={() => setOpen((v) => !v)}
            className="md:hidden flex flex-col gap-[5px] p-2"
            aria-label="menu"
          >
            <motion.span animate={{ rotate: open ? 45 : 0, y: open ? 7 : 0 }} className="w-6 h-[2px] bg-ink block" />
            <motion.span animate={{ opacity: open ? 0 : 1 }} className="w-6 h-[2px] bg-ink block" />
            <motion.span animate={{ rotate: open ? -45 : 0, y: open ? -7 : 0 }} className="w-6 h-[2px] bg-ink block" />
          </button>
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden overflow-hidden mx-6 mt-3 rounded-2xl bg-white/90 backdrop-blur border border-black/5 shadow-sm"
          >
            <div className="flex flex-col p-4 gap-3">
              {LINKS.map((l) => (
                <a key={l} href={`#${l}`} onClick={() => setOpen(false)} className="text-zinc-700 lowercase">
                  {l}
                </a>
              ))}
              <Link to="/simulation" className="bg-ink text-white text-center rounded-full py-2 mt-1">
                run simulation →
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}
