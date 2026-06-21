import { Routes, Route } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import Landing from './pages/Landing'

// Code-split the 3D sim (three.js + drei + GLB preloads) so it only loads when
// the user opens /simulation — keeps the landing page light and isolated.
const Simulation = lazy(() => import('./pages/Simulation'))

export default function App() {
  return (
    <div className="min-h-screen bg-bg-base selection:bg-brand-green selection:text-black">
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route
          path="/simulation"
          element={
            <Suspense fallback={<div className="fixed inset-0 flex items-center justify-center bg-[#0d1018] text-white/70">Loading simulation…</div>}>
              <Simulation />
            </Suspense>
          }
        />
      </Routes>
    </div>
  )
}
