import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { OrbitControls, RoundedBox, Html, useGLTF, Environment, Lightformer } from '@react-three/drei'
import * as THREE from 'three'
import { WarehouseSim, COLS, ROWS, skuColor } from './engine'

export type Sel = { type: 'shelf' | 'robot' | 'monitor'; i: number } | null

const SP = 1.5
const STEP = 0.5
type Cell = [number, number]
const wx = (c: Cell) => (c[0] - (COLS - 1) / 2) * SP
const wz = (c: Cell) => (c[1] - (ROWS - 1) / 2) * SP
const lerp = (a: number, b: number, t: number) => a + (b - a) * t
const eq = (a: Cell, b: Cell) => a[0] === b[0] && a[1] === b[1]
const k = (c: Cell) => c[0] + ',' + c[1]

function refs<T>(n: number) {
  return useMemo(() => Array.from({ length: n }, () => ({ current: null as T | null })), [n])
}

// ---- GLB models (tweak these if scale/orientation looks off) ----------------
const ROBOT_URL = '/models/robot.glb'
const PICKER_URL = '/models/picker.glb'
const ROBOT_SIZE = 1.5        // target max-dimension (world units) for the AMR
const ROBOT_FACE_ROT = 0      // rotate if the model's "front" isn't +Z (try Math.PI, ±Math.PI/2)
const PICKER_SIZE = 3.0       // target max-dimension for the picker workstation
const PICKER_ROT = Math.PI    // face the workstation toward the warehouse
const POD_URL = '/models/pod.glb'
const POD_SIZE = 1.35         // storage pod (kiva) footprint width (≈ robot width)
const WORKER_URL = '/models/worker.glb'
const WORKER_SIZE = 1.7       // person height
const WORKER_ROT = 0          // rotate if the worker's "front" isn't +Z
const RESTOCK_URL = '/models/pallet.glb'      // pallet of boxes at the stock area
const BOXING_URL = '/models/packing.glb'      // packing station at the boxing area
const SHELL_URL = '/models/shell.glb'         // the warehouse building everything sits in
const SHELL_SIZE = 26                          // footprint span (grid is ~19.5 x 16.5)
const SHELL_TINT = '#7c8493'                   // concrete grey (model ships untextured)
useGLTF.preload(ROBOT_URL)
useGLTF.preload(PICKER_URL)
useGLTF.preload(POD_URL)
useGLTF.preload(WORKER_URL)
useGLTF.preload(RESTOCK_URL)
useGLTF.preload(BOXING_URL)

// exponentially damp an angle toward a target along the shortest arc (smooth turn)
const dampAngle = (cur: number, target: number, lambda: number, dt: number) => {
  const d = Math.atan2(Math.sin(target - cur), Math.cos(target - cur))
  return cur + d * (1 - Math.exp(-lambda * dt))
}

// Loads a GLB, deep-clones it (so it can appear many times), auto-scales it to a
// target size, centers it on x/z and sits it on the floor. castShadow on.
function GLBModel({ url, target, rotY = 0, fitFootprint = false, tint, shadow = true }:
  { url: string; target: number; rotY?: number; fitFootprint?: boolean; tint?: string; shadow?: boolean }) {
  const { scene } = useGLTF(url)
  const obj = useMemo(() => {
    const c = scene.clone(true)
    const box = new THREE.Box3().setFromObject(c)
    const size = new THREE.Vector3(); box.getSize(size)
    const center = new THREE.Vector3(); box.getCenter(center)
    // fitFootprint: scale by horizontal width (good for tall shelving so its
    // footprint matches a robot); otherwise scale by the largest dimension.
    const denom = (fitFootprint ? Math.max(size.x, size.z) : Math.max(size.x, size.y, size.z)) || 1
    const s = target / denom
    c.scale.setScalar(s)
    c.position.set(-center.x * s, -box.min.y * s, -center.z * s)
    c.traverse((o: THREE.Object3D) => {
      const m = o as THREE.Mesh
      if (!m.isMesh) return
      m.castShadow = shadow; m.receiveShadow = true
      // GLB exports are often fully metallic with no env map -> they render
      // black. Tame metalness so direct lights illuminate them properly.
      const mats = Array.isArray(m.material) ? m.material : [m.material]
      mats.forEach((mat) => {
        const sm = mat as THREE.MeshStandardMaterial
        if (sm && 'metalness' in sm) {
          sm.metalness = Math.min(sm.metalness ?? 0, 0.2)
          sm.roughness = Math.max(sm.roughness ?? 1, 0.55)
          sm.envMapIntensity = 1
          // some GLBs ship untextured (default white) -> let callers tint them
          if (tint && sm.color) sm.color.set(tint)
          sm.needsUpdate = true
        }
      })
    })
    return c
  }, [scene, target, fitFootprint, tint, shadow])
  return (
    <group rotation={[0, rotY, 0]}>
      <primitive object={obj} />
    </group>
  )
}

export default function Warehouse3D({ sim, selected, onSelect }: {
  sim: WarehouseSim; selected: Sel; onSelect: (s: Sel) => void
}) {
  const robotG = refs<THREE.Group>(sim.robots.length)
  const carriedG = refs<THREE.Group>(sim.robots.length)
  const highlight = useRef<THREE.Group>(null)
  const monitorBox = useRef<THREE.Mesh>(null)
  const podG = refs<THREE.Group>(sim.shelves.length)
  const humanG = useRef<THREE.Group>(null)
  const FALLEN_POOL = 16
  const fallenM = refs<THREE.Mesh>(FALLEN_POOL)

  const podKeys = useMemo(() => sim.shelves.map((s) => k(s.cell)), [sim])
  const acc = useRef(0)
  const prev = useRef<Cell[]>(sim.robots.map((r) => [...r.cell] as Cell))
  const cur = useRef<Cell[]>(sim.robots.map((r) => [...r.cell] as Cell))
  const hPrev = useRef<Cell>([...sim.human] as Cell)
  const hCur = useRef<Cell>([...sim.human] as Cell)

  useFrame((_, dt) => {
    acc.current += Math.min(dt, 0.05)
    if (acc.current >= STEP) {
      acc.current -= STEP
      prev.current = cur.current
      hPrev.current = hCur.current
      sim.stepSim()
      cur.current = sim.robots.map((r) => [...r.cell] as Cell)
      hCur.current = [...sim.human] as Cell
    }
    const t = Math.min(1, acc.current / STEP)

    robotG.forEach((g, i) => {
      if (!g.current) return
      const p = prev.current[i], c = cur.current[i]
      const x = lerp(wx(p), wx(c), t), z = lerp(wz(p), wz(c), t)
      g.current.position.set(x, 0, z)
      const dx = wx(c) - wx(p), dz = wz(c) - wz(p)
      if (Math.abs(dx) + Math.abs(dz) > 0.0001)
        g.current.rotation.y = dampAngle(g.current.rotation.y, Math.atan2(dx, dz), 12, Math.min(dt, 0.05))
      const r = sim.robots[i]
      if (carriedG[i].current) carriedG[i].current!.visible = r.carrying
    })

    if (humanG.current) {
      const hp = hPrev.current, hc = hCur.current
      humanG.current.position.set(lerp(wx(hp), wx(hc), t), 0, lerp(wz(hp), wz(hc), t))
      const hdx = wx(hc) - wx(hp), hdz = wz(hc) - wz(hp)
      if (Math.abs(hdx) + Math.abs(hdz) > 0.0001)
        humanG.current.rotation.y = dampAngle(humanG.current.rotation.y, Math.atan2(hdx, hdz), 10, Math.min(dt, 0.05))
    }

    const lifted = sim.liftedCells()
    podG.forEach((g, i) => { if (g.current) g.current.visible = !lifted.has(podKeys[i]) })

    for (let i = 0; i < FALLEN_POOL; i++) {
      const m = fallenM[i].current
      if (!m) continue
      const f = sim.fallen[i]
      m.visible = !!f
      if (f) { m.position.set(wx(f.cell), 0.35, wz(f.cell)); (m.material as THREE.MeshStandardMaterial).color.set(skuColor(f.sku)) }
    }

    // monitor carries the boxed item between fall site -> boxing -> restock
    if (monitorBox.current) {
      const c = sim.humanCarry
      monitorBox.current.visible = !!c
      if (c) (monitorBox.current.material as THREE.MeshStandardMaterial).color.set(skuColor(c.sku))
    }

    // selection ring follows the selected rack (static) or robot (moving)
    if (highlight.current) {
      if (!selected) highlight.current.visible = false
      else {
        highlight.current.visible = true
        if (selected.type === 'robot') {
          const p = robotG[selected.i]?.current
          if (p) highlight.current.position.set(p.position.x, 0.06, p.position.z)
        } else if (selected.type === 'monitor') {
          if (humanG.current) highlight.current.position.set(humanG.current.position.x, 0.06, humanG.current.position.z)
        } else {
          const sh = sim.shelves[selected.i]
          highlight.current.position.set(wx(sh.cell), 0.06, wz(sh.cell))
        }
      }
    }
  })

  return (
    <>
      <color attach="background" args={['#0d1018']} />
      <fog attach="fog" args={['#0d1018', 30, 70]} />
      <ambientLight intensity={0.4} />
      <hemisphereLight intensity={0.4} groundColor="#1a2030" color="#eaf0ff" />
      <directionalLight position={[-8, 12, -6]} intensity={0.3} />
      <directionalLight position={[12, 18, 10]} intensity={0.9} castShadow
        shadow-mapSize={[1024, 1024]} shadow-camera-left={-24} shadow-camera-right={24}
        shadow-camera-top={24} shadow-camera-bottom={-24} />
      {/* gentle highlight on the picker workstation */}
      <pointLight position={[wx(sim.picker), 4.5, wz(sim.picker) + 1]} intensity={2.5} decay={0} color="#eafff2" />
      {/* image-based lighting (procedural, offline) carries the base illumination */}
      <Environment resolution={128} frames={1}>
        <Lightformer intensity={1.4} color="#ffffff" position={[0, 9, 0]} scale={[14, 14, 1]} rotation={[Math.PI / 2, 0, 0]} />
        <Lightformer intensity={0.8} color="#cfe0ff" position={[8, 4, 8]} scale={[7, 7, 1]} />
        <Lightformer intensity={0.7} color="#ffe9c8" position={[-8, 4, -8]} scale={[7, 7, 1]} />
      </Environment>

      {/* selection ring (follows the clicked rack or robot) */}
      <group ref={highlight} visible={false}>
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.85, 1.05, 40]} />
          <meshBasicMaterial color="#9fff00" transparent opacity={0.9} side={THREE.DoubleSide} />
        </mesh>
      </group>

      {/* the warehouse building everything sits inside */}
      <GLBModel url={SHELL_URL} target={SHELL_SIZE} fitFootprint tint={SHELL_TINT} shadow={false} />

      {/* floor + grid (grid floats just above the shell floor) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]} receiveShadow>
        <planeGeometry args={[COLS * SP + 4, ROWS * SP + 4]} />
        <meshStandardMaterial color="#171d2b" roughness={0.95} />
      </mesh>
      <gridHelper args={[Math.max(COLS, ROWS) * SP + 4, Math.max(COLS, ROWS) + 3, '#2b3650', '#222a3c']} position={[0, 0.04, 0]} />

      {/* picker = workstation GLB model */}
      <group position={[wx(sim.picker), 0, wz(sim.picker)]}>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
          <planeGeometry args={[1.6, 1.6]} />
          <meshStandardMaterial color="#46d68a" transparent opacity={0.18} />
        </mesh>
        <GLBModel url={PICKER_URL} target={PICKER_SIZE} rotY={PICKER_ROT} />
        <Html position={[0, 2.6, 0]} center distanceFactor={22}>
          <div style={{ color: '#9fffc8', font: '700 11px Inter', letterSpacing: '.08em', whiteSpace: 'nowrap' }}>PICKER</div>
        </Html>
      </group>
      <StationModel cell={sim.boxing} color="#f0b85c" label="BOXING" url={BOXING_URL} size={1.3} />
      <StationModel cell={sim.restock} color="#c08bff" label="RESTOCK" url={RESTOCK_URL} size={1.5} />

      {/* pods (click to inspect) */}
      {sim.shelves.map((sh, i) => (
        <group key={i} ref={podG[i]} position={[wx(sh.cell), 0, wz(sh.cell)]}
          onClick={(e) => { e.stopPropagation(); onSelect({ type: 'shelf', i }) }}
          onPointerOver={() => (document.body.style.cursor = 'pointer')}
          onPointerOut={() => (document.body.style.cursor = 'auto')}>
          <GLBModel url={POD_URL} target={POD_SIZE} fitFootprint />
        </group>
      ))}

      {/* robots */}
      {sim.robots.map((_, i) => (
        <group key={i} ref={robotG[i]}
          onClick={(e) => { e.stopPropagation(); onSelect({ type: 'robot', i }) }}
          onPointerOver={() => (document.body.style.cursor = 'pointer')}
          onPointerOut={() => (document.body.style.cursor = 'auto')}>
          {/* the AMR model */}
          <GLBModel url={ROBOT_URL} target={ROBOT_SIZE} rotY={ROBOT_FACE_ROT} />
          {/* carried pod (the kiva pod, sits on the deck) */}
          <group ref={carriedG[i]} position={[0, 0.35, 0]} visible={false}>
            <GLBModel url={POD_URL} target={1.2} fitFootprint />
          </group>
        </group>
      ))}

      {/* the human monitor ("the guy") — click to inspect */}
      <group ref={humanG}
        onClick={(e) => { e.stopPropagation(); onSelect({ type: 'monitor', i: 0 }) }}
        onPointerOver={() => (document.body.style.cursor = 'pointer')}
        onPointerOut={() => (document.body.style.cursor = 'auto')}>
        <GLBModel url={WORKER_URL} target={WORKER_SIZE} rotY={WORKER_ROT} tint="#f08a28" />
        {/* boxed item the monitor is carrying (fall -> box -> restock) */}
        <mesh ref={monitorBox} position={[0, 0.7, 0.35]} visible={false} castShadow>
          <boxGeometry args={[0.32, 0.32, 0.32]} />
          <meshStandardMaterial color="#f0b85c" />
        </mesh>
        <Html position={[0, 1.7, 0]} center distanceFactor={22}>
          <div style={{ color: '#cfe3ff', font: '700 11px Inter', letterSpacing: '.08em', whiteSpace: 'nowrap' }}>MONITOR</div>
        </Html>
      </group>

      {/* fallen-item pool */}
      {Array.from({ length: FALLEN_POOL }).map((_, i) => (
        <mesh key={i} ref={fallenM[i]} visible={false} rotation={[0, Math.PI / 4, 0]}>
          <octahedronGeometry args={[0.28]} />
          <meshStandardMaterial color="#ef6b6b" emissive="#5a0d0d" />
        </mesh>
      ))}

      <OrbitControls makeDefault target={[0, 0, 0]} minDistance={8} maxDistance={55} maxPolarAngle={Math.PI / 2.15} />
    </>
  )
}

function StationModel({ cell, color, label, url, size }:
  { cell: Cell; color: string; label: string; url: string; size: number }) {
  return (
    <group position={[wx(cell), 0, wz(cell)]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
        <planeGeometry args={[1.5, 1.5]} />
        <meshStandardMaterial color={color} transparent opacity={0.18} />
      </mesh>
      <GLBModel url={url} target={size} fitFootprint />
      <Html position={[0, 1.6, 0]} center distanceFactor={22}>
        <div style={{ color, font: '700 11px Inter', letterSpacing: '.08em', whiteSpace: 'nowrap' }}>{label}</div>
      </Html>
    </group>
  )
}

function Station({ cell, color, label }: { cell: Cell; color: string; label: string }) {
  return (
    <group position={[wx(cell), 0, wz(cell)]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
        <planeGeometry args={[1.4, 1.4]} />
        <meshStandardMaterial color={color} transparent opacity={0.22} />
      </mesh>
      <Html position={[0, 0.4, 0.9]} center distanceFactor={22}>
        <div style={{ color, font: '700 11px Inter', letterSpacing: '.08em', whiteSpace: 'nowrap' }}>{label}</div>
      </Html>
    </group>
  )
}
