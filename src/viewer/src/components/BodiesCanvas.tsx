import { useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";
import type { CourtPolyline } from "../utils/courtModel";

/**
 * The 3-D mode of the People map: fitted SMPL-X bodies with their skeletons.
 * Data from people/3D_People/export_bodies.py, in placer-world (Z up, metres).
 */
// https://smpl-x.is.tue.mpg.de/
// https://docs.pmnd.rs/react-three-fiber/api/canvas
// https://threejs.org/docs/#api/en/core/BufferGeometry
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Float32Array

/** One body in one frame: which track it belongs to and where its joints are. */
interface BodyFrameObs {
  track_id: number;
  rec: number;
  joints: ([number, number, number] | null)[] | null;
  /** Root from the roster trajectory, pose slerped between the fitted frames.
   *  https://en.wikipedia.org/wiki/Slerp */
  interp?: boolean;
}
/** bodies.json: how to read the vertex, face and colour files that go with it. */
interface BodiesManifest {
  n_verts: number;
  /** Floats per record in verts.bin. */
  stride: number;
  verts_url: string;
  faces_url: string;
  color_url?: string | null;     // per-vertex uint8 RGB, null if flat-coloured
  has_color?: boolean;
  n_records: number;
  /** Roster-locked exports only: track id -> athlete label ("A11"). */
  slots?: Record<string, string> | null;
  frames: Record<string, BodyFrameObs[]>;
  bones: [number, number][];
}
/** The manifest plus the buffers it points at, once they are loaded. */
interface BodiesData {
  m: BodiesManifest;
  verts: Float32Array;
  colors: Uint8Array | null;
  index: THREE.BufferAttribute;
}

// Track colours. utils/palette.ts has 16, so ids over 7 differ from the 2D map.
const PALETTE = [
  "#ff7b00", "#00c2ff", "#ff3df0", "#7CFF45",
  "#ffd400", "#ff4545", "#9b8cff", "#00ffa3",
];

/** Fetches bodies.json and the binary files it names. */
async function loadBodies(): Promise<BodiesData> {
  const m = (await fetch("/bodies.json").then(function (r) {
    if (!r.ok) throw new Error("no /bodies.json");
    return r.json();
  })) as BodiesManifest;
  const [vbuf, fbuf] = await Promise.all([
    fetch(`/${m.verts_url}`).then(function (r) { return r.arrayBuffer(); }),
    fetch(`/${m.faces_url}`).then(function (r) { return r.arrayBuffer(); }),
  ]);
  const verts = new Float32Array(vbuf);
  // The face file is uint32, one index per item.
  // https://threejs.org/docs/#api/en/core/BufferAttribute
  const index = new THREE.BufferAttribute(new Uint32Array(fbuf), 1);
  let colors: Uint8Array | null = null;
  if (m.has_color && m.color_url) {
    const cbuf = await fetch(`/${m.color_url}`).then(function (r) { return r.arrayBuffer(); });
    colors = new Uint8Array(cbuf);
  }
  return { m, verts, colors, index };
}

/** One geometry per record: its own vertex slice, the shared face index. */
function useBodyGeometries(data: BodiesData | null): THREE.BufferGeometry[] {
  return useMemo(function () {
    if (!data) return [];
    const { m, verts, colors, index } = data;
    const cstride = m.n_verts * 3;
    const out: THREE.BufferGeometry[] = [];
    for (let r = 0; r < m.n_records; r++) {
      const g = new THREE.BufferGeometry();
      const slice = verts.slice(r * m.stride, (r + 1) * m.stride);
      g.setAttribute("position", new THREE.BufferAttribute(slice, 3));
      if (colors) {
        const cslice = colors.slice(r * cstride, (r + 1) * cstride);
        // The `true` is the normalized flag: uint8 0..255 is read as 0..1.
        g.setAttribute("color", new THREE.BufferAttribute(cslice, 3, true));
      }
      // https://threejs.org/docs/#api/en/core/BufferGeometry.setIndex
      g.setIndex(index); // one GPU index buffer shared by every body
      g.computeVertexNormals();
      out.push(g);
    }
    return out;
  }, [data]);
}

/** Bone segments and joint dots for one body, drawn on top of its mesh. */
function Skeleton({ joints, bones }: { joints: BodyFrameObs["joints"]; bones: [number, number][] }) {
  const segPos = useMemo(function () {
    if (!joints) return new Float32Array(0);
    const seg: number[] = [];
    for (const [a, b] of bones) {
      const ja = joints[a], jb = joints[b];
      if (ja && jb) seg.push(ja[0], ja[1], ja[2], jb[0], jb[1], jb[2]);
    }
    return new Float32Array(seg);
  }, [joints, bones]);
  if (!joints) return null;
  return (
    <group>
      {segPos.length > 0 && (
        <lineSegments renderOrder={10}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[segPos, 3]} />
          </bufferGeometry>
          {/* depthTest off so the skeleton shows through the mesh.
              https://threejs.org/docs/#api/en/materials/Material.depthTest */}
          <lineBasicMaterial color="#ffffff" depthTest={false} transparent opacity={0.95} />
        </lineSegments>
      )}
      {joints.map(function (j, k) {
        return j ? (
          <mesh key={k} position={j} renderOrder={11}>
            <sphereGeometry args={[0.028, 8, 8]} />
            <meshBasicMaterial color="#ffe24a" depthTest={false} transparent />
          </mesh>
        ) : null;
      },
      )}
    </group>
  );
}

/** One fitted body: the mesh, plus its skeleton when that is switched on. */
function Body({
  geom, joints, color, bones, textured, showSkeleton,
}: {
  geom: THREE.BufferGeometry; joints: BodyFrameObs["joints"];
  color: string; bones: [number, number][]; textured: boolean; showSkeleton: boolean;
}) {
  return (
    <group>
      <mesh geometry={geom}>
        {textured ? (
          // per-vertex colour sampled from the source frames
          <meshStandardMaterial vertexColors roughness={0.9} metalness={0.0} side={THREE.DoubleSide} />
        ) : (
          // flat per-track colour, translucent so the skeleton reads through
          <meshStandardMaterial color={color} transparent opacity={0.62} roughness={0.85}
            metalness={0.0} side={THREE.DoubleSide} />
        )}
      </mesh>
      {showSkeleton && <Skeleton joints={joints} bones={bones} />}
    </group>
  );
}

/** Floor slab, court lines and hoop rings, in world coords at z=0 (rim height for hoops). */
function CourtFloor({ court, hoops }: { court: CourtPolyline[]; hoops: Hoop[] }) {
  return (
    <group>
      {/* floor slab */}
      <mesh position={[0, 0, -0.005]}>
        <planeGeometry args={[32, 18]} />
        <meshStandardMaterial color="#16301c" roughness={1} metalness={0} />
      </mesh>
      {court.map(function (pl, k) {
        return (
          <Line
            key={k}
            points={pl.pts.map(function ([x, y]) { return [x, y, 0.01] as [number, number, number]; })}
            color={pl.court === "basketball" ? "#e8e8e8" : "#55aaff"}
            lineWidth={1.4}
            transparent
            opacity={0.85}
          />
        );
      })}
      {hoops.map(function (h) {
        return (
          <mesh key={h.id} position={[h.x, h.y, h.z]}>
            <torusGeometry args={[0.23, 0.02, 8, 28]} />
            <meshStandardMaterial color="#ffae42" emissive="#7a4d10" />
          </mesh>
        );
      })}
    </group>
  );
}

/** One basket ring in world metres, z being the rim height. */
export interface Hoop { id: string; x: number; y: number; z: number; }

/** The People map in 3-D: the court plus every fitted body at frame `t`. */
export default function BodiesCanvas({
  court, hoops, t,
}: {
  court: CourtPolyline[]; hoops: Hoop[]; t: number | undefined;
}) {
  const [data, setData] = useState<BodiesData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [texOn, setTexOn] = useState(true);
  const [skelOn, setSkelOn] = useState(false);
  useEffect(function () {
    let live = true;
    loadBodies().then(function (d) {
      if (!live) return;
      setData(d);
      setTexOn(!!d.colors);          // texture on when the export has colours
      setSkelOn(!d.colors);          // skeleton on only when it does not
    }).catch(function (e) { if (live) setErr(String(e)); });
    return function () { live = false; };
  }, []);
  const geoms = useBodyGeometries(data);

  if (err) {
    return (
      <div style={{ padding: 24, color: "#cccccc", maxWidth: 640 }}>
        <h3 style={{ color: "#fc6" }}>No 3D bodies yet</h3>
        <p style={{ color: "#9d9d9d", lineHeight: 1.6 }}>
          Couldn't load <code>/bodies.json</code> ({err}). The bodies are made by the pipeline.
          From the project folder:
        </p>
        <pre style={{ background: "#181818", border: "1px solid #2b2b2b", borderRadius: 4, padding: 12, color: "#bdbdbd", fontSize: 12 }}>
{`uv run pipeline.py`}
        </pre>
      </div>
    );
  }
  if (!data) {
    return <div style={{ padding: 24, color: "#9d9d9d" }}>Loading SMPL-X bodies…</div>;
  }

  const people = (t !== undefined ? data.m.frames[String(t)] : undefined) ?? [];
  const nInterp = people.reduce(function (n, p) { return n + (p.interp ? 1 : 0); }, 0);
  const textured = texOn && !!data.colors;
  // the sampled colours are dim, so lift the ambient in texture mode
  const amb = textured ? 1.15 : 0.7;

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <Canvas
        camera={{ position: [11, 9, 16], fov: 50, near: 0.05, far: 500 }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={["#1a1a1a"]} />
        <ambientLight intensity={amb} />
        <directionalLight position={[8, 18, 12]} intensity={textured ? 0.6 : 0.9} />
        <directionalLight position={[-10, 6, -8]} intensity={textured ? 0.3 : 0.35} />
        {/* https://threejs.org/docs/#examples/en/controls/OrbitControls */}
        <OrbitControls makeDefault target={[0, 1, 0]} enableDamping dampingFactor={0.08} />

        {/* world Z-up to three.js Y-up */}
        <group rotation={[-Math.PI / 2, 0, 0]}>
          <CourtFloor court={court} hoops={hoops} />
          {people.map(function (p) {
            return (
              <Body
                key={`${p.track_id}-${p.rec}`}
                geom={geoms[p.rec]}
                joints={p.joints}
                color={PALETTE[p.track_id % PALETTE.length]}
                bones={data.m.bones}
                textured={textured}
                showSkeleton={skelOn}
              />
            );
          })}
        </group>
      </Canvas>

      {/* render controls */}
      <div style={{
        position: "absolute", top: 10, left: 10, display: "flex", gap: 12,
        background: "rgba(20,20,20,0.72)", border: "1px solid #3c3c3c", borderRadius: 5,
        padding: "5px 10px", fontSize: 12, color: "#ddd", userSelect: "none",
      }}>
        {data.m.has_color && (
          <label style={{ display: "flex", alignItems: "center", gap: 5, cursor: "pointer" }}>
            <input type="checkbox" checked={texOn} onChange={function (e) { setTexOn(e.target.checked); }} />
            Photo texture
          </label>
        )}
        <label style={{ display: "flex", alignItems: "center", gap: 5, cursor: "pointer" }}>
          <input type="checkbox" checked={skelOn} onChange={function (e) { setSkelOn(e.target.checked); }} />
          Skeleton
        </label>
      </div>

      {/* Roster legend, roster-locked exports only. */}
      {data.m.slots && (
        <div style={{
          position: "absolute", top: 10, right: 10, background: "rgba(20,20,20,0.78)",
          border: "1px solid #3c3c3c", borderRadius: 5, padding: "6px 9px",
          fontSize: 11, color: "#ddd", userSelect: "none", lineHeight: 1.5,
          maxHeight: "calc(100% - 20px)", overflowY: "auto",
        }}>
          <div style={{ color: "#9a9a9a", marginBottom: 3 }}>
            {people.length} athletes
            {nInterp > 0 && <> · <span style={{ color: "#ffd400" }}>{nInterp} interpolated</span></>}
          </div>
          {people.map(function (p) {
            const label = data.m.slots?.[String(p.track_id)] ?? `track ${p.track_id}`;
            return (
              <div key={p.track_id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{
                  width: 9, height: 9, borderRadius: 2, flex: "0 0 auto",
                  background: PALETTE[p.track_id % PALETTE.length],
                }} />
                <span style={{ color: p.interp ? "#9a9a9a" : "#eee" }}>{label}</span>
                {p.interp && <span style={{ color: "#ffd400", fontSize: 10 }}>interp</span>}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
