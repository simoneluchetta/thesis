/**
 * The reconstructed arena: the fused Depth Anything 3 world, the temporal 4D world and the
 * NeRF floor cloud. Everything is read from public/nerf/, which the pipeline fills.
 */
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import DensePly from "../components/DensePly";
import TemporalPly from "../components/TemporalPly";

// The clouds are PLY files written by the pipeline.
// https://en.wikipedia.org/wiki/PLY_(file_format)
interface CloudInfo { file: string; points: number; source: string; point_size: number; }
/** One instant of the 4D world: its frame number, PLY file and point count. */
interface Frame4D { t: number; file: string; points: number; }
/** public/nerf/da3_4d/index.json: the 4D frame list and its fps. */
interface Index4D { fps: number; frames: Frame4D[]; }
/** public/nerf/manifest.json, written by arenaDA3/export_webapp_assets.py. */
interface Manifest {
  cloud: CloudInfo | null;
  da3_world?: CloudInfo | null;
}

/** The NeRF tab: the reconstructed arena as point clouds, with a 4D timeline. */
export default function NeRFResults() {
  const [man, setMan] = useState<Manifest | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [pointSize, setPointSize] = useState(0.03);
  // TSDF-fused DA3 world. When present, the floor-only NeRF cloud starts hidden.
  // https://en.wikipedia.org/wiki/Signed_distance_function
  const [showWorld, setShowWorld] = useState(true);
  const [showFloorCloud, setShowFloorCloud] = useState(true);
  // Temporal 4D world under /nerf/da3_4d/; turning it on hides the static world.
  const [idx4d, setIdx4d] = useState<Index4D | null>(null);
  const [show4D, setShow4D] = useState(false);
  const [i4d, setI4d] = useState(0);
  const [shown4d, setShown4d] = useState(-1);
  const [playing4d, setPlaying4d] = useState(false);
  const [speed4d, setSpeed4d] = useState(1);
  const worldBefore4D = useRef(true);

  // https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
  useEffect(function () {
    fetch("/nerf/manifest.json")
      .then(function (r) { return (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))); })
      .then(function (d: Manifest) {
        setMan(d);
        if (d.cloud) setPointSize(d.cloud.point_size);
        if (d.da3_world) { setPointSize(d.da3_world.point_size); setShowFloorCloud(false); }
      })
      .catch(function (e: Error) { return setErr(e.message); });
  }, []);

  // 4D manifest. Re-fetched on each enable: fusion appends frames while the page is open.
  // https://react.dev/reference/react/useCallback
  const load4D = useCallback(function () {
    fetch("/nerf/da3_4d/index.json")
      .then(function (r) { return (r.ok ? r.json() : null); })
      .then(function (d: Index4D | null) { if (d?.frames?.length) setIdx4d(d); })
      .catch(function () {});
  }, []);
  useEffect(function () { load4D(); }, [load4D]);
  useEffect(function () { if (show4D) load4D(); }, [show4D, load4D]);

  const frames4d = idx4d?.frames ?? [];
  const i4c = Math.max(0, Math.min(i4d, frames4d.length - 1));
  const f4d = frames4d[i4c];
  const urls4d = useMemo(function () {
    return frames4d.map(function (f) { return `/nerf/${f.file}`; });
  }, [frames4d]);
  const fps4d = idx4d?.fps ?? 25;
  // Real-time hold per step (timestamps are frame numbers of the 25 fps clip).
  const hold4dMs = frames4d.length > 1
    ? Math.max(30, Math.round(((frames4d[1].t - frames4d[0].t) / fps4d) * 1000)) : 200;
  // Playback is a plain interval that steps the index and wraps around.
  // https://developer.mozilla.org/en-US/docs/Web/API/Window/setInterval
  useEffect(function () {
    if (!playing4d || !show4D || frames4d.length < 2) return;
    const h = window.setInterval(
      function () {
        return setI4d(function (c) { return (c + 1) % frames4d.length; });
      },
      Math.max(30, Math.round(hold4dMs / speed4d)),
    );
    return function () { return window.clearInterval(h); };
  }, [playing4d, show4D, frames4d.length, hold4dMs, speed4d]);

  function toggle4D(on: boolean) {
    setShow4D(on);
    if (on) {
      worldBefore4D.current = showWorld;
      setShowWorld(false);
    } else {
      setShowWorld(worldBefore4D.current);
      setPlaying4d(false);
    }
  }

  const cloudUrl = useMemo(
    function () { return (man?.cloud ? `/nerf/${man.cloud.file}` : null); },
    [man],
  );
  const worldUrl = useMemo(
    function () { return (man?.da3_world ? `/nerf/${man.da3_world.file}` : null); },
    [man],
  );

  if (err) {
    return (
      <div style={{ padding: 24, color: "#ccc" }}>
        <h2>NeRF results unavailable</h2>
        <p style={{ color: "#f88" }}>Could not load <code>/nerf/manifest.json</code> ({err}).</p>
        <p>
          The files are made by the pipeline. From the project folder:
        </p>
        <pre style={pre}>
{`uv run pipeline.py`}
        </pre>
      </div>
    );
  }
  if (!man) return <div style={{ padding: 24, color: "#ccc" }}>Loading NeRF results…</div>;

  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", background: "#1f1f1f" }}>
      {/* Main: interactive NeRF point cloud */}
      <div style={{ flex: 1, minWidth: 0, position: "relative" }}>
        {/* up: [0,0,1] because the world is Z-up, not three.js' default Y-up.
            https://docs.pmnd.rs/react-three-fiber/api/canvas
            https://github.com/pmndrs/react-three-fiber */}
        {cloudUrl || worldUrl || idx4d ? (
          <Canvas camera={{ position: [26, -26, 16], up: [0, 0, 1], fov: 50, near: 0.05, far: 5000 }}>
            <color attach="background" args={["#141414"]} />
            <ambientLight intensity={0.6} />
            <directionalLight position={[20, 20, 40]} intensity={0.7} />
            {/* Suspense keeps the canvas alive while a PLY is still loading.
                https://react.dev/reference/react/Suspense */}
            {cloudUrl && showFloorCloud && (
              <Suspense fallback={null}>
                <DensePly url={cloudUrl} pointSize={pointSize} />
              </Suspense>
            )}
            {worldUrl && showWorld && (
              <Suspense fallback={null}>
                <DensePly url={worldUrl} pointSize={pointSize} />
              </Suspense>
            )}
            {/* TemporalPly streams the frames through a bounded cache. */}
            {show4D && urls4d.length > 0 && (
              <TemporalPly urls={urls4d} index={i4c} pointSize={pointSize} onShownIndex={setShown4d} />
            )}
            {/* GridHelper sits on the XZ plane, so rotate a quarter turn onto the Z-up floor.
                https://threejs.org/docs/#api/en/helpers/GridHelper
                https://threejs.org/docs/#api/en/helpers/AxesHelper */}
            <gridHelper args={[60, 30, "#333", "#262626"]} rotation={[Math.PI / 2, 0, 0]} />
            <axesHelper args={[3]} />
            {/* drei's OrbitControls; makeDefault lets other drei helpers find them.
                https://github.com/pmndrs/drei */}
            <OrbitControls makeDefault target={[0, 0, 2]} />
          </Canvas>
        ) : (
          <div style={{ padding: 24, color: "#ccc" }}>
            No 3D results yet (run <code>uv run pipeline.py</code>).
          </div>
        )}
        {/* overlay controls */}
        <div style={overlay}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Arena 3D layers</div>
          {/* toLocaleString puts thousands separators in the point counts.
              https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/toLocaleString */}
          {man.da3_world && (
            <div style={{ color: "#9ab", marginBottom: 6 }}>
              {man.da3_world.points.toLocaleString()} pts · DA3 fused world
            </div>
          )}
          {man.cloud && !man.da3_world && (
            <div style={{ color: "#9ab", marginBottom: 6 }}>
              {man.cloud.points.toLocaleString()} pts · {man.cloud.source}
            </div>
          )}
          <label style={{ display: "block", color: "#bbb" }}>
            point size {pointSize.toFixed(3)}
            <input
              type="range" min={0.005} max={0.12} step={0.005} value={pointSize}
              onChange={function (e) { setPointSize(parseFloat(e.target.value)); }}
              style={{ width: "100%" }}
            />
          </label>
          {man.da3_world && (
            <label style={{ display: "block", color: "#bbb", marginTop: 6 }}>
              <input type="checkbox" checked={showWorld} onChange={function (e) {
                setShowWorld(e.target.checked);
              }} />
              {" "}DA3 fused world (TSDF)
            </label>
          )}
          {man.cloud && (
            <label style={{ display: "block", color: "#bbb", marginTop: 4 }}>
              <input type="checkbox" checked={showFloorCloud} onChange={function (e) {
                setShowFloorCloud(e.target.checked);
              }} />
              {" "}NeRF floor cloud ({man.cloud.source})
            </label>
          )}
          {idx4d && (
            <div style={{ borderTop: "1px solid #333", marginTop: 8, paddingTop: 6 }}>
              <div style={{ fontWeight: 600, marginBottom: 2 }}>Temporal world · 4D</div>
              <label style={{ display: "block", color: "#bbb" }}>
                <input type="checkbox" checked={show4D} onChange={function (e) { toggle4D(e.target.checked); }} />
                {" "}sliding DA3 world (per instant)
              </label>
              {show4D && (
                <div style={{ color: "#9ab", marginTop: 3 }}>
                  {frames4d.length} instants · game clip · fused per frame
                </div>
              )}
            </div>
          )}
          <div style={{ color: "#777", marginTop: 4 }}>drag = orbit · scroll = zoom · right-drag = pan</div>
        </div>

        {/* Timeline bar for the 4D sliding world */}
        {show4D && f4d && (
          <div
            style={timelineBar}
            title={"One TSDF-fused DA3 world per instant of the game clip — "
              + "the reconstruction itself stepped through time."}
          >
            <button
              onClick={function () {
                setPlaying4d(function (v) { return !v; });
              }}
              disabled={frames4d.length < 2}
              style={{
                background: playing4d ? "#fc6" : "#252526", color: playing4d ? "#1f1f1f" : "#ccc",
                border: "1px solid #3c3c3c", borderRadius: 3, padding: "3px 12px",
                cursor: frames4d.length < 2 ? "default" : "pointer", fontSize: 12, fontWeight: 600, minWidth: 62,
              }}
            >
              {playing4d ? "❚❚ Pause" : "▶ Play"}
            </button>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }} title="Playback speed multiplier">
              <span style={{ fontSize: 11, color: "#9d9d9d" }}>Speed</span>
              <input
                type="range" min={0.25} max={8} step={0.25} value={speed4d}
                onChange={function (e) { setSpeed4d(parseFloat(e.target.value)); }} style={{ width: 80 }}
              />
              <span style={{ fontSize: 12, color: "#fc6", minWidth: 34, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                {speed4d}×
              </span>
            </span>
            <button onClick={function () { setI4d(Math.max(0, i4c - 1)); }} style={stepBtn} title="Previous instant">‹</button>
            <input
              type="range" min={0} max={frames4d.length - 1} step={1} value={i4c}
              onChange={function (e) { setI4d(parseInt(e.target.value, 10)); }} style={{ flex: 1 }}
            />
            <button onClick={function () { setI4d(Math.min(frames4d.length - 1, i4c + 1)); }} style={stepBtn} title="Next instant">›</button>
            {/* tabular-nums keeps the readout from jittering as the digits change.
                https://developer.mozilla.org/en-US/docs/Web/CSS/font-variant-numeric
                https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/padStart */}
            <span style={{ fontSize: 12, color: "#fc6", minWidth: 210, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
              {shown4d !== i4c ? "⏳ " : ""}{i4c + 1} / {frames4d.length} · t{String(f4d.t).padStart(6, "0")} · {(f4d.t / fps4d).toFixed(1)} s · {f4d.points.toLocaleString()} pts
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

const pre: React.CSSProperties = {
  background: "#111", padding: 12, borderRadius: 4, color: "#9c9", fontSize: 12, overflow: "auto",
};
const overlay: React.CSSProperties = {
  position: "absolute", top: 10, left: 10, background: "rgba(20,20,20,0.82)",
  border: "1px solid #333", borderRadius: 5, padding: "8px 10px", fontSize: 12, color: "#ccc", width: 200,
};
const timelineBar: React.CSSProperties = {
  position: "absolute", left: 10, right: 10, bottom: 10, display: "flex", alignItems: "center",
  gap: 10, background: "rgba(20,20,20,0.82)", border: "1px solid #333", borderRadius: 5,
  padding: "6px 10px", fontSize: 12, color: "#ccc",
};
const stepBtn: React.CSSProperties = {
  background: "#252526", color: "#ccc", border: "1px solid #3c3c3c", borderRadius: 3,
  padding: "2px 8px", cursor: "pointer", fontSize: 12,
};
