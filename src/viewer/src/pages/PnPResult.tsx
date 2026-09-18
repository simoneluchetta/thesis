/**
 * Compares the hand-placed cameras with the ones solved from the court: both sets of
 * frustums in one scene, plus a per-camera table of how far apart they are.
 */
import { useMemo, useState } from "react";
import type { SceneJson, SceneCameraEntry, SceneMeta } from "../types";
import SceneCanvas from "../components/SceneCanvas";
import { useViewer } from "../state/store";

interface Props {
  scene: SceneJson;
}

/** Per-cam comparison row data. */
interface Row {
  cam: SceneCameraEntry;
  hasPnp: boolean;
  posDriftM: number | null;
  rotDriftDeg: number | null;
}

// Angle in degrees between two 3x3 rotations.
function rotAngleDeg(a: number[][], b: number[][]): number {
  // Angle between two rotations: acos((tr(A^T B) - 1)/2).
  // https://en.wikipedia.org/wiki/Rotation_matrix
  // https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
  let trace = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      trace += a[j][i] * b[j][i]; // A^T B = sum a[j][i] * b[j][i]
    }
  }
  // Clamp: rounding can push the ratio outside [-1, 1] and give NaN.
  const cosT = Math.max(-1, Math.min(1, (trace - 1) / 2));
  return (Math.acos(cosT) * 180) / Math.PI;
}

// Straight-line distance between two 3-vectors.
function vecDist(a: number[], b: number[]): number {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** The PnP result tab: solved cameras against hand-placed ones, with the drift per camera. */
export default function PnPResult({ scene }: Props) {
  const [showManual, setShowManual] = useState(true);
  const [showPnp, setShowPnp] = useState(true);
  // The selection is shared with the 3D canvas, so it lives in the zustand store.
  // https://zustand.docs.pmnd.rs/
  const selectedCamId = useViewer(function (s) { return s.selectedCamId; });
  const setSelectedCam = useViewer(function (s) { return s.setSelectedCam; });

  // Pose solved from the court keypoints vs the pose placed by hand.
  // https://en.wikipedia.org/wiki/Perspective-n-Point
  const rows: Row[] = useMemo(function () {
    return scene.cameras.map(function (cam) {
      const hasPnp = !!(cam.pnp_position && cam.pnp_rotation_c2w);
      const posDriftM = hasPnp ? vecDist(cam.pnp_position!, cam.position) : null;
      const rotDriftDeg = hasPnp
        ? rotAngleDeg(cam.pnp_rotation_c2w!, cam.rotation_c2w)
        : null;
      return { cam, hasPnp, posDriftM, rotDriftDeg };
    });
  }, [scene.cameras]);

  const nPnp = rows.filter(function (r) { return r.hasPnp; }).length;
  const meanPos = rows.filter(function (r) { return r.posDriftM != null; });
  const meanPosVal = meanPos.length
    ? meanPos.reduce(function (s, r) { return s + r.posDriftM!; }, 0) / meanPos.length
    : null;
  const meanRotVal = meanPos.length
    ? meanPos.reduce(function (s, r) { return s + r.rotDriftDeg!; }, 0) / meanPos.length
    : null;

  if (nPnp === 0) {
    return (
      <div style={{ padding: 20, color: "#cccccc" }}>
        <h2>PnP Result</h2>
        <p style={{ color: "#f88" }}>No PnP poses found in scene.json.</p>
        <p style={{ fontSize: 13, color: "#9d9d9d" }}>
          Add a set of poses to scene.json, e.g.:
        </p>
        <pre style={{
          background: "#181818", padding: 10, fontSize: 12, color: "#cccccc",
          border: "1px solid #2b2b2b", borderRadius: 3,
        }}>
{`cd viewer
uv --project .. run python update_pnp_overlay.py \\
  --placer ../../placements/placed_cameras_redfox_repaired.json`}
        </pre>
        <p style={{ fontSize: 13, color: "#9d9d9d", marginTop: 14 }}>
          Then refresh this page.
        </p>
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: "100%", display: "flex" }}>
      {/* Left panel: per-cam delta table */}
      <div style={{
        width: 360, background: "#181818", borderRight: "1px solid #2b2b2b",
        color: "#cccccc", overflowY: "auto", padding: "12px 10px",
      }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
          PnP vs manual placement
        </div>

        <div style={{
          display: "flex", gap: 8, marginBottom: 12, fontSize: 12,
        }}>
          <ToggleChip
            label="manual"
            color="#5fff8a"
            on={showManual}
            onClick={function () {
              setShowManual(function (v) { return !v; });
            }}
          />
          <ToggleChip
            label="PnP"
            color="#ff45d6"
            on={showPnp}
            onClick={function () {
              setShowPnp(function (v) { return !v; });
            }}
          />
        </div>

        <div style={{
          fontSize: 11, color: "#9d9d9d", marginBottom: 10,
          padding: "6px 8px", background: "#222", borderRadius: 3,
        }}>
          {nPnp}/{scene.cameras.length} cams have PnP poses.
          {meanPosVal != null && (
            <> Mean drift: {meanPosVal.toFixed(2)} m, {meanRotVal!.toFixed(1)}°.</>
          )}
        </div>

        <PnpSourceCard meta={scene.meta} />

        <table style={{
          width: "100%", fontSize: 11, borderCollapse: "collapse",
        }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #2b2b2b" }}>
              <th style={{ textAlign: "left", padding: "4px 4px" }}>cam</th>
              <th style={{ textAlign: "right", padding: "4px 4px" }}>Δpos (m)</th>
              <th style={{ textAlign: "right", padding: "4px 4px" }}>Δrot (°)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(function (r) {
              const isSel = selectedCamId === r.cam.id;
              const driftBad = r.posDriftM != null && r.posDriftM > 5;
              const rotBad = r.rotDriftDeg != null && r.rotDriftDeg > 15;
              return (
                <tr
                  key={r.cam.id}
                  onClick={function () { setSelectedCam(isSel ? null : r.cam.id); }}
                  style={{
                    cursor: "pointer",
                    background: isSel ? "#3a3322" : "transparent",
                    color: r.hasPnp ? "#cccccc" : "#666",
                  }}
                >
                  <td style={{ padding: "3px 4px" }}>
                    {r.cam.name}{r.cam.is_stereo ? " ⌬" : ""}
                  </td>
                  <td style={{
                    padding: "3px 4px", textAlign: "right",
                    color: driftBad ? "#ff8866" : undefined,
                  }}>
                    {r.posDriftM != null ? r.posDriftM.toFixed(2) : "—"}
                  </td>
                  <td style={{
                    padding: "3px 4px", textAlign: "right",
                    color: rotBad ? "#ff8866" : undefined,
                  }}>
                    {r.rotDriftDeg != null ? r.rotDriftDeg.toFixed(1) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <div style={{
          marginTop: 14, fontSize: 11, color: "#888", lineHeight: 1.5,
        }}>
          Green frustums = your manual placement<br />
          Magenta frustums = PnP-computed pose<br />
          Click a row to focus that cam in 3D.
        </div>
      </div>

      {/* Right: 3D canvas with both layers overlaid */}
      <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
        <SceneCanvas
          scene={scene}
          showManualCams={showManual}
          showPnpOverlay={showPnp}
          hidePoints
        />
      </div>
    </div>
  );
}

/** Which placed_cameras*.json update_pnp_overlay.py folded in. */
function PnpSourceCard({ meta }: { meta: SceneMeta | undefined }) {
  if (!meta) {
    return (
      <div style={{
        fontSize: 11, color: "#9d9d9d", marginBottom: 10,
        padding: "6px 8px", background: "#1f1a14",
        border: "1px solid #3a2e1a", borderRadius: 3,
      }}>
        <b style={{ color: "#ffc04f" }}>Source unknown.</b> This scene.json does not record
        which placement file the cameras came from.
      </div>
    );
  }
  const src = meta.pnp_overlay_source;
  if (!src) {
    return (
      <div style={{
        fontSize: 11, color: "#9d9d9d", marginBottom: 10,
        padding: "6px 8px", background: "#222", borderRadius: 3,
      }}>
        No PnP overlay in scene.json. Run <code>update_pnp_overlay.py</code> to add one.
      </div>
    );
  }
  const fileName = src.path.replace(/\\/g, "/").split("/").pop() ?? src.path;
  return (
    <div style={{
      fontSize: 11, color: "#cccccc", marginBottom: 10,
      padding: "6px 8px", background: "#1a2a1f",
      border: "1px solid #2a4a35", borderRadius: 3, lineHeight: 1.5,
    }}>
      <div style={{ color: "#9d9d9d", fontSize: 10, textTransform: "uppercase",
        letterSpacing: "0.06em", marginBottom: 3 }}>
        PnP source
      </div>
      <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        color: "#cfe9d6", wordBreak: "break-all" }}>
        {fileName}
      </div>
      <div style={{ color: "#9d9d9d", marginTop: 3 }}>
        file mtime: {src.mtime}
      </div>
      <div style={{ color: "#9d9d9d" }}>
        baked: {meta.baked_at}
      </div>
      <div style={{ color: "#6d6d6d", marginTop: 4, fontSize: 10,
        wordBreak: "break-all" }}>
        {src.path}
      </div>
    </div>
  );
}

interface ToggleChipProps {
  label: string;
  color: string;
  on: boolean;
  onClick: () => void;
}
/** A small on/off button that takes its colour from the layer it toggles. */
function ToggleChip({ label, color, on, onClick }: ToggleChipProps) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        background: on ? color : "#252526",
        color: on ? "#1f1f1f" : "#cccccc",
        border: `1px solid ${on ? color : "#3c3c3c"}`,
        padding: "4px 8px", borderRadius: 3, fontSize: 11,
        fontWeight: on ? 600 : 400, cursor: "pointer",
      }}
    >
      {label}
    </button>
  );
}
