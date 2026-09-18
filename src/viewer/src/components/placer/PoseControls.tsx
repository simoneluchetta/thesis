/**
 * The placer's controls: joysticks and number boxes for the selected camera's pose.
 */
// https://docs.pmnd.rs/zustand
// https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/number
import { useMemo } from "react";
import type { SceneCameraEntry } from "../../types";
import { usePlacer } from "../../state/placerStore";
import XYPad from "./widgets/XYPad";
import ZPad from "./widgets/ZPad";
import HPad from "./widgets/HPad";
import RotationReference from "./widgets/RotationReference";

interface Props {
  cameras: SceneCameraEntry[];
}

// Radians to degrees, for the number boxes.
const DEG = 180 / Math.PI;
// Degrees back to radians, which is what the store holds.
const RAD = Math.PI / 180;

// Sensitivities at full stick deflection.
const SPEED_XY = 3; // m/s
const SPEED_Z = 1;  // m/s
const SPEED_YAW = 60 * RAD;   // 60 deg/s in rad/s
const SPEED_PITCH = 45 * RAD; // 45 deg/s
const SPEED_ROLL = 45 * RAD;  // 45 deg/s

/** Spring-centred joysticks, plus number boxes for the same six degrees of freedom. */
export default function PoseControls({ cameras }: Props) {
  const selected = usePlacer(function (s) { return s.selected; });
  const poses = usePlacer(function (s) { return s.poses; });
  const setPosition = usePlacer(function (s) { return s.setPosition; });
  const setPositionXY = usePlacer(function (s) { return s.setPositionXY; });
  const setRotation = usePlacer(function (s) { return s.setRotation; });
  const resetCam = usePlacer(function (s) { return s.resetCam; });
  const selectCam = usePlacer(function (s) { return s.selectCam; });

  const selectedCam = useMemo(
    function () {
      return cameras.find(function (c) { return c.name === selected; }) ?? null;
    },
    [cameras, selected],
  );

  const name = selectedCam?.name ?? null;

  // Read the pose via getState(), so each tick integrates fresh values.
  // https://github.com/pmndrs/zustand
  function onXYTick(dx: number, dy: number, dt: number) {
    if (!name) return;
    const p = usePlacer.getState().poses[name]?.position;
    if (!p) return;
    setPositionXY(
      name,
      p[0] + dx * SPEED_XY * dt,
      p[1] + dy * SPEED_XY * dt,
    );
  }
  function onZTick(dz: number, dt: number) {
    if (!name) return;
    const p = usePlacer.getState().poses[name]?.position;
    if (!p) return;
    setPosition(name, 2, p[2] + dz * SPEED_Z * dt);
  }
  function makeRotTick(axis: 0 | 1 | 2, speed: number) {
    return function (d: number, dt: number) {
      if (!name) return;
      const r = usePlacer.getState().poses[name]?.rotation;
      if (!r) return;
      setRotation(name, axis, r[axis] + d * speed * dt);
    };
  }
  const onYawTick = makeRotTick(0, SPEED_YAW);
  const onPitchTick = makeRotTick(1, SPEED_PITCH);
  const onRollTick = makeRotTick(2, SPEED_ROLL);

  return (
    <div style={{
      background: "#181818",
      border: "1px solid #2b2b2b",
      borderRadius: 4,
      padding: 8,
      minHeight: 0,
      overflowY: "auto",
      display: "flex",
      flexDirection: "column",
      gap: 8,
    }}>
      <div style={{ fontSize: 12, color: "#9d9d9d" }}>
        {selectedCam
          ? `Controls — ${selectedCam.name}${selectedCam.is_stereo ? " (stereo eye)" : ""}`
          : "Controls"}
      </div>

      {!selectedCam && (
        <div style={{ color: "#6d6d6d", fontSize: 11, padding: 8 }}>
          Click a camera on the map to enable controls.
        </div>
      )}

      {selectedCam && (function () {
        const pose = poses[selectedCam.name];
        if (!pose) return null;
        const [px, py, pz] = pose.position;
        const [yaw, pitch, roll] = pose.rotation;
        return (
          <>
            <div style={{
              display: "flex", justifyContent: "center", alignItems: "center", gap: 12,
            }}>
              <XYPad size={180} onTick={onXYTick} />
              <ZPad height={180} width={48} onTick={onZTick} />
            </div>
            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6,
            }}>
              <NumberField label="X (m)" value={px} step={0.01}
                onChange={function (v) { setPosition(selectedCam.name, 0, v); }} />
              <NumberField label="Y (m)" value={py} step={0.01}
                onChange={function (v) { setPosition(selectedCam.name, 1, v); }} />
              <NumberField label="Z (m)" value={pz} step={0.01}
                onChange={function (v) { setPosition(selectedCam.name, 2, v); }} />
              <NumberField label="Yaw (°)" value={yaw * DEG} step={0.5}
                onChange={function (v) { setRotation(selectedCam.name, 0, v * RAD); }} />
              <NumberField label="Pitch (°)" value={pitch * DEG} step={0.5}
                onChange={function (v) { setRotation(selectedCam.name, 1, v * RAD); }} />
              <NumberField label="Roll (°)" value={roll * DEG} step={0.5}
                onChange={function (v) { setRotation(selectedCam.name, 2, v * RAD); }} />
            </div>

            {/* Rotation joysticks. */}
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <PadRow label="Yaw" onTick={onYawTick} />
              <PadRow label="Pitch" onTick={onPitchTick} />
              <PadRow label="Roll" onTick={onRollTick} />
            </div>

            <div style={{ display: "flex", gap: 6 }}>
              <button onClick={function () { resetCam(selectedCam.name); }} style={btnStyle}>Reset this cam</button>
              <button onClick={function () { selectCam(null); }} style={btnStyle}>Deselect</button>
            </div>
          </>
        );
      })()}

      {/* Rotation reference, shown even with no camera selected. */}
      <div style={{
        marginTop: "auto",
        borderTop: "1px solid #2b2b2b",
        paddingTop: 6,
      }}>
        <div style={{ fontSize: 10, color: "#6d6d6d", marginBottom: 2, textAlign: "center" }}>
          Rotation reference (Z-up, cam-forward = +Y)
        </div>
        <RotationReference />
      </div>
    </div>
  );
}

interface PadRowProps {
  label: string;
  onTick: (d: number, dt: number) => void;
}
/** A label and one horizontal joystick, one row per rotation axis. */
function PadRow({ label, onTick }: PadRowProps) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ width: 44, fontSize: 11, color: "#9d9d9d" }}>{label}</div>
      <HPad onTick={onTick} />
    </div>
  );
}

interface NumberFieldProps {
  label: string;
  value: number;
  step: number;
  onChange: (v: number) => void;
}
/** Labelled number box, for typing a pose value exactly. */
function NumberField({ label, value, step, onChange }: NumberFieldProps) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2, fontSize: 11, color: "#9d9d9d" }}>
      {label}
      <input type="number" step={step} value={value.toFixed(2)}
        onChange={function (e) {
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
        style={{
          width: "100%", background: "#141414", border: "1px solid #2b2b2b",
          color: "#cccccc", padding: "3px 4px", fontSize: 12, boxSizing: "border-box",
        }} />
    </label>
  );
}

// The two small buttons under the pads.
const btnStyle: React.CSSProperties = {
  background: "#2b2b2b", color: "#cccccc", border: "1px solid #3c3c3c",
  padding: "4px 10px", fontSize: 12, cursor: "pointer", borderRadius: 3,
};
