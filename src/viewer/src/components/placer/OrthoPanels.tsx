/**
 * The placer's three orthographic panels: XY, XZ and YZ. The plan view has no height.
 */
// https://threejs.org/docs/#api/en/math/Euler
// https://threejs.org/docs/#api/en/math/Matrix4.makeRotationFromEuler
// https://en.wikipedia.org/wiki/Euler_angles
import { useMemo } from "react";
import { Euler, Matrix4, Vector3 } from "three";
import type { SceneCameraEntry } from "../../types";
import { usePlacer } from "../../state/placerStore";
import { colorForPlacerCam } from "./camColors";

interface Props {
  cameras: SceneCameraEntry[];
}

/** Index into a world XYZ triple. */
type Axis = 0 | 1 | 2;

// Scratch objects reused across cameras; each result is read immediately.
const FWD_LOCAL = new Vector3(0, 0, 1);
const SCRATCH_EULER = new Euler(0, 0, 0, "ZXY");
const SCRATCH_MATRIX = new Matrix4();
const SCRATCH_FWD = new Vector3();
// Length of a forward arrow, in the panel's 240-unit viewBox.
const ARROW_LEN_SVG = 14;

interface PlanePanelProps {
  cameras: SceneCameraEntry[];
  hAxis: Axis;
  vAxis: Axis;
  label: string;
  /** Horizontal axis world label (e.g. "X"). */
  hLabel: string;
  /** Vertical axis world label (e.g. "Y"). */
  vLabel: string;
  /** Slot of `rotation` shown by the corner gauge: 0=yaw, 1=pitch, 2=roll. */
  gaugeAxis: Axis;
  gaugeName: string;
}

/** One ortho panel: every camera as a dot with a forward arrow, plus the angle gauge. */
function PlanePanel({ cameras, hAxis, vAxis, label, hLabel, vLabel, gaugeAxis, gaugeName }: PlanePanelProps) {
  const poses = usePlacer(function (s) { return s.poses; });
  const selected = usePlacer(function (s) { return s.selected; });
  const clickCam = usePlacer(function (s) { return s.clickCam; });

  const VBOX = 240;
  const { minH, maxH, minV, maxV } = useMemo(function () {
    let minH = Infinity, maxH = -Infinity, minV = Infinity, maxV = -Infinity;
    for (const cam of cameras) {
      const p = poses[cam.name]?.position ?? cam.position;
      minH = Math.min(minH, p[hAxis]); maxH = Math.max(maxH, p[hAxis]);
      minV = Math.min(minV, p[vAxis]); maxV = Math.max(maxV, p[vAxis]);
    }
    const ch = (minH + maxH) / 2;
    const cv = (minV + maxV) / 2;
    const span = Math.max(maxH - minH, maxV - minV, 2);
    const half = span * 1.2;
    return { minH: ch - half, maxH: ch + half, minV: cv - half, maxV: cv + half };
  }, [cameras, poses, hAxis, vAxis]);

  function toSvg(wH: number, wV: number) {
    const u = (wH - minH) / (maxH - minH);
    const v = (wV - minV) / (maxV - minV);
    return [u * VBOX, (1 - v) * VBOX] as const;
  }

  const [zeroH, zeroV] = toSvg(0, 0);

  return (
    <div style={{ background: "#181818", border: "1px solid #2b2b2b", borderRadius: 4, padding: 6 }}>
      <div style={{ fontSize: 12, color: "#9d9d9d", marginBottom: 4 }}>
        {label} <span style={{ color: "#6d6d6d" }}>(horiz={hLabel}, vert={vLabel})</span>
      </div>
      <svg viewBox={`0 0 ${VBOX} ${VBOX}`} style={{ width: "100%", aspectRatio: "1 / 1", display: "block" }}>
        {/* Axes through world origin */}
        <line x1={0} x2={VBOX} y1={zeroV} y2={zeroV} stroke="#3c3c3c" strokeWidth={1} />
        <line x1={zeroH} x2={zeroH} y1={0} y2={VBOX} stroke="#3c3c3c" strokeWidth={1} />
        {/* Border */}
        <rect x={0} y={0} width={VBOX} height={VBOX} fill="none" stroke="#2b2b2b" strokeWidth={1} />
        {/* Cameras */}
        {cameras.map(function (cam) {
          const pose = poses[cam.name];
          const p = pose?.position ?? cam.position;
          const [sx, sy] = toSvg(p[hAxis], p[vAxis]);
          const isSel = selected === cam.name;
          const isStereo = !!cam.is_stereo;
          const fill = colorForPlacerCam(p[2], isSel, isStereo);

          // World forward = R*(0,0,1) from the stored ZXY Euler, projected onto the panel.
          // https://threejs.org/docs/#api/en/math/Euler
          // https://en.wikipedia.org/wiki/Euler_angles
          let arrow: React.ReactElement | null = null;
          if (pose) {
            SCRATCH_EULER.set(pose.rotation[1], pose.rotation[2], pose.rotation[0], "ZXY");
            SCRATCH_MATRIX.makeRotationFromEuler(SCRATCH_EULER);
            SCRATCH_FWD.copy(FWD_LOCAL).applyMatrix4(SCRATCH_MATRIX);
            const components = [SCRATCH_FWD.x, SCRATCH_FWD.y, SCRATCH_FWD.z];
            const fwdH = components[hAxis];
            const fwdV = components[vAxis];
            const fwdLen = Math.hypot(fwdH, fwdV);
            if (fwdLen > 1e-6) {
              const tipX = sx + (fwdH / fwdLen) * ARROW_LEN_SVG;
              const tipY = sy - (fwdV / fwdLen) * ARROW_LEN_SVG;
              arrow = (
                <line x1={sx} y1={sy} x2={tipX} y2={tipY}
                  stroke={isSel ? "#fc6" : "#969696"}
                  strokeWidth={isSel ? 1.6 : 1}
                  strokeLinecap="round" />
              );
            }
          }

          return (
            <g key={cam.name} style={{ cursor: "pointer" }} onClick={function () { clickCam(cam.name); }}>
              {arrow}
              <circle cx={sx} cy={sy} r={isSel ? 5 : 3.5}
                fill={fill} stroke={isSel ? "#fff" : "#000a"} strokeWidth={1} />
              {isSel && (
                <text x={sx + 7} y={sy + 4} fontSize={10} fill="#fc6">{cam.name}</text>
              )}
            </g>
          );
        })}

        {/* Gauge for this panel's normal axis: yaw for XY, roll for XZ, pitch for YZ. */}
        {(function () {
          const sel = cameras.find(function (c) { return c.name === selected; });
          const pose = sel ? poses[sel.name] : null;
          if (!sel || !pose) return null;
          return (
            <NativeRotationGauge
              vbox={VBOX}
              angleRad={pose.rotation[gaugeAxis]}
              label={gaugeName}
            />
          );
        })()}
      </svg>
    </div>
  );
}

/** Clock-face gauge, top-right. Angle 0 points up, positive turns counter-clockwise.
 *  https://en.wikipedia.org/wiki/Right-hand_rule */
function NativeRotationGauge({ vbox, angleRad, label }: {
  vbox: number; angleRad: number; label: string;
}) {
  const r = 18;
  const cx = vbox - r - 6;
  const cy = r + 6;
  const tipX = cx - r * Math.sin(angleRad);
  const tipY = cy - r * Math.cos(angleRad);
  const deg = (angleRad * 180 / Math.PI).toFixed(1);
  return (
    <g pointerEvents="none">
      {/* backing disc, so the gauge reads over camera dots */}
      <circle cx={cx} cy={cy} r={r} fill="#181818" fillOpacity={0.92}
        stroke="#3c3c3c" strokeWidth={1} />
      {/* 12-o'clock tick is angle 0 */}
      <line x1={cx} y1={cy - r + 1} x2={cx} y2={cy - r + 5}
        stroke="#5a5a5a" strokeWidth={1} />
      {/* needle */}
      <line x1={cx} y1={cy} x2={tipX} y2={tipY}
        stroke="#fc6" strokeWidth={1.6} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={1.6} fill="#fc6" />
      <text x={cx} y={cy + r + 9} fontSize={9} fill="#9d9d9d" textAnchor="middle">
        {label} {deg}°
      </text>
    </g>
  );
}

/** The three ortho panels in a column, beside the placer's map. */
export default function OrthoPanels({ cameras }: Props) {
  return (
    <div style={{ display: "grid", gap: 6 }}>
      {/* XY normal is +Z, so the matching slider is yaw (rot[0]). */}
      <PlanePanel cameras={cameras} hAxis={0} vAxis={1} label="XY plane"
        hLabel="X" vLabel="Y" gaugeAxis={0} gaugeName="Yaw" />
      {/* XZ normal is +Y, so the matching slider is roll (rot[2]). */}
      <PlanePanel cameras={cameras} hAxis={0} vAxis={2} label="XZ plane"
        hLabel="X" vLabel="Z" gaugeAxis={2} gaugeName="Roll" />
      {/* YZ normal is +X, so the matching slider is pitch (rot[1]). */}
      <PlanePanel cameras={cameras} hAxis={1} vAxis={2} label="YZ plane"
        hLabel="Y" vLabel="Z" gaugeAxis={1} gaugeName="Pitch" />
    </div>
  );
}
