/**
 * The placer's plan view: the court from above with every camera on it, draggable.
 * Wheel zooms around the cursor, dragging empty space pans.
 */
// https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events
// https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver
// https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/viewBox
// https://threejs.org/docs/#api/en/math/Euler
import { useEffect, useMemo, useRef, useState } from "react";
import { Euler, Matrix4, Vector3 } from "three";
import type { SceneCameraEntry } from "../../types";
import { usePlacer } from "../../state/placerStore";
import { useViewer } from "../../state/store";
import CourtOverlay, { BASKETBALL_COLOR, VOLLEYBALL_COLOR } from "./CourtOverlay";
import Compass from "../Compass";
import {
  CAM_COLOR_BELOW_FLOOR, CAM_COLOR_NORMAL, CAM_COLOR_SELECTED, CAM_COLOR_STEREO,
  colorForPlacerCam,
} from "./camColors";

interface Props {
  cameras: SceneCameraEntry[];
}

// Half a 28x15 m FIBA court, in metres.
const BASKETBALL_HALF = { x: 14, y: 7.5 };
// Half an 18x9 m FIVB court, in metres.
const VOLLEYBALL_HALF = { x: 9, y: 4.5 };

// SVG viewBox width; the height follows the container's aspect.
const VBOX_W = 1000;
// Furthest the wheel can zoom out, 1 being the fitted view.
const MIN_ZOOM = 0.1;
// Closest the wheel can zoom in.
const MAX_ZOOM = 50;
// Zoom factor per wheel notch.
const WHEEL_ZOOM_STEP = 1.2;

// Scratch objects for the yaw arrow, reused so a drag frame allocates nothing.
const FWD_LOCAL = new Vector3(0, 0, 1);
const SCRATCH_EULER = new Euler(0, 0, 0, "ZXY");
const SCRATCH_MATRIX = new Matrix4();
const SCRATCH_FWD = new Vector3();

/** World X runs left to right, world Y bottom to top. Height (Z) is in the controls.
 *  VBOX_H follows the container's aspect; 1 metre is the same on both axes. */
export default function TopDownMap({ cameras }: Props) {
  const poses = usePlacer(function (s) { return s.poses; });
  const initialPoses = usePlacer(function (s) { return s.initialPoses; });
  const selected = usePlacer(function (s) { return s.selected; });
  const setPositionXY = usePlacer(function (s) { return s.setPositionXY; });
  const clickCam = usePlacer(function (s) { return s.clickCam; });

  const svgRef = useRef<SVGSVGElement | null>(null);
  const [dragging, setDragging] = useState<string | null>(null);
  const [showBasketball, setShowBasketball] = useState(true);
  const [showVolleyball, setShowVolleyball] = useState(true);

  // Court placement shared with the 3-D viewer's HUD. This view is X-Y only.
  const courtAnchorX = useViewer(function (s) { return s.courtAnchorX; });
  const courtAnchorY = useViewer(function (s) { return s.courtAnchorY; });
  const courtYawDeg = useViewer(function (s) { return s.courtYawDeg; });
  const courtYawRad = (courtYawDeg * Math.PI) / 180;
  const courtCos = Math.cos(courtYawRad);
  const courtSin = Math.sin(courtYawRad);

  // zoom > 1 is zoomed in. `pan` is a metre offset from the base extent's centre.
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  // World point grabbed on an empty-space drag; it stays under the cursor.
  const [panAnchor, setPanAnchor] = useState<{ wx: number; wy: number } | null>(null);

  // Rendered aspect ratio of the SVG, so the viewBox follows the real DOM size.
  const [aspect, setAspect] = useState(1);
  useEffect(function () {
    const node = svgRef.current;
    if (!node) return;
    const ro = new ResizeObserver(function (entries) {
      const r = entries[0]?.contentRect;
      if (r && r.height > 0 && r.width > 0) setAspect(r.width / r.height);
    });
    ro.observe(node);
    return function () { return ro.disconnect(); };
  }, []);
  const VBOX_H = VBOX_W / Math.max(aspect, 0.0001);

  // Bbox of the cameras plus the enabled courts, padded so a drag can go past them.
  // From `initialPoses`: live poses would grow the bbox mid-drag and shift the view.
  const baseExtent = useMemo(function () {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const cam of cameras) {
      const p = initialPoses[cam.name]?.position ?? cam.position;
      minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
    }
    // Court corners after anchor and yaw, so the fit still frames it off-origin.
    function includeRotatedRect(hx: number, hy: number) {
      const corners: [number, number][] = [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]];
      for (const [cx, cy] of corners) {
        const tx = cx * courtCos - cy * courtSin + courtAnchorX;
        const ty = cx * courtSin + cy * courtCos + courtAnchorY;
        minX = Math.min(minX, tx); maxX = Math.max(maxX, tx);
        minY = Math.min(minY, ty); maxY = Math.max(maxY, ty);
      }
    }
    if (showBasketball) includeRotatedRect(BASKETBALL_HALF.x, BASKETBALL_HALF.y);
    if (showVolleyball) includeRotatedRect(VOLLEYBALL_HALF.x, VOLLEYBALL_HALF.y);
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    // Half-span of the shorter axis; the longer one covers halfSpan * aspect.
    const halfSpan = Math.max(maxX - minX, maxY - minY, 4) * 1.5;
    return { centerX, centerY, halfSpan };
  }, [cameras, initialPoses, showBasketball, showVolleyball,
      courtAnchorX, courtAnchorY, courtCos, courtSin]);

  /** Half-extents per world axis, so 1 metre is the same on both axes. */
  const halfX = useMemo(function () {
    const base = baseExtent.halfSpan / zoom;
    return aspect >= 1 ? base * aspect : base;
  }, [baseExtent, zoom, aspect]);
  const halfY = useMemo(function () {
    const base = baseExtent.halfSpan / zoom;
    return aspect >= 1 ? base : base / aspect;
  }, [baseExtent, zoom, aspect]);

  const { minX, maxX, minY, maxY } = useMemo(function () {
    const cx = baseExtent.centerX + pan.x;
    const cy = baseExtent.centerY + pan.y;
    return { minX: cx - halfX, maxX: cx + halfX, minY: cy - halfY, maxY: cy + halfY };
  }, [baseExtent, pan, halfX, halfY]);

  function worldToSvg(wx: number, wy: number) {
    const u = (wx - minX) / (maxX - minX);
    const v = (wy - minY) / (maxY - minY);
    return [u * VBOX_W, (1 - v) * VBOX_H] as const;
  }
  function svgToWorld(sx: number, sy: number) {
    const u = sx / VBOX_W;
    const v = 1 - sy / VBOX_H;
    return [minX + u * (maxX - minX), minY + v * (maxY - minY)] as const;
  }
  // Scale for radii (circles, arcs). X and Y share a scale, so one scalar works.
  const metersToSvg = VBOX_W / (maxX - minX);

  /** Pan that puts world (wx, wy) under SVG (sx, sy) at the current zoom. */
  function panThatPlaces(wx: number, wy: number, sx: number, sy: number) {
    return {
      x: wx - baseExtent.centerX + halfX * (1 - 2 * sx / VBOX_W),
      y: wy - baseExtent.centerY + halfY * (2 * sy / VBOX_H - 1),
    };
  }

  function onWheel(e: React.WheelEvent<SVGSVGElement>) {
    if (!svgRef.current) return;
    e.preventDefault();
    // preserveAspectRatio="none", so a client point converts with a plain ratio.
    // https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect
    const rect = svgRef.current.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * VBOX_W;
    const sy = ((e.clientY - rect.top) / rect.height) * VBOX_H;
    const [wxBefore, wyBefore] = svgToWorld(sx, sy);
    const factor = e.deltaY < 0 ? WHEEL_ZOOM_STEP : 1 / WHEEL_ZOOM_STEP;
    const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom * factor));
    // Recompute the halves: the memoized halfX/halfY are still at the old zoom.
    const baseHalf = baseExtent.halfSpan / newZoom;
    const newHalfX = aspect >= 1 ? baseHalf * aspect : baseHalf;
    const newHalfY = aspect >= 1 ? baseHalf : baseHalf / aspect;
    setZoom(newZoom);
    setPan({
      x: wxBefore - baseExtent.centerX + newHalfX * (1 - 2 * sx / VBOX_W),
      y: wyBefore - baseExtent.centerY + newHalfY * (2 * sy / VBOX_H - 1),
    });
  }

  function onSvgPointerDown(e: React.PointerEvent<SVGSVGElement>) {
    // Only fires on empty SVG; camera handlers call stopPropagation().
    if (dragging || panAnchor || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * VBOX_W;
    const sy = ((e.clientY - rect.top) / rect.height) * VBOX_H;
    const [wx, wy] = svgToWorld(sx, sy);
    setPanAnchor({ wx, wy });
    // https://developer.mozilla.org/en-US/docs/Web/API/Element/setPointerCapture
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* ignore */ }
  }

  // Throttle store writes to one per animation frame; pointermove fires faster.
  // https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame
  const pendingDragRef = useRef<{ name: string; wx: number; wy: number } | null>(null);
  const pendingPanRef = useRef<{ wx: number; wy: number; sx: number; sy: number } | null>(null);
  const rafIdRef = useRef<number | null>(null);

  function scheduleFlush() {
    if (rafIdRef.current != null) return;
    rafIdRef.current = requestAnimationFrame(function () {
      rafIdRef.current = null;
      const d = pendingDragRef.current;
      const p = pendingPanRef.current;
      pendingDragRef.current = null;
      pendingPanRef.current = null;
      if (d) setPositionXY(d.name, d.wx, d.wy);
      if (p) setPan(panThatPlaces(p.wx, p.wy, p.sx, p.sy));
    });
  }

  useEffect(function () {
    return function () {
      if (rafIdRef.current != null) cancelAnimationFrame(rafIdRef.current);
    };
  }, []);

  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * VBOX_W;
    const sy = ((e.clientY - rect.top) / rect.height) * VBOX_H;
    if (dragging) {
      const [wx, wy] = svgToWorld(sx, sy);
      pendingDragRef.current = { name: dragging, wx, wy };
      scheduleFlush();
    } else if (panAnchor) {
      pendingPanRef.current = { wx: panAnchor.wx, wy: panAnchor.wy, sx, sy };
      scheduleFlush();
    }
  }
  function onPointerUp(e: React.PointerEvent<SVGSVGElement>) {
    if (dragging) {
      // Flush now, so the final position matches the cursor at release.
      const d = pendingDragRef.current;
      if (d) setPositionXY(d.name, d.wx, d.wy);
      pendingDragRef.current = null;
      try { (e.target as Element).releasePointerCapture?.(e.pointerId); } catch { /* ignore */ }
      setDragging(null);
    }
    if (panAnchor) {
      const p = pendingPanRef.current;
      if (p) setPan(panThatPlaces(p.wx, p.wy, p.sx, p.sy));
      pendingPanRef.current = null;
      try { e.currentTarget.releasePointerCapture?.(e.pointerId); } catch { /* ignore */ }
      setPanAnchor(null);
    }
  }

  function onFitView() { setZoom(1); setPan({ x: 0, y: 0 }); }

  // Grid lines every 1 m, labelled every 5 m.
  const gridLines: { x?: number; y?: number; label: string }[] = [];
  const startX = Math.ceil(minX);
  const endX = Math.floor(maxX);
  for (let x = startX; x <= endX; x++) {
    const [sx] = worldToSvg(x, 0);
    gridLines.push({ x: sx, label: x % 5 === 0 ? `${x}` : "" });
  }
  const startY = Math.ceil(minY);
  const endY = Math.floor(maxY);
  for (let y = startY; y <= endY; y++) {
    const [, sy] = worldToSvg(0, y);
    gridLines.push({ y: sy, label: y % 5 === 0 ? `${y}` : "" });
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", height: "100%" }}>
      <div style={{
        padding: "4px 8px", color: "#9d9d9d", fontSize: 12, background: "#1f1f1f",
        display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center",
      }}>
        <span>Top-down (X-Y). Drag a camera to move it. Drag empty space to pan, wheel to zoom. Z (height) is in the controls panel.</span>
        <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer", color: BASKETBALL_COLOR }}>
          <input type="checkbox" checked={showBasketball}
            onChange={function (e) { setShowBasketball(e.target.checked); }} />
          Basketball (28x15m)
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer", color: VOLLEYBALL_COLOR }}>
          <input type="checkbox" checked={showVolleyball}
            onChange={function (e) { setShowVolleyball(e.target.checked); }} />
          Volleyball (18x9m)
        </label>
        <button onClick={onFitView} style={{
          background: "#2b2b2b", color: "#cccccc", border: "1px solid #3c3c3c",
          padding: "2px 8px", fontSize: 11, cursor: "pointer", borderRadius: 3,
        }}>
          Fit ({zoom.toFixed(2)}x)
        </button>
      </div>
      <div style={{ position: "relative", flex: 1, minHeight: 0 }}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${VBOX_W} ${VBOX_H}`}
        preserveAspectRatio="none"
        style={{
          position: "absolute", inset: 0,
          width: "100%", height: "100%", background: "#181818",
          // touch-action none, or a touch drag would scroll the page instead.
          // https://developer.mozilla.org/en-US/docs/Web/CSS/touch-action
          touchAction: "none", userSelect: "none",
          cursor: panAnchor ? "grabbing" : "grab",
        }}
        onWheel={onWheel}
        onPointerDown={onSvgPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        {gridLines.filter(function (g) { return g.x !== undefined; }).map(function (g, i) {
          return (
            <line key={`vx-${i}`} x1={g.x} x2={g.x} y1={0} y2={VBOX_H}
              stroke="#2b2b2b" strokeWidth={g.label ? 1.5 : 0.5} />
          );
        })}
        {gridLines.filter(function (g) { return g.y !== undefined; }).map(function (g, i) {
          return (
            <line key={`vy-${i}`} x1={0} x2={VBOX_W} y1={g.y} y2={g.y}
              stroke="#2b2b2b" strokeWidth={g.label ? 1.5 : 0.5} />
          );
        })}
        {gridLines.filter(function (g) { return g.label && g.x !== undefined; }).map(function (g, i) {
          return (
            <text key={`lx-${i}`} x={g.x} y={VBOX_H - 4} fontSize={14} fill="#5a5a5a" textAnchor="middle">
              {g.label}
            </text>
          );
        })}
        {gridLines.filter(function (g) { return g.label && g.y !== undefined; }).map(function (g, i) {
          return (
            <text key={`ly-${i}`} x={4} y={g.y! - 4} fontSize={14} fill="#5a5a5a">
              {g.label}
            </text>
          );
        })}

        {/* Origin marker */}
        {(function () {
          const [ox, oy] = worldToSvg(0, 0);
          return (
            <g>
              <circle cx={ox} cy={oy} r={6} fill="none" stroke="#6d6d6d" strokeWidth={1.5} />
              <text x={ox + 10} y={oy + 4} fontSize={12} fill="#6d6d6d">origin</text>
            </g>
          );
        })()}

        {/* Court overlays. Anchor and yaw are folded into the worldToSvg below. */}
        <CourtOverlay
          basketball={showBasketball}
          volleyball={showVolleyball}
          worldToSvg={function (x, y) {
            return worldToSvg(
              x * courtCos - y * courtSin + courtAnchorX,
              x * courtSin + y * courtCos + courtAnchorY,
            );
          }}
          metersToSvg={metersToSvg}
        />

        {/* Cameras */}
        {cameras.map(function (cam) {
          const pose = poses[cam.name];
          if (!pose) return null;
          const [sx, sy] = worldToSvg(pose.position[0], pose.position[1]);
          // Rebuild c2w from the stored ZXY Euler (yaw, pitch, roll) and rotate local
          // https://threejs.org/docs/#api/en/math/Euler
          // https://en.wikipedia.org/wiki/Euler_angles
          // +Z, the COLMAP forward axis, into world. X-Y is the look on the floor.
          SCRATCH_EULER.set(pose.rotation[1], pose.rotation[2], pose.rotation[0], "ZXY");
          SCRATCH_MATRIX.makeRotationFromEuler(SCRATCH_EULER);
          SCRATCH_FWD.copy(FWD_LOCAL).applyMatrix4(SCRATCH_MATRIX);
          const fwdLen = Math.hypot(SCRATCH_FWD.x, SCRATCH_FWD.y);
          const norm = fwdLen > 1e-6 ? fwdLen : 1;
          const tipX = sx + (SCRATCH_FWD.x / norm) * 28;
          const tipY = sy - (SCRATCH_FWD.y / norm) * 28;  // SVG y flipped vs world Y
          const isSel = selected === cam.name;
          const isStereo = !!cam.is_stereo;
          const fill = colorForPlacerCam(pose.position[2], isSel, isStereo);
          return (
            <g
              key={cam.name}
              style={{ cursor: dragging === cam.name ? "grabbing" : "grab" }}
              onPointerDown={function (ev) {
                ev.stopPropagation();
                (ev.target as Element).setPointerCapture?.(ev.pointerId);
                setDragging(cam.name);
                clickCam(cam.name);
              }}
            >
              {fwdLen > 1e-6 && (
                <line x1={sx} y1={sy} x2={tipX} y2={tipY}
                  stroke={isSel ? "#fc6" : "#969696"} strokeWidth={isSel ? 3 : 2} />
              )}
              <circle cx={sx} cy={sy} r={isSel ? 10 : 7}
                fill={fill} stroke={isSel ? "#fff" : "#000a"} strokeWidth={isSel ? 2 : 1} />
              <text x={sx + 12} y={sy + 4} fontSize={13} fill="#cccccc" pointerEvents="none">
                {cam.name}
              </text>
            </g>
          );
        })}
      </svg>
      <Legend />
      {/* Same world-axes compass as the Court Calibrator page. */}
      <div style={{ position: "absolute", top: 12, right: 12 }}>
        <Compass size={76} variant="basketball" caption="World axes" panel />
      </div>
      </div>
    </div>
  );
}

/** Colour key, bottom-left of the map. */
function Legend() {
  return (
    <div style={{
      position: "absolute",
      bottom: 12, left: 12,
      padding: "10px 12px 8px",
      background: "rgba(24, 24, 24, 0.82)",
      backdropFilter: "blur(8px)",
      WebkitBackdropFilter: "blur(8px)",
      border: "1px solid #3c3c3c",
      borderRadius: 8,
      boxShadow: "0 6px 22px rgba(0, 0, 0, 0.45)",
      fontSize: 11,
      color: "#cccccc",
      pointerEvents: "none",
      userSelect: "none",
      minWidth: 168,
    }}>
      <div style={{
        fontSize: 9.5,
        textTransform: "uppercase",
        letterSpacing: "0.1em",
        color: "#969696",
        fontWeight: 600,
        marginBottom: 6,
      }}>
        Camera legend
      </div>
      <LegendRow color={CAM_COLOR_SELECTED} label="Selected" />
      <LegendRow color={CAM_COLOR_BELOW_FLOOR} label="Below floor (Z < 0)" />
      <LegendRow color={CAM_COLOR_STEREO} label="Stereo eye" />
      <LegendRow color={CAM_COLOR_NORMAL} label="Above floor" />
    </div>
  );
}

/** One line of the legend: a coloured dot and what it means. */
function LegendRow({ color, label }: { color: string; label: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "2px 0" }}>
      <span style={{
        width: 11, height: 11, borderRadius: 6,
        background: color,
        boxShadow: `0 0 8px ${color}55, inset 0 0 0 1px rgba(0,0,0,0.35)`,
        flex: "0 0 auto",
      }} />
      <span>{label}</span>
    </div>
  );
}
