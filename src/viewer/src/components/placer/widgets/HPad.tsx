/**
 * A horizontal joystick that springs back to the middle, used for the angles.
 * It keeps reporting while held. Right on screen is positive.
 */
// https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events
// https://developer.mozilla.org/en-US/docs/Web/API/Element/setPointerCapture
// https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame
// https://developer.mozilla.org/en-US/docs/Web/API/Performance/now
// https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect
import { useEffect, useRef, useState } from "react";

const STICK = "#fc6";   // colour of the stick while it is held

interface Props {
  width?: number;
  height?: number;
  /** Deflection in [-1, 1] per frame, positive is stick right. dt in seconds. */
  onTick?: (d: number, dt: number) => void;
}

/** Horizontal joystick for one angle, used by the yaw, pitch and roll rows. */
export default function HPad({ width = 200, height = 28, onTick }: Props) {
  const padRef = useRef<HTMLDivElement>(null);
  const [held, setHeld] = useState(false);
  const [deflection, setDeflection] = useState(0);
  const deflectionRef = useRef(0);
  const onTickRef = useRef(onTick);
  const rafIdRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);

  useEffect(function () { onTickRef.current = onTick; }, [onTick]);
  useEffect(function () { deflectionRef.current = deflection; }, [deflection]);

  function updateFromClient(clientX: number) {
    const pad = padRef.current;
    if (!pad) return;
    const rect = pad.getBoundingClientRect();
    const half = rect.width / 2;
    const d = (clientX - (rect.left + half)) / half;
    setDeflection(Math.max(-1, Math.min(1, d)));
  }

  function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    setHeld(true);
    updateFromClient(e.clientX);
  }
  function onPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!held) return;
    updateFromClient(e.clientX);
  }
  function onPointerUp(e: React.PointerEvent<HTMLDivElement>) {
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    setHeld(false);
    setDeflection(0);
  }

  useEffect(function () {
    if (!held) return;
    lastTimeRef.current = performance.now();
    function tick() {
      const now = performance.now();
      // Cap dt, so a long pause doesn't launch the value when the loop resumes.
      const dt = Math.min(0.1, (now - lastTimeRef.current) / 1000);
      lastTimeRef.current = now;
      onTickRef.current?.(deflectionRef.current, dt);
      rafIdRef.current = requestAnimationFrame(tick);
    }
    rafIdRef.current = requestAnimationFrame(tick);
    return function () {
      if (rafIdRef.current != null) cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    };
  }, [held]);

  const stickR = 11;
  const center = width / 2;
  const travel = center - stickR;
  const stickX = center + deflection * travel;

  return (
    <div
      ref={padRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      style={{
        width, height,
        background: "#141414",
        border: "1px solid #2b2b2b",
        borderRadius: height / 2,
        touchAction: "none", userSelect: "none",
        position: "relative",
        cursor: held ? "grabbing" : "grab",
      }}
    >
      {/* Center axis + center tick */}
      <div style={{ position: "absolute", top: height / 2, left: 4, right: 4, height: 1, background: "#2b2b2b" }} />
      <div style={{ position: "absolute", left: center, top: 3, bottom: 3, width: 1, background: "#2b2b2b" }} />
      <div style={{
        position: "absolute",
        left: stickX - stickR, top: height / 2 - stickR,
        width: stickR * 2, height: stickR * 2,
        borderRadius: stickR,
        background: held ? STICK : "#969696",
        boxShadow: held ? `0 0 8px ${STICK}` : "none",
        transition: held ? "none" : "left 150ms, background 150ms",
        pointerEvents: "none",
      }} />
    </div>
  );
}
