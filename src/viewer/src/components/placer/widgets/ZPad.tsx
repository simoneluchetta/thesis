/**
 * The same joystick again for height. Up on the screen is positive.
 */
// https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events
// https://developer.mozilla.org/en-US/docs/Web/API/Element/setPointerCapture
// https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame
// https://developer.mozilla.org/en-US/docs/Web/API/Performance/now
// https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect
import { useEffect, useRef, useState } from "react";

const STICK = "#fc6";   // colour of the stick while it is held

interface Props {
  height?: number;
  width?: number;
  /** Deflection in [-1, 1] per frame, positive is up (raise height). dt in seconds. */
  onTick?: (d: number, dt: number) => void;
}

/** Upright joystick that raises and lowers the selected camera. */
export default function ZPad({ height = 140, width = 40, onTick }: Props) {
  const padRef = useRef<HTMLDivElement>(null);
  const [held, setHeld] = useState(false);
  const [deflection, setDeflection] = useState(0);
  const deflectionRef = useRef(0);
  const onTickRef = useRef(onTick);
  const rafIdRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);

  useEffect(function () { onTickRef.current = onTick; }, [onTick]);
  useEffect(function () { deflectionRef.current = deflection; }, [deflection]);

  function updateFromClient(clientY: number) {
    const pad = padRef.current;
    if (!pad) return;
    const rect = pad.getBoundingClientRect();
    const half = rect.height / 2;
    const dyClient = (clientY - (rect.top + half)) / half;
    // Flip so smaller client y (screen up) reports positive deflection.
    const d = -dyClient;
    setDeflection(Math.max(-1, Math.min(1, d)));
  }

  function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    setHeld(true);
    updateFromClient(e.clientY);
  }
  function onPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!held) return;
    updateFromClient(e.clientY);
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

  const stickR = 12;
  const center = height / 2;
  const travel = center - stickR;
  const stickY = center - deflection * travel;

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
        borderRadius: width / 2,
        touchAction: "none", userSelect: "none",
        position: "relative",
        cursor: held ? "grabbing" : "grab",
      }}
    >
      {/* Center reference line + corner labels */}
      <div style={{ position: "absolute", left: 4, right: 4, top: center, height: 1, background: "#2b2b2b" }} />
      <div style={{
        position: "absolute", left: 0, right: 0, top: 3,
        fontSize: 9, color: "#6d6d6d", textAlign: "center", pointerEvents: "none",
      }}>▲</div>
      <div style={{
        position: "absolute", left: 0, right: 0, bottom: 3,
        fontSize: 9, color: "#6d6d6d", textAlign: "center", pointerEvents: "none",
      }}>▼</div>
      <div style={{
        position: "absolute", left: 0, right: 0, top: center - 6,
        fontSize: 9, color: "#6d6d6d", textAlign: "center", pointerEvents: "none",
      }}>Z</div>
      {/* Stick */}
      <div style={{
        position: "absolute",
        left: width / 2 - stickR, top: stickY - stickR,
        width: stickR * 2, height: stickR * 2,
        borderRadius: stickR,
        background: held ? STICK : "#969696",
        boxShadow: held ? `0 0 8px ${STICK}` : "none",
        transition: held ? "none" : "top 150ms, background 150ms",
        pointerEvents: "none",
      }} />
    </div>
  );
}
