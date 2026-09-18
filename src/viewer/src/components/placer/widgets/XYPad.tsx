/**
 * The same joystick in two dimensions, for moving a camera around the floor.
 * Press anywhere in the disc to grab the stick, release to recentre.
 */
// https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events
// https://developer.mozilla.org/en-US/docs/Web/API/Element/setPointerCapture
// https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame
// https://developer.mozilla.org/en-US/docs/Web/API/Performance/now
// https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect
import { useEffect, useRef, useState } from "react";

const STICK = "#fc6";   // colour of the stick while it is held

interface Props {
  size?: number;
  /** Deflection in [-1, 1] per frame, dy positive is screen up. dt in seconds. */
  onTick?: (dx: number, dy: number, dt: number) => void;
}

/** Round joystick that walks the selected camera across the floor. */
export default function XYPad({ size = 140, onTick }: Props) {
  const padRef = useRef<HTMLDivElement>(null);
  const [held, setHeld] = useState(false);
  const [deflection, setDeflection] = useState({ x: 0, y: 0 });
  const deflectionRef = useRef({ x: 0, y: 0 });
  const onTickRef = useRef(onTick);
  const rafIdRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);

  useEffect(function () { onTickRef.current = onTick; }, [onTick]);
  useEffect(function () { deflectionRef.current = deflection; }, [deflection]);

  function updateFromClient(clientX: number, clientY: number) {
    const pad = padRef.current;
    if (!pad) return;
    const rect = pad.getBoundingClientRect();
    const r = rect.width / 2;
    const dx = (clientX - (rect.left + r)) / r;
    const dyClient = (clientY - (rect.top + r)) / r;
    // Flip so screen-up (smaller client y) reports as positive dy.
    const dy = -dyClient;
    const len = Math.hypot(dx, dy);
    if (len > 1) setDeflection({ x: dx / len, y: dy / len });
    else setDeflection({ x: dx, y: dy });
  }

  function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    setHeld(true);
    updateFromClient(e.clientX, e.clientY);
  }
  function onPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!held) return;
    updateFromClient(e.clientX, e.clientY);
  }
  function onPointerUp(e: React.PointerEvent<HTMLDivElement>) {
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    setHeld(false);
    setDeflection({ x: 0, y: 0 });
  }

  useEffect(function () {
    if (!held) return;
    lastTimeRef.current = performance.now();
    function tick() {
      const now = performance.now();
      // Cap dt, so a long pause doesn't launch the value when the loop resumes.
      const dt = Math.min(0.1, (now - lastTimeRef.current) / 1000);
      lastTimeRef.current = now;
      const d = deflectionRef.current;
      onTickRef.current?.(d.x, d.y, dt);
      rafIdRef.current = requestAnimationFrame(tick);
    }
    rafIdRef.current = requestAnimationFrame(tick);
    return function () {
      if (rafIdRef.current != null) cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    };
  }, [held]);

  const r = size / 2;
  const stickR = 12;
  const stickX = r + deflection.x * (r - stickR);
  const stickY = r - deflection.y * (r - stickR);

  return (
    <div
      ref={padRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      style={{
        width: size, height: size,
        background: "#141414",
        border: "1px solid #2b2b2b",
        borderRadius: "50%",
        touchAction: "none", userSelect: "none",
        position: "relative",
        cursor: held ? "grabbing" : "grab",
      }}
    >
      <div style={{ position: "absolute", top: r, left: 0, right: 0, height: 1, background: "#2b2b2b" }} />
      <div style={{ position: "absolute", left: r, top: 0, bottom: 0, width: 1, background: "#2b2b2b" }} />
      <div style={{
        position: "absolute",
        left: stickX - stickR, top: stickY - stickR,
        width: stickR * 2, height: stickR * 2,
        borderRadius: stickR,
        background: held ? STICK : "#969696",
        boxShadow: held ? `0 0 10px ${STICK}` : "none",
        transition: held ? "none" : "left 150ms, top 150ms, background 150ms",
        pointerEvents: "none",
      }} />
      <div style={{
        position: "absolute", left: 4, top: 4,
        fontSize: 10, color: "#6d6d6d", pointerEvents: "none",
      }}>X-Y</div>
    </div>
  );
}
