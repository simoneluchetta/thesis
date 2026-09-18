/**
 * Constrains orbiting to one world axis at a time, so the court stays level.
 */
// OrbitControls angle limits:
// https://threejs.org/docs/#examples/en/controls/OrbitControls
// https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events
import { useEffect, useLayoutEffect, useRef } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { useViewer } from "../state/store";

// Polar angle range in radians, kept off the poles so the view cannot flip over.
const FULL_POLAR = { min: 0.05, max: Math.PI - 0.05 };

/** Locks the OrbitControls angles for X/Y orbit, full range for free and Z.
 *  Z is handled by WorldZOrbitDrag below. */
export function OrbitAxisSync() {
  const orbitAxis = useViewer(function (s) { return s.orbitAxis; });
  const controls = useThree(function (s) { return s.controls; }) as OrbitControlsImpl | null;
  const resetViewKey = useViewer(function (s) { return s.resetViewKey; });
  const selectedCamId = useViewer(function (s) { return s.selectedCamId; });

  useLayoutEffect(function () {
    if (!controls) return;

    if (orbitAxis === "y") {
      // Pin the polar angle: the eye only goes round the horizon.
      const phi = controls.getPolarAngle();
      controls.minPolarAngle = phi;
      controls.maxPolarAngle = phi;
      controls.minAzimuthAngle = -Infinity;
      controls.maxAzimuthAngle = Infinity;
    } else if (orbitAxis === "x") {
      // Pin the azimuth: the eye only goes up and over.
      const theta = controls.getAzimuthalAngle();
      controls.minAzimuthAngle = theta;
      controls.maxAzimuthAngle = theta;
      controls.minPolarAngle = FULL_POLAR.min;
      controls.maxPolarAngle = FULL_POLAR.max;
    } else {
      // "free", and "z" which WorldZOrbitDrag drives by hand.
      controls.minPolarAngle = FULL_POLAR.min;
      controls.maxPolarAngle = FULL_POLAR.max;
      controls.minAzimuthAngle = -Infinity;
      controls.maxAzimuthAngle = Infinity;
    }
    controls.update();
  }, [orbitAxis, controls, resetViewKey, selectedCamId]);

  return null;
}

// World up in the placer convention.
const WORLD_Z = new THREE.Vector3(0, 0, 1);

/** With orbitAxis "z", left-drag spins the eye around world +Z through the orbit target. */
export function WorldZOrbitDrag() {
  const orbitAxis = useViewer(function (s) { return s.orbitAxis; });
  const active = orbitAxis === "z";
  const { camera, gl } = useThree();
  const controls = useThree(function (s) { return s.controls; }) as OrbitControlsImpl | null;
  const dragging = useRef(false);

  useEffect(function () {
    const el = gl.domElement;
    const rotPerPx = 0.008;

    function onPointerDown(e: PointerEvent) {
      if (!active || e.button !== 0) return;
      dragging.current = true;
      // Capture, so a drag that leaves the canvas keeps sending moves here.
      // https://developer.mozilla.org/en-US/docs/Web/API/Element/setPointerCapture
      el.setPointerCapture(e.pointerId);
    }

    function endDrag(e: PointerEvent) {
      if (dragging.current) {
        try {
          el.releasePointerCapture(e.pointerId);
        } catch {
          /* ignore */
        }
      }
      dragging.current = false;
    }

    function onPointerMove(e: PointerEvent) {
      if (!active || !dragging.current || !controls) return;
      // https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/movementX
      const dx = e.movementX;
      // Swing the eye around world +Z through the orbit target.
      // https://threejs.org/docs/#api/en/math/Vector3.applyAxisAngle
      const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
      offset.applyAxisAngle(WORLD_Z, -dx * rotPerPx);
      camera.position.copy(controls.target).add(offset);
      controls.update();
    }

    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointerup", endDrag);
    el.addEventListener("pointercancel", endDrag);
    el.addEventListener("pointermove", onPointerMove);
    return function () {
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointerup", endDrag);
      el.removeEventListener("pointercancel", endDrag);
      el.removeEventListener("pointermove", onPointerMove);
    };
  }, [active, camera, controls, gl.domElement]);

  return null;
}
