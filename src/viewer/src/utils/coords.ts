/**
 * Conversions between this project's world and three.js's.
 * The project is Z-up, three.js is Y-up.
 */
// https://threejs.org/docs/#api/en/math/Matrix4
// https://threejs.org/docs/#api/en/math/Quaternion
// https://en.wikipedia.org/wiki/Rotation_matrix
import * as THREE from "three";
import type { UpAxis } from "../state/store";

/**
 * COLMAP cam-to-world 3x3 rotation -> three.js Matrix4 at `position`.
 * COLMAP looks down +Z with +Y down, three.js down -Z with +Y up: a 180 deg turn about X.
 *
 * https://colmap.github.io/format.html
 * https://threejs.org/docs/#api/en/math/Matrix4.set (row-major arguments)
 */
export function colmapCamToWorldMatrix(
  rotationC2W: number[][],
  position: [number, number, number],
): THREE.Matrix4 {
  const r = rotationC2W;
  const R = new THREE.Matrix4().set(
    r[0][0], r[0][1], r[0][2], 0,
    r[1][0], r[1][1], r[1][2], 0,
    r[2][0], r[2][1], r[2][2], 0,
    0,       0,       0,       1,
  );
  // Flip Y and Z basis vectors (180-deg rotation about X).
  // https://threejs.org/docs/#api/en/math/Matrix4.makeRotationX
  const flipYZ = new THREE.Matrix4().makeRotationX(Math.PI);
  R.multiply(flipYZ);
  R.setPosition(position[0], position[1], position[2]);
  return R;
}

/** The world-up vector: the user's override, or the hint from scene.json. */
export function resolveUp(
  upAxis: UpAxis,
  auto: [number, number, number],
): THREE.Vector3 {
  switch (upAxis) {
    case "+x": return new THREE.Vector3(1, 0, 0);
    case "+y": return new THREE.Vector3(0, 1, 0);
    case "+z": return new THREE.Vector3(0, 0, 1);
    case "auto":
    default:   return new THREE.Vector3(auto[0], auto[1], auto[2]).normalize();
  }
}

/** Quaternion rotating `from` onto three.js +Y. Antiparallel gets a half turn about X.
 *  https://threejs.org/docs/#api/en/math/Quaternion.setFromUnitVectors */
export function alignUpToY(from: THREE.Vector3): THREE.Quaternion {
  const src = from.clone().normalize();
  const dst = new THREE.Vector3(0, 1, 0);
  const q = new THREE.Quaternion();
  const dot = src.dot(dst);
  if (dot > 0.999999) return q;
  if (dot < -0.999999) {
    return q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
  }
  return q.setFromUnitVectors(src, dst);
}

/**
 * Frustum lines: apex rays to the far-plane corners, the far rectangle, an up tick.
 * Camera-local space: apex at the origin, looking down -Z.
 *
 * https://threejs.org/docs/#api/en/helpers/CameraHelper
 * https://threejs.org/docs/#api/en/core/BufferGeometry
 * https://threejs.org/docs/#api/en/core/BufferAttribute
 */
export function buildFrustumGeometry(
  fovYDeg: number,
  aspect: number,
  far: number,
): THREE.BufferGeometry {
  const halfH = Math.tan((fovYDeg * Math.PI) / 360) * far;
  const halfW = halfH * aspect;
  // Three.js cam looks down -Z, so the far plane sits at z = -far.
  const zFar = -far;
  const a = new THREE.Vector3(0, 0, 0);
  const tl = new THREE.Vector3(-halfW,  halfH, zFar);
  const tr = new THREE.Vector3( halfW,  halfH, zFar);
  const bl = new THREE.Vector3(-halfW, -halfH, zFar);
  const br = new THREE.Vector3( halfW, -halfH, zFar);
  // Tick above the top edge, showing camera roll.
  const tu = new THREE.Vector3(0, halfH + 0.25 * halfH, zFar);
  const lt = new THREE.Vector3(-0.4 * halfW, halfH, zFar);
  const rt = new THREE.Vector3( 0.4 * halfW, halfH, zFar);

  const segments: THREE.Vector3[] = [
    // Apex rays
    a, tl, a, tr, a, bl, a, br,
    // Far plane rectangle
    tl, tr, tr, br, br, bl, bl, tl,
    // Up tick
    lt, tu, rt, tu,
  ];
  const positions = new Float32Array(segments.length * 3);
  for (let i = 0; i < segments.length; i++) {
    const v = segments[i];
    positions[i * 3 + 0] = v.x;
    positions[i * 3 + 1] = v.y;
    positions[i * 3 + 2] = v.z;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  return g;
}
