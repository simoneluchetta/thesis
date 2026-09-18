/**
 * The hand-placed camera poses, plus the SfM poses they started from.
 * In memory only: a reload goes back to the SfM poses, so export before leaving.
 */
// https://docs.pmnd.rs/zustand
// https://en.wikipedia.org/wiki/Euler_angles
// https://threejs.org/docs/#api/en/math/Euler
// https://threejs.org/docs/#api/en/math/Matrix4
import { create } from "zustand";
import { Euler, Matrix4 } from "three";
import type { SceneCameraEntry } from "../types";

/** An [x,y,z] triple. */
export type Vec3 = [number, number, number];

const TWO_PI = 2 * Math.PI;

/** Wrap an angle into (-pi, pi], since joystick ticks just add to it. */
function wrapPi(rad: number): number {
  // Twice, so negative inputs land in range: JS % keeps the left operand's sign.
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Remainder
  const t = ((rad + Math.PI) % TWO_PI + TWO_PI) % TWO_PI;
  return t - Math.PI;
}

/** A hand-placed pose in the world frame: X-Y is the floor, Z is height.
 *  Euler order is ZXY, so the first slot is yaw. */
// https://threejs.org/docs/#api/en/math/Euler
interface PlacedPose {
  position: Vec3;
  /** ZXY Euler in radians: [yaw=e.z, pitch=e.x, roll=e.y]. */
  rotation: Vec3;
}

/** The placer tab's state: one pose per camera, plus the selection. */
interface PlacerState {
  /** Edited pose per camera, keyed by camera name like "cam01". */
  poses: Record<string, PlacedPose>;
  /** SfM starting pose per camera, used by `resetCam`. */
  initialPoses: Record<string, PlacedPose>;
  selected: string | null;

  /** Seed from scene.json. No-op if the same camera set is already seeded. */
  initFromScene: (cameras: SceneCameraEntry[]) => void;
  selectCam: (name: string | null) => void;
  /** Toggles selection: clicking the selected camera deselects it. */
  clickCam: (name: string) => void;
  setPosition: (name: string, axis: 0 | 1 | 2, value: number) => void;
  /** One write per floor drag step, so a drag doesn't spread `poses` twice. */
  setPositionXY: (name: string, x: number, y: number) => void;
  setRotation: (name: string, axis: 0 | 1 | 2, value: number) => void;
  resetCam: (name: string) => void;
  resetAll: () => void;
  exportSnapshot: () => Record<string, PlacedPose>;
  /** Apply a snapshot from `exportSnapshot`. Unknown or malformed entries are
   *  counted in the result, not applied. */
  importSnapshot: (data: unknown) => ImportResult;
}

/** What importSnapshot made of a loaded JSON file. */
interface ImportResult {
  ok: boolean;
  /** Cameras whose pose was updated. */
  matched: number;
  /** Entries skipped: unknown camera name or malformed pose. */
  ignored: number;
  /** Set when ok is false. */
  error?: string;
}

// Turns a scene.json camera into an editable position and ZXY Euler pose.
function poseFromCamera(cam: SceneCameraEntry): PlacedPose {
  const r = cam.rotation_c2w;
  const m = new Matrix4().set(
    r[0][0], r[0][1], r[0][2], 0,
    r[1][0], r[1][1], r[1][2], 0,
    r[2][0], r[2][1], r[2][2], 0,
    0, 0, 0, 1,
  );
  // ZXY decomposes c2w as Rz * Rx * Ry: e.z yaw, e.x pitch, e.y roll.
  // https://threejs.org/docs/#api/en/math/Euler.setFromRotationMatrix
  const e = new Euler().setFromRotationMatrix(m, "ZXY");
  return {
    position: [...cam.position] as Vec3,
    rotation: [wrapPi(e.z), wrapPi(e.x), wrapPi(e.y)] as Vec3,
  };
}

/** Hook for the hand-placed camera poses. */
export const usePlacer = create<PlacerState>(function (set, get) {
  return {
    poses: {},
    initialPoses: {},
    selected: null,

    initFromScene(cameras) {
      const current = get().initialPoses;
      const sameSet =
        Object.keys(current).length === cameras.length &&
        cameras.every(function (c) { return current[c.name] !== undefined; });
      if (sameSet) return;
      const initial: Record<string, PlacedPose> = {};
      const poses: Record<string, PlacedPose> = {};
      for (const cam of cameras) {
        const p = poseFromCamera(cam);
        initial[cam.name] = p;
        // Keep an existing edit if the same cam already had a pose.
        poses[cam.name] = get().poses[cam.name] ?? p;
      }
      set({ initialPoses: initial, poses });
    },

    selectCam(selected) { return set({ selected }); },

    clickCam(name) {
      return set(function (s) { return { selected: s.selected === name ? null : name }; });
    },

    setPosition(name, axis, value) {
      return set(function (s) {
        const cur = s.poses[name];
        if (!cur) return s;
        const next: Vec3 = [...cur.position] as Vec3;
        next[axis] = value;
        return { poses: { ...s.poses, [name]: { ...cur, position: next } } };
      });
    },

    setPositionXY(name, x, y) {
      return set(function (s) {
        const cur = s.poses[name];
        if (!cur) return s;
        const p = cur.position;
        if (p[0] === x && p[1] === y) return s;
        return { poses: { ...s.poses, [name]: { ...cur, position: [x, y, p[2]] } } };
      });
    },

    setRotation(name, axis, value) {
      return set(function (s) {
        const cur = s.poses[name];
        if (!cur) return s;
        const next: Vec3 = [...cur.rotation] as Vec3;
        next[axis] = wrapPi(value);
        return { poses: { ...s.poses, [name]: { ...cur, rotation: next } } };
      });
    },

    resetCam(name) {
      return set(function (s) {
        const init = s.initialPoses[name];
        if (!init) return s;
        return { poses: { ...s.poses, [name]: { ...init, position: [...init.position] as Vec3, rotation: [...init.rotation] as Vec3 } } };
      });
    },

    resetAll() {
      return set(function (s) {
        const fresh: Record<string, PlacedPose> = {};
        for (const [k, v] of Object.entries(s.initialPoses)) {
          fresh[k] = { position: [...v.position] as Vec3, rotation: [...v.rotation] as Vec3 };
        }
        return { poses: fresh };
      });
    },

    importSnapshot(data) {
      if (!data || typeof data !== "object" || Array.isArray(data)) {
        return { ok: false, matched: 0, ignored: 0, error: "Expected a JSON object at the top level." };
      }
      const entries = Object.entries(data as Record<string, unknown>);
      const known = get().initialPoses;
      const updates: Record<string, PlacedPose> = {};
      let matched = 0, ignored = 0;
      for (const [name, raw] of entries) {
        if (!(name in known) || !raw || typeof raw !== "object") {
          ignored++;
          continue;
        }
        const pose = raw as { position?: unknown; rotation?: unknown };
        const pos = pose.position;
        const rot = pose.rotation;
        // Type predicate, so the checked value narrows for the caller:
        // https://www.typescriptlang.org/docs/handbook/2/narrowing.html#using-type-predicates
        function validVec(v: unknown): v is number[] {
          return Array.isArray(v) && v.length === 3 && v.every(function (n) {
            return typeof n === "number" && Number.isFinite(n);
          });
        }
        if (!validVec(pos) || !validVec(rot)) {
          ignored++;
          continue;
        }
        updates[name] = {
          position: [pos[0], pos[1], pos[2]] as Vec3,
          rotation: [wrapPi(rot[0]), wrapPi(rot[1]), wrapPi(rot[2])] as Vec3,
        };
        matched++;
      }
      if (matched === 0) {
        return { ok: false, matched: 0, ignored, error: "No cameras from the JSON matched the current scene." };
      }
      set(function (s) { return { poses: { ...s.poses, ...updates } }; });
      return { ok: true, matched, ignored };
    },

    exportSnapshot() {
      const out: Record<string, PlacedPose> = {};
      for (const [k, v] of Object.entries(get().poses)) {
        out[k] = { position: [...v.position] as Vec3, rotation: [...v.rotation] as Vec3 };
      }
      return out;
    },
  };
});
