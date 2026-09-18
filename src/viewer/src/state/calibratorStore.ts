/**
 * The calibrator's clicks: court keypoints and off-floor tie points, per camera.
 * In memory only. `exportSnapshot` produces the JSON the Python solvers read.
 */
// https://docs.pmnd.rs/zustand
// What these clicks get solved for: https://en.wikipedia.org/wiki/Perspective-n-Point
import { create } from "zustand";

/** One click. `image_xy` is in source-sensor pixels, not thumbnail pixels.
 *  Older exports used [1920,2160] for the ceiling eyes; import rescales those. */
export interface CalibratorAnnotation {
  keypoint_id: string;
  image_xy: [number, number];
}

/** One camera's view of an off-floor landmark, clicked in every camera that sees it.
 *  `image_xy` is in sensor pixels. */
export interface TiePointObs {
  cam: string;
  image_xy: [number, number];
}

/** An extra tie sub-point the user added beyond the catalog slots. */
interface ExtraTiePoint {
  id: string;
  label: string;
  group: string;
}

/** Everything the calibrator tab holds while the user clicks. */
interface CalibratorState {
  /** All per-camera annotations. Survives page navigation. */
  annotations: Record<string, CalibratorAnnotation[]>;
  /** Selected camera name, e.g. "cam07". */
  selectedCam: string | null;
  /** Court keypoint the next click will be tagged with. */
  selectedKeypoint: string | null;

  /** Tie-point observations keyed by landmark id, at most one per camera. */
  tieObs: Record<string, TiePointObs[]>;
  /** When set, the next image click adds a tie observation, not a keypoint. */
  selectedTieId: string | null;
  extraTiePoints: ExtraTiePoint[];

  setSelectedCam: (name: string | null) => void;
  setSelectedKeypoint: (id: string | null) => void;
  setSelectedTie: (id: string | null) => void;

  /** One click per keypoint per camera: a second click on the same pair overwrites. */
  addAnnotation: (camName: string, keypointId: string, imageXy: [number, number]) => void;
  removeAnnotation: (camName: string, keypointId: string) => void;
  /** Wipe one camera's annotations, or all of them when camName is null. */
  clearAnnotations: (camName: string | null) => void;
  /** Replace the whole annotations map. Used by import. */
  setAnnotations: (next: Record<string, CalibratorAnnotation[]>) => void;

  /** One click per camera per landmark: a second click overwrites. */
  addTieObs: (tieId: string, camName: string, xy: [number, number]) => void;
  removeTieObs: (tieId: string, camName: string) => void;
  clearTieObs: (tieId: string | null) => void;
  setTieObs: (next: Record<string, TiePointObs[]>) => void;
  /** No-op if the id already exists. */
  addExtraTiePoint: (p: ExtraTiePoint) => void;

  /** Snapshot in the PnP solver's input schema. The page supplies image_size
   *  and the per-cam k1. */
  exportSnapshot: (
    imageSizes: Record<string, [number, number]>,
    k1ByCam?: Record<string, number>,
  ) => Record<string, {
    image_size: [number, number];
    division_k1: number;
    annotations: CalibratorAnnotation[];
  }>;
}

/** Hook for the calibrator's clicks. */
export const useCalibrator = create<CalibratorState>(function (set, get) {
  return {
    annotations: {},
    tieObs: {},
    selectedCam: null,
    selectedKeypoint: null,
    selectedTieId: null,
    extraTiePoints: [],

    setSelectedCam(selectedCam) { return set({ selectedCam }); },
    setSelectedKeypoint(selectedKeypoint) { return set({ selectedKeypoint }); },
    setSelectedTie(selectedTieId) { return set({ selectedTieId }); },

    addTieObs(tieId, camName, xy) {
      return set(function (s) {
        const obs = (s.tieObs[tieId] ?? []).filter(function (o) { return o.cam !== camName; });
        obs.push({ cam: camName, image_xy: xy });
        return { tieObs: { ...s.tieObs, [tieId]: obs } };
      });
    },

    removeTieObs(tieId, camName) {
      return set(function (s) {
        const obs = s.tieObs[tieId];
        if (!obs) return s;
        return { tieObs: { ...s.tieObs, [tieId]: obs.filter(function (o) { return o.cam !== camName; }) } };
      });
    },

    clearTieObs(tieId) {
      return set(function (s) {
        if (tieId == null) return { tieObs: {} };
        const next = { ...s.tieObs };
        delete next[tieId];
        return { tieObs: next };
      });
    },

    setTieObs(tieObs) { return set({ tieObs }); },

    addExtraTiePoint(p) {
      return set(function (s) {
        return (s.extraTiePoints.some(function (e) { return e.id === p.id; }) ? s : { extraTiePoints: [...s.extraTiePoints, p] });
      });
    },

    addAnnotation(camName, keypointId, imageXy) {
      return set(function (s) {
        const camAnns = s.annotations[camName] ?? [];
        const filtered = camAnns.filter(function (a) { return a.keypoint_id !== keypointId; });
        filtered.push({ keypoint_id: keypointId, image_xy: imageXy });
        return { annotations: { ...s.annotations, [camName]: filtered } };
      });
    },

    removeAnnotation(camName, keypointId) {
      return set(function (s) {
        const camAnns = s.annotations[camName];
        if (!camAnns) return s;
        const filtered = camAnns.filter(function (a) { return a.keypoint_id !== keypointId; });
        return { annotations: { ...s.annotations, [camName]: filtered } };
      });
    },

    clearAnnotations(camName) {
      return set(function (s) {
        if (camName == null) return { annotations: {} };
        const next = { ...s.annotations };
        delete next[camName];
        return { annotations: next };
      });
    },

    setAnnotations(next) { return set({ annotations: next }); },

    exportSnapshot(imageSizes, k1ByCam) {
      const out: Record<string, {
        image_size: [number, number];
        division_k1: number;
        annotations: CalibratorAnnotation[];
      }> = {};
      for (const [name, anns] of Object.entries(get().annotations)) {
        if (anns.length === 0) continue;
        const size = imageSizes[name] ?? [3840, 2160];
        out[name] = { image_size: size, division_k1: k1ByCam?.[name] ?? 0, annotations: anns };
      }
      return out;
    },
  };
});
