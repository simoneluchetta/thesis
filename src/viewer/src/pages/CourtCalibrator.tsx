/**
 * Click court keypoints on each camera thumbnail and export them for the PnP solver.
 * A homography fitted from the clicks draws the court back, so a bad click shows at once.
 * https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
 * https://en.wikipedia.org/wiki/Homography_(computer_vision)
 * https://en.wikipedia.org/wiki/Perspective-n-Point
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Euler, Matrix4, Vector3 } from "three";
import type { SceneJson, SceneCameraEntry } from "../types";
import { useCalibrator, type CalibratorAnnotation, type TiePointObs } from "../state/calibratorStore";
import { usePlacer } from "../state/placerStore";
import Compass, { type CameraPoseHint } from "../components/Compass";
import { ransacHomography, applyH, convexHull, expandPolygon, pointInConvexPoly, type Pt, type Mat3 } from "../utils/homography";
import { snapToIntersection, type LineColor } from "../utils/intersectionSnap";
import { divisionUndistort, divisionDistort, estimateDivisionK1, estimateDivisionCenterK1, projectWorld, type DivNorm } from "../utils/distortion";
import { buildCourtModel, courtDimsFromCatalog } from "../utils/courtModel";

/** Sensor size per camera when scene.json has no image_size. Must match the solve. */
const CAM_RAW_SIZE: Record<string, [number, number]> = {
  cam01: [3840, 2160], cam02: [3840, 2160], cam03: [3840, 2160], cam04: [3840, 2160],
  cam05: [3840, 2160], cam06: [3840, 2160], cam07: [3840, 2160], cam08: [3840, 2160],
  cam12: [3840, 2160], cam13: [3840, 2160],
  cam09L: [2048, 1792], cam09R: [2048, 1792], cam10L: [2048, 1800], cam10R: [2048, 1800],
  cam11L: [2048, 1792], cam11R: [2048, 1792],
};

/** Off-floor landmarks, one group per object. Scale comes from the court, not from these.
 *  https://en.wikipedia.org/wiki/Bundle_adjustment */
interface TieSubPoint { id: string; label: string }
/** One group of off-floor landmarks, such as the net or the bench. */
interface TieGroup {
  key: string;
  title: string;
  color: string; // accent + on-image marker colour
  hint: string;
  points: TieSubPoint[];
}

// The landmarks offered in the tie-point panel.
const TIE_GROUPS: TieGroup[] = [
  {
    key: "hoop_E", title: "Hoop East (perimeter cams)", color: "#ffae42",
    hint: "EAST hoop (X≈+12.8 m). Corners are named by WORLD coordinates, NOT your view — so every camera clicks the SAME physical corner regardless of which side it films from. TOP/BOTTOM = upper/lower board edge; +Y = NORTH, −Y = SOUTH (check the compass). Click view-stable features only (board corners + rim front/back), not the round rim silhouette.",
    points: [
      { id: "hoop_E/board_TR", label: "Backboard TOP · +Y (North) side" },
      { id: "hoop_E/board_TL", label: "Backboard TOP · −Y (South) side" },
      { id: "hoop_E/board_BR", label: "Backboard BOTTOM · +Y (North) side" },
      { id: "hoop_E/board_BL", label: "Backboard BOTTOM · −Y (South) side" },
      { id: "hoop_E/rim_front", label: "Rim tip (toward court centre, −X)" },
      { id: "hoop_E/rim_back", label: "Rim base (at the backboard, +X)" },
    ],
  },
  {
    key: "hoop_W", title: "Hoop West (perimeter cams)", color: "#ffae42",
    hint: "WEST hoop (X≈−12.8 m). Same world-coordinate naming as Hoop East: TOP/BOTTOM = upper/lower board edge; +Y = NORTH, −Y = SOUTH (check the compass) — NOT camera-relative left/right. Click view-stable features only (board corners + rim front/back), not the round rim silhouette.",
    points: [
      { id: "hoop_W/board_TR", label: "Backboard TOP · +Y (North) side" },
      { id: "hoop_W/board_TL", label: "Backboard TOP · −Y (South) side" },
      { id: "hoop_W/board_BR", label: "Backboard BOTTOM · +Y (North) side" },
      { id: "hoop_W/board_BL", label: "Backboard BOTTOM · −Y (South) side" },
      { id: "hoop_W/rim_front", label: "Rim tip (toward court centre, +X)" },
      { id: "hoop_W/rim_back", label: "Rim base (at the backboard, −X)" },
    ],
  },
  {
    key: "vbnet", title: "Volleyball net on the stands", color: "#3ff",
    hint: "The volleyball net LAID ON THE STANDS (NOT the court). Pick crisp corners/junctions (post tops, a net corner, a tensioner) and click each in cam07 + the ceiling eyes — same physical point across cameras.",
    points: [
      { id: "vbnet/p1", label: "Corner / junction 1" },
      { id: "vbnet/p2", label: "Corner / junction 2" },
      { id: "vbnet/p3", label: "Corner / junction 3" },
      { id: "vbnet/p4", label: "Corner / junction 4" },
    ],
  },
  {
    key: "bench", title: "Bench in the middle", color: "#3ff",
    hint: "The bench in the MIDDLE, visible from cam13 + the ceiling eyes. Click crisp corners/legs (backrest top corners, leg-to-floor contacts) — same physical point across cameras.",
    points: [
      { id: "bench/fL", label: "Front-left corner / leg" },
      { id: "bench/fR", label: "Front-right corner / leg" },
      { id: "bench/bL", label: "Back-left corner / leg" },
      { id: "bench/bR", label: "Back-right corner / leg" },
    ],
  },
];

/** Label, colour and hint shown for one tie sub-point. */
interface TiePointMeta { label: string; color: string; hint: string }
/** id -> metadata for the predefined sub-points. Extras merge in at runtime. */
const TIE_POINT_INFO: Record<string, TiePointMeta> = {};
for (const g of TIE_GROUPS) {
  for (const p of g.points) {
    TIE_POINT_INFO[p.id] = { label: p.label, color: g.color, hint: g.hint };
  }
}

/** Residual -> colour. Thresholds are a fraction of the image width. */
function residualColor(resPx: number, imgWidth: number): string {
  const f = resPx / imgWidth;
  if (f < 0.0025) return "#5f5";   // green: ~cm-accurate
  if (f < 0.006) return "#fd6";    // amber: borderline
  return "#f66";                   // red: re-click this one
}

// Scratch for the ZXY Euler -> forward direction, so nothing allocates per render.
// https://threejs.org/docs/#api/en/math/Euler
// https://threejs.org/docs/#api/en/math/Matrix4
// https://threejs.org/docs/#api/en/math/Vector3
const FWD_LOCAL = new Vector3(0, 0, 1);
const SCRATCH_EULER = new Euler(0, 0, 0, "ZXY");
const SCRATCH_MATRIX = new Matrix4();
const SCRATCH_FWD = new Vector3();

/** One keypoint from public/court_keypoints.json, with its world position. */
interface CourtKeypoint {
  id: string;
  court: string;
  world: [number, number, number];
  label: string;
}

/** public/court_keypoints.json: the court dimensions plus every keypoint. */
interface KeypointsCatalog {
  courts: Record<string, unknown>;
  keypoints: CourtKeypoint[];
}

interface Props {
  scene: SceneJson;
}

/** The Court calibrator tab: click court keypoints on each camera and export them for the PnP solve. */
export default function CourtCalibrator({ scene }: Props) {
  // One zustand selector per field: a click only re-renders the parts that read it.
  // https://zustand.docs.pmnd.rs/
  const annotations = useCalibrator(function (s) { return s.annotations; });
  const selectedCam = useCalibrator(function (s) { return s.selectedCam; });
  const selectedKeypoint = useCalibrator(function (s) { return s.selectedKeypoint; });
  const setSelectedCam = useCalibrator(function (s) { return s.setSelectedCam; });
  const setSelectedKeypoint = useCalibrator(function (s) { return s.setSelectedKeypoint; });
  const tieObs = useCalibrator(function (s) { return s.tieObs; });
  const selectedTieId = useCalibrator(function (s) { return s.selectedTieId; });
  const setSelectedTie = useCalibrator(function (s) { return s.setSelectedTie; });
  const addTieObs = useCalibrator(function (s) { return s.addTieObs; });
  const removeTieObs = useCalibrator(function (s) { return s.removeTieObs; });
  const clearTieObs = useCalibrator(function (s) { return s.clearTieObs; });
  const setTieObs = useCalibrator(function (s) { return s.setTieObs; });
  const extraTiePoints = useCalibrator(function (s) { return s.extraTiePoints; });
  const addExtraTiePoint = useCalibrator(function (s) { return s.addExtraTiePoint; });
  const addAnnotation = useCalibrator(function (s) { return s.addAnnotation; });
  const removeAnnotation = useCalibrator(function (s) { return s.removeAnnotation; });
  const clearAnnotations = useCalibrator(function (s) { return s.clearAnnotations; });
  const setAnnotations = useCalibrator(function (s) { return s.setAnnotations; });
  const exportSnapshot = useCalibrator(function (s) { return s.exportSnapshot; });
  /** Placer-edited poses, so the compass shows where this camera was placed. */
  const placerPoses = usePlacer(function (s) { return s.poses; });

  const [catalog, setCatalog] = useState<KeypointsCatalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  // public/court_overlay.json, through the full PnP + Brown model. Points are sensor px.
  const [calibOverlay, setCalibOverlay] = useState<Record<string, {
    image_size: [number, number]; rms_px: number;
    polylines: { court: string; pts: [number, number][] }[];
  }> | null>(null);
  const [imgSize, setImgSize] = useState<[number, number] | null>(null);
  /** Thumbnail zoom. No extra pixels, just easier aiming; the wrapper scrolls.
   *  https://developer.mozilla.org/en-US/docs/Web/CSS/overflow */
  const [zoom, setZoom] = useState(1.0);
  /** Compass overlay size in px. */
  const [compassSize, setCompassSize] = useState(220);
  /** Compass overlay top in px, measured so it sits below the header. */
  const [overlayTop, setOverlayTop] = useState(52);
  /** Optional public/reclick_hints.json: expected_xy is where a flagged click should land. */
  type ReclickHint = { expected_xy: [number, number]; err_px: number; kind: string; note?: string };
  const [reclickHints, setReclickHints] = useState<Record<string, Record<string, ReclickHint>>>({});
  /** Snap each click to the nearest court-line intersection. */
  const [snapEnabled, setSnapEnabled] = useState(false);
  const [lastSnap, setLastSnap] = useState<number | null>(null);
  /** Project the court line model onto the image, distortion included. */
  const [showOverlay, setShowOverlay] = useState(false);
  // Clip the overlay to the convex hull of the clicks; off draws the whole court.
  const [clipToCoverage, setClipToCoverage] = useState(false);
  // Draw court_overlay.json for the selected cam when the file has it.
  const [showCalibrated, setShowCalibrated] = useState(false);
  /** Also snap clicks to the projected court-line model. */
  const [snapProjected, setSnapProjected] = useState(false);
  /** Magnifier loupe that follows the cursor over the image. */
  const [lensEnabled, setLensEnabled] = useState(false);
  /** Cursor position and image size in rendered px; null when the cursor is off-image. */
  const [lensPos, setLensPos] = useState<{ xR: number; yR: number; w: number; h: number } | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const tieFileInputRef = useRef<HTMLInputElement | null>(null);
  /** RGBA pixels of the current thumbnail, for colour-aware snapping. */
  const imgDataRef = useRef<{ rgba: Uint8ClampedArray; w: number; h: number } | null>(null);
  /** Per-camera division-model k1, debounced by a signature and kept for export. */
  const k1CacheRef = useRef<Record<string, { k1: number; sig: string; cx?: number; cy?: number }>>({});
  /** Scrollable viewport around the zoomable image. */
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  // Keep the image centred in the scroller; the rAF reads scrollWidth after layout.
  // https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver
  // https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame
  // https://developer.mozilla.org/en-US/docs/Web/API/Element/scrollLeft
  useEffect(function () {
    const sc = scrollerRef.current;
    if (!sc) return;
    let rafId: number | null = null;
    const recenter = function () {
      if (rafId != null) return;
      rafId = requestAnimationFrame(function () {
        rafId = null;
        const left = (sc.scrollWidth - sc.clientWidth) / 2;
        const top = (sc.scrollHeight - sc.clientHeight) / 2;
        if (left > 0) sc.scrollLeft = left;
        if (top > 0) sc.scrollTop = top;
      });
    };
    recenter();
    const ro = new ResizeObserver(recenter);
    ro.observe(sc);
    const img = imgRef.current;
    if (img) ro.observe(img);
    return function () {
      if (rafId != null) cancelAnimationFrame(rafId);
      ro.disconnect();
    };
    // Re-run on cam change to re-attach the observer to the new IMG element.
  }, [selectedCam, zoom]);

  // Measure where the image area starts, before paint, so the overlay clears the header.
  // https://react.dev/reference/react/useLayoutEffect
  useLayoutEffect(function () {
    const sc = scrollerRef.current;
    if (!sc) return;
    const measure = function () { setOverlayTop(sc.offsetTop + 8); };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(sc);
    if (sc.parentElement) ro.observe(sc.parentElement);
    window.addEventListener("resize", measure);
    return function () { ro.disconnect(); window.removeEventListener("resize", measure); };
  }, [selectedCam, zoom, imgSize]);

  // Load the keypoint catalog once. Everything here is served from public/.
  // https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
  useEffect(function () {
    fetch("/court_keypoints.json")
      .then(function (r) {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(function (data: KeypointsCatalog) { return setCatalog(data); })
      .catch(function (e: Error) { return setCatalogError(e.message); });
  }, []);

  // Load the accurate court-PnP overlay if present (optional; silent if missing).
  useEffect(function () {
    fetch("/court_overlay.json")
      .then(function (r) { return (r.ok ? r.json() : null); })
      .then(function (data) { if (data) setCalibOverlay(data); })
      .catch(function () { /* no calibrated overlay yet, the live one still works */ });
  }, []);

  // Load re-click hints if present (optional; silent if no flagged tie points).
  useEffect(function () {
    fetch("/reclick_hints.json")
      .then(function (r) { return (r.ok ? r.json() : null); })
      .then(function (data) { if (data) setReclickHints(data); })
      .catch(function () { /* no hints file, nothing flagged */ });
  }, []);

  // Nothing selected yet (first visit): start on the first camera of the scene.
  useEffect(function () {
    if (selectedCam) return;
    if (scene.cameras.length > 0) setSelectedCam(scene.cameras[0].name);
  }, [scene.cameras, selectedCam, setSelectedCam]);

  const selectedCamObj: SceneCameraEntry | null = useMemo(
    function () {
      return scene.cameras.find(function (c) { return c.name === selectedCam; }) ?? null;
    },
    [scene.cameras, selectedCam],
  );

  /** Camera pose flattened onto X-Y for the compass. Local +Z rotated = where it looks.
   *  https://en.wikipedia.org/wiki/Euler_angles
   *  https://en.wikipedia.org/wiki/Rotation_matrix */
  const cameraPoseHint: CameraPoseHint | null = useMemo(function () {
    if (!selectedCamObj) return null;
    const placed = placerPoses[selectedCamObj.name];
    if (placed) {
      SCRATCH_EULER.set(placed.rotation[1], placed.rotation[2], placed.rotation[0], "ZXY");
      SCRATCH_MATRIX.makeRotationFromEuler(SCRATCH_EULER);
    } else {
      // Placer not initialised for this scene: use the c2w matrix from scene.json.
      const r = selectedCamObj.rotation_c2w;
      SCRATCH_MATRIX.set(
        r[0][0], r[0][1], r[0][2], 0,
        r[1][0], r[1][1], r[1][2], 0,
        r[2][0], r[2][1], r[2][2], 0,
        0, 0, 0, 1,
      );
    }
    SCRATCH_FWD.copy(FWD_LOCAL).applyMatrix4(SCRATCH_MATRIX);
    const pos = placed ? placed.position : selectedCamObj.position;
    return {
      position: [pos[0], pos[1]],
      forward: [SCRATCH_FWD.x, SCRATCH_FWD.y],
      label: selectedCamObj.name,
    };
  }, [selectedCamObj, placerPoses]);

  const camAnnotations: CalibratorAnnotation[] = useMemo(
    function () { return (selectedCam ? annotations[selectedCam] ?? [] : []); },
    [annotations, selectedCam],
  );

  /** Runtime sub-point lookup = predefined catalog + user-added extras. */
  const tieInfoById = useMemo(function () {
    const m: Record<string, TiePointMeta> = { ...TIE_POINT_INFO };
    for (const e of extraTiePoints) {
      const g = TIE_GROUPS.find(function (gr) { return gr.key === e.group; });
      m[e.id] = { label: e.label, color: g?.color ?? "#3ff", hint: g?.hint ?? "" };
    }
    return m;
  }, [extraTiePoints]);

  /** The sub-points of a group, predefined first then any user extras. */
  function pointsForGroup(g: TieGroup): TieSubPoint[] {
    return [
      ...g.points,
      ...extraTiePoints.filter(function (e) { return e.group === g.key; }).map(function (e) {
        return { id: e.id, label: e.label };
      }),
    ];
  }

  /** Tie-point observations for the current cam (one per sub-point, at most). */
  const camTieObs = useMemo(function () {
    const out: { tieId: string; label: string; color: string; xy: [number, number] }[] = [];
    if (!selectedCam) return out;
    for (const [id, info] of Object.entries(tieInfoById)) {
      const obs = (tieObs[id] ?? []).find(function (o) { return o.cam === selectedCam; });
      if (obs) out.push({ tieId: id, label: info.label, color: info.color, xy: obs.image_xy });
    }
    return out;
  }, [tieObs, selectedCam, tieInfoById]);

  /** Pick a tie sub-point to place (clears the court-keypoint selection). */
  function pickTie(id: string | null) {
    setSelectedTie(id);
    if (id) setSelectedKeypoint(null);
  }

  /** Append a fresh extra sub-point to a group and select it. */
  function addExtraToGroup(g: TieGroup) {
    const n = extraTiePoints.filter(function (e) { return e.group === g.key; }).length + 1;
    const id = `${g.key}/extra${n}`;
    addExtraTiePoint({ id, label: `Extra point ${n}`, group: g.key });
    pickTie(id);
  }
  /** Pick a court keypoint (clears the tie-point selection). */
  function pickKeypoint(id: string) {
    setSelectedKeypoint(id);
    setSelectedTie(null);
  }

  /** Snap target colour: basketball lines are white, the inner volleyball lines blue. */
  const lineColor: LineColor = useMemo(function () {
    if (!catalog || !selectedKeypoint) return "any";
    const kp = catalog.keypoints.find(function (k) { return k.id === selectedKeypoint; });
    const court =
      kp?.court ??
      (selectedKeypoint.startsWith("bb_") ? "basketball"
        : selectedKeypoint.startsWith("vb_") ? "volleyball" : "");
    return court === "basketball" ? "white" : court === "volleyball" ? "blue" : "any";
  }, [catalog, selectedKeypoint]);

  // Group keypoints by court for the picker.
  const keypointsByCourt = useMemo(function () {
    const groups: Record<string, CourtKeypoint[]> = {};
    if (catalog) {
      for (const kp of catalog.keypoints) {
        (groups[kp.court] ??= []).push(kp);
      }
    }
    return groups;
  }, [catalog]);

  // Same data as the picker, flattened to what Compass wants (id + world).
  const basketballKeypoints = useMemo(
    function () {
      return (keypointsByCourt["basketball"] ?? []).map(function (kp) { return { id: kp.id, world: kp.world }; });
    },
    [keypointsByCourt],
  );
  const volleyballKeypoints = useMemo(
    function () {
      return (keypointsByCourt["volleyball"] ?? []).map(function (kp) { return { id: kp.id, world: kp.world }; });
    },
    [keypointsByCourt],
  );

  // True sensor size [W,H]. Annotations are stored in this space, not in thumbnail px.
  function camOriginalSize(cam: SceneCameraEntry): [number, number] {
    return cam.image_size ?? CAM_RAW_SIZE[cam.name] ?? [3840, 2160];
  }

  // Division-model normalisation: centre of the sensor, scale = W/2.
  // https://en.wikipedia.org/wiki/Distortion_(optics)
  // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
  function sensorSizeOf(cam: SceneCameraEntry): [number, number] {
    return camOriginalSize(cam);
  }
  function sensorNorm(cam: SceneCameraEntry): DivNorm {
    const [w, h] = sensorSizeOf(cam);
    return { cx: w / 2, cy: h / 2, s: w / 2 };
  }

  // Court line model (world floor polylines) for the projected overlay.
  const courtModel = useMemo(
    function () { return (catalog ? buildCourtModel(courtDimsFromCatalog(catalog.courts)) : []); },
    [catalog],
  );

  function onImgLoad() {
    const el = imgRef.current;
    if (!el) return;
    setImgSize([el.naturalWidth, el.naturalHeight]);
    // Rasterise the thumbnail so the snap code can read pixels; getImageData throws if tainted.
    // https://stackoverflow.com/questions/22097747/how-to-fix-getimagedata-error-the-canvas-has-been-tainted-by-cross-origin-data
    // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/getImageData
    // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage
    // https://developer.mozilla.org/en-US/docs/Web/API/HTMLImageElement/naturalWidth
    try {
      const cv = document.createElement("canvas");
      cv.width = el.naturalWidth;
      cv.height = el.naturalHeight;
      const ctx = cv.getContext("2d", { willReadFrequently: true });
      if (ctx) {
        ctx.drawImage(el, 0, 0);
        const data = ctx.getImageData(0, 0, cv.width, cv.height);
        imgDataRef.current = { rgba: data.data, w: cv.width, h: cv.height };
      }
    } catch {
      imgDataRef.current = null; // tainted canvas / read failure -> snapping off
    }
    setLastSnap(null);
  }

  function onImgClick(e: React.MouseEvent<HTMLImageElement>) {
    if (!selectedCam || !selectedCamObj || !imgSize) return;
    // The image's on-screen box turns a page click into a position inside the image.
    // https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect
    const rect = e.currentTarget.getBoundingClientRect();
    // Click position in the rendered image's pixel coordinates.
    const click_xRendered = e.clientX - rect.left;
    const click_yRendered = e.clientY - rect.top;
    // Scale from rendered px to natural thumbnail px.
    let xThumb = (click_xRendered / rect.width) * imgSize[0];
    let yThumb = (click_yRendered / rect.height) * imgSize[1];
    // Tie-point mode: one observation per (landmark, cam), no snapping.
    if (selectedTieId) {
      const [tw, th] = camOriginalSize(selectedCamObj);
      addTieObs(selectedTieId, selectedCam,
        [(xThumb / imgSize[0]) * tw, (yThumb / imgSize[1]) * th]);
      return;
    }
    if (!selectedKeypoint) return;
    // Snap to the nearest court-line intersection, in thumbnail space.
    if (snapEnabled && imgDataRef.current) {
      const { rgba, w, h } = imgDataRef.current;
      const snap = snapToIntersection(rgba, w, h, xThumb, yThumb, lineColor);
      if (snap.snapped) {
        xThumb = snap.xy[0];
        yThumb = snap.xy[1];
        // report the move in original-image px so it's comparable to residuals
        setLastSnap((snap.delta / imgSize[0]) * camOriginalSize(selectedCamObj)[0]);
      } else {
        setLastSnap(null);
      }
    }
    // Snap to the projected court model, a fallback where the paint is faint or hidden.
    if (snapProjected && reproj && reproj.status === "ok" && reproj.H) {
      const H = reproj.H as Mat3; const { norm, k1, sw, sh } = reproj;
      const tol = imgSize[0] * 0.012;
      let bestD = tol, bx = xThumb, by = yThumb;
      for (const pl of courtModel) {
        let prev: [number, number] | null = null;
        for (const w of pl.pts) {
          const qD = projectWorld(w, H, norm, k1);
          if (!qD) { prev = null; continue; }
          const tx = (qD[0] / sw) * imgSize[0], ty = (qD[1] / sh) * imgSize[1];
          if (prev) {
            const [d, fx, fy] = pointSegFoot(xThumb, yThumb, prev[0], prev[1], tx, ty);
            if (d < bestD) { bestD = d; bx = fx; by = fy; }
          }
          prev = [tx, ty];
        }
      }
      if (bestD < tol) { xThumb = bx; yThumb = by; }
    }
    // Thumbnail -> original-image px (the space annotations are stored in).
    const [origW, origH] = camOriginalSize(selectedCamObj);
    const xOrig = (xThumb / imgSize[0]) * origW;
    const yOrig = (yThumb / imgSize[1]) * origH;
    addAnnotation(selectedCam, selectedKeypoint, [xOrig, yOrig]);
  }

  function onImgMouseMove(e: React.MouseEvent<HTMLImageElement>) {
    if (!lensEnabled) return;
    const rect = e.currentTarget.getBoundingClientRect();
    setLensPos({
      xR: e.clientX - rect.left,
      yR: e.clientY - rect.top,
      w: rect.width,
      h: rect.height,
    });
  }
  function onImgMouseLeave() { return setLensPos(null); }

  /** Live fit: estimate k1, undistort the clicks, fit H, report residuals in original px.
   *  https://react.dev/reference/react/useMemo
   *  https://en.wikipedia.org/wiki/Homography_(computer_vision)
   *  https://en.wikipedia.org/wiki/Distortion_(optics)
   *  https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html */
  const reproj = useMemo(function (): {
    status: "ok" | "few" | "failed";
    perKp: Record<string, number>;
    origW: number;
    rms: number;
    nInliers: number;
    nFloor: number;
    k1: number;
    H: Mat3 | null;
    norm: DivNorm;
    sw: number;
    sh: number;
    /** Convex hull of the clicked floor points, world XY plus a margin.
     *  https://en.wikipedia.org/wiki/Convex_hull */
    hull: Pt[] | null;
  } | null {
    if (!catalog || !selectedCamObj) return null;
    const worldById = new Map(catalog.keypoints.map(function (k) { return [k.id, k.world]; }));
    const items = camAnnotations
      .map(function (a) { return { id: a.keypoint_id, w: worldById.get(a.keypoint_id), xy: a.image_xy }; })
      .filter(function (x): x is { id: string; w: [number, number, number]; xy: [number, number] } {
        return !!x.w && Math.abs(x.w[2]) < 0.05;
      }); // floor (z=0) keypoints only
    const [origW, origH] = camOriginalSize(selectedCamObj);
    const [sw, sh] = sensorSizeOf(selectedCamObj);
    let norm = sensorNorm(selectedCamObj);
    function toSensor(p: Pt): Pt { return [(p[0] * sw) / origW, (p[1] * sh) / origH]; }
    const base = { perKp: {} as Record<string, number>, origW, rms: NaN, nInliers: 0, nFloor: items.length, k1: 0, H: null, norm, sw, sh, hull: null as Pt[] | null };
    if (items.length < 4) return { ...base, status: "few" as const };
    const src: Pt[] = items.map(function (x) { return [x.w[0], x.w[1]]; });
    const dstSensor: Pt[] = items.map(function (x) { return toSensor(x.xy); });

    // Estimate k1 (gated on enough, well-spread points; debounced per cam).
    // https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    // https://stackoverflow.com/questions/51895602/opencv-are-lens-distortion-coefficients-inverted-for-projectpoints
    let k1 = 0;
    const radii = dstSensor.map(function (p) {
      return Math.hypot((p[0] - norm.cx) / norm.s, (p[1] - norm.cy) / norm.s);
    });
    const spread = Math.max(...radii) - Math.min(...radii);
    if (items.length >= 6 && spread >= 0.25) {
      const sig = `${items.length}:${Math.round(spread * 20)}`;
      const cached = k1CacheRef.current[selectedCamObj.name];
      if (cached && cached.sig === sig) {
        k1 = cached.k1;
        // Restore a freed (off-centre) principal point if one was estimated.
        // https://en.wikipedia.org/wiki/Pinhole_camera_model
        if (cached.cx !== undefined && cached.cy !== undefined) norm = { cx: cached.cx, cy: cached.cy, s: norm.s };
      } else if (selectedCamObj.is_stereo && items.length >= 7) {
        // A ceiling eye is half a shared optic, so solve its centre instead of pinning it.
        const est = estimateDivisionCenterK1(src, dstSensor, norm, sw * 0.006, {});
        k1 = est.k1;
        norm = est.norm;
        k1CacheRef.current[selectedCamObj.name] = { k1, sig, cx: norm.cx, cy: norm.cy };
      } else {
        const est = estimateDivisionK1(src, dstSensor, norm, sw * 0.006, { warmStart: cached?.k1 });
        k1 = est.k1;
        k1CacheRef.current[selectedCamObj.name] = { k1, sig };
      }
    }

    // RANSAC fits on the undistorted clicks and drops the outliers.
    // https://en.wikipedia.org/wiki/Random_sample_consensus
    const dstU = dstSensor.map(function (p) { return divisionUndistort(p, norm, k1); });
    const fit = ransacHomography(src, dstU, sw * 0.006);
    if (!fit) return { ...base, status: "failed" as const, k1 };

    // Residuals in original (distorted) px so existing colour thresholds hold.
    const perKp: Record<string, number> = {};
    items.forEach(function (x, i) {
      const qD = divisionDistort(applyH(fit.H, src[i]), norm, k1) ?? applyH(fit.H, src[i]);
      const qOrig: Pt = [(qD[0] * origW) / sw, (qD[1] * origH) / sh];
      perKp[x.id] = Math.hypot(qOrig[0] - x.xy[0], qOrig[1] - x.xy[1]);
    });
    // RMS over the inliers only.
    // https://en.wikipedia.org/wiki/Root_mean_square
    const inl = Object.values(perKp).filter(function (r) { return r < origW * 0.006; });
    const rms = inl.length ? Math.sqrt(inl.reduce(function (a, r) { return a + r * r; }, 0) / inl.length) : NaN;
    // Hull used by the "clip to clicked area" overlay mode.
    const hull = src.length >= 3 ? expandPolygon(convexHull(src), 1.12, 0.8) : null;
    return { status: "ok", perKp, origW, rms, nInliers: inl.length, nFloor: items.length, k1, H: fit.H, norm, sw, sh, hull };
  }, [catalog, camAnnotations, selectedCamObj]);

  const thumbUrl = selectedCamObj?.thumb ? `/${selectedCamObj.thumb}` : null;

  // Camera summary for the left panel: name + #annotations.
  const camRows = useMemo(
    function () {
      return scene.cameras.map(function (c) {
        return {
          name: c.name,
          count: annotations[c.name]?.length ?? 0,
          isStereo: !!c.is_stereo,
        };
      });
    },
    [scene.cameras, annotations],
  );

  const totalAnnotations = useMemo(
    function () {
      return Object.values(annotations).reduce(function (sum, a) { return sum + a.length; }, 0);
    },
    [annotations],
  );

  function onExport() {
    const sizes: Record<string, [number, number]> = {};
    for (const c of scene.cameras) sizes[c.name] = camOriginalSize(c);
    // Cached k1 per cam, plus the live value for the selected cam.
    const k1ByCam: Record<string, number> = {};
    for (const [name, v] of Object.entries(k1CacheRef.current)) k1ByCam[name] = v.k1;
    if (selectedCam && reproj?.status === "ok") k1ByCam[selectedCam] = reproj.k1;
    downloadJson(exportSnapshot(sizes, k1ByCam), "court_annotations.json");
  }

  /** Total tie-point clicks across all landmarks + cams (for the section header). */
  const totalTieClicks = useMemo(
    function () {
      return Object.values(tieObs).reduce(function (sum, o) { return sum + o.length; }, 0);
    },
    [tieObs],
  );

  /** Export the tie points to court_tiepoints.json for the bundle adjustment. */
  function onExportTie() {
    const usedCams = new Set<string>();
    for (const obs of Object.values(tieObs)) for (const o of obs) usedCams.add(o.cam);
    const image_sizes: Record<string, [number, number]> = {};
    for (const c of scene.cameras) if (usedCams.has(c.name)) image_sizes[c.name] = camOriginalSize(c);
    const tie_points = TIE_GROUPS.flatMap(function (g) {
      return pointsForGroup(g)
        .map(function (p) {
          return {
            id: p.id,
            label: p.label,
            group: g.key,
            world: null as [number, number, number] | null, // all free; scale comes from the floor
            observations: (tieObs[p.id] ?? []).map(function (o) { return { cam: o.cam, image_xy: o.image_xy }; }),
          };
        })
        .filter(function (t) { return t.observations.length > 0; });
    },
    );
    downloadJson(
      { tie_points, image_sizes, intrinsics_ref: "diagnostics/court_pnp_cameras_redfox_repaired.json" },
      "court_tiepoints.json",
    );
  }

  // The file picker is a hidden <input type="file">; File.text() reads it as a string.
  // https://developer.mozilla.org/en-US/docs/Web/API/HTMLInputElement/files
  // https://developer.mozilla.org/en-US/docs/Web/API/Blob/text
  async function onImportTieFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (tieFileInputRef.current) tieFileInputRef.current.value = "";
    if (!file) return;
    try {
      const raw = JSON.parse(await file.text());
      const next: Record<string, TiePointObs[]> = {};
      const tps = (raw.tie_points ?? []) as Array<{ id: string; label?: string; group?: string; observations?: TiePointObs[] }>;
      for (const t of tps) {
        if (!t.id) continue;
        next[t.id] = (t.observations ?? []).map(function (o) { return { cam: o.cam, image_xy: o.image_xy }; });
        // Recreate any non-catalog sub-points as extras so they render + round-trip.
        if (!TIE_POINT_INFO[t.id] && t.group) {
          addExtraTiePoint({ id: t.id, label: t.label ?? t.id, group: t.group });
        }
      }
      if (Object.keys(next).length === 0) {
        window.alert("No usable tie_points entries in this JSON.");
        return;
      }
      setTieObs(next);
    } catch (err) {
      window.alert(`Couldn't parse tie-points JSON: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  function onImportClick() { return fileInputRef.current?.click(); }
  async function onImportFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    // Clear the input, or picking the same file twice fires no `change` event.
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (!file) return;
    try {
      const raw = JSON.parse(await file.text());
      // Accept the export schema or a bare {cam: [annotations]} map.
      const next: Record<string, CalibratorAnnotation[]> = {};
      let matched = 0;
      for (const [name, entry] of Object.entries(raw)) {
        if (Array.isArray(entry)) {
          next[name] = entry as CalibratorAnnotation[];
          matched++;
        } else if (entry && typeof entry === "object" && Array.isArray((entry as { annotations?: unknown }).annotations)) {
          // Older exports used a [1920,2160] stereo-eye space; rescale if image_size differs.
          const cam = scene.cameras.find(function (c) { return c.name === name; });
          const trueSize = cam ? camOriginalSize(cam) : null;
          const fileSize = (entry as { image_size?: [number, number] }).image_size;
          const needRescale = !!fileSize && !!trueSize &&
            (Math.abs(fileSize[0] - trueSize[0]) > 0.5 || Math.abs(fileSize[1] - trueSize[1]) > 0.5);
          const sx = needRescale ? trueSize![0] / fileSize![0] : 1;
          const sy = needRescale ? trueSize![1] / fileSize![1] : 1;
          const anns = (entry as { annotations: CalibratorAnnotation[] }).annotations;
          next[name] = needRescale
            ? anns.map(function (a) {
              return { ...a, image_xy: [a.image_xy[0] * sx, a.image_xy[1] * sy] as [number, number] };
            })
            : anns;
          matched++;
        }
      }
      if (matched === 0) {
        window.alert("No usable annotation entries in this JSON.");
        return;
      }
      setAnnotations(next);
    } catch (err) {
      window.alert(`Couldn't parse JSON: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  if (catalogError) {
    return (
      <div style={{ padding: 24, color: "#f88" }}>
        Failed to load /court_keypoints.json: {catalogError}
      </div>
    );
  }
  if (!catalog) {
    return <div style={{ padding: 24, color: "#cccccc" }}>Loading court keypoints...</div>;
  }

  return (
    <div
      style={{
        display: "grid",
        // minmax(0, 1fr) lets the middle column shrink below its content.
        // https://developer.mozilla.org/en-US/docs/Web/CSS/minmax
        gridTemplateColumns: "280px minmax(0, 1fr) 280px",
        gap: 8,
        width: "100%",
        height: "100%",
        padding: 8,
        boxSizing: "border-box",
        background: "#1f1f1f",
        color: "#cccccc",
      }}
    >
      {/* Left: camera picker */}
      <div
        style={{
          background: "#181818",
          border: "1px solid #2b2b2b",
          borderRadius: 4,
          overflowY: "auto",
          padding: 6,
        }}
      >
        <div style={{ fontSize: 12, color: "#9d9d9d", marginBottom: 6 }}>
          Cameras ({camRows.length})
        </div>
        {camRows.map(function (row) {
          const sel = row.name === selectedCam;
          return (
            <button
              key={row.name}
              onClick={function () { setSelectedCam(row.name); }}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                width: "100%",
                padding: "4px 8px",
                margin: "2px 0",
                background: sel ? "#fc6" : "#2b2b2b",
                color: sel ? "#1f1f1f" : row.isStereo ? "#f88" : "#cccccc",
                border: `1px solid ${sel ? "#fc6" : "#3c3c3c"}`,
                borderRadius: 3,
                cursor: "pointer",
                fontSize: 12,
                fontWeight: sel ? 600 : 400,
              }}
            >
              <span>{row.name}</span>
              <span style={{ fontSize: 11, opacity: 0.8 }}>{row.count}</span>
            </button>
          );
        })}
        <div style={{ marginTop: 12, paddingTop: 8, borderTop: "1px solid #2b2b2b" }}>
          <div style={{ fontSize: 11, color: "#6d6d6d", marginBottom: 4 }}>
            Total: {totalAnnotations} clicks
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <button onClick={onImportClick} style={btnStyle}>Import JSON</button>
            <button onClick={onExport} style={btnStyle}>Export JSON</button>
            <button
              onClick={function () {
                if (selectedCam && window.confirm(`Clear all clicks for ${selectedCam}?`)) {
                  clearAnnotations(selectedCam);
                }
              }}
              style={{ ...btnStyle, color: "#f88" }}
            >
              Clear this cam
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/json,.json"
              onChange={onImportFile}
              style={{ display: "none" }}
            />
          </div>
        </div>

        {/* Legend: what the clicked-point colours mean. */}
        <div style={{ marginTop: 12, paddingTop: 8, borderTop: "1px solid #2b2b2b" }}>
          <div style={{ fontSize: 11, color: "#9d9d9d", marginBottom: 6 }}>Point colours</div>
          {([
            ["#5f5", "Accurate — low reprojection error"],
            ["#fd6", "Borderline — double-check this click"],
            ["#f66", "Off — re-click (often the wrong court line)"],
            ["#5cf", "Not checked yet (need ≥4 floor points)"],
          ] as const).map(function ([c, label]) {
            return (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 6, margin: "3px 0", fontSize: 11 }}>
                <span style={{ width: 10, height: 10, borderRadius: "50%", background: c, border: "2px solid #1f1f1f", flexShrink: 0 }} />
                <span style={{ color: "#bdbdbd" }}>{label}</span>
              </div>
            );
          })}
          <div style={{ display: "flex", alignItems: "center", gap: 6, margin: "3px 0", fontSize: 11 }}>
            <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#5cf", border: "2px solid #fc6", boxShadow: "0 0 0 1px #fc6", flexShrink: 0 }} />
            <span style={{ color: "#bdbdbd" }}>Selected keypoint (amber ring)</span>
          </div>
        </div>
      </div>

      {/* Middle: clickable image */}
      <div
        style={{
          background: "#181818",
          border: "1px solid #2b2b2b",
          borderRadius: 4,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
          // Anchors the compass overlay to this column, not to the scroller below.
          position: "relative",
        }}
      >
        <div
          style={{
            padding: "6px 10px",
            borderBottom: "1px solid #2b2b2b",
            fontSize: 12,
            color: "#9d9d9d",
            display: "flex",
            gap: 12,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <span>
            {selectedCam ?? "(no cam selected)"}
            {selectedCamObj?.is_stereo ? " (stereo eye)" : ""}
          </span>
          <span style={{ color: "#6d6d6d" }}>
            {selectedTieId
              ? <>Placing tie point: <b style={{ color: "#3ff" }}>{tieInfoById[selectedTieId]?.label ?? selectedTieId}</b></>
              : <>Selected keypoint: {selectedKeypoint ? <b style={{ color: "#fc6" }}>{selectedKeypoint}</b> : "(none — pick one on the right)"}</>}
          </span>
          {imgSize && (
            <span style={{ color: "#6d6d6d" }}>
              Thumbnail {imgSize[0]}x{imgSize[1]} px
            </span>
          )}
          {/* Live fit quality from the floor-plane homography. */}
          {/* https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html */}
          {/* https://stackoverflow.com/questions/22389896/finding-the-real-world-coordinates-of-an-image-point */}
          {reproj && (
            <span
              style={{
                fontSize: 11,
                padding: "1px 7px",
                borderRadius: 3,
                background: "#222",
                border: "1px solid #3c3c3c",
                color: reproj.status === "ok" ? residualColor(reproj.rms, reproj.origW) : "#9d9d9d",
              }}
              title="Floor-plane homography fit from your clicks. A wrong-line click reprojects far (red dot). No intrinsics needed."
            >
              {reproj.status === "few"
                ? `PnP check: need ≥4 floor pts (have ${reproj.nFloor})`
                : reproj.status === "failed"
                  ? "PnP check: fit failed"
                  : `PnP rms ${reproj.rms.toFixed(1)} px · ${reproj.nInliers}/${reproj.nFloor} inliers${reproj.k1 ? ` · k1 ${reproj.k1.toFixed(2)}` : ""}`}
            </span>
          )}
          <label
            style={{ fontSize: 11, color: "#9d9d9d", display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}
            title="Snap each click to the nearest court-line crossing (Förstner junction). Disable for free-hand clicks."
          >
            <input type="checkbox" checked={snapEnabled} onChange={function (e) {
              setSnapEnabled(e.target.checked);
            }} />
            Snap to lines
            {lineColor !== "any" && (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 3 }} title={`Snapping to ${lineColor} ${lineColor === "white" ? "(basketball)" : "(volleyball)"} lines`}>
                <span style={{
                  width: 9, height: 9, borderRadius: "50%",
                  background: lineColor === "white" ? "#fff" : "#5af",
                  border: "1px solid #555",
                }} />
                {lineColor}
              </span>
            )}
            {lastSnap != null && (
              <span style={{ color: "#5cf" }}>· snapped {lastSnap.toFixed(0)} px</span>
            )}
          </label>
          <label
            style={{ fontSize: 11, color: "#9d9d9d", display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}
            title="Show a circular magnifier that follows the cursor over the image for pixel-precise clicking."
          >
            <input type="checkbox" checked={lensEnabled} onChange={function (e) { setLensEnabled(e.target.checked); if (!e.target.checked) setLensPos(null); }} />
            Magnifier
          </label>
          <label
            style={{ fontSize: 11, color: "#9d9d9d", display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}
            title="Project the court line model onto the image (white=basketball, blue=volleyball). Lines curve with the lens distortion to hug the real painted lines. Needs ≥6 floor clicks."
          >
            <input type="checkbox" checked={showOverlay} onChange={function (e) {
              setShowOverlay(e.target.checked);
            }} />
            Court overlay
            {reproj && reproj.status !== "ok" && (
              <span style={{ color: "#6d6d6d" }}>· need ≥6 floor pts</span>
            )}
          </label>
          <label
            style={{ fontSize: 11, color: showOverlay ? "#9d9d9d" : "#5d5d5d", display: "flex", alignItems: "center", gap: 4, cursor: showOverlay ? "pointer" : "default", paddingLeft: 14 }}
            title="OFF: project the whole court, including beyond the image edges (preview the rest of the court). ON: only draw the overlay inside the convex hull of your clicks, hiding the extrapolation into the unclicked half (useful for the ceiling eyes, which only see one court half)."
          >
            <input type="checkbox" checked={clipToCoverage} disabled={!showOverlay} onChange={function (e) {
              setClipToCoverage(e.target.checked);
            }} />
            Clip to clicked area
          </label>
          <label
            style={{ fontSize: 11, color: "#9d9d9d", display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}
            title="Draw the court projected through the full PnP + regularized Brown distortion solve. Accurate for the strongly-barrel ceiling eyes (the live homography+1-param fit is too weak)."
          >
            <input type="checkbox" checked={showCalibrated} onChange={function (e) {
              setShowCalibrated(e.target.checked);
            }} />
            Calibrated overlay (court-PnP)
            {selectedCam && (calibOverlay?.[selectedCam]
              ? <span style={{ color: "#7c7" }}>· rms {calibOverlay[selectedCam].rms_px}px</span>
              : <span style={{ color: "#6d6d6d" }}>· none for this cam</span>)}
          </label>
          <label
            style={{ fontSize: 11, color: "#9d9d9d", display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}
            title="Also snap clicks onto the projected court-line model (AutoCAD-style) — useful where the painted line is faint or occluded."
          >
            <input type="checkbox" checked={snapProjected} onChange={function (e) {
              setSnapProjected(e.target.checked);
            }} />
            Snap to model
          </label>
          <span style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: "auto" }}>
            <span style={{ color: "#6d6d6d", fontSize: 11 }}>Zoom</span>
            <button
              onClick={function () {
                setZoom(function (z) { return Math.max(1, +(z - 0.5).toFixed(1)); });
              }}
              style={zoomBtnStyle}
              title="Zoom out"
            >
              −
            </button>
            <input
              type="range"
              min={1}
              max={8}
              step={0.5}
              value={zoom}
              onChange={function (e) { setZoom(parseFloat(e.target.value)); }}
              style={{ width: 120 }}
            />
            <button
              onClick={function () {
                setZoom(function (z) { return Math.min(8, +(z + 0.5).toFixed(1)); });
              }}
              style={zoomBtnStyle}
              title="Zoom in"
            >
              +
            </button>
            <span style={{ color: "#fc6", fontSize: 11, minWidth: 30, textAlign: "right" }}>
              {zoom.toFixed(1)}x
            </span>
            <button
              onClick={function () { setZoom(1); }}
              style={{ ...zoomBtnStyle, fontSize: 10 }}
              title="Reset zoom"
            >
              fit
            </button>
            <span style={{ width: 1, alignSelf: "stretch", background: "#3c3c3c", margin: "0 4px" }} />
            <span style={{ color: "#6d6d6d", fontSize: 11 }}>Compass</span>
            <input
              type="range"
              min={120}
              max={420}
              step={10}
              value={compassSize}
              onChange={function (e) { setCompassSize(parseInt(e.target.value, 10)); }}
              style={{ width: 100 }}
              title="Adjust on-court compass size"
            />
            <span style={{ color: "#fc6", fontSize: 11, minWidth: 36, textAlign: "right" }}>
              {compassSize}px
            </span>
          </span>
        </div>
        <div
          ref={scrollerRef}
          style={{
            flex: 1,
            minHeight: 0,
            position: "relative",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            overflow: "auto",
          }}
        >
          {thumbUrl && selectedCamObj ? (
            <div style={{ position: "relative", display: "inline-block" }}>
              <img
                ref={imgRef}
                src={thumbUrl}
                alt={selectedCam!}
                onLoad={onImgLoad}
                onClick={onImgClick}
                onMouseMove={onImgMouseMove}
                onMouseLeave={onImgMouseLeave}
                style={{
                  display: "block",
                  cursor: selectedKeypoint || selectedTieId ? "crosshair" : "default",
                  userSelect: "none",
                  ...(zoom === 1
                    ? { maxWidth: "100%", maxHeight: "calc(100vh - 140px)" }
                    : {
                        width: `${100 * zoom}%`,
                        height: "auto",
                        maxWidth: "none",
                        maxHeight: "none",
                      }),
                }}
              />
              {/* court_overlay.json, in sensor px. Each line is drawn twice, black under colour.
                  https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d */}
              {showCalibrated && selectedCam && selectedCamObj && calibOverlay?.[selectedCam] && imgRef.current && (function () {
                const rect = imgRef.current.getBoundingClientRect();
                const entry = calibOverlay[selectedCam];
                const [ow, oh] = entry.image_size ?? camOriginalSize(selectedCamObj);
                return (
                  <svg style={{ position: "absolute", left: 0, top: 0, width: rect.width, height: rect.height, pointerEvents: "none", overflow: "visible" }}>
                    {entry.polylines.map(function (pl, i) {
                      const d = pl.pts.map(function (p, j) {
                        return `${j ? "L" : "M"}${(p[0] / ow * rect.width).toFixed(1)},${(p[1] / oh * rect.height).toFixed(1)}`;
                      }).join(" ");
                      const color = pl.court === "bb" ? "#ffffff" : "#55aaff";
                      return (
                        <g key={i}>
                          <path d={d} fill="none" stroke="#000" strokeWidth={3} opacity={0.45} />
                          <path d={d} fill="none" stroke={color} strokeWidth={1.6} opacity={0.95} />
                        </g>
                      );
                    })}
                  </svg>
                );
              })()}
              {/* Live overlay: the court through the fitted H + k1, so lines curve with the lens. */}
              {showOverlay && reproj?.status === "ok" && reproj.H && imgRef.current && (function () {
                const rect = imgRef.current.getBoundingClientRect();
                const H = reproj.H; const { norm, k1, sw, sh, hull } = reproj;
                if (!H) return null;
                return (
                  <svg style={{ position: "absolute", left: 0, top: 0, width: rect.width, height: rect.height, pointerEvents: "none", overflow: "visible" }}>
                    {courtModel.map(function (pl, idx) {
                      // projectWorld returns null past the lens model's valid radius; that ends a run.
                      const runs: Pt[][] = []; let cur: Pt[] = [];
                      for (const w of pl.pts) {
                        if (clipToCoverage && hull && !pointInConvexPoly(w, hull)) {
                          if (cur.length > 1) runs.push(cur); cur = []; continue;
                        }
                        const qD = projectWorld(w, H, norm, k1);
                        if (!qD) { if (cur.length > 1) runs.push(cur); cur = []; continue; }
                        cur.push([(qD[0] / sw) * rect.width, (qD[1] / sh) * rect.height]);
                      }
                      if (cur.length > 1) runs.push(cur);
                      const color = pl.court === "basketball" ? "#ffffff" : "#55aaff";
                      return runs.map(function (run, ri) {
                        const d = run.map(function (p, i) {
                          return `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`;
                        }).join(" ");
                        return (
                          <g key={`${idx}-${ri}`}>
                            <path d={d} fill="none" stroke="#000" strokeWidth={3} opacity={0.4} />
                            <path d={d} fill="none" stroke={color} strokeWidth={1.4} opacity={0.75} />
                          </g>
                        );
                      });
                    })}
                  </svg>
                );
              })()}
              {imgSize &&
                camAnnotations.map(function (a) {
                  const [origW, origH] = camOriginalSize(selectedCamObj);
                  // Original px -> thumbnail px -> rendered px.
                  const el = imgRef.current;
                  if (!el) return null;
                  const rect = el.getBoundingClientRect();
                  const xThumb = (a.image_xy[0] / origW) * imgSize[0];
                  const yThumb = (a.image_xy[1] / origH) * imgSize[1];
                  const xRendered = (xThumb / imgSize[0]) * rect.width;
                  const yRendered = (yThumb / imgSize[1]) * rect.height;
                  const isSel = a.keypoint_id === selectedKeypoint;
                  // Colour by live reprojection residual when available.
                  // https://stackoverflow.com/questions/16265714/camera-pose-estimation-opencv-pnp
                  const res = reproj?.perKp[a.keypoint_id];
                  const fill = res !== undefined && reproj
                    ? residualColor(res, reproj.origW)
                    : "#5cf";
                  return (
                    <div
                      key={a.keypoint_id}
                      style={{
                        position: "absolute",
                        left: xRendered - 6,
                        top: yRendered - 6,
                        width: 12,
                        height: 12,
                        borderRadius: "50%",
                        background: fill,
                        border: `2px solid ${isSel ? "#fc6" : "#1f1f1f"}`,
                        pointerEvents: "none",
                        boxShadow: isSel ? "0 0 0 2px #fc6" : "0 0 0 1px #1f1f1f",
                      }}
                      title={res !== undefined ? `${a.keypoint_id}: ${res.toFixed(1)} px residual` : a.keypoint_id}
                    />
                  );
                })}
              {/* Tie-point markers. Square, so they stand apart from the round keypoint dots. */}
              {imgSize && selectedCamObj && imgRef.current && camTieObs.map(function (t) {
                const [origW, origH] = camOriginalSize(selectedCamObj);
                const el = imgRef.current;
                if (!el) return null;
                const rect = el.getBoundingClientRect();
                const xRendered = (t.xy[0] / origW) * rect.width;
                const yRendered = (t.xy[1] / origH) * rect.height;
                const isSel = t.tieId === selectedTieId;
                const color = t.color;
                return (
                  <div
                    key={t.tieId}
                    style={{
                      position: "absolute",
                      left: xRendered - 6,
                      top: yRendered - 6,
                      width: 12,
                      height: 12,
                      background: color,
                      border: `2px solid ${isSel ? "#fff" : "#1f1f1f"}`,
                      pointerEvents: "none",
                      boxShadow: isSel ? "0 0 0 2px #fff" : "0 0 0 1px #1f1f1f",
                    }}
                    title={`${t.label} (tie point)`}
                  />
                );
              })}
              {/* Pulsing ring (SMIL) where a flagged tie point should have been clicked.
                  https://developer.mozilla.org/en-US/docs/Web/SVG/Element/animate */}
              {imgSize && selectedCam && selectedTieId && selectedCamObj && imgRef.current &&
                reclickHints[selectedCam]?.[selectedTieId] &&
                (reclickHints[selectedCam][selectedTieId].kind === "reclick" ||
                 reclickHints[selectedCam][selectedTieId].kind === "swap") && (function () {
                const h = reclickHints[selectedCam][selectedTieId];
                const [origW, origH] = camOriginalSize(selectedCamObj);
                const rect = imgRef.current.getBoundingClientRect();
                const x = (h.expected_xy[0] / origW) * rect.width;
                const y = (h.expected_xy[1] / origH) * rect.height;
                return (
                  <svg style={{ position: "absolute", left: 0, top: 0, width: rect.width, height: rect.height, pointerEvents: "none", overflow: "visible" }}>
                    <circle cx={x} cy={y} r={14} fill="none" stroke="#000" strokeWidth={3} opacity={0.5}>
                      <animate attributeName="r" values="10;20;10" dur="1.4s" repeatCount="indefinite" />
                    </circle>
                    <circle cx={x} cy={y} r={14} fill="none" stroke="#ffd400" strokeWidth={2}>
                      <animate attributeName="r" values="10;20;10" dur="1.4s" repeatCount="indefinite" />
                    </circle>
                    <circle cx={x} cy={y} r={3} fill="#ffd400" stroke="#000" strokeWidth={1} />
                    <text x={x + 22} y={y - 12} fill="#ffd400" stroke="#000" strokeWidth={0.7} fontSize={12} style={{ paintOrder: "stroke" }}>
                      re-click here (was off {h.err_px.toFixed(0)}px)
                    </text>
                  </svg>
                );
              })()}
              {/* Loupe: the same image blown up, moved with background-position.
                  https://developer.mozilla.org/en-US/docs/Web/CSS/background-position
                  https://developer.mozilla.org/en-US/docs/Web/CSS/background-size
                  https://developer.mozilla.org/en-US/docs/Web/CSS/pointer-events */}
              {lensEnabled && lensPos && (
                <div
                  style={{
                    position: "absolute",
                    left: lensPos.xR - LENS_DIAM / 2,
                    top: lensPos.yR - LENS_DIAM / 2,
                    width: LENS_DIAM,
                    height: LENS_DIAM,
                    borderRadius: "50%",
                    border: "2px solid #fc6",
                    boxShadow: "0 0 0 1px #1f1f1f, 0 3px 10px rgba(0,0,0,0.55)",
                    pointerEvents: "none",
                    overflow: "hidden",
                    backgroundImage: `url(${thumbUrl})`,
                    backgroundRepeat: "no-repeat",
                    // Enlarge by LENS_ZOOM, then offset so the cursor sits at the loupe centre.
                    backgroundSize: `${lensPos.w * LENS_ZOOM}px ${lensPos.h * LENS_ZOOM}px`,
                    backgroundPosition:
                      `${-(lensPos.xR * LENS_ZOOM - LENS_DIAM / 2)}px ` +
                      `${-(lensPos.yR * LENS_ZOOM - LENS_DIAM / 2)}px`,
                    // https://developer.mozilla.org/en-US/docs/Web/CSS/image-rendering
                    imageRendering: "auto",
                    zIndex: 20,
                  }}
                >
                  {/* centre crosshair marks the exact click point */}
                  <div style={{ position: "absolute", left: LENS_DIAM / 2 - 0.5, top: 8, bottom: 8, width: 1, background: "rgba(255,80,80,0.8)" }} />
                  <div style={{ position: "absolute", top: LENS_DIAM / 2 - 0.5, left: 8, right: 8, height: 1, background: "rgba(255,80,80,0.8)" }} />
                </div>
              )}
            </div>
          ) : (
            <div style={{ color: "#6d6d6d" }}>
              No thumbnail available for this camera.
            </div>
          )}
        </div>
        {/* Compass, then one keypoint map per court. All three share the size slider. */}
        <div style={{
          position: "absolute",
          // overlayTop keeps this below the header.
          left: 12, top: overlayTop, bottom: 12,
          display: "flex", flexDirection: "column", gap: 8,
          overflowY: "auto",
          // Clicks fall through to the image; each Compass card re-enables its own.
          pointerEvents: "none",
        }}>
          <Compass
            size={compassSize}
            variant="basketball"
            caption={selectedCam ? `${selectedCam} on court` : "World axes"}
            panel
            cameraPose={cameraPoseHint}
          />
          <Compass
            size={compassSize}
            variant="basketball"
            caption="Basketball points"
            panel
            keypoints={basketballKeypoints}
            highlightKeypoint={selectedKeypoint}
          />
          <Compass
            size={compassSize}
            variant="volleyball"
            caption="Volleyball points"
            panel
            keypoints={volleyballKeypoints}
            highlightKeypoint={selectedKeypoint}
          />
        </div>
      </div>

      {/* Right: keypoint picker + annotations list */}
      <div
        style={{
          background: "#181818",
          border: "1px solid #2b2b2b",
          borderRadius: 4,
          overflowY: "auto",
          padding: 6,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Off-floor tie points: click the same feature in every camera that sees it. */}
        <div style={{ marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid #2b2b2b" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
            <span style={{ fontSize: 12, color: "#3ff" }}>Off-floor tie points</span>
            <span style={{ fontSize: 10, color: "#6d6d6d" }}>{totalTieClicks} clicks</span>
          </div>
          <div style={{ fontSize: 10, color: "#8d8d8d", lineHeight: 1.45, marginBottom: 6 }}>
            Click the SAME physical feature in <b>every camera that sees it</b> (≥3 ideally).
            These break the floor-only coplanarity so the off-floor 3D becomes metric. No court snap.
          </div>
          {TIE_GROUPS.map(function (g) {
            return (
              <div key={g.key} style={{ marginBottom: 6 }}>
                <div style={{ fontSize: 10, color: g.color, textTransform: "uppercase", letterSpacing: 0.5, margin: "4px 0" }}>
                  {g.title}
                </div>
                {pointsForGroup(g).map(function (p) {
                  const sel = p.id === selectedTieId;
                  const obs = tieObs[p.id] ?? [];
                  const n = obs.length;
                  const hasCam = obs.some(function (o) { return o.cam === selectedCam; });
                  const flag = selectedCam ? reclickHints[selectedCam]?.[p.id] : undefined;
                  const flagged = !!flag;
                  const flagColor = flag?.kind === "delete" ? "#f88" : flag?.kind === "relabel" ? "#fb6" : "#ffd400";
                  return (
                    <button
                      key={p.id}
                      onClick={function () { pickTie(sel ? null : p.id); }}
                      title={flag ? `Flagged in ${selectedCam}: ${flag.kind}${flag.note ? ` — ${flag.note}` : ""} (off ${flag.err_px.toFixed(0)}px)` : g.hint}
                      style={{
                        display: "flex", justifyContent: "space-between", alignItems: "center",
                        width: "100%", textAlign: "left", margin: "2px 0", padding: "3px 6px",
                        background: sel ? g.color : hasCam ? "#143a3a" : "#2b2b2b",
                        color: sel ? "#1f1f1f" : hasCam ? "#9ff" : "#cccccc",
                        border: `1px solid ${flagged && !sel ? flagColor : sel ? g.color : "#3c3c3c"}`,
                        borderRadius: 3, cursor: "pointer", fontSize: 11, fontWeight: sel ? 600 : 400,
                      }}
                    >
                      <span>{flagged ? "⚠ " : hasCam ? "✓ " : "  "}{p.label}</span>
                      <span style={{ fontSize: 10, opacity: 0.8 }}>{n} cam{n === 1 ? "" : "s"}</span>
                    </button>
                  );
                })}
                <button
                  onClick={function () { addExtraToGroup(g); }}
                  style={{ ...btnStyle, width: "100%", marginTop: 2, fontSize: 10, color: "#9d9d9d", borderStyle: "dashed" }}
                  title="Add another sub-point to this group (e.g. a 2nd/3rd edge or corner of the same object)"
                >
                  + add point
                </button>
              </div>
            );
          })}
          {selectedTieId && (function () {
            const info = tieInfoById[selectedTieId];
            const obs = tieObs[selectedTieId] ?? [];
            return (
              <div style={{ marginTop: 4, padding: 6, background: "#151515", border: "1px solid #2b2b2b", borderRadius: 3 }}>
                <div style={{ fontSize: 10, color: "#9d9d9d", lineHeight: 1.45, marginBottom: 6 }}><b style={{ color: info?.color ?? "#3ff" }}>{info?.label ?? selectedTieId}</b> — {info?.hint}</div>
              {selectedCam && reclickHints[selectedCam]?.[selectedTieId] && (function () {
                const h = reclickHints[selectedCam][selectedTieId];
                const c = h.kind === "delete" ? "#f88" : h.kind === "relabel" ? "#fb6" : "#ffd400";
                const msg = h.kind === "delete"
                  ? `DELETE this ${selectedCam} click — ${h.note} (was off ${h.err_px.toFixed(0)}px).`
                  : h.kind === "relabel"
                  ? `RELABEL: this ${selectedCam} click ${h.note}.`
                  : `RE-CLICK in ${selectedCam} toward the yellow ring on the image (was off ${h.err_px.toFixed(0)}px).`;
                return (
                  <div style={{ fontSize: 10, color: c, lineHeight: 1.45, marginBottom: 6, padding: "3px 5px", border: `1px solid ${c}`, borderRadius: 3 }}>
                    ⚠ {msg}
                  </div>
                );
              })()}
                {obs.length === 0 ? (
                  <div style={{ fontSize: 11, color: "#6d6d6d" }}>No clicks yet — pick a camera (left) and click the feature in the image.</div>
                ) : (
                  obs.map(function (o) {
                    return (
                      <div key={o.cam} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: "2px 0", padding: "2px 4px", background: o.cam === selectedCam ? "#143a3a" : "#2b2b2b", borderRadius: 3, fontSize: 11 }}>
                        <span title={`(${o.image_xy[0].toFixed(0)}, ${o.image_xy[1].toFixed(0)}) px`}>{o.cam}</span>
                        <button onClick={function () { removeTieObs(selectedTieId, o.cam); }} style={{ background: "transparent", color: "#f88", border: "none", cursor: "pointer", fontSize: 12, padding: "0 4px" }} title="Delete this click">×</button>
                      </div>
                    );
                  })
                )}
                {obs.length > 0 && (
                  <button
                    onClick={function () { if (window.confirm(`Clear all ${obs.length} clicks for ${info?.label ?? selectedTieId}?`)) clearTieObs(selectedTieId); }}
                    style={{ ...btnStyle, color: "#f88", marginTop: 6, width: "100%" }}
                  >
                    Clear this point (all cams)
                  </button>
                )}
              </div>
            );
          })()}
          <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
            <button onClick={function () { tieFileInputRef.current?.click(); }} style={{ ...btnStyle, flex: 1 }}>Import ties</button>
            <button onClick={onExportTie} style={{ ...btnStyle, flex: 1 }}>Export ties</button>
            <input ref={tieFileInputRef} type="file" accept="application/json,.json" onChange={onImportTieFile} style={{ display: "none" }} />
          </div>
        </div>

        <div style={{ fontSize: 12, color: "#9d9d9d", marginBottom: 6 }}>
          1. Pick the court keypoint you're about to click
        </div>
        {Object.entries(keypointsByCourt).map(function ([court, kps]) {
          return (
            <div key={court} style={{ marginBottom: 8 }}>
              <div
                style={{
                  fontSize: 11,
                  color: court === "basketball" ? "#fa6" : "#6af",
                  margin: "4px 0",
                  textTransform: "uppercase",
                  letterSpacing: 0.5,
                }}
              >
                {court}
              </div>
              {kps.map(function (kp) {
                const sel = kp.id === selectedKeypoint;
                const annotated = camAnnotations.some(function (a) { return a.keypoint_id === kp.id; });
                return (
                  <button
                    key={kp.id}
                    onClick={function () { pickKeypoint(kp.id); }}
                    style={{
                      display: "block",
                      width: "100%",
                      textAlign: "left",
                      margin: "2px 0",
                      padding: "3px 6px",
                      background: sel ? "#fc6" : annotated ? "#1d3a1d" : "#2b2b2b",
                      color: sel ? "#1f1f1f" : annotated ? "#9f9" : "#cccccc",
                      border: `1px solid ${sel ? "#fc6" : "#3c3c3c"}`,
                      borderRadius: 3,
                      cursor: "pointer",
                      fontSize: 11,
                      fontWeight: sel ? 600 : 400,
                    }}
                    title={`world: [${kp.world.map(function (v) { return v.toFixed(2); }).join(", ")}]\n${kp.label}`}
                  >
                    {annotated ? "✓ " : "  "}
                    {kp.id}
                  </button>
                );
              })}
            </div>
          );
        })}

        {camAnnotations.length > 0 && (
          <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid #2b2b2b" }}>
            <div style={{ fontSize: 12, color: "#9d9d9d", marginBottom: 6 }}>
              2. Recorded clicks for {selectedCam} ({camAnnotations.length})
            </div>
            {camAnnotations.map(function (a) {
              const res = reproj?.perKp[a.keypoint_id];
              return (
                <div
                  key={a.keypoint_id}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    margin: "2px 0",
                    padding: "2px 4px",
                    background: "#2b2b2b",
                    borderRadius: 3,
                    fontSize: 11,
                  }}
                >
                  <span title={`(${a.image_xy[0].toFixed(1)}, ${a.image_xy[1].toFixed(1)}) px in original`}>
                    {a.keypoint_id}
                  </span>
                  <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    {res !== undefined && (
                      <span
                        style={{ color: residualColor(res, reproj!.origW), fontVariantNumeric: "tabular-nums" }}
                        title="reprojection residual under the floor-plane homography"
                      >
                        {res.toFixed(0)}px
                      </span>
                    )}
                    <button
                      onClick={function () { selectedCam && removeAnnotation(selectedCam, a.keypoint_id); }}
                      style={{
                        background: "transparent",
                        color: "#f88",
                        border: "none",
                        cursor: "pointer",
                        fontSize: 12,
                        padding: "0 4px",
                      }}
                      title="Delete this click"
                    >
                      ×
                    </button>
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

/** Save an object as a JSON file: a Blob, an object URL, a throwaway <a download>.
 *  https://developer.mozilla.org/en-US/docs/Web/API/Blob
 *  https://developer.mozilla.org/en-US/docs/Web/API/File_API/Using_files_from_web_applications
 *  https://developer.mozilla.org/en-US/docs/Web/HTML/Element/a#download */
function downloadJson(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** Foot of the perpendicular from (px,py) to segment a-b; returns [dist, fx, fy].
 *  https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line */
function pointSegFoot(px: number, py: number, ax: number, ay: number, bx: number, by: number): [number, number, number] {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 > 1e-9 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  const fx = ax + t * dx, fy = ay + t * dy;
  return [Math.hypot(px - fx, py - fy), fx, fy];
}

/** Magnifier loupe geometry. */
const LENS_DIAM = 190; // px diameter of the circular loupe
const LENS_ZOOM = 2.5; // magnification over the currently-rendered image

const btnStyle: React.CSSProperties = {
  background: "#2b2b2b",
  color: "#cccccc",
  border: "1px solid #3c3c3c",
  padding: "4px 8px",
  fontSize: 12,
  cursor: "pointer",
  borderRadius: 3,
};

const zoomBtnStyle: React.CSSProperties = {
  background: "#252526",
  color: "#cccccc",
  border: "1px solid #3c3c3c",
  padding: "2px 8px",
  fontSize: 12,
  cursor: "pointer",
  borderRadius: 3,
  lineHeight: 1,
  minWidth: 22,
};
