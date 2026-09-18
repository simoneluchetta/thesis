/**
 * Shared viewer state: open tab, visible layers, selected camera, court placement.
 * The HUD writes it and SceneCanvas reads it, so neither needs the other's props.
 */
// https://docs.pmnd.rs/zustand
// https://github.com/pmndrs/zustand
import { create } from "zustand";

/** World up axis, 'auto' meaning the hint in scene.json. */
export type UpAxis = "auto" | "+x" | "+y" | "+z";

/** Which world axis orbit is locked to. */
export type OrbitAxisMode = "free" | "x" | "y" | "z";

/** Which tab is open. */
type Page = "viewer" | "placer" | "calibrator" | "pnpResult" | "peopleMap" | "annotate" | "nerf";

/** Everything the viewer tab shares between the HUD and the canvas. */
interface ViewerState {
  page: Page;
  setPage: (p: Page) => void;

  showFrustums: boolean;
  showLabels: boolean;
  showThumbs: boolean;
  showPeople: boolean;
  /** Index into scene.people.timestamps for the people currently shown. */
  peopleTimeIndex: number;
  showBasketballCourt: boolean;
  showVolleyballCourt: boolean;
  /** Where the court overlay sits in world coords, before worldQuat. */
  courtAnchorX: number;
  courtAnchorY: number;
  courtAnchorZ: number;
  /** Rotation of the court overlay around the world up axis (Z), in degrees. */
  courtYawDeg: number;
  /** Per-layer visibility, keyed by `PlyLayer.id`. */
  plyLayersOn: Record<string, boolean>;
  /** Highlighted camera, or null. */
  selectedCamId: number | null;
  /** Multiplier on every frustum's near/far. 1 = the values in scene.json. */
  frustumScale: number;
  /** Multiplier on every camera label's on-screen size. */
  labelScale: number;
  /** World up-axis override. 'auto' uses world.up from scene.json. */
  upAxis: UpAxis;
  /** Bumped on every "reset view" click so the canvas can react. */
  resetViewKey: number;
  orbitAxis: OrbitAxisMode;

  toggle: (key: "showFrustums" | "showLabels" | "showThumbs" | "showPeople"
    | "showBasketballCourt" | "showVolleyballCourt") => void;
  setPeopleTimeIndex: (i: number) => void;
  togglePly: (id: string) => void;
  setSelectedCam: (id: number | null) => void;
  setPlyLayers: (initial: Record<string, boolean>) => void;
  setFrustumScale: (s: number) => void;
  setLabelScale: (s: number) => void;
  setUpAxis: (a: UpAxis) => void;
  setOrbitAxis: (m: OrbitAxisMode) => void;
  setCourtAnchorX: (v: number) => void;
  setCourtAnchorY: (v: number) => void;
  setCourtAnchorZ: (v: number) => void;
  setCourtYawDeg: (v: number) => void;
  resetCourtAnchor: () => void;
  resetView: () => void;
}

// Tab names accepted in the URL hash.
const PAGES: Page[] = ["viewer", "placer", "calibrator", "pnpResult", "peopleMap", "annotate", "nerf"];

/** Open the tab named in the URL hash, e.g. /#peopleMap.
 *  https://developer.mozilla.org/en-US/docs/Web/API/Location/hash */
function initialPage(): Page {
  const h = typeof window !== "undefined" ? window.location.hash.slice(1) : "";
  return (PAGES as string[]).includes(h) ? (h as Page) : "viewer";
}

/** Hook for the shared viewer state. */
export const useViewer = create<ViewerState>(function (set) {
  return {
    page: initialPage(),
    setPage(page) { return set({ page }); },

    showFrustums: true,
    showLabels: true,
    showThumbs: true,
    showPeople: true,
    peopleTimeIndex: 0,
    showBasketballCourt: true,
    showVolleyballCourt: true,
    courtAnchorX: 0,
    courtAnchorY: 0,
    courtAnchorZ: 0,
    courtYawDeg: 0,
    plyLayersOn: {},
    selectedCamId: null,
    frustumScale: 2.5,
    labelScale: 4.0,
    // +z matches the placer's Z-up convention. The HUD can override it.
    upAxis: "+z",
    resetViewKey: 0,
    orbitAxis: "free",

    toggle(key) {
      return set(function (s) { return { ...s, [key]: !s[key] }; });
    },
    setPeopleTimeIndex(peopleTimeIndex) { return set({ peopleTimeIndex }); },
    togglePly(id) {
      return set(function (s) {
        return {
          plyLayersOn: { ...s.plyLayersOn, [id]: !s.plyLayersOn[id] },
        };
      });
    },
    setSelectedCam(id) { return set({ selectedCamId: id }); },
    setPlyLayers(initial) { return set({ plyLayersOn: initial }); },
    setFrustumScale(frustumScale) { return set({ frustumScale }); },
    setLabelScale(labelScale) { return set({ labelScale }); },
    setUpAxis(upAxis) { return set({ upAxis }); },
    setOrbitAxis(orbitAxis) { return set({ orbitAxis }); },
    setCourtAnchorX(courtAnchorX) { return set({ courtAnchorX }); },
    setCourtAnchorY(courtAnchorY) { return set({ courtAnchorY }); },
    setCourtAnchorZ(courtAnchorZ) { return set({ courtAnchorZ }); },
    setCourtYawDeg(courtYawDeg) { return set({ courtYawDeg }); },
    resetCourtAnchor() {
      return set({ courtAnchorX: 0, courtAnchorY: 0, courtAnchorZ: 0, courtYawDeg: 0 });
    },
    resetView() {
      return set(function (s) { return { resetViewKey: s.resetViewKey + 1 }; });
    },
  };
});
