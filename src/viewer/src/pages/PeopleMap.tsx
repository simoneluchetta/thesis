/**
 * The athletes seen from above, with a timeline and a camera inset. "Download" records
 * the map to a video file. The 3D SMPL-X bodies are a separate mode, in BodiesCanvas.
 */
import { useEffect, useMemo, useState } from "react";
import type { SceneJson, ScenePersonObs } from "../types";
import { useViewer } from "../state/store";
import { buildCourtModel, courtDimsFromCatalog } from "../utils/courtModel";
import type { CourtPolyline } from "../utils/courtModel";
import BodiesCanvas from "../components/BodiesCanvas";
import { PALETTE } from "../utils/palette";

interface Props {
  scene: SceneJson;
}

/** One court keypoint from public/court_keypoints.json. */
interface KP {
  id: string;
  world: [number, number, number];
}
/** public/court_keypoints.json: court dimensions plus the named keypoints. */
interface Catalog {
  courts: Record<string, unknown>;
  keypoints: KP[];
}


/** The track id when the block has one, else the per-frame fuse index. */
function pidOf(p: ScenePersonObs) { return p.track_id ?? p.id; }

/** Officials, coaches and spectators stay on the map, drawn muted. */
function isAthlete(p: ScenePersonObs) {
  return !p.off_team && (!p.role || p.role === "player");
}
// Grey, for everyone who is not an athlete.
const NON_ATHLETE = "#7a7a7a";
// Palette colour for an athlete, grey for anyone else.
function colourOf(p: ScenePersonObs) {
  return isAthlete(p) ? PALETTE[pidOf(p) % PALETTE.length] : NON_ATHLETE;
}

/** A polyline history for one track: pre-split into gap-free segments. */
interface TrailDraw {
  color: string;
  segs: [number, number][][];
}

// Top-down world window in metres. X = long axis, Y = short axis, origin at centre.
const XMIN = -16, XMAX = 16, YMIN = -9.5, YMAX = 9.5;
// Size of that window, which is also the SVG viewBox size.
const VW = XMAX - XMIN, VH = YMAX - YMIN;

/** World (X,Y) -> SVG viewBox coords (Y flipped so +Y is up on screen). */
function vx(x: number) { return x - XMIN; }
function vy(y: number) { return YMAX - y; }

// An SVG path "d" string: M moves, L draws a line to the next point.
// https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d
// https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths
function polyPath(pts: [number, number][]): string {
  return pts.map(function (p, i) { return `${i ? "L" : "M"}${vx(p[0]).toFixed(3)},${vy(p[1]).toFixed(3)}`; }).join(" ");
}

// Video export: redraw the map frame by frame onto a 2D canvas and record that canvas.
// Everything drawn must stay same-origin, or captureStream() refuses the tainted canvas.
// https://stackoverflow.com/questions/22097747/how-to-fix-getimagedata-error-the-canvas-has-been-tainted-by-cross-origin-data
// https://stackoverflow.com/questions/39874867/canvas-recording-using-capturestream-and-mediarecorder
// https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder
// https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/captureStream
// https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial

// 40 px per world unit over the padded viewBox (matches the on-screen aspect).
const EXPORT_PX_PER_UNIT = 40;
const EXPORT_W = (VW + 2) * EXPORT_PX_PER_UNIT; // 1360
const EXPORT_H = (VH + 2) * EXPORT_PX_PER_UNIT; // 840
// Fallback hold per timestamp in the recorded video (ms).
const EXPORT_HOLD_MS = 800;

// https://developer.mozilla.org/en-US/docs/Web/API/Window/setTimeout
function sleep(ms: number) {
  return new Promise<void>(function (r) { return window.setTimeout(r, ms); });
}

// Image load as a promise; the timeout stops one missing thumbnail stalling the export.
// https://developer.mozilla.org/en-US/docs/Web/API/HTMLImageElement/Image
function loadImage(src: string, timeoutMs = 8000): Promise<HTMLImageElement> {
  return new Promise(function (resolve, reject) {
    const img = new Image();
    const to = window.setTimeout(function () { return reject(new Error(`image load timed out: ${src}`)); }, timeoutMs);
    img.onload = function () { window.clearTimeout(to); resolve(img); };
    img.onerror = function () { window.clearTimeout(to); reject(new Error(`failed to load image: ${src}`)); };
    img.src = src;
  });
}

// World (viewBox) -> export-canvas px, at EXPORT_PX_PER_UNIT px per world unit.
function cX(wx: number) { return (vx(wx) + 1) * EXPORT_PX_PER_UNIT; }
function cY(wy: number) { return (vy(wy) + 1) * EXPORT_PX_PER_UNIT; }

/** Draw one frame's map onto a 2D canvas (an SVG route hangs in some webviews).
 *  https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D */
function drawMap(
  c: CanvasRenderingContext2D,
  frame: ScenePersonObs[],
  court: CourtPolyline[],
  hoops: { id: string; x: number; y: number }[],
  trails: TrailDraw[] = [],
  slots?: Record<string, string>,
): void {
  const PX = EXPORT_PX_PER_UNIT;

  // background + court floor fill
  c.fillStyle = "#15301a";
  c.fillRect(0, 0, EXPORT_W, EXPORT_H);
  c.fillStyle = "#1c3a24";
  c.fillRect(cX(-14), cY(7.5), 28 * PX, 15 * PX);

  // court lines: basketball white, volleyball blue
  c.lineJoin = "round";
  c.globalAlpha = 0.9;
  for (const pl of court) {
    c.beginPath();
    pl.pts.forEach(function (p, idx) {
      const X = cX(p[0]), Y = cY(p[1]);
      if (idx === 0) c.moveTo(X, Y); else c.lineTo(X, Y);
    });
    c.strokeStyle = pl.court === "basketball" ? "#e8e8e8" : "#55aaff";
    c.lineWidth = (pl.court === "basketball" ? 0.07 : 0.06) * PX;
    c.stroke();
  }
  c.globalAlpha = 1;

  // hoops. arc(..., 0, 2*PI) is a full circle.
  // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/arc
  // https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Drawing_text
  for (const h of hoops) {
    c.beginPath();
    c.arc(cX(h.x), cY(h.y), 0.28 * PX, 0, Math.PI * 2);
    c.strokeStyle = "#ffae42";
    c.lineWidth = 0.07 * PX;
    c.stroke();
    c.fillStyle = "#ffae42";
    c.font = `600 ${0.62 * PX}px system-ui, sans-serif`;
    c.textAlign = "center";
    c.textBaseline = "alphabetic";
    c.fillText(h.id === "bb_hoop_E" ? "Hoop E" : "Hoop W", cX(h.x), cY(h.y) - 0.5 * PX);
  }

  // track history, under the markers
  c.lineCap = "round";
  c.globalAlpha = 0.55;
  for (const tr of trails) {
    c.strokeStyle = tr.color;
    c.lineWidth = 0.09 * PX;
    for (const seg of tr.segs) {
      c.beginPath();
      seg.forEach(function (p, idx) {
        const X = cX(p[0]), Y = cY(p[1]);
        if (idx === 0) c.moveTo(X, Y); else c.lineTo(X, Y);
      });
      c.stroke();
    }
  }
  c.globalAlpha = 1;

  // people
  for (const p of frame) {
    const color = colourOf(p);
    const X = cX(p.pos[0]), Y = cY(p.pos[1]);
    const conf = Math.max(2, p.n_cams);
    // confidence halo, drawn see-through under the marker
    // https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/globalAlpha
    c.globalAlpha = 0.18;
    c.fillStyle = color;
    c.beginPath();
    c.arc(X, Y, (0.35 + 0.06 * conf) * PX, 0, Math.PI * 2);
    c.fill();
    c.globalAlpha = 1;
    // marker
    c.beginPath();
    c.arc(X, Y, 0.42 * PX, 0, Math.PI * 2);
    c.fillStyle = color;
    c.fill();
    c.strokeStyle = "#0c0c0c";
    c.lineWidth = 0.06 * PX;
    c.stroke();
    // jersey label on a roster-locked export, else the track id
    const tag = slots?.[String(pidOf(p))] ?? String(pidOf(p));
    c.fillStyle = "#0c0c0c";
    c.font = `700 ${(tag.length > 2 ? 0.40 : 0.55) * PX}px system-ui, sans-serif`;
    c.textAlign = "center";
    c.textBaseline = "middle";
    c.fillText(tag, X, Y);
    // "Ncam" tag
    c.fillStyle = color;
    c.font = `${0.42 * PX}px system-ui, sans-serif`;
    c.textAlign = "left";
    c.textBaseline = "alphabetic";
    c.fillText(`${p.n_cams}cam`, X + 0.6 * PX, Y - 0.5 * PX);
  }

  // origin marker
  c.fillStyle = "#888";
  c.beginPath();
  c.arc(cX(0), cY(0), 0.08 * PX, 0, Math.PI * 2);
  c.fill();
}

/** The People map tab: athletes from above, with a timeline and a camera inset. */
export default function PeopleMap({ scene }: Props) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  // The timeline position is shared with the rest of the viewer, so it lives in zustand.
  // https://zustand.docs.pmnd.rs/
  const idx = useViewer(function (s) { return s.peopleTimeIndex; });
  const setIdx = useViewer(function (s) { return s.setPeopleTimeIndex; });
  const [playing, setPlaying] = useState(false);
  /** Playback and export speed multiplier (1x = wall-clock). */
  const [speed, setSpeed] = useState(1);

  // Everything under public/ is served next to the app, so a plain fetch is enough.
  // https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
  useEffect(function () {
    fetch("/court_keypoints.json")
      .then(function (r) { return (r.ok ? r.json() : null); })
      .then(function (d) { if (d) setCatalog(d as Catalog); })
      .catch(function () { /* courts just won't draw */ });
  }, []);

  const court = useMemo(
    function () { return (catalog ? buildCourtModel(courtDimsFromCatalog(catalog.courts)) : []); },
    [catalog],
  );
  const hoops = useMemo(
    function () {
      return (catalog?.keypoints ?? [])
        .filter(function (k) { return k.id === "bb_hoop_E" || k.id === "bb_hoop_W"; })
        .map(function (k) { return { id: k.id, x: k.world[0], y: k.world[1], z: k.world[2] }; });
    },
    [catalog],
  );

  // 2D top-down map (default) vs 3D SMPL-X bodies.
  const [mode, setMode] = useState<"2d" | "3d">("2d");

  const people = scene.people;
  const peopleTs = useMemo(function () { return people?.timestamps ?? []; }, [people]);
  const pmeta = people?.meta;
  const fps = pmeta?.fps && pmeta.fps > 0 ? pmeta.fps : 25;
  const sessionLabel = pmeta?.label ?? null;
  const tracked = !!pmeta?.tracked;

  // Track id -> [x,y] per timeline slot, null where that track is absent.
  // https://react.dev/reference/react/useMemo
  const trailsById = useMemo(function () {
    const m = new Map<number, ([number, number] | null)[]>();
    if (!people) return m;
    peopleTs.forEach(function (tt, k) {
      for (const p of people.frames[String(tt)] ?? []) {
        if (!isAthlete(p)) continue;      // a coach standing still leaves a dot, not a trail
        const id = pidOf(p);
        let arr = m.get(id);
        if (!arr) { arr = new Array(peopleTs.length).fill(null); m.set(id, arr); }
        arr[k] = [p.pos[0], p.pos[1]];
      }
    });
    return m;
  }, [people, peopleTs]);

  /** How much history to draw behind each athlete, in seconds (0 = off). */
  const [trailSec, setTrailSec] = useState(3);
  /** Timeline step in seconds - the sampling rate the fuse actually ran at. */
  const stepSec = peopleTs.length > 1 ? (peopleTs[1] - peopleTs[0]) / fps : 1 / fps;
  /** One timeline step held for its real duration = 1x is wall-clock speed. */
  const holdBaseMs = peopleTs.length > 1 ? Math.round(stepSec * 1000) : EXPORT_HOLD_MS;
  const trailSamples = trailSec >= 999 ? peopleTs.length : Math.round(trailSec / Math.max(stepSec, 1e-6));

  /** Trail polylines ending at index k, split at gaps. Only tracks present at k. */
  function buildTrails(k: number): TrailDraw[] {
    if (trailSamples <= 0) return [];
    const from = Math.max(0, k - trailSamples);
    const out: TrailDraw[] = [];
    trailsById.forEach(function (arr, id) {
      if (!arr[k]) return;
      const segs: [number, number][][] = [];
      let cur: [number, number][] = [];
      for (let j = from; j <= k; j++) {
        const p = arr[j];
        if (p) cur.push(p);
        else if (cur.length) { segs.push(cur); cur = []; }
      }
      if (cur.length) segs.push(cur);
      const keep = segs.filter(function (s) { return s.length > 1; });
      if (keep.length) out.push({ color: PALETTE[id % PALETTE.length], segs: keep });
    });
    return out;
  }

  // Bodies manifest without the mesh data. In 3D mode it drives the timeline and the count.
  const [bodies, setBodies] = useState<
    {
      timestamps: number[];
      frames: Record<string, { track_id: number }[]>;
      /** Which tracking run the bodies were fitted from. */
      tracked_source?: string | null;
    } | null
  >(null);
  useEffect(function () {
    fetch("/bodies.json")
      .then(function (r) { return (r.ok ? r.json() : null); })
      .then(function (d) { if (d) setBodies(d); })
      .catch(function () { /* 3D mode shows how to create the bodies */ });
  }, []);

  // Warn when the bodies came from a different session than the 2D block.
  const bodiesForeign = useMemo(function () {
    if (!bodies || peopleTs.length === 0) return false;
    // Best check: which tracking run each side came from.
    const bsrc = bodies.tracked_source;
    const psrc = pmeta?.source;
    if (bsrc && psrc) return bsrc !== psrc;
    // Older exports carry no stamp: fall back to a disjoint timeline (weak).
    return !bodies.timestamps.some(function (bt) { return peopleTs.includes(bt); });
  }, [bodies, pmeta, peopleTs]);

  const ts = (mode === "3d" && bodies && bodies.timestamps.length) ? bodies.timestamps : peopleTs;
  const i = Math.max(0, Math.min(idx, ts.length - 1));
  const t = ts[i];
  const frame = (people && t !== undefined) ? (people.frames[String(t)] ?? []) : [];
  const headerCount = (mode === "3d" && bodies && t !== undefined)
    ? (bodies.frames[String(t)]?.length ?? 0)
    : frame.length;
  const curTrails = useMemo(
    function () { return (mode === "2d" ? buildTrails(i) : []); },
    // buildTrails is rebuilt every render, so it is not a dep.
    [mode, i, trailsById, trailSamples],
  );
  const thumbCam = people?.thumb_cam ?? "cam13";
  const frameThumbs = people?.frame_thumbs ?? {};
  const curThumb = t !== undefined ? frameThumbs[String(t)] : undefined;

  // Width (px) of the bottom-right reference-camera inset; user-adjustable.
  const INSET_MIN = 150, INSET_MAX = 620, INSET_STEP = 80;
  const [insetW, setInsetW] = useState(220);

  // Auto-advance on play. getState() keeps the interval off a stale closure.
  // https://zustand.docs.pmnd.rs/
  // https://developer.mozilla.org/en-US/docs/Web/API/Window/setInterval
  useEffect(function () {
    if (!playing || ts.length < 2) return;
    const intervalMs = Math.max(30, Math.round(holdBaseMs / speed));
    const h = window.setInterval(function () {
      const cur = useViewer.getState().peopleTimeIndex;
      useViewer.getState().setPeopleTimeIndex((cur + 1) % ts.length);
    }, intervalMs);
    return function () { return window.clearInterval(h); };
  }, [playing, ts.length, speed, holdBaseMs]);

  // Video export: record the map + cam thumbnail across the timeline
  const [exporting, setExporting] = useState(false);
  const [exportPct, setExportPct] = useState(0);
  /** Best MediaRecorder format this browser supports, MP4 first then WebM.
   *  https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/mimeType */
  const exportFormat = useMemo(function () {
    const fallback = { mime: "", ext: "webm" };
    if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
      return fallback;
    }
    const candidates = [
      { mime: "video/mp4;codecs=avc1.42E01E", ext: "mp4" },
      { mime: "video/mp4;codecs=avc1", ext: "mp4" },
      { mime: "video/mp4", ext: "mp4" },
      { mime: "video/webm;codecs=vp9", ext: "webm" },
      { mime: "video/webm;codecs=vp8", ext: "webm" },
      { mime: "video/webm", ext: "webm" },
    ];
    return candidates.find(function (c) { return MediaRecorder.isTypeSupported(c.mime); }) ?? fallback;
  }, []);

  async function handleDownload() {
    if (exporting || !people || ts.length === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = EXPORT_W;
    canvas.height = EXPORT_H;
    const ctx = canvas.getContext("2d");
    if (
      !ctx ||
      typeof canvas.captureStream !== "function" ||
      typeof MediaRecorder === "undefined"
    ) {
      window.alert(
        "Video export needs a Chromium- or Firefox-based browser " +
        "(requires MediaRecorder + canvas.captureStream).",
      );
      return;
    }

    /** Composite one frame: map, caption, camera thumbnail. Synchronous, for the rAF loop. */
    function paintFrame(k: number, thumbs: Map<string, HTMLImageElement>) {
      const tk = ts[k];
      const fr = people!.frames[String(tk)] ?? [];

      drawMap(ctx!, fr, court, hoops, buildTrails(k), people?.slots);

      // caption (top-left)
      const tStr = `t${String(tk).padStart(6, "0")}`;
      const cap = `${sessionLabel ? sessionLabel + " · " : "People on the court · "}${tStr} · `
        + `${(tk / fps).toFixed(1)} s · ${fr.length} ${fr.length === 1 ? "person" : "people"}`;
      ctx!.font = "600 20px system-ui, sans-serif";
      ctx!.textAlign = "left";
      ctx!.textBaseline = "middle";
      const capW = ctx!.measureText(cap).width;
      ctx!.fillStyle = "rgba(12,12,12,0.55)";
      ctx!.fillRect(14, 14, capW + 24, 34);
      ctx!.fillStyle = "#ffcc66";
      ctx!.fillText(cap, 26, 14 + 17);

      // reference-camera inset (bottom-right), mirrors the on-screen overlay
      const rel = frameThumbs[String(tk)];
      const im = rel ? thumbs.get(rel) : undefined;
      if (im && im.naturalWidth > 0) {
        const margin = 18;
        const iw = Math.round(EXPORT_W * 0.30);
        const ih = Math.round(iw * (im.naturalHeight / im.naturalWidth));
        const x = EXPORT_W - iw - margin;
        const y = EXPORT_H - ih - margin;
        ctx!.fillStyle = "#0c0c0c";
        ctx!.fillRect(x - 2, y - 2, iw + 4, ih + 4);
        ctx!.drawImage(im, x, y, iw, ih);
        const barH = 24;
        ctx!.fillStyle = "rgba(12,12,12,0.78)";
        ctx!.fillRect(x, y + ih - barH, iw, barH);
        ctx!.fillStyle = "#dddddd";
        ctx!.font = "600 13px system-ui, sans-serif";
        ctx!.fillText(`${thumbCam}  ${tStr}`, x + 8, y + ih - barH / 2);
        ctx!.strokeStyle = "#3c3c3c";
        ctx!.lineWidth = 2;
        ctx!.strokeRect(x - 1, y - 1, iw + 2, ih + 2);
      }
    }

    setExporting(true);
    setExportPct(0);
    try {
      // Preload every thumbnail up front, so the capture loop never waits on the network.
      // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise/all
      const cache = new Map<string, HTMLImageElement>();
      const thumbsInOrder = ts.map(function (tt) { return frameThumbs[String(tt)]; });
      const uniqueThumbs = Array.from(
        new Set(thumbsInOrder.filter(function (u): u is string { return !!u; })),
      );
      await Promise.all(
        uniqueThumbs.map(async function (rel) {
          try { cache.set(rel, await loadImage(`/${rel}`)); } catch { /* frame just shows no inset */ }
        }),
      );

      const { mime, ext } = exportFormat;
      // Repaint from rAF and let the stream sample: track.requestFrame() is missing in some browsers.
      // https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/captureStream
      // https://developer.mozilla.org/en-US/docs/Web/API/MediaStream
      const FPS = 25;
      const stream = canvas.captureStream(FPS);
      const rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 6_000_000 } : undefined);
      // The recorder hands back the video in pieces; collect them and join them at the end.
      // https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/dataavailable_event
      const chunks: BlobPart[] = [];
      rec.ondataavailable = function (e) { if (e.data.size) chunks.push(e.data); };
      const stopped = new Promise<void>(function (res) {
        rec.onstop = function () { return res(); };
      });

      // Paint frame 0, then hold each timestamp for holdMs while the stream samples the canvas.
      const holdMs = Math.max(30, holdBaseMs / speed);
      paintFrame(0, cache);
      rec.start();
      const totalMs = holdMs * ts.length;
      // Index comes from elapsed time, so the video keeps real time if a frame draws slowly.
      // https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame
      // https://developer.mozilla.org/en-US/docs/Web/API/Performance/now
      await new Promise<void>(function (resolve) {
        const start = performance.now();
        let lastK = -1;
        function tick(now: number) {
          const elapsed = now - start;
          const k = Math.min(ts.length - 1, Math.floor(elapsed / holdMs));
          if (k !== lastK) {
            lastK = k;
            setExportPct((k + 1) / ts.length);
          }
          paintFrame(k, cache);
          if (elapsed >= totalMs) { resolve(); return; }
          requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
      });
      // Let the encoder flush the final frames before stopping.
      await sleep(150);
      rec.stop();
      await stopped;

      // Save with no server: a Blob, an object URL, a throwaway <a download>.
      // https://developer.mozilla.org/en-US/docs/Web/API/Blob
      // https://developer.mozilla.org/en-US/docs/Web/API/File_API/Using_files_from_web_applications
      // https://developer.mozilla.org/en-US/docs/Web/HTML/Element/a#download
      const blob = new Blob(chunks, { type: (mime || "video/webm").split(";")[0] });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const first = String(ts[0]).padStart(6, "0");
      const last = String(ts[ts.length - 1]).padStart(6, "0");
      a.href = url;
      a.download = `people_${thumbCam}_t${first}-t${last}.${ext}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.setTimeout(function () { return URL.revokeObjectURL(url); }, 5000);
    } catch (err) {
      console.error("[people-map] video export failed", err);
      window.alert("Video export failed: " + (err instanceof Error ? err.message : String(err)));
    } finally {
      setExporting(false);
      setExportPct(0);
    }
  }

  if (!people || ts.length === 0) {
    return (
      <div style={{ padding: 24, color: "#cccccc", maxWidth: 720 }}>
        <h2 style={{ color: "#fc6" }}>People map</h2>
        <p>No people data yet.</p>
        <p style={{ color: "#9d9d9d", lineHeight: 1.6 }}>
          Put the videos in <code>input_videos/</code> and run, from the project folder:
        </p>
        <pre style={{ background: "#181818", border: "1px solid #2b2b2b", borderRadius: 4, padding: 12, color: "#bdbdbd", fontSize: 12 }}>
{`uv run pipeline.py`}
        </pre>
        <p style={{ color: "#9d9d9d" }}>Then reload this page.</p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", width: "100%", height: "100%", background: "#1f1f1f", color: "#cccccc" }}>
      {/* ---------------- left column: how to make new data, then the legend ---------------- */}
      <aside
        style={{
          width: 300, flex: "0 0 300px", minWidth: 0, height: "100%",
          borderRight: "1px solid #2b2b2b", background: "#1a1a1a",
          display: "flex", flexDirection: "column",
        }}
      >
        {/* Legend */}
        <div style={{ padding: "12px 14px", overflowY: "auto", flex: 1, minHeight: 0 }}>
          <div style={{ fontSize: 14, color: "#fc6", fontWeight: 700, marginBottom: 8 }}>People map</div>

          <p style={{ fontSize: 11, color: "#bdbdbd", lineHeight: 1.45, margin: "0 0 14px" }}>
            To see other videos, put them in <code>input_videos/</code>, run{" "}
            <code>uv run pipeline.py</code> from the project folder, then reload this page.
          </p>

          <LegendRow swatch={<PersonGlyph color="#ff7b00" label="0" />}>
            Person ({tracked ? "track id" : "per-frame id"})
          </LegendRow>

          {tracked && (
            <LegendRow swatch={
              <svg width={26} height={18}>
                <path d="M2 14 C 8 4, 16 16, 24 5" fill="none" stroke="#00c2ff" strokeWidth={1.6} opacity={0.6} strokeLinecap="round" />
              </svg>
            }>
              Recent path
            </LegendRow>
          )}

          <LegendRow swatch={<HaloGlyph />}>
            Halo / “N cam” = cameras that saw the person
          </LegendRow>

          <LegendRow swatch={<svg width={26} height={18}><circle cx={13} cy={9} r={6} fill="none" stroke="#ffae42" strokeWidth={1.4} /></svg>}>
            Hoop
          </LegendRow>

          <LegendRow swatch={
            <svg width={26} height={18}>
              <line x1={2} y1={6} x2={24} y2={6} stroke="#e8e8e8" strokeWidth={1.6} />
              <line x1={2} y1={13} x2={24} y2={13} stroke="#55aaff" strokeWidth={1.6} />
            </svg>
          }>
            Basketball / volleyball lines
          </LegendRow>
        </div>
      </aside>

      {/* ---------------- right: header + map + timeline ---------------- */}
      <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0, height: "100%" }}>
        {/* Header */}
        <div style={{ padding: "8px 14px", borderBottom: "1px solid #2b2b2b", display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
          <span style={{ fontSize: 14, color: "#fc6", fontWeight: 600 }}>People on the court</span>
          {/* 2D map vs 3D SMPL-X bodies */}
          <span style={{ display: "inline-flex", border: "1px solid #3c3c3c", borderRadius: 4, overflow: "hidden" }}>
            {(["2d", "3d"] as const).map(function (md) {
              return (
                <button
                  key={md}
                  onClick={function () { setMode(md); }}
                  style={{
                    background: mode === md ? "#fc6" : "#252526",
                    color: mode === md ? "#1f1f1f" : "#cccccc",
                    border: "none", padding: "4px 12px", cursor: "pointer",
                    fontSize: 12, fontWeight: 600,
                  }}
                  title={md === "2d" ? "Top-down floor map" : "3D SMPL-X bodies"}
                >
                  {md === "2d" ? "2D map" : "3D bodies"}
                </button>
              );
            })}
          </span>
          <span style={{ fontSize: 12, color: "#9d9d9d" }}>
            frame <b style={{ color: "#fff" }}>t{String(t).padStart(6, "0")}</b> · {(t / fps).toFixed(1)} s · {headerCount} {headerCount === 1 ? "person" : "people"}
          </span>
          {sessionLabel && (
            <span
              style={{
                fontSize: 11, color: "#8fd18f", border: "1px solid #2f5e35", borderRadius: 3,
                padding: "2px 8px", background: "#1b2a1d",
              }}
              title={
                `session ${pmeta?.session ?? "?"}\n`
                + `positions: ${pmeta?.source ?? "?"}\n`
                + `calibration: ${pmeta?.calibration ?? "?"}\n`
                + `inset frames: ${thumbCam}, same footage`
              }
            >
              {sessionLabel}
            </span>
          )}
          <span style={{ fontSize: 11, color: "#6d6d6d", marginLeft: "auto" }}>
            {mode === "2d"
              ? "top-down · metres · X = long axis (28 m), Y = short axis (15 m)"
              : "SMPL-X mesh + triangulated 3D skeleton · drag to orbit · metres"}
          </span>
        </div>

        {/* Map + current-frame inset */}
        <div style={{ flex: 1, minHeight: 0, display: "flex", justifyContent: "center", alignItems: "center", padding: mode === "3d" ? 0 : 12, position: "relative" }}>
          {mode === "3d" ? (
            <>
              <BodiesCanvas court={court} hoops={hoops} t={t} />
              {bodiesForeign && (
                <div style={{
                  position: "absolute", left: 14, top: 12, maxWidth: 460, zIndex: 5,
                  background: "rgba(40,28,12,0.92)", border: "1px solid #6b4b1c", borderRadius: 4,
                  padding: "7px 10px", fontSize: 11, color: "#ffcc66", lineHeight: 1.5,
                }}>
                  These SMPL-X bodies are from a <b>different session</b> than the 2D map
                  {sessionLabel ? <> (<i>{sessionLabel}</i>)</> : null} — they have their own
                  timeline. Re-run <code>people/3D_People/export_bodies.py</code> on this session
                  to fit bodies to it.
                </div>
              )}
            </>
          ) : (
            <>
          {/* viewBox units are metres, so the shapes below are in world units. meet = fit inside.
              https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/viewBox
              https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/preserveAspectRatio */}
          <svg
            viewBox={`-1 -1 ${VW + 2} ${VH + 2}`}
            preserveAspectRatio="xMidYMid meet"
            style={{ width: "100%", height: "100%", background: "#15301a", border: "1px solid #2b2b2b", borderRadius: 6 }}
          >
            {/* court floor fill */}
            <rect x={vx(-14)} y={vy(7.5)} width={28} height={15} fill="#1c3a24" stroke="none" />

            {/* court lines: basketball white, volleyball blue */}
            {court.map(function (pl, k) {
              return (
                <path
                  key={k}
                  d={polyPath(pl.pts)}
                  fill="none"
                  stroke={pl.court === "basketball" ? "#e8e8e8" : "#55aaff"}
                  strokeWidth={pl.court === "basketball" ? 0.07 : 0.06}
                  strokeLinejoin="round"
                  opacity={0.9}
                />
              );
            })}

            {/* hoops */}
            {hoops.map(function (h) {
              return (
                <g key={h.id}>
                  <circle cx={vx(h.x)} cy={vy(h.y)} r={0.28} fill="none" stroke="#ffae42" strokeWidth={0.07} />
                  <text
                    x={vx(h.x)} y={vy(h.y) - 0.5}
                    fontSize={0.62} fill="#ffae42" textAnchor="middle"
                    style={{ fontWeight: 600 }}
                  >
                    {h.id === "bb_hoop_E" ? "Hoop E" : "Hoop W"}
                  </text>
                </g>
              );
            })}

            {/* track history (under the markers) */}
            {curTrails.map(function (tr, k) {
              return tr.segs.map(function (seg, j) {
                return (
                  <path
                    key={`${k}-${j}`}
                    d={polyPath(seg)}
                    fill="none"
                    stroke={tr.color}
                    strokeWidth={0.09}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    opacity={0.55}
                  />
                );
              });
            },
            )}

            {/* people. The key is the person id, so a marker keeps its DOM node.
                https://react.dev/learn/rendering-lists */}
            {frame.map(function (p) {
              const pid = pidOf(p);
              const color = colourOf(p);
              const athlete = isAthlete(p);
              const cx = vx(p.pos[0]), cy = vy(p.pos[1]);
              const conf = Math.max(2, p.n_cams);
              return (
                <g key={pid} opacity={athlete ? 1 : 0.45}>
                  {/* confidence halo: bigger/softer = more cameras agreeing */}
                  <circle cx={cx} cy={cy} r={0.35 + 0.06 * conf} fill={color} opacity={0.18} />
                  <circle cx={cx} cy={cy} r={0.42} fill={color} stroke="#0c0c0c" strokeWidth={0.06} />
                  <text
                    x={cx}
                    y={cy + (people?.slots?.[String(pid)] ? 0.16 : 0.22)}
                    fontSize={people?.slots?.[String(pid)] ? 0.40 : 0.55}
                    fill="#0c0c0c"
                    textAnchor="middle"
                    style={{ fontWeight: 700 }}
                  >
                    {people?.slots?.[String(pid)] ?? pid}
                  </text>
                  <text x={cx + 0.6} y={cy - 0.5} fontSize={0.42} fill={color} textAnchor="start">
                    {p.n_cams}cam
                  </text>
                </g>
              );
            })}

            {/* axis hint at origin */}
            <circle cx={vx(0)} cy={vy(0)} r={0.08} fill="#888" />
          </svg>

          {/* current-frame camera inset (overlay, bottom-right) - resizable */}
          {curThumb && (
            <div
              style={{
                position: "absolute", right: 22, bottom: 22, width: insetW, maxWidth: "calc(100% - 44px)",
                border: "1px solid #3c3c3c", borderRadius: 5, overflow: "hidden",
                background: "#0c0c0c", boxShadow: "0 2px 10px rgba(0,0,0,0.5)",
              }}
              title={`${thumbCam} at t${String(t).padStart(6, "0")}`}
            >
              <img src={`/${curThumb}`} alt={`${thumbCam} current frame`} style={{ display: "block", width: "100%", height: "auto" }} />
              <div style={{ padding: "2px 4px 2px 7px", fontSize: 10, color: "#9d9d9d", display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontWeight: 600, color: "#ddd" }}>{thumbCam}</span>
                <span style={{ fontVariantNumeric: "tabular-nums" }}>t{String(t).padStart(6, "0")}</span>
                <span style={{ marginLeft: "auto", display: "flex", gap: 3 }}>
                  <button
                    onClick={function () {
                      setInsetW(function (w) { return Math.max(INSET_MIN, w - INSET_STEP); });
                    }}
                    disabled={insetW <= INSET_MIN}
                    style={insetBtn}
                    title="Smaller"
                  >−</button>
                  <button
                    onClick={function () {
                      setInsetW(function (w) { return Math.min(INSET_MAX, w + INSET_STEP); });
                    }}
                    disabled={insetW >= INSET_MAX}
                    style={insetBtn}
                    title="Bigger"
                  >+</button>
                </span>
              </div>
            </div>
          )}
            </>
          )}
        </div>

        {/* Timeline controls */}
        <div style={{ padding: "8px 14px", borderTop: "1px solid #2b2b2b", display: "flex", alignItems: "center", gap: 12 }}>
          <button
            onClick={function () {
              setPlaying(function (v) { return !v; });
            }}
            disabled={ts.length < 2}
            style={{
              background: playing ? "#fc6" : "#252526", color: playing ? "#1f1f1f" : "#cccccc",
              border: "1px solid #3c3c3c", borderRadius: 3, padding: "4px 14px", cursor: ts.length < 2 ? "default" : "pointer",
              fontSize: 12, fontWeight: 600, minWidth: 64,
            }}
            title={ts.length < 2 ? "Need ≥2 timestamps to animate" : playing ? "Pause" : "Play"}
          >
            {playing ? "❚❚ Pause" : "▶ Play"}
          </button>
          <span
            style={{ display: "flex", alignItems: "center", gap: 6 }}
            title="Sequence speed — scales both on-screen playback and the downloaded video (higher × = shorter)."
          >
            <span style={{ fontSize: 11, color: "#9d9d9d" }}>Speed</span>
            <input
              type="range"
              min={0.25}
              max={8}
              step={0.25}
              value={speed}
              onChange={function (e) { setSpeed(Number.parseFloat(e.target.value)); }}
              style={{ width: 96 }}
            />
            <span style={{ fontSize: 12, color: "#fc6", minWidth: 40, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
              {speed}×
            </span>
          </span>
          {mode === "2d" && (
            <span
              style={{ display: "flex", alignItems: "center", gap: 6 }}
              title="How much of each athlete's recent path to draw behind them. Also applies to the downloaded video."
            >
              <span style={{ fontSize: 11, color: "#9d9d9d" }}>Trails</span>
              <select
                value={trailSec}
                onChange={function (e) { setTrailSec(Number.parseFloat(e.target.value)); }}
                style={{
                  background: "#252526", color: "#cccccc", border: "1px solid #3c3c3c",
                  borderRadius: 3, fontSize: 11, padding: "2px 4px",
                }}
              >
                <option value={0}>off</option>
                <option value={1}>1 s</option>
                <option value={3}>3 s</option>
                <option value={10}>10 s</option>
                <option value={999}>all</option>
              </select>
            </span>
          )}
          <button onClick={function () { setIdx(Math.max(0, i - 1)); }} style={stepBtn} title="Previous frame">‹</button>
          <input
            type="range"
            min={0}
            max={ts.length - 1}
            step={1}
            value={i}
            onChange={function (e) { setIdx(Number.parseInt(e.target.value, 10)); }}
            style={{ flex: 1 }}
          />
          <button onClick={function () { setIdx(Math.min(ts.length - 1, i + 1)); }} style={stepBtn} title="Next frame">›</button>
          {/* tabular-nums stops the counter shifting; padStart pads to six digits.
              https://developer.mozilla.org/en-US/docs/Web/CSS/font-variant-numeric
              https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/padStart */}
          <span style={{ fontSize: 12, color: "#fc6", minWidth: 130, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
            {i + 1} / {ts.length} · t{String(t).padStart(6, "0")}
          </span>
          {mode === "2d" && (
            <button
              onClick={handleDownload}
              disabled={exporting || ts.length === 0}
              style={{
                background: exporting ? "#2a2a2a" : "#2e7d32", color: exporting ? "#9d9d9d" : "#eaffea",
                border: "1px solid #3c3c3c", borderRadius: 3, padding: "4px 14px",
                cursor: exporting || ts.length === 0 ? "default" : "pointer",
                fontSize: 12, fontWeight: 600, minWidth: 132,
              }}
              title={
                exporting
                  ? `Recording the timeline to a .${exportFormat.ext} video…`
                  : `Download a .${exportFormat.ext} video of the ${thumbCam} view + floor reconstruction across all ${ts.length} frames `
                    + `(~${Math.max(1, Math.round((holdBaseMs / speed) * ts.length / 1000))} s at ${speed}×)`
                    + (exportFormat.ext === "mp4" ? "" : " — this browser can't record MP4; open in Chrome/Edge for MP4")
              }
            >
              {exporting ? `Recording… ${Math.round(exportPct * 100)}%` : `⬇ Download .${exportFormat.ext}`}
            </button>
          )}
        </div>

        {/* Footnote */}
        <div style={{ padding: "4px 14px 10px", fontSize: 10, color: "#6d6d6d", lineHeight: 1.5 }}>
          Positions are where each person's feet touch the court, combined from all the cameras that
          saw them. The halo size shows how many cameras agreed.{" "}
          {tracked
            ? "A colour and label stay with the same athlete for the whole clip (team letter and shirt number; u = number not read)."
            : "Ids are per frame, so a colour is not the same person across timestamps."}
        </div>
      </div>
    </div>
  );
}

/** One legend line: swatch on the left, text on the right. */
function LegendRow({ swatch, children }: { swatch: React.ReactNode; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: 9, margin: "0 0 9px" }}>
      <div style={{ flex: "0 0 26px", paddingTop: 1 }}>{swatch}</div>
      <div style={{ fontSize: 11, color: "#bdbdbd", lineHeight: 1.45 }}>{children}</div>
    </div>
  );
}

/** Legend swatch: a coloured dot with the athlete's label on it. */
function PersonGlyph({ color, label }: { color: string; label: string }) {
  return (
    <svg width={26} height={18}>
      <circle cx={13} cy={9} r={5.6} fill={color} stroke="#0c0c0c" strokeWidth={0.8} />
      <text x={13} y={12} fontSize={8} fill="#0c0c0c" textAnchor="middle" style={{ fontWeight: 700 }}>{label}</text>
    </svg>
  );
}

/** Legend swatch: a dot with the confidence halo around it. */
function HaloGlyph() {
  return (
    <svg width={26} height={18}>
      <circle cx={13} cy={9} r={8.5} fill="#ff7b00" opacity={0.18} />
      <circle cx={13} cy={9} r={5} fill="#ff7b00" stroke="#0c0c0c" strokeWidth={0.8} />
    </svg>
  );
}

const stepBtn: React.CSSProperties = {
  background: "#252526", color: "#cccccc", border: "1px solid #3c3c3c",
  borderRadius: 3, padding: "2px 10px", cursor: "pointer", fontSize: 14, lineHeight: 1,
};

const insetBtn: React.CSSProperties = {
  background: "#252526", color: "#cccccc", border: "1px solid #3c3c3c",
  borderRadius: 3, width: 20, height: 18, padding: 0, cursor: "pointer",
  fontSize: 13, lineHeight: 1, fontWeight: 700,
};
