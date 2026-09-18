/**
 * Label who is who at 30 instants of a clip, from the camera frames and a top-down map.
 * Saved to localStorage as you go, exported as gt_answers.json. Task: public/gt_task.json.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { buildCourtModel, courtDimsFromCatalog } from "../utils/courtModel";
import type { CourtPolyline } from "../utils/courtModel";
import { PALETTE } from "../utils/palette";

// gt_task.json schema
interface GtMeta {
  source: string;
  created: string;
  court: { x_half: number; y_half: number };
}
/** What the tracker guessed about one track, before a human checks it. */
interface GtTrackSeed {
  role: string;
  off_team: boolean | null;   // null = tracker could not split this track by team
  number: string | null;
  number_margin: number | null;
  n: number;
}
/** One person as seen by one camera: box and foot point, in image px. */
interface GtView {
  cam: string;
  bbox: [number, number, number, number];
  foot: [number, number] | null;
}
/** One person at one instant: map position plus every camera view of them. */
interface GtPerson {
  pid: number;
  track_id: number;
  pos: [number, number];
  views: GtView[];
}
/** One camera's frame for an instant, with the image size it was saved at. */
interface GtCam { cam: string; img: string; w: number; h: number; }
/** One instant to label: its frame number, the camera frames and the people. */
interface GtInstant { t: number; sec: number; cams: GtCam[]; people: GtPerson[]; }
/** public/gt_task.json: the frames and tracks to label. */
interface GtTask {
  meta: GtMeta;
  tracks: Record<string, GtTrackSeed>;
  instants: GtInstant[];
}

// What the annotator has decided so far. Saved continuously, never uploaded anywhere.
const TEAMS = ["A", "B", "official", "other"] as const;
/** Which side a person belongs to. */
type Team = (typeof TEAMS)[number];
/** Whether a detection is kept or marked a ghost. */
type Status = "ok" | "ghost";

/** GT label for a whole track - the default that propagates to every instant. */
interface TrackLabel { team?: Team; number?: string; }
/** Per-(instant,person) corrections. team/number present = per-instant override. */
interface PersonAnn {
  status?: Status;
  team?: Team;
  number?: string;
  pos?: [number, number];
  posEdited?: boolean;
}
/** Someone the annotator added by hand, missing from the detections. */
interface AddedPerson { id: number; pos: [number, number]; team?: Team; number?: string; }
/** One instant's annotations. */
interface InstantAnn {
  done?: boolean;
  people?: Record<string, PersonAnn>;
  added?: AddedPerson[];
}
/** The whole annotation file, kept in localStorage and exported as gt_answers.json. */
interface AnnState {
  tracks: Record<string, TrackLabel>;
  instants: Record<string, InstantAnn>;
}
// Starting point when nothing has been annotated yet.
const EMPTY_ANN: AnnState = { tracks: {}, instants: {} };

// Live consistency checks: the same number on two athletes metres apart is a mistake.
interface DupConflict { team: Team; number: string; names: string[]; dist: number; }

/** Rows of one instant under the current annotations: ghosts out, added people in. */
function labeledRowsAt(ins: GtInstant, ann: AnnState):
    { name: string; team: Team | null; number: string; pos: [number, number] }[] {
  const ia = ann.instants[String(ins.t)] ?? {};
  const rows: { name: string; team: Team | null; number: string; pos: [number, number] }[] = [];
  for (const p of ins.people) {
    const pa = ia.people?.[String(p.pid)];
    if ((pa?.status ?? "ok") === "ghost") continue;
    const tl = ann.tracks[String(p.track_id)];
    rows.push({
      name: `track ${p.track_id}`,
      team: pa?.team ?? tl?.team ?? null,
      number: pa?.number ?? tl?.number ?? "",
      pos: pa?.pos ?? p.pos,
    });
  }
  for (const a of ia.added ?? [])
    rows.push({ name: "added", team: a.team ?? null, number: a.number ?? "", pos: a.pos });
  return rows;
}

/** Same team and number twice: closer than this a duplicate, further apart a wrong label. */
const DUP_NEAR_M = 2.0;
const TWO_LABELS_M = 0.6;   // same team, different numbers closer than this = likely one athlete
/** Two different numbers on people standing too close to be two athletes. */
interface NearConflict { team: Team; na: string; nb: string; a: string; b: string; dist: number; }
/** Every warning for one instant, grouped by severity. */
interface InstantChecks { red: DupConflict[]; yellow: DupConflict[]; orange: NearConflict[]; over: string[]; }

// Distances are in metres.
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/hypot
function dupConflictsAt(ins: GtInstant, ann: AnnState): InstantChecks {
  const rows = labeledRowsAt(ins, ann).filter(
    function (r) { return (r.team === "A" || r.team === "B") && r.number; },
  ) as { name: string; team: Team; number: string; pos: [number, number] }[];
  const groups = new Map<string, { name: string; pos: [number, number] }[]>();
  for (const r of rows) {
    const key = `${r.team}|${r.number}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push({ name: r.name, pos: r.pos });
  }
  const red: DupConflict[] = [], yellow: DupConflict[] = [];
  for (const [key, v] of groups) {
    if (v.length < 2) continue;
    let dist = 0;
    for (let a = 0; a < v.length; a++)
      for (let b = a + 1; b < v.length; b++)
        dist = Math.max(dist, Math.hypot(v[a].pos[0] - v[b].pos[0], v[a].pos[1] - v[b].pos[1]));
    const [team, number] = key.split("|") as [Team, string];
    (dist >= DUP_NEAR_M ? red : yellow).push({ team, number, names: v.map(function (x) { return x.name; }), dist });
  }
  // same team, different numbers, nearly the same spot: one athlete wearing two labels?
  const orange: NearConflict[] = [];
  for (let a = 0; a < rows.length; a++)
    for (let b = a + 1; b < rows.length; b++) {
      const ra = rows[a], rb = rows[b];
      if (ra.team !== rb.team || ra.number === rb.number) continue;
      const d = Math.hypot(ra.pos[0] - rb.pos[0], ra.pos[1] - rb.pos[1]);
      if (d < TWO_LABELS_M)
        orange.push({ team: ra.team, na: ra.number, nb: rb.number, a: ra.name, b: rb.name, dist: d });
    }
  // more than five numbered players per team on court
  const over: string[] = [];
  for (const tm of ["A", "B"] as Team[]) {
    const teamRows = rows.filter(function (r) { return r.team === tm; });
    const nums = new Set(teamRows.map(function (r) { return r.number; }));
    if (nums.size > 5)
      over.push(`${nums.size} distinct ${tm} numbers (${[...nums].sort().join(", ")}) — only five can be on court; one row is mislabelled`);
  }
  return { red, yellow, orange, over };
}
// How many warnings count as real problems; yellow ones do not.
function nIssues(c: InstantChecks) { return c.red.length + c.orange.length + c.over.length; }

// The top-down map, using the same conventions as the People map so the two look alike
const XMIN = -15.5, XMAX = 15.5, YMIN = -8.5, YMAX = 8.5;
const VW = XMAX - XMIN, VH = YMAX - YMIN;
// World (X,Y) -> SVG viewBox coords, Y flipped so +Y is up on screen.
function vx(x: number) { return x - XMIN; }
function vy(y: number) { return YMAX - y; }
// An SVG path "d" string: M moves, L draws a line to the next point.
// https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d
// https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths
function polyPath(pts: [number, number][]): string {
  return pts.map(function (p, i) { return `${i ? "L" : "M"}${vx(p[0]).toFixed(3)},${vy(p[1]).toFixed(3)}`; }).join(" ");
}


// Palette colour for a track id, wrapping round the palette.
function trackColor(tid: number) {
  return PALETTE[((tid % PALETTE.length) + PALETTE.length) % PALETTE.length];
}

/** Ring colour once a team is assigned. */
const TEAM_RING: Record<Team, string> = {
  A: "#00c2ff", B: "#ff4545", official: "#e8e8e8", other: "#8a8a8a",
};
// Green, for a person added by hand.
const ADDED_COLOR = "#7CFF45";
// Yellow, for a label overridden in this instant only.
const OVERRIDE_BADGE = "#ffd400";

/** What is selected on the map: a detected person, an added one, or nothing. */
type Sel = { kind: "p"; pid: number } | { kind: "a"; id: number } | null;

/** The team and number in force for one person, after any override. */
interface Effective { team: Team | null; number: string; overridden: boolean; }

/** The Annotate tab: label who is who at each instant, from the camera frames and the map. */
export default function Annotate() {
  const [task, setTask] = useState<GtTask | null>(null);
  const [taskErr, setTaskErr] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<{ courts?: Record<string, unknown>; keypoints?: { id: string; world: number[] }[] } | null>(null);

  // Fetch once (never re-fetched on instant change). Both files are served from public/.
  // https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
  useEffect(function () {
    fetch("/gt_task.json")
      .then(function (r) { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(function (d: GtTask) { return setTask(d); })
      .catch(function (e: Error) { return setTaskErr(e.message); });
    fetch("/court_keypoints.json")
      .then(function (r) { return (r.ok ? r.json() : null); })
      .then(function (d) { if (d) setCatalog(d); })
      .catch(function () { /* court lines just won't draw */ });
  }, []);

  const court: CourtPolyline[] = useMemo(
    function () { return (catalog ? buildCourtModel(courtDimsFromCatalog(catalog.courts)) : []); },
    [catalog],
  );
  const hoops = useMemo(
    function () {
      return (catalog?.keypoints ?? [])
        .filter(function (k) { return k.id === "bb_hoop_E" || k.id === "bb_hoop_W"; })
        .map(function (k) { return { id: k.id, x: k.world[0], y: k.world[1] }; });
    },
    [catalog],
  );

  // Kept across a reload. The key has the task's stamps, so a new task starts clean.
  // https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage
  const [ann, setAnn] = useState<AnnState>(EMPTY_ANN);
  const [annLoaded, setAnnLoaded] = useState(false);
  const storageKey = task ? `gt_annot::${task.meta.source}::${task.meta.created}` : null;

  useEffect(function () {
    if (!storageKey) return;
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw) {
        const parsed = JSON.parse(raw) as AnnState;
        if (parsed && typeof parsed === "object") {
          setAnn({ tracks: parsed.tracks ?? {}, instants: parsed.instants ?? {} });
        }
      }
    } catch { /* start fresh */ }
    setAnnLoaded(true);
  }, [storageKey]);

  // setItem throws when the origin is out of quota, so save on a best-effort basis.
  // https://developer.mozilla.org/en-US/docs/Web/API/Storage/setItem
  useEffect(function () {
    if (!storageKey || !annLoaded) return;
    try { localStorage.setItem(storageKey, JSON.stringify(ann)); } catch { /* out of quota, keep working in memory */ }
  }, [ann, storageKey, annLoaded]);

  // where we are in the task
  const [idx, setIdx] = useState(0);
  const [sel, setSel] = useState<Sel>(null);
  const [overrideMode, setOverrideMode] = useState(false);
  const [addMode, setAddMode] = useState(false);
  const [numBuf, setNumBuf] = useState<string | null>(null);
  const [zoomCam, setZoomCam] = useState<string | null>(null);
  const [showGuide, setShowGuide] = useState(false);
  const [showJson, setShowJson] = useState(false);

  const nInst = task?.instants.length ?? 0;
  const i = Math.max(0, Math.min(idx, Math.max(0, nInst - 1)));
  const inst: GtInstant | null = task ? task.instants[i] : null;
  const instKey = inst ? String(inst.t) : "";
  const instAnn: InstantAnn = (inst && ann.instants[instKey]) || {};

  function gotoInstant(k: number) {
    if (!task) return;
    const kk = Math.max(0, Math.min(task.instants.length - 1, k));
    setIdx(kk);
    setSel(null);
    setAddMode(false);
    setNumBuf(null);
    setZoomCam(null);
  }

  // the edits
  function updTrack(tid: number, patch: TrackLabel) {
    return setAnn(function (a) {
      return {
        ...a,
        tracks: { ...a.tracks, [String(tid)]: { ...a.tracks[String(tid)], ...patch } },
      };
    });
  }

  function updPerson(tKey: string, pid: number, fn: (cur: PersonAnn) => PersonAnn) {
    return setAnn(function (a) {
      const ia = a.instants[tKey] ?? {};
      const people = { ...(ia.people ?? {}) };
      people[String(pid)] = fn(people[String(pid)] ?? {});
      return { ...a, instants: { ...a.instants, [tKey]: { ...ia, people } } };
    });
  }

  function updInstant(tKey: string, patch: Partial<InstantAnn>) {
    return setAnn(function (a) {
      return {
        ...a,
        instants: { ...a.instants, [tKey]: { ...(a.instants[tKey] ?? {}), ...patch } },
      };
    });
  }

  function updAdded(tKey: string, id: number, fn: (cur: AddedPerson) => AddedPerson) {
    return setAnn(function (a) {
      const ia = a.instants[tKey] ?? {};
      const added = (ia.added ?? []).map(function (x) { return (x.id === id ? fn(x) : x); });
      return { ...a, instants: { ...a.instants, [tKey]: { ...ia, added } } };
    });
  }

  function addPerson(tKey: string, pos: [number, number]) {
    // Id from the current state: the updater below may not run synchronously.
    const existing = ann.instants[tKey]?.added ?? [];
    const newId = existing.reduce(function (m, x) { return Math.max(m, x.id); }, 0) + 1;
    setAnn(function (a) {
      const ia = a.instants[tKey] ?? {};
      const added = ia.added ?? [];
      return {
        ...a,
        instants: { ...a.instants, [tKey]: { ...ia, added: [...added, { id: newId, pos }] } },
      };
    });
    setSel({ kind: "a", id: newId });
  }

  function deleteAdded(tKey: string, id: number) {
    setAnn(function (a) {
      const ia = a.instants[tKey] ?? {};
      return {
        ...a,
        instants: { ...a.instants, [tKey]: { ...ia, added: (ia.added ?? []).filter(function (x) {
          return x.id !== id;
        }) } },
      };
    });
    setSel(null);
  }

  function resetInstant() {
    if (!inst) return;
    if (!window.confirm(`Reset every annotation on instant ${i + 1} (t${inst.t})?`)) return;
    setAnn(function (a) {
      const instants = { ...a.instants };
      delete instants[instKey];
      return { ...a, instants };
    });
    setSel(null);
  }

  // what a person is actually labelled as here: a per-instant override beats the track's own
  function effectiveOf(p: GtPerson): Effective {
    const pa = instAnn.people?.[String(p.pid)];
    const tl = ann.tracks[String(p.track_id)];
    const overridden = pa?.team !== undefined || pa?.number !== undefined;
    return {
      team: pa?.team ?? tl?.team ?? null,
      number: pa?.number ?? tl?.number ?? "",
      overridden,
    };
  }
  function statusOf(p: GtPerson): Status {
    return instAnn.people?.[String(p.pid)]?.status ?? "ok";
  }
  function posOf(p: GtPerson): [number, number] {
    return instAnn.people?.[String(p.pid)]?.pos ?? p.pos;
  }
  function posEdited(p: GtPerson): boolean {
    return !!instAnn.people?.[String(p.pid)]?.posEdited;
  }

  /** Route an edit: the added person, this (t,pid) in override mode, else the whole track. */
  function applyLabel(patch: { team?: Team; number?: string }) {
    if (!inst || !sel) return;
    if (sel.kind === "a") {
      updAdded(instKey, sel.id, function (c) { return { ...c, ...patch }; });
      return;
    }
    const p = inst.people.find(function (pp) { return pp.pid === sel.pid; });
    if (!p) return;
    if (overrideMode) updPerson(instKey, p.pid, function (c) { return { ...c, ...patch }; });
    else updTrack(p.track_id, patch);
  }

  function clearOverride(pid: number) {
    return updPerson(instKey, pid, function (c) {
      const rest: PersonAnn = { ...c };
      delete rest.team;
      delete rest.number;
      return rest;
    });
  }

  function toggleGhost() {
    if (!inst || !sel || sel.kind !== "p") return;
    updPerson(instKey, sel.pid, function (c) {
      return { ...c, status: (c.status ?? "ok") === "ghost" ? "ok" : "ghost" };
    });
  }

  function toggleDone() {
    if (!inst) return;
    updInstant(instKey, { done: !instAnn.done });
  }

  // dragging somebody to the right place, or adding one the tracker missed
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<{ kind: "p" | "a"; key: number; startX: number; startY: number; moved: boolean } | null>(null);

  /** Screen px -> world metres, clamped to the map and rounded to the centimetre.
   *  https://developer.mozilla.org/en-US/docs/Web/API/SVGGraphicsElement/getScreenCTM
   *  https://developer.mozilla.org/en-US/docs/Web/API/DOMPoint
   *  https://developer.mozilla.org/en-US/docs/Web/API/DOMMatrix */
  function clientToWorld(cx: number, cy: number): [number, number] | null {
    const svg = svgRef.current;
    if (!svg) return null;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const p = new DOMPoint(cx, cy).matrixTransform(ctm.inverse());
    const wx = Math.max(XMIN + 0.2, Math.min(XMAX - 0.2, p.x + XMIN));
    const wy = Math.max(YMIN + 0.2, Math.min(YMAX - 0.2, YMAX - p.y));
    return [Math.round(wx * 100) / 100, Math.round(wy * 100) / 100];
  }

  // Pointer capture keeps the move events coming when the cursor leaves the marker.
  // https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events
  // https://developer.mozilla.org/en-US/docs/Web/API/Element/setPointerCapture
  function startDrag(e: React.PointerEvent, kind: "p" | "a", key: number) {
    e.stopPropagation();
    setSel(kind === "p" ? { kind: "p", pid: key } : { kind: "a", id: key });
    setNumBuf(null);
    dragRef.current = { kind, key, startX: e.clientX, startY: e.clientY, moved: false };
    try { svgRef.current?.setPointerCapture(e.pointerId); } catch { /* older engines */ }
  }

  function onMapPointerMove(e: React.PointerEvent) {
    const d = dragRef.current;
    if (!d || !inst) return;
    if (!d.moved && Math.abs(e.clientX - d.startX) + Math.abs(e.clientY - d.startY) < 5) return;
    d.moved = true;
    const w = clientToWorld(e.clientX, e.clientY);
    if (!w) return;
    if (d.kind === "p") updPerson(instKey, d.key, function (c) { return { ...c, pos: w, posEdited: true }; });
    else updAdded(instKey, d.key, function (c) { return { ...c, pos: w }; });
  }

  // Pointer capture retargets the post-drag click to the svg, so suppress that one click.
  // https://developer.mozilla.org/en-US/docs/Web/API/Element/setPointerCapture
  const suppressClickRef = useRef(false);
  function onMapPointerUp() {
    if (dragRef.current) {
      suppressClickRef.current = true;
      // Clear on the next tick so the flag can never swallow a later background click.
      window.setTimeout(function () { suppressClickRef.current = false; }, 0);
    }
    dragRef.current = null;
  }

  function onMapClick(e: React.MouseEvent) {
    if (suppressClickRef.current) { suppressClickRef.current = false; return; }
    if (!inst) return;
    if (addMode) {
      const w = clientToWorld(e.clientX, e.clientY);
      if (w) addPerson(instKey, w);
      setAddMode(false);
    } else {
      setSel(null);
      setNumBuf(null);
    }
  }

  function onMapDblClick(e: React.MouseEvent) {
    if (!inst) return;
    const w = clientToWorld(e.clientX, e.clientY);
    if (w) addPerson(instKey, w);
  }

  // Keyboard shortcuts. e.key is the character produced, so it follows the user's layout.
  // https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key
  // https://developer.mozilla.org/en-US/docs/Web/API/Element/keydown_event
  useEffect(function () {
    function onKey(e: KeyboardEvent) {
      // Never steal keys while the user is typing in a field.
      const tag = (e.target as HTMLElement | null)?.tagName ?? "";
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (!task || !inst) return;

      if (e.key === "Escape") {
        if (zoomCam) { setZoomCam(null); return; }
        if (numBuf !== null) { setNumBuf(null); return; }
        setSel(null);
        return;
      }
      if (e.key === "ArrowRight" || e.key === "n") { gotoInstant(i + 1); return; }
      if (e.key === "ArrowLeft" || e.key === "p") { gotoInstant(i - 1); return; }
      if (e.key === "d") { toggleDone(); return; }
      if (e.key === "o") { setOverrideMode(function (v) { return !v; }); return; }

      if (!sel) return;
      if (e.key === "a") { applyLabel({ team: "A" }); return; }
      if (e.key === "b") { applyLabel({ team: "B" }); return; }
      if (e.key === "r") { applyLabel({ team: "official" }); return; }
      if (e.key === "x") { applyLabel({ team: "other" }); return; }
      if (e.key === "g") { toggleGhost(); return; }
      if (/^[0-9]$/.test(e.key)) { setNumBuf(function (b) { return (b ?? "") + e.key; }); return; }
      if (e.key === "Enter") {
        if (numBuf !== null) { applyLabel({ number: numBuf }); setNumBuf(null); }
        return;
      }
      if (e.key === "Backspace" && numBuf !== null) { setNumBuf(numBuf.slice(0, -1) || null); }
    }
    window.addEventListener("keydown", onKey);
    return function () { return window.removeEventListener("keydown", onKey); };
    // Short deps on purpose: the handler only reads these.
  }, [task, inst, i, sel, overrideMode, numBuf, zoomCam, instKey, instAnn]);

  // the exported answers file
  function buildAnswers() {
    if (!task) return null;
    // Role comes from the annotator's team choice only, never from the tracker seed.
    function teamRole(team: Team | null): string | null {
      if (team === "official") return "official";
      if (team === "other") return "other";
      if (team === "A" || team === "B") return "player";
      return null;
    }
    const tracksOut: Record<string, { team: Team | null; number: string; role: string | null }> = {};
    for (const tid of Object.keys(task.tracks)) {
      const tl = ann.tracks[tid];
      tracksOut[tid] = {
        team: tl?.team ?? null,
        number: tl?.number ?? "",
        role: teamRole(tl?.team ?? null),
      };
    }
    const instantsOut = task.instants.map(function (ins) {
      const tKey = String(ins.t);
      const ia = ann.instants[tKey] ?? {};
      const people = ins.people.map(function (p) {
        const pa = ia.people?.[String(p.pid)];
        const tl = ann.tracks[String(p.track_id)];
        const overridden = pa?.team !== undefined || pa?.number !== undefined;
        return {
          pid: p.pid,
          track_id: p.track_id,
          status: pa?.status ?? "ok",
          team: pa?.team ?? tl?.team ?? null,
          number: pa?.number ?? tl?.number ?? "",
          pos: pa?.pos ?? p.pos,
          pos_edited: !!pa?.posEdited,
          overridden,
        };
      });
      const added = (ia.added ?? []).map(function (a) {
        return {
          pos: a.pos, team: a.team ?? null, number: a.number ?? "",
        };
      });
      return { t: ins.t, done: !!ia.done, people, added };
    });
    return {
      source: task.meta.source,
      task_created: task.meta.created,
      exported: new Date().toISOString(),
      tracks: tracksOut,
      instants: instantsOut,
    };
  }

  const answersJson = useMemo(
    function () { return (showJson ? JSON.stringify(buildAnswers(), null, 1) : ""); },
    // buildAnswers is rebuilt every render, so it is not a dep.
    [showJson, ann, task],
  );

  // Save with no server: a Blob, an object URL, a throwaway <a download>.
  // https://developer.mozilla.org/en-US/docs/Web/API/Blob
  // https://developer.mozilla.org/en-US/docs/Web/API/File_API/Using_files_from_web_applications
  // https://developer.mozilla.org/en-US/docs/Web/HTML/Element/a#download
  function saveAnswers() {
    const out = buildAnswers();
    if (!out) return;
    const blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "gt_answers.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(function () { return URL.revokeObjectURL(url); }, 5000);
  }

  // a summary of everything overridden, so a mistake can be found again
  const overridesList = useMemo(function () {
    if (!task) return [];
    const out: { instIdx: number; t: number; pid: number; tid: number; team?: Team; number?: string }[] = [];
    task.instants.forEach(function (ins, k) {
      const ia = ann.instants[String(ins.t)];
      if (!ia?.people) return;
      for (const p of ins.people) {
        const pa = ia.people[String(p.pid)];
        if (pa && (pa.team !== undefined || pa.number !== undefined)) {
          out.push({ instIdx: k, t: ins.t, pid: p.pid, tid: p.track_id, team: pa.team, number: pa.number });
        }
      }
    });
    return out;
  }, [task, ann]);

  const doneCount = useMemo(
    function () {
      return (task ? task.instants.filter(function (ins) { return ann.instants[String(ins.t)]?.done; }).length : 0);
    },
    [task, ann],
  );

  // live duplicate-number checks (current instant, whole task, one-instant identities)
  const curConf = useMemo<InstantChecks>(
    function () { return (inst ? dupConflictsAt(inst, ann) : { red: [], yellow: [], orange: [], over: [] }); },
    [inst, ann],
  );
  const allRed = useMemo(function () {
    const m = new Map<number, number>();
    if (task) for (const ins of task.instants) m.set(ins.t, nIssues(dupConflictsAt(ins, ann)));
    return m;
  }, [task, ann]);
  const totalRed = useMemo(function () {
    let s = 0;
    allRed.forEach(function (v) { s += v; });
    return s;
  }, [allRed]);
  const singletons = useMemo(function () {
    if (!task) return [] as string[];
    const count = new Map<string, number>();
    for (const ins of task.instants)
      for (const r of labeledRowsAt(ins, ann))
        if ((r.team === "A" || r.team === "B") && r.number)
          count.set(`${r.team}${r.number}`, (count.get(`${r.team}${r.number}`) ?? 0) + 1);
    const once = [...count].filter(function ([, n]) { return n === 1; });
    return once.map(function ([k]) { return k; }).sort();
  }, [task, ann]);

  // nothing loaded yet, or nothing to load
  if (taskErr) {
    return (
      <div style={{ padding: 24, color: "#cccccc", maxWidth: 760 }}>
        <h2 style={{ color: "#fc6" }}>Annotate: no task file</h2>
        <p>Could not load <code>/gt_task.json</code> ({taskErr}).</p>
        <p style={{ color: "#9d9d9d", lineHeight: 1.6 }}>
          This tab was used during the thesis to label a first game by hand. Its task file is not
          made by the pipeline.
        </p>
      </div>
    );
  }
  if (!task || !inst) {
    return <div style={{ padding: 24, color: "#cccccc" }}>Loading gt_task.json…</div>;
  }

  const selPerson: GtPerson | null =
    sel?.kind === "p" ? inst.people.find(function (p) { return p.pid === sel.pid; }) ?? null : null;
  const selAdded: AddedPerson | null =
    sel?.kind === "a" ? (instAnn.added ?? []).find(function (a) { return a.id === sel.id; }) ?? null : null;

  // Per-camera draw entries for the frame overlays.
  const drawForCam = function (cam: string): DrawEntry[] {
    const out: DrawEntry[] = [];
    for (const p of inst.people) {
      const v = p.views.find(function (vv) { return vv.cam === cam; });
      if (!v) continue;
      const eff = effectiveOf(p);
      out.push({
        pid: p.pid,
        tid: p.track_id,
        color: trackColor(p.track_id),
        bbox: v.bbox,
        foot: v.foot,
        team: eff.team,
        number: eff.number,
        overridden: eff.overridden,
        ghost: statusOf(p) === "ghost",
        selected: sel?.kind === "p" && sel.pid === p.pid,
      });
    }
    return out;
  };

  const zoomCamInfo = zoomCam ? inst.cams.find(function (c) { return c.cam === zoomCam; }) ?? null : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", height: "100%", background: "#1f1f1f", color: "#cccccc", overflow: "hidden" }}>
      {/* ---------------- header ---------------- */}
      <div style={{ padding: "6px 14px", borderBottom: "1px solid #2b2b2b", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <span style={{ fontSize: 14, color: "#fc6", fontWeight: 700 }}>Ground-truth annotation</span>
        <span style={{ fontSize: 11, color: "#9d9d9d" }}>
          {task.meta.source} · {task.instants.length} instants · {Object.keys(task.tracks).length} tracks
        </span>
        {/* progress */}
        <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 140, height: 8, background: "#2b2b2b", borderRadius: 4, overflow: "hidden" }}>
            <span style={{ display: "block", width: `${(doneCount / task.instants.length) * 100}%`, height: "100%", background: "#2e7d32" }} />
          </span>
          <span style={{ fontSize: 11, color: doneCount === task.instants.length ? "#7CFF45" : "#9d9d9d" }}>
            {doneCount} of {task.instants.length} instants done
          </span>
        </span>
        {(totalRed > 0 || singletons.length > 0) && (
          <span
            style={{ fontSize: 11, color: "#ff8a80" }}
            title="Labels that need fixing. 'Seen once' means a player labelled in only one moment, often a typo."
          >
            {totalRed > 0 && <>⚠ {totalRed} to fix</>}
            {totalRed > 0 && singletons.length > 0 && " · "}
            {singletons.length > 0 && <>seen once: {singletons.join(", ")}</>}
          </span>
        )}
        <span style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          <button style={btn(showGuide)} onClick={function () {
            setShowGuide(function (v) { return !v; });
          }}>
            {showGuide ? "Hide instructions" : "Instructions"}
          </button>
          <button style={btn(showJson)} onClick={function () {
            setShowJson(function (v) { return !v; });
          }} title="Read-only JSON preview (copy fallback)">
            {showJson ? "Hide JSON" : "Show JSON"}
          </button>
          <button
            style={{ ...btn(false), background: "#2e7d32", color: "#eaffea", fontWeight: 700 }}
            onClick={saveAnswers}
            title="Download gt_answers.json"
          >
            Save gt_answers.json
          </button>
        </span>
      </div>

      {/* ---------------- collapsible instructions ---------------- */}
      {showGuide && (
        <div style={{ maxHeight: "42%", overflowY: "auto", borderBottom: "1px solid #2b2b2b", background: "#181818", padding: "10px 18px" }}>
          <GuideContent />
        </div>
      )}

      {/* ---------------- JSON preview ---------------- */}
      {showJson && (
        <div style={{ borderBottom: "1px solid #2b2b2b", background: "#181818", padding: "8px 14px" }}>
          <textarea
            readOnly
            value={answersJson}
            style={{ width: "100%", height: 140, background: "#141414", color: "#bdbdbd", border: "1px solid #2b2b2b", borderRadius: 4, fontSize: 11, fontFamily: "monospace", resize: "vertical" }}
            onFocus={function (e) { e.currentTarget.select(); }}
          />
        </div>
      )}

      {/* ---------------- main row ---------------- */}
      <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
        {/* ---- map ---- */}
        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", padding: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6, flexWrap: "wrap" }}>
            <span style={{ fontSize: 12, color: "#9d9d9d" }}>
              Instant <b style={{ color: "#fff" }}>{i + 1}</b>/{task.instants.length} · t{String(inst.t).padStart(6, "0")} · {inst.sec.toFixed(1)} s · {inst.people.length} people
            </span>
            <button
              style={btn(addMode)}
              onClick={function () {
                setAddMode(function (v) { return !v; });
              }}
              title="Then click the map where the missed person stands (or just double-click the map)"
            >
              {addMode ? "Click map to add…" : "+ Add person"}
            </button>
            <button style={btn(false)} onClick={resetInstant} title="Clear every annotation on this instant">Reset instant</button>
            <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 5, marginLeft: "auto", cursor: "pointer", color: instAnn.done ? "#7CFF45" : "#cccccc" }}>
              <input type="checkbox" checked={!!instAnn.done} onChange={toggleDone} />
              done <span style={{ color: "#6d6d6d" }}>(d)</span>
            </label>
          </div>

          {(curConf.red.length > 0 || curConf.yellow.length > 0 || curConf.orange.length > 0 || curConf.over.length > 0) && (
            <div style={{ marginBottom: 6, display: "flex", flexDirection: "column", gap: 3 }}>
              {curConf.over.map(function (msg, k) {
                return (
                  <div key={`o${k}`} style={{ fontSize: 12, color: "#ff8a80", background: "#3a1f1f", border: "1px solid #a33", borderRadius: 4, padding: "4px 8px" }}>
                    ⚠ {msg}
                  </div>
                );
              })}
              {curConf.orange.map(function (c, k) {
                return (
                  <div key={`n${k}`} style={{ fontSize: 12, color: "#ffb74d", background: "#3a2c1a", border: "1px solid #a73", borderRadius: 4, padding: "4px 8px" }}>
                    ❓ <b>{c.team}{c.na}</b> ({c.a}) and <b>{c.team}{c.nb}</b> ({c.b}) only {c.dist.toFixed(2)} m apart —
                    likely ONE athlete wearing two labels: read the shirt and give both the same number
                    (or delete the redundant added person).
                  </div>
                );
              })}
              {curConf.red.map(function (c, k) {
                return (
                  <div key={`r${k}`} style={{ fontSize: 12, color: "#ff8a80", background: "#3a1f1f", border: "1px solid #a33", borderRadius: 4, padding: "4px 8px" }}>
                    ⚠ <b>{c.team}{c.number}</b> on {c.names.join(" and ")}, {c.dist.toFixed(1)} m apart — one athlete cannot be in two places:
                    read both shirts and fix the wrong one (select it, then <b>o</b> for a per-instant override if only this instant is wrong).
                  </div>
                );
              })}
              {curConf.yellow.map(function (c, k) {
                return (
                  <div key={`y${k}`} style={{ fontSize: 12, color: "#ffd54f", background: "#33301f", border: "1px solid #886", borderRadius: 4, padding: "4px 8px" }}>
                    ℹ <b>{c.team}{c.number}</b> twice within {c.dist.toFixed(1)} m ({c.names.join(", ")}) — fine if the tracker duplicated one athlete;
                    same number on both is the correct annotation (ghost one only if it marks empty floor).
                  </div>
                );
              })}
            </div>
          )}

          {/* viewBox units are metres. touchAction none stops panning instead of dragging.
              https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/viewBox
              https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/preserveAspectRatio
              https://developer.mozilla.org/en-US/docs/Web/CSS/touch-action */}
          <svg
            ref={svgRef}
            viewBox={`-0.5 -0.5 ${VW + 1} ${VH + 1}`}
            preserveAspectRatio="xMidYMid meet"
            style={{ flex: 1, minHeight: 0, width: "100%", background: "#15301a", border: "1px solid #2b2b2b", borderRadius: 6, cursor: addMode ? "crosshair" : "default", touchAction: "none" }}
            onClick={onMapClick}
            onDoubleClick={onMapDblClick}
            onPointerMove={onMapPointerMove}
            onPointerUp={onMapPointerUp}
          >
            <rect x={vx(-task.meta.court.x_half)} y={vy(task.meta.court.y_half)} width={2 * task.meta.court.x_half} height={2 * task.meta.court.y_half} fill="#1c3a24" stroke="none" />
            {court.map(function (pl, k) {
              return (
                <path key={k} d={polyPath(pl.pts)} fill="none"
                  stroke={pl.court === "basketball" ? "#e8e8e8" : "#55aaff"}
                  strokeWidth={pl.court === "basketball" ? 0.07 : 0.06}
                  strokeLinejoin="round" opacity={0.9} />
              );
            })}
            {hoops.map(function (h) {
              return (
                <circle key={h.id} cx={vx(h.x)} cy={vy(h.y)} r={0.28} fill="none" stroke="#ffae42" strokeWidth={0.07} />
              );
            })}

            {/* people from the tracker */}
            {inst.people.map(function (p) {
              const eff = effectiveOf(p);
              const ghost = statusOf(p) === "ghost";
              const [px, py] = posOf(p);
              const cx = vx(px), cy = vy(py);
              const color = trackColor(p.track_id);
              const isSel = sel?.kind === "p" && sel.pid === p.pid;
              return (
                <g key={p.pid} style={{ cursor: "grab" }}
                  onPointerDown={function (e) { startDrag(e, "p", p.pid); }}
                  onClick={function (e) { e.stopPropagation(); }}
                  onDoubleClick={function (e) { e.stopPropagation(); }}
                >
                  {/* dashed ring = selected; solid ring = team colour
                      https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/stroke-dasharray */}
                  {isSel && <circle cx={cx} cy={cy} r={0.72} fill="none" stroke="#ffffff" strokeWidth={0.06} strokeDasharray="0.14 0.1" />}
                  {eff.team && <circle cx={cx} cy={cy} r={0.55} fill="none" stroke={TEAM_RING[eff.team]} strokeWidth={0.1} opacity={ghost ? 0.4 : 1} />}
                  <circle cx={cx} cy={cy} r={0.42} fill={color} stroke="#0c0c0c" strokeWidth={0.05} opacity={ghost ? 0.3 : 1} />
                  <text x={cx} y={cy + 0.2} fontSize={0.5} fill="#0c0c0c" textAnchor="middle" style={{ fontWeight: 700, userSelect: "none" }} opacity={ghost ? 0.5 : 1}>
                    {p.track_id}
                  </text>
                  {ghost && (
                    <g stroke="#ff4545" strokeWidth={0.09} strokeLinecap="round">
                      <line x1={cx - 0.4} y1={cy - 0.4} x2={cx + 0.4} y2={cy + 0.4} />
                      <line x1={cx - 0.4} y1={cy + 0.4} x2={cx + 0.4} y2={cy - 0.4} />
                    </g>
                  )}
                  {eff.number && (
                    <text x={cx + 0.62} y={cy + 0.05} fontSize={0.42} fill="#ffffff" textAnchor="start" style={{ fontWeight: 600, userSelect: "none" }}>
                      #{eff.number}
                    </text>
                  )}
                  {/* override badge = tracker identity swap marked here */}
                  {eff.overridden && (
                    <g>
                      <circle cx={cx + 0.48} cy={cy - 0.48} r={0.22} fill={OVERRIDE_BADGE} stroke="#0c0c0c" strokeWidth={0.04} />
                      <text x={cx + 0.48} y={cy - 0.38} fontSize={0.32} fill="#0c0c0c" textAnchor="middle" style={{ fontWeight: 700, userSelect: "none" }}>!</text>
                    </g>
                  )}
                  {posEdited(p) && <circle cx={cx - 0.48} cy={cy - 0.48} r={0.1} fill="#7CFF45" stroke="#0c0c0c" strokeWidth={0.03} />}
                </g>
              );
            })}

            {/* people the annotator added */}
            {(instAnn.added ?? []).map(function (a) {
              const cx = vx(a.pos[0]), cy = vy(a.pos[1]);
              const isSel = sel?.kind === "a" && sel.id === a.id;
              return (
                <g key={`add${a.id}`} style={{ cursor: "grab" }}
                  onPointerDown={function (e) { startDrag(e, "a", a.id); }}
                  onClick={function (e) { e.stopPropagation(); }}
                  onDoubleClick={function (e) { e.stopPropagation(); }}
                >
                  {isSel && <circle cx={cx} cy={cy} r={0.72} fill="none" stroke="#ffffff" strokeWidth={0.06} strokeDasharray="0.14 0.1" />}
                  {a.team && <circle cx={cx} cy={cy} r={0.55} fill="none" stroke={TEAM_RING[a.team]} strokeWidth={0.1} />}
                  <circle cx={cx} cy={cy} r={0.42} fill="none" stroke={ADDED_COLOR} strokeWidth={0.1} strokeDasharray="0.16 0.1" />
                  <text x={cx} y={cy + 0.18} fontSize={0.5} fill={ADDED_COLOR} textAnchor="middle" style={{ fontWeight: 700, userSelect: "none" }}>+</text>
                  {a.number && (
                    <text x={cx + 0.62} y={cy + 0.05} fontSize={0.42} fill="#ffffff" textAnchor="start" style={{ fontWeight: 600, userSelect: "none" }}>
                      #{a.number}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>

          {/* overrides summary */}
          {overridesList.length > 0 && (
            <div style={{ marginTop: 6, maxHeight: 74, overflowY: "auto", fontSize: 10, color: "#bdbdbd", background: "#181818", border: "1px solid #2b2b2b", borderRadius: 4, padding: "4px 8px" }}>
              <b style={{ color: OVERRIDE_BADGE }}>Per-instant overrides ({overridesList.length})</b> — identity swaps marked so far:{" "}
              {overridesList.map(function (o, k) {
                return (
                  <button
                    key={k}
                    onClick={function () { gotoInstant(o.instIdx); setSel({ kind: "p", pid: o.pid }); }}
                    style={{ background: "#252526", color: "#ffd400", border: "1px solid #3c3c3c", borderRadius: 3, padding: "0 5px", margin: "1px 3px", fontSize: 10, cursor: "pointer" }}
                    title="Jump to this override"
                  >
                    inst {o.instIdx + 1} · track {o.tid}{o.team ? ` → ${o.team}` : ""}{o.number !== undefined ? ` #${o.number || "∅"}` : ""}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* ---- camera frames ---- */}
        <div style={{ width: 430, flex: "0 0 430px", borderLeft: "1px solid #2b2b2b", overflowY: "auto", padding: 8, display: "flex", flexDirection: "column", gap: 8 }}>
          {inst.cams.map(function (c) {
            return (
              <FrameView
                key={c.cam}
                cam={c}
                entries={drawForCam(c.cam)}
                onSelect={function (pid) { setSel({ kind: "p", pid }); setNumBuf(null); }}
                onImageClick={function () { setZoomCam(c.cam); }}
              />
            );
          })}
          <div style={{ fontSize: 10, color: "#6d6d6d", lineHeight: 1.5 }}>
            Click a box to select that person; click the image itself to enlarge (read the shirt numbers there).
          </div>
        </div>

        {/* ---- side panel ---- */}
        <div style={{ width: 268, flex: "0 0 268px", borderLeft: "1px solid #2b2b2b", overflowY: "auto", padding: "10px 12px", background: "#1a1a1a" }}>
          {!sel && (
            <div style={{ fontSize: 11, color: "#9d9d9d", lineHeight: 1.6 }}>
              <b style={{ color: "#fc6" }}>No person selected</b>
              <p>Click a marker on the map or a box in a frame.</p>
              <p style={{ color: "#6d6d6d" }}>
                Keys: a/b team · r official · x other · g ghost · digits+Enter number ·
                o override · ←/→ instants · d done. Full map in Instructions.
              </p>
            </div>
          )}

          {selPerson && (function () {
            const seed = task.tracks[String(selPerson.track_id)];
            const eff = effectiveOf(selPerson);
            const ghost = statusOf(selPerson) === "ghost";
            const [px, py] = posOf(selPerson);
            return (
              <div style={{ fontSize: 12 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                  <span style={{ width: 16, height: 16, borderRadius: "50%", background: trackColor(selPerson.track_id), border: "1px solid #0c0c0c" }} />
                  <b style={{ color: "#fff" }}>Track {selPerson.track_id}</b>
                  <span style={{ color: "#6d6d6d", fontSize: 11 }}>pid {selPerson.pid}</span>
                  {eff.overridden && (
                    <span style={{ background: OVERRIDE_BADGE, color: "#1f1f1f", borderRadius: 3, padding: "0 5px", fontSize: 10, fontWeight: 700 }}>OVERRIDE</span>
                  )}
                </div>

                {seed && (
                  <div style={{ background: "#241f14", border: "1px solid #5a4a1e", borderRadius: 4, padding: "6px 8px", fontSize: 10, color: "#d8c48a", lineHeight: 1.5, marginBottom: 10 }}>
                    <b style={{ color: "#fc6" }}>Tracker's guess — read the shirt, don't trust this:</b><br />
                    role <code>{seed.role}</code> · team-split <code>{seed.off_team == null ? "?" : seed.off_team ? "off" : "home"}</code> ·
                    number <code>{seed.number ?? "—"}</code>
                    {seed.number != null && seed.number_margin != null && <> (margin {seed.number_margin.toFixed(2)})</>} · seen in {seed.n} frames
                  </div>
                )}

                <LabelEditor
                  team={eff.team}
                  number={eff.number}
                  onTeam={function (t) { applyLabel({ team: t }); }}
                  onNumber={function (n) { applyLabel({ number: n }); }}
                  numBuf={numBuf}
                />

                <div style={{ margin: "10px 0" }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", color: overrideMode ? OVERRIDE_BADGE : "#cccccc" }}>
                    <input type="checkbox" checked={overrideMode} onChange={function (e) {
                      setOverrideMode(e.target.checked);
                    }} />
                    only this instant <span style={{ color: "#6d6d6d" }}>(o)</span>
                  </label>
                  <div style={{ fontSize: 10, color: "#6d6d6d", lineHeight: 1.4, marginTop: 2 }}>
                    Off: team/number apply to the whole track. On: recorded at this instant only —
                    use it exactly where the tracker swapped identities.
                  </div>
                  {eff.overridden && (
                    <button style={{ ...btn(false), marginTop: 5 }} onClick={function () {
                      clearOverride(selPerson.pid);
                    }}>
                      Clear override (fall back to track label)
                    </button>
                  )}
                </div>

                <div style={{ margin: "10px 0" }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", color: ghost ? "#ff4545" : "#cccccc" }}>
                    <input type="checkbox" checked={ghost} onChange={toggleGhost} />
                    ghost — tracker invented this person <span style={{ color: "#6d6d6d" }}>(g)</span>
                  </label>
                </div>

                <div style={{ fontSize: 11, color: "#9d9d9d" }}>
                  pos ({px.toFixed(2)}, {py.toFixed(2)}) m
                  {posEdited(selPerson) && <span style={{ color: "#7CFF45" }}> · edited</span>}
                  <div style={{ fontSize: 10, color: "#6d6d6d" }}>Drag the marker on the map to correct it (only if off by &gt; ~0.5 m).</div>
                </div>
              </div>
            );
          })()}

          {selAdded && (
            <div style={{ fontSize: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ width: 16, height: 16, borderRadius: "50%", border: `2px dashed ${ADDED_COLOR}` }} />
                <b style={{ color: ADDED_COLOR }}>Added person {selAdded.id}</b>
                <span style={{ color: "#6d6d6d", fontSize: 11 }}>this instant only</span>
              </div>
              <LabelEditor
                team={selAdded.team ?? null}
                number={selAdded.number ?? ""}
                onTeam={function (t) { applyLabel({ team: t }); }}
                onNumber={function (n) { applyLabel({ number: n }); }}
                numBuf={numBuf}
              />
              <div style={{ fontSize: 11, color: "#9d9d9d", margin: "10px 0" }}>
                pos ({selAdded.pos[0].toFixed(2)}, {selAdded.pos[1].toFixed(2)}) m — drag to move.
              </div>
              <button
                style={{ ...btn(false), color: "#ff8080", borderColor: "#5a2b2b" }}
                onClick={function () { deleteAdded(instKey, selAdded.id); }}
              >
                Delete this person
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ---------------- instant timeline ---------------- */}
      <div style={{ padding: "7px 14px", borderTop: "1px solid #2b2b2b", display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <button style={btn(false)} onClick={function () { gotoInstant(i - 1); }} disabled={i === 0} title="Previous instant (p / ←)">‹ prev</button>
        <div style={{ display: "flex", gap: 3, flexWrap: "wrap", flex: 1 }}>
          {task.instants.map(function (ins, k) {
            const done = !!ann.instants[String(ins.t)]?.done;
            const cur = k === i;
            const nred = allRed.get(ins.t) ?? 0;
            return (
              <button
                key={ins.t}
                onClick={function () { gotoInstant(k); }}
                title={`t${ins.t} · ${ins.sec.toFixed(1)} s${done ? " · done" : ""}${nred > 0 ? ` · ⚠ ${nred} label check${nred > 1 ? "s" : ""} needed` : ""}`}
                style={{
                  width: 26, height: 22, fontSize: 10, cursor: "pointer", borderRadius: 3,
                  background: done ? "#2e7d32" : "#252526",
                  color: done ? "#eaffea" : "#cccccc",
                  border: cur ? "2px solid #fc6" : nred > 0 ? "1px solid #d33" : "1px solid #3c3c3c",
                  boxShadow: nred > 0 ? "0 0 0 1px #d33" : undefined,
                  fontWeight: cur ? 700 : 400, padding: 0,
                }}
              >
                {k + 1}
              </button>
            );
          })}
        </div>
        <button style={btn(false)} onClick={function () { gotoInstant(i + 1); }} disabled={i === task.instants.length - 1} title="Next instant (n / →)">next ›</button>
      </div>

      {/* ---------------- lightbox ---------------- */}
      {zoomCamInfo && (
        <div
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.88)", zIndex: 60, display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}
          onClick={function () { setZoomCam(null); }}
        >
          <div style={{ width: "min(94vw, 152vh)", maxHeight: "94vh" }} onClick={function (e) {
            e.stopPropagation();
          }}>
            <div style={{ display: "flex", alignItems: "center", color: "#ddd", fontSize: 13, marginBottom: 6 }}>
              <b>{zoomCamInfo.cam}</b>&nbsp;· t{String(inst.t).padStart(6, "0")} — click a box to select, Esc or ✕ to close
              <button style={{ ...btn(false), marginLeft: "auto" }} onClick={function () { setZoomCam(null); }}>✕ close</button>
            </div>
            <FrameView
              cam={zoomCamInfo}
              entries={drawForCam(zoomCamInfo.cam)}
              onSelect={function (pid) { setSel({ kind: "p", pid }); setNumBuf(null); }}
              big
            />
          </div>
        </div>
      )}
    </div>
  );
}

// A frame with the boxes and foot markers drawn over it, in the strip and enlarged
interface DrawEntry {
  pid: number;
  tid: number;
  color: string;
  bbox: [number, number, number, number];
  foot: [number, number] | null;
  team: Team | null;
  number: string;
  overridden: boolean;
  ghost: boolean;
  selected: boolean;
}

/** One camera frame with the boxes and foot markers drawn over it. */
function FrameView({ cam, entries, onSelect, onImageClick, big }: {
  cam: GtCam;
  entries: DrawEntry[];
  onSelect: (pid: number) => void;
  onImageClick?: () => void;
  big?: boolean;
}) {
  const [imgErr, setImgErr] = useState(false);
  // Keyed by camera name, so clear the error whenever the src changes.
  // https://developer.mozilla.org/en-US/docs/Web/HTML/Element/img
  useEffect(function () { setImgErr(false); }, [cam.img]);
  // A cam's frame file can be missing without the whole task being broken.
  if (imgErr) {
    return (
      <div style={{ border: "1px solid #2b2b2b", borderRadius: 4, padding: 10, fontSize: 11, color: "#9d9d9d", background: "#181818" }}>
        <b style={{ color: "#ddd" }}>{cam.cam}</b> — image <code>/{cam.img}</code> missing.
      </div>
    );
  }
  // viewBox units are source px; the small stacked frames need fatter strokes and text.
  const strokeW = big ? 2 : 3.5;
  const fontSize = big ? 16 : 30;
  const footR = big ? 4 : 6;
  return (
    <div style={{ position: "relative", border: "1px solid #2b2b2b", borderRadius: 4, overflow: "hidden", background: "#0c0c0c" }}>
      <img
        src={`/${cam.img}`}
        alt={cam.cam}
        onError={function () { setImgErr(true); }}
        onClick={onImageClick}
        style={{ display: "block", width: "100%", height: "auto", cursor: onImageClick ? "zoom-in" : "default" }}
        draggable={false}
      />
      {/* preserveAspectRatio none stretches the overlay onto the image box at any size.
          https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/preserveAspectRatio */}
      <svg
        viewBox={`0 0 ${cam.w} ${cam.h}`}
        preserveAspectRatio="none"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
      >
        {entries.map(function (e) {
          const [x0, y0, x1, y1] = e.bbox;
          const stroke = e.selected ? "#ffffff" : e.color;
          return (
            <g key={e.pid} opacity={e.ghost ? 0.45 : 1}>
              {e.selected && (
                <rect x={x0 - 3} y={y0 - 3} width={x1 - x0 + 6} height={y1 - y0 + 6} fill="none" stroke="#ffffff" strokeWidth={strokeW + 2} opacity={0.35} />
              )}
              <rect
                x={x0} y={y0} width={x1 - x0} height={y1 - y0}
                fill={e.selected ? "rgba(255,255,255,0.08)" : "transparent"}
                stroke={stroke} strokeWidth={strokeW}
                strokeDasharray={e.ghost ? "6 5" : undefined}
                style={{ pointerEvents: "auto", cursor: "pointer" }}
                onClick={function (ev) { ev.stopPropagation(); onSelect(e.pid); }}
              />
              {e.team && (
                <rect x={x0} y={y0} width={x1 - x0} height={4} fill={TEAM_RING[e.team]} />
              )}
              {e.foot && <circle cx={e.foot[0]} cy={e.foot[1]} r={footR} fill={e.color} stroke="#0c0c0c" strokeWidth={1.2} />}
              {/* paint-order stroke draws the outline behind the glyphs, so labels stay readable.
                  https://developer.mozilla.org/en-US/docs/Web/CSS/paint-order */}
              <text
                x={x0} y={Math.max(fontSize, y0 - 5)}
                fontSize={fontSize}
                fill={stroke}
                style={{ fontWeight: 700, paintOrder: "stroke", stroke: "#0c0c0c", strokeWidth: 3 }}
              >
                {e.tid}{e.number ? ` #${e.number}` : ""}{e.team ? ` ${e.team}` : ""}{e.overridden ? " !" : ""}{e.ghost ? " ghost" : ""}
              </text>
            </g>
          );
        })}
      </svg>
      <div style={{ position: "absolute", left: 4, bottom: 3, fontSize: 10, color: "#ddd", background: "rgba(12,12,12,0.65)", padding: "1px 6px", borderRadius: 3, pointerEvents: "none" }}>
        {cam.cam}
      </div>
    </div>
  );
}

// The little team-and-number editor, used for whoever is currently selected
function LabelEditor({ team, number, onTeam, onNumber, numBuf }: {
  team: Team | null;
  number: string;
  onTeam: (t: Team) => void;
  onNumber: (n: string) => void;
  numBuf: string | null;
}) {
  return (
    <div>
      <div style={{ fontSize: 11, color: "#9d9d9d", marginBottom: 4 }}>Team (a/b/r/x)</div>
      <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
        {TEAMS.map(function (t) {
          return (
            <button
              key={t}
              onClick={function () { onTeam(t); }}
              style={{
                flex: 1, padding: "4px 0", fontSize: 11, cursor: "pointer", borderRadius: 3,
                background: team === t ? TEAM_RING[t] : "#252526",
                color: team === t ? "#1f1f1f" : "#cccccc",
                border: `1px solid ${team === t ? TEAM_RING[t] : "#3c3c3c"}`,
                fontWeight: team === t ? 700 : 400,
              }}
            >
              {t}
            </button>
          );
        })}
      </div>
      <div style={{ fontSize: 11, color: "#9d9d9d", marginBottom: 4 }}>
        Shirt number (type digits, Enter commits{numBuf !== null && <> — typing <b style={{ color: "#fc6" }}>{numBuf}</b>_</>})
      </div>
      <input
        type="text"
        value={number}
        placeholder="empty = not legible"
        onChange={function (e) { onNumber(e.target.value.replace(/[^0-9]/g, "")); }}
        style={{ width: "100%", background: "#141414", color: "#fff", border: "1px solid #3c3c3c", borderRadius: 3, padding: "4px 8px", fontSize: 13, boxSizing: "border-box" }}
      />
    </div>
  );
}

// The instructions panel. Same text as ANNOTATION_GUIDE.md, kept in step by hand.
function GuideContent() {
  const h: React.CSSProperties = { color: "#fc6", fontSize: 13, margin: "12px 0 4px" };
  const p: React.CSSProperties = { fontSize: 12, color: "#bdbdbd", lineHeight: 1.6, margin: "0 0 6px" };
  const kbd: React.CSSProperties = { background: "#252526", border: "1px solid #3c3c3c", borderRadius: 3, padding: "0 5px", color: "#fff", fontFamily: "monospace", fontSize: 11 };
  return (
    <div style={{ maxWidth: 900 }}>
      <h3 style={{ ...h, marginTop: 0 }}>What this is</h3>
      <p style={p}>
        You say who is who at 30 moments of the game. The result, gt_answers.json, was used during
        the thesis to measure how well the tracking works. The pipeline does not need it.
      </p>

      <h3 style={h}>Important</h3>
      <p style={{ ...p, background: "#2a1c1c", border: "1px solid #6b2b2b", borderRadius: 4, padding: "8px 10px", color: "#ffb0b0" }}>
        The team and number already filled in are the tracker's guesses. Always check the shirt in
        the photos. If you cannot read the number, leave it empty.
      </p>

      <h3 style={h}>How to do it</h3>
      <p style={p}>Go through the moments one by one. For each one:</p>
      <ul style={{ ...p, paddingLeft: 20 }}>
        <li>Click a person on the map or in a photo. Click a photo to zoom in.</li>
        <li>Set the team and the number. The label is copied to the same person in the other moments.</li>
        <li>If the label is wrong only in this moment, tick <span style={{ color: OVERRIDE_BADGE }}>"only this instant"</span> and fix it.</li>
        <li>If nobody is really there, mark it as a <b style={{ color: "#ff4545" }}>ghost</b>.</li>
        <li>If someone is missing, double-click the map where they stand, or use <b style={{ color: ADDED_COLOR }}>+ Add person</b>.</li>
        <li>Only drag a marker if it is more than about half a metre off.</li>
        <li>Tick <b style={{ color: "#7CFF45" }}>done</b>.</li>
      </ul>
      <p style={p}>
        A <b style={{ color: "#ff8a80" }}>red warning</b> means the same player is in two places, so one
        label is wrong. A <b style={{ color: "#ffd54f" }}>yellow note</b> means the same player appears
        twice close together; that is usually fine.
      </p>

      <h3 style={h}>Keys</h3>
      <p style={p}>
        With a person selected: <span style={kbd}>a</span> <span style={kbd}>b</span> team,{" "}
        <span style={kbd}>r</span> referee, <span style={kbd}>x</span> other,{" "}
        <span style={kbd}>g</span> ghost, <span style={kbd}>0-9</span> then <span style={kbd}>Enter</span> number,{" "}
        <span style={kbd}>o</span> only this instant.
        <br />
        Any time: <span style={kbd}>n</span> next, <span style={kbd}>p</span> previous,{" "}
        <span style={kbd}>d</span> done.
      </p>

      <h3 style={h}>When you finish</h3>
      <p style={p}>
        Click <b style={{ color: "#7CFF45" }}>Save gt_answers.json</b> and keep the file. Your work
        is also saved in the browser as you go, so you can close the tab and come back.
      </p>
    </div>
  );
}

function btn(active: boolean): React.CSSProperties {
  return {
    background: active ? "#fc6" : "#252526",
    color: active ? "#1f1f1f" : "#cccccc",
    border: `1px solid ${active ? "#fc6" : "#3c3c3c"}`,
    padding: "3px 10px", fontSize: 11, cursor: "pointer", borderRadius: 3,
    fontWeight: active ? 600 : 400,
  };
}
