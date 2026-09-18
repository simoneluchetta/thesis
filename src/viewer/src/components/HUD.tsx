/**
 * The panel over the 3-D view: layer toggles, sliders and the camera picker.
 * It writes to the viewer store, which SceneCanvas reads.
 */
// https://docs.pmnd.rs/zustand
// https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/range
// https://react.dev/reference/react-dom/components/common#common-props (style)
import { useEffect } from "react";
import type { SceneJson } from "../types";
import { useViewer, type OrbitAxisMode, type UpAxis } from "../state/store";

interface Props {
  scene: SceneJson;
}

/** The only PLY layers offered in the panel. The others in scene.json stay hidden. */
const SHOWN_PLY_LAYERS = ["unistands_floor_ortho"];

// System UI font, so the viewer ships no font files.
const baseFont ="ui-sans-serif, system-ui, -apple-system, sans-serif";
// Monospace stack for numbers and file names.
const monoFont = "ui-monospace, SFMono-Regular, Menlo, monospace";

// Inline styles, so there is no stylesheet to ship with the viewer.
// https://developer.mozilla.org/en-US/docs/Web/CSS/backdrop-filter
const panelBase: React.CSSProperties = {
  position: "absolute",
  top: 12,
  bottom: 12,
  width: 300,
  overflowY: "auto",
  padding: 14,
  background: "rgba(24, 24, 24, 0.88)",
  border: "1px solid #3c3c3c",
  borderRadius: 8,
  color: "#cccccc",
  fontFamily: baseFont,
  fontSize: 12,
  lineHeight: 1.5,
  backdropFilter: "blur(6px)",
  WebkitBackdropFilter: "blur(6px)",
  boxShadow: "0 10px 30px rgba(0,0,0,0.45)",
  pointerEvents: "auto",
};

// The left panel, the one that explains how the cameras were placed.
const leftPanelStyle: React.CSSProperties = { ...panelBase, left: 12 };
// The right panel, the one with the controls.
const rightPanelStyle: React.CSSProperties = { ...panelBase, right: 12 };

// One block of controls, with a rule above it.
const sectionStyle: React.CSSProperties = {
  marginTop: 14,
  paddingTop: 10,
  borderTop: "1px solid #2f2f2f",
};

// Small uppercase heading at the top of a block.
const headStyle: React.CSSProperties = {
  fontSize: 10.5,
  textTransform: "uppercase",
  letterSpacing: "0.09em",
  color: "#9d9d9d",
  marginBottom: 8,
  fontWeight: 600,
};

// One row: label on the left, value or control on the right.
const rowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 8,
  padding: "3px 0",
};

// The numeric readouts, monospace so they do not jump as digits change.
const numStyle: React.CSSProperties = {
  fontFamily: monoFont,
  color: "#fff",
};

// A row of pill buttons sharing the width.
const buttonRow: React.CSSProperties = {
  display: "flex",
  gap: 6,
  marginTop: 4,
};

// Grey explanatory text.
const hintTextStyle: React.CSSProperties = {
  color: "#9d9d9d",
  fontSize: 11.5,
  lineHeight: 1.55,
};

// Gap between the numbered steps on the left panel.
const stepStyle: React.CSSProperties = { marginBottom: 6 };

// Inline code, used for file names.
const codeStyle: React.CSSProperties = { fontFamily: monoFont, color: "#cccccc" };

/** Small button that stays lit while active, used for the axis pickers. */
function PillButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        padding: "5px 0",
        fontSize: 11,
        background: active ? "#0078d4" : "#252526",
        color: active ? "#fff" : "#cccccc",
        border: `1px solid ${active ? "#0078d4" : "#3c3c3c"}`,
        borderRadius: 4,
        cursor: "pointer",
        fontFamily: "inherit",
      }}
    >
      {label}
    </button>
  );
}

/** Both panels over the 3-D view: the explainer on the left, the controls on the right. */
export default function HUD({ scene }: Props) {
  const showFrustums = useViewer(function (s) { return s.showFrustums; });
  const showLabels = useViewer(function (s) { return s.showLabels; });
  const showThumbs = useViewer(function (s) { return s.showThumbs; });
  const showPeople = useViewer(function (s) { return s.showPeople; });
  const peopleTimeIndex = useViewer(function (s) { return s.peopleTimeIndex; });
  const setPeopleTimeIndex = useViewer(function (s) { return s.setPeopleTimeIndex; });
  const showBasketballCourt = useViewer(function (s) { return s.showBasketballCourt; });
  const showVolleyballCourt = useViewer(function (s) { return s.showVolleyballCourt; });
  const courtAnchorX = useViewer(function (s) { return s.courtAnchorX; });
  const courtAnchorY = useViewer(function (s) { return s.courtAnchorY; });
  const courtAnchorZ = useViewer(function (s) { return s.courtAnchorZ; });
  const courtYawDeg = useViewer(function (s) { return s.courtYawDeg; });
  const setCourtAnchorX = useViewer(function (s) { return s.setCourtAnchorX; });
  const setCourtAnchorY = useViewer(function (s) { return s.setCourtAnchorY; });
  const setCourtAnchorZ = useViewer(function (s) { return s.setCourtAnchorZ; });
  const setCourtYawDeg = useViewer(function (s) { return s.setCourtYawDeg; });
  const resetCourtAnchor = useViewer(function (s) { return s.resetCourtAnchor; });
  const plyLayersOn = useViewer(function (s) { return s.plyLayersOn; });
  const selectedCamId = useViewer(function (s) { return s.selectedCamId; });
  const frustumScale = useViewer(function (s) { return s.frustumScale; });
  const labelScale = useViewer(function (s) { return s.labelScale; });
  const upAxis = useViewer(function (s) { return s.upAxis; });
  const toggle = useViewer(function (s) { return s.toggle; });
  const togglePly = useViewer(function (s) { return s.togglePly; });
  const setSelectedCam = useViewer(function (s) { return s.setSelectedCam; });
  const setPlyLayers = useViewer(function (s) { return s.setPlyLayers; });
  const setFrustumScale = useViewer(function (s) { return s.setFrustumScale; });
  const setLabelScale = useViewer(function (s) { return s.setLabelScale; });
  const setUpAxis = useViewer(function (s) { return s.setUpAxis; });
  const resetView = useViewer(function (s) { return s.resetView; });
  const orbitAxis = useViewer(function (s) { return s.orbitAxis; });
  const setOrbitAxis = useViewer(function (s) { return s.setOrbitAxis; });

  // Seed the per-layer toggles from scene.json the first time the panel sees it.
  useEffect(function () {
    const initial: Record<string, boolean> = {};
    for (const l of scene.ply_layers) initial[l.id] = !!l.default_on;
    setPlyLayers(initial);
  }, [scene.ply_layers, setPlyLayers]);

  return (
    <>
      {/* Left panel: how the camera placement was obtained. */}
      <div style={leftPanelStyle}>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#fff" }}>
          Arena camera viewer
        </div>

        <div style={sectionStyle}>
          <div style={headStyle}>How the cameras were placed</div>
          <ol style={{ ...hintTextStyle, margin: 0, paddingLeft: 18 }}>
            <li style={stepStyle}>
              <b>Place.</b> In the <i>Camera placer</i> tab, each camera was put on the floor plan
              by hand (position, height, yaw / pitch / roll) until the court drawn on top matched
              its video frame.
            </li>
            <li style={stepStyle}>
              <b>Export.</b> The placement was saved and baked into{" "}
              <code style={codeStyle}>public/scene.json</code>, which this tab shows.
            </li>
            <li style={stepStyle}>
              <b>Refine.</b> That first guess was solved against the court with PnP, then refined
              by repairing the distortion model of each camera: that gave the final placement.
            </li>
          </ol>
        </div>
      </div>

      {/* Right panel: the controls. */}
      <div style={rightPanelStyle}>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#fff" }}>
          Commands
        </div>
        <div style={{ color: "#9d9d9d", marginTop: 4 }}>
          Toggles and sliders that drive the 3D view.
        </div>

        <div style={sectionStyle}>
          <div style={headStyle}>View controls</div>
          <div style={{ ...rowStyle, marginBottom: 4 }}>
            <span>up axis</span>
            <span style={{ ...numStyle, color: "#9d9d9d" }}>
              {upAxis === "auto"
                ? `auto (${scene.world.up.map(function (x) { return x.toFixed(2); }).join(", ")})`
                : upAxis}
            </span>
          </div>
          <div style={buttonRow}>
            {(["auto", "+x", "+y", "+z"] as UpAxis[]).map(function (a) {
              return (
                <PillButton
                  key={a}
                  label={a}
                  active={upAxis === a}
                  onClick={function () { setUpAxis(a); }}
                />
              );
            })}
          </div>
          <div style={{ ...headStyle, marginTop: 12 }}>Orbit axis</div>
          <div style={buttonRow}>
            {(["free", "x", "y", "z"] as OrbitAxisMode[]).map(function (a) {
              return (
                <PillButton
                  key={a}
                  label={a}
                  active={orbitAxis === a}
                  onClick={function () { setOrbitAxis(a); }}
                />
              );
            })}
          </div>
          <div style={{ ...rowStyle, marginTop: 12 }}>
            <span>frustum size</span>
            <span style={numStyle}>{frustumScale.toFixed(2)}x</span>
          </div>
          <input
            type="range"
            min={0.2}
            max={3.0}
            step={0.1}
            value={frustumScale}
            onChange={function (e) { setFrustumScale(Number.parseFloat(e.target.value)); }}
            style={{ width: "100%" }}
          />
          <div style={{ ...rowStyle, marginTop: 8 }}>
            <span>label size</span>
            <span style={numStyle}>{labelScale.toFixed(2)}x</span>
          </div>
          <input
            type="range"
            min={0.3}
            max={5.0}
            step={0.1}
            value={labelScale}
            onChange={function (e) { setLabelScale(Number.parseFloat(e.target.value)); }}
            style={{ width: "100%" }}
          />
          <div style={buttonRow}>
            <PillButton label="reset view" active={false} onClick={resetView} />
          </div>
        </div>

        <div style={sectionStyle}>
          <div style={headStyle}>Layers</div>
          <Toggle label="frustums" on={showFrustums} onClick={function () { toggle("showFrustums"); }} />
          <Toggle label="labels" on={showLabels} onClick={function () { toggle("showLabels"); }} />
          <Toggle label="thumbnails" on={showThumbs} onClick={function () { toggle("showThumbs"); }} />
          {scene.people && (
            <Toggle label="people" on={showPeople} onClick={function () { toggle("showPeople"); }} />
          )}
          <Toggle label="basketball court" on={showBasketballCourt} onClick={function () {
            toggle("showBasketballCourt");
          }} />
          <Toggle label="volleyball court" on={showVolleyballCourt} onClick={function () {
            toggle("showVolleyballCourt");
          }} />
        </div>

        {scene.people && scene.people.timestamps.length > 0 && (
          <div style={sectionStyle}>
            <div style={headStyle}>People timeline</div>
            {(function () {
              const ts = scene.people.timestamps;
              const i = Math.min(peopleTimeIndex, ts.length - 1);
              const t = ts[i];
              const n = (scene.people.frames[String(t)] ?? []).length;
              return (
                <>
                  <div style={rowStyle}>
                    <span>frame t{String(t).padStart(6, "0")}</span>
                    <span style={numStyle}>{n} ppl · {(t / (scene.people?.meta?.fps || 25)).toFixed(1)}s</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={ts.length - 1}
                    step={1}
                    value={i}
                    onChange={function (e) { setPeopleTimeIndex(Number.parseInt(e.target.value, 10)); }}
                    style={{ width: "100%" }}
                    disabled={ts.length < 2}
                  />
                </>
              );
            })()}
          </div>
        )}

        {(showBasketballCourt || showVolleyballCourt) && (
          <div style={sectionStyle}>
            <div style={headStyle}>Court placement</div>
            <CourtSlider label="X (m)" value={courtAnchorX} min={-50} max={50} step={0.05}
              onChange={setCourtAnchorX} />
            <CourtSlider label="Y (m)" value={courtAnchorY} min={-50} max={50} step={0.05}
              onChange={setCourtAnchorY} />
            <CourtSlider label="Z (m)" value={courtAnchorZ} min={-20} max={20} step={0.05}
              onChange={setCourtAnchorZ} />
            <CourtSlider label="Yaw (°)" value={courtYawDeg} min={-180} max={180} step={0.5}
              onChange={setCourtYawDeg} />
            <div style={{ ...buttonRow, marginTop: 6 }}>
              <PillButton label="reset court" active={false} onClick={resetCourtAnchor} />
            </div>
          </div>
        )}

        {scene.ply_layers.some(function (l) { return SHOWN_PLY_LAYERS.includes(l.id); }) && (
          <div style={sectionStyle}>
            <div style={headStyle}>Dense PLY layers</div>
            {scene.ply_layers.filter(function (l) { return SHOWN_PLY_LAYERS.includes(l.id); }).map(function (l) {
              return (
                <Toggle
                  key={l.id}
                  label={l.label}
                  on={!!plyLayersOn[l.id]}
                  onClick={function () { togglePly(l.id); }}
                />
              );
            })}
          </div>
        )}

        <div style={sectionStyle}>
          <div style={headStyle}>Focus camera</div>
          <select
            value={selectedCamId ?? ""}
            onChange={function (e) {
              const v = e.target.value;
              setSelectedCam(v === "" ? null : Number.parseInt(v, 10));
            }}
            style={{
              width: "100%",
              background: "#252526",
              color: "#fff",
              border: "1px solid #3c3c3c",
              borderRadius: 4,
              padding: "5px 6px",
              fontFamily: "inherit",
            }}
          >
            <option value="">(none)</option>
            {scene.cameras.map(function (c) {
              const r = c.diag?.reproj_error_px;
              const rTxt = r === null || r === undefined ? "—" : `${r.toFixed(1)}px`;
              return (
                <option key={c.id} value={c.id}>
                  {c.name}
                  {c.is_stereo ? " ⌬" : ""} — {rTxt},{" "}
                  {c.diag?.n_visible_points ?? 0}pts
                </option>
              );
            })}
          </select>
        </div>
      </div>
    </>
  );
}

interface CourtSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}
/** Slider and number box on one row, used for the court anchor and yaw. */
function CourtSlider({ label, value, min, max, step, onChange }: CourtSliderProps) {
  return (
    <div style={{ ...rowStyle, padding: "2px 0", flexWrap: "wrap" }}>
      <span style={{ width: 50 }}>{label}</span>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={function (e) { onChange(Number.parseFloat(e.target.value)); }}
        style={{ flex: 1, minWidth: 80 }}
      />
      <input
        type="number"
        step={step} value={value.toFixed(2)}
        onChange={function (e) {
          const v = Number.parseFloat(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
        style={{
          width: 56, background: "#141414", color: "#cccccc",
          border: "1px solid #2b2b2b", padding: "2px 4px", fontSize: 11,
          boxSizing: "border-box", fontFamily: monoFont,
        }}
      />
    </div>
  );
}

/** A label with a switch, used for every layer in the panel. */
function Toggle({
  label,
  on,
  onClick,
}: {
  label: string;
  on: boolean;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      style={{
        ...rowStyle,
        cursor: "pointer",
        userSelect: "none",
      }}
    >
      <span>{label}</span>
      <span
        style={{
          width: 28,
          height: 16,
          borderRadius: 8,
          background: on ? "#0078d4" : "#3c3c3c",
          position: "relative",
          flex: "0 0 auto",
          transition: "background 120ms",
        }}
      >
        <span
          style={{
            width: 12,
            height: 12,
            borderRadius: 6,
            background: on ? "#fff" : "#6d6d6d",
            position: "absolute",
            top: 2,
            left: on ? 14 : 2,
            transition: "left 120ms",
          }}
        />
      </span>
    </div>
  );
}
