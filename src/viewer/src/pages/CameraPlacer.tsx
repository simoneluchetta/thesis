/**
 * Place the cameras by hand: drag a pose and watch the court overlay slide across the real
 * photograph. The result seeds the PnP solve.
 */
import { useEffect, useRef } from "react";
import type { SceneJson } from "../types";
import TopDownMap from "../components/placer/TopDownMap";
import OrthoPanels from "../components/placer/OrthoPanels";
import CameraView from "../components/placer/CameraView";
import PoseControls from "../components/placer/PoseControls";
import { usePlacer } from "../state/placerStore";

interface Props {
  scene: SceneJson;
}

/** Four panes: camera view + pose sliders, floor plan, ortho scatters. */
export default function CameraPlacer({ scene }: Props) {
  // One zustand selector per field, so editing a pose only re-renders what reads it.
  // https://zustand.docs.pmnd.rs/
  const initFromScene = usePlacer(function (s) { return s.initFromScene; });
  const resetAll = usePlacer(function (s) { return s.resetAll; });
  const exportSnapshot = usePlacer(function (s) { return s.exportSnapshot; });
  const importSnapshot = usePlacer(function (s) { return s.importSnapshot; });

  useEffect(function () { initFromScene(scene.cameras); }, [scene.cameras, initFromScene]);

  const cameras = scene.cameras;

  function onExport() {
    const data = exportSnapshot();
    // Save with no server: a Blob, an object URL, a throwaway <a download>.
    // https://developer.mozilla.org/en-US/docs/Web/API/Blob
    // https://developer.mozilla.org/en-US/docs/Web/API/File_API/Using_files_from_web_applications
    // https://developer.mozilla.org/en-US/docs/Web/HTML/Element/a#download
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "placed_cameras.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  // Hidden file input behind the "Import JSON" button.
  // https://developer.mozilla.org/en-US/docs/Web/API/HTMLInputElement/files
  const fileInputRef = useRef<HTMLInputElement>(null);
  function onImportClick() { fileInputRef.current?.click(); }
  async function onImportFile(e: React.ChangeEvent<HTMLInputElement>) {
    // File.text() reads the picked file as a string.
    // https://developer.mozilla.org/en-US/docs/Web/API/Blob/text
    const file = e.target.files?.[0];
    // Reset the input so re-selecting the same file still fires `change`.
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (!file) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(await file.text());
    } catch (err) {
      window.alert(`Couldn't parse "${file.name}" as JSON: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    const result = importSnapshot(parsed);
    if (!result.ok) {
      window.alert(`Import failed: ${result.error ?? "unknown error"}`
        + (result.ignored > 0 ? ` (${result.ignored} entries skipped)` : ""));
    } else if (result.ignored > 0) {
      window.alert(`Imported ${result.matched} camera${result.matched === 1 ? "" : "s"}.\n`
        + `${result.ignored} entr${result.ignored === 1 ? "y" : "ies"} skipped `
        + `(unknown camera name or malformed pose).`);
    }
  }

  return (
    <div style={{
      display: "flex", flexDirection: "column",
      height: "100%", width: "100%",
      background: "#1f1f1f", color: "#cccccc", boxSizing: "border-box",
    }}>
      <div style={{
        display: "flex", gap: 6, padding: 8,
        borderBottom: "1px solid #2b2b2b",
      }}>
        <button onClick={resetAll} style={btnStyle}>Reset all to SfM</button>
        <button onClick={onImportClick} style={btnStyle}>Import JSON</button>
        <button onClick={onExport} style={btnStyle}>Export JSON</button>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/json,.json"
          onChange={onImportFile}
          style={{ display: "none" }}
        />
      </div>

      {/* minmax(0, 1fr) lets the middle column shrink below its content.
          https://developer.mozilla.org/en-US/docs/Web/CSS/minmax */}
      <div style={{
        flex: 1, minHeight: 0,
        display: "grid",
        gridTemplateColumns: "480px minmax(0, 1fr) 340px",
        gap: 8, padding: 8, boxSizing: "border-box",
      }}>
        <div style={{
          display: "grid",
          // The thumbnail only needs image-aspect height; the controls panel gets the rest.
          gridTemplateRows: "minmax(180px, 2fr) 5fr",
          gap: 8,
          minHeight: 0, minWidth: 0,
        }}>
          <CameraView cameras={cameras} />
          <PoseControls cameras={cameras} />
        </div>

        <div style={{
          minHeight: 0, minWidth: 0,
          border: "1px solid #2b2b2b", borderRadius: 4, overflow: "hidden",
        }}>
          <TopDownMap cameras={cameras} />
        </div>

        <div style={{ minHeight: 0, minWidth: 0, overflowY: "auto" }}>
          <OrthoPanels cameras={cameras} />
        </div>
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  background: "#2b2b2b", color: "#cccccc", border: "1px solid #3c3c3c",
  padding: "4px 10px", fontSize: 12, cursor: "pointer", borderRadius: 3,
};
