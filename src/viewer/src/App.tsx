/**
 * App shell. Loads scene.json once and hands it to whichever tab is open.
 * Everything comes from static files, so there is no backend.
 */
// https://react.dev/reference/react/useState
// https://react.dev/reference/react/useEffect
// https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
import { useEffect, useState } from "react";
import type { SceneJson } from "./types";
import SceneCanvas from "./components/SceneCanvas";
import HUD from "./components/HUD";
import CameraPlacer from "./pages/CameraPlacer";
import CourtCalibrator from "./pages/CourtCalibrator";
import PnPResult from "./pages/PnPResult";
import PeopleMap from "./pages/PeopleMap";
import Annotate from "./pages/Annotate";
import NeRFResults from "./pages/NeRFResults";
import { useViewer } from "./state/store";

/** Root of the app: the tab bar, plus whichever tab is open. */
export default function App() {
  const [scene, setScene] = useState<SceneJson | null>(null);
  const [error, setError] = useState<string | null>(null);
  const page = useViewer(function (s) { return s.page; });
  const setPage = useViewer(function (s) { return s.setPage; });

  // Runs once on mount.
  useEffect(function () {
    fetch("/scene.json")
      .then(function (r) {
        if (!r.ok) throw new Error(`HTTP ${r.status} fetching scene.json`);
        return r.json();
      })
      .then(function (data: SceneJson) { return setScene(data); })
      .catch(function (e: Error) { return setError(e.message); });
  }, []);

  if (error) {
    return (
      <div style={{ padding: 24 }}>
        <h1>Failed to load scene.json</h1>
        <p style={{ color: "#f88" }}>{error}</p>
        <p>
          <code>public/scene.json</code> is part of the project and must not be deleted. Restore it
          from the delivered copy of the project.
        </p>
      </div>
    );
  }
  if (!scene) {
    return <div style={{ padding: 24 }}>Loading scene.json...</div>;
  }
  return (
    <div style={{
      position: "relative", width: "100vw", height: "100vh",
      display: "flex", flexDirection: "column", background: "#1f1f1f",
    }}>
      <nav style={{
        display: "flex", gap: 6, padding: "6px 10px",
        background: "#181818", borderBottom: "1px solid #2b2b2b",
      }}>
        <TabButton active={page === "viewer"} onClick={function () { setPage("viewer"); }}>
          3D viewer
        </TabButton>
        <TabButton active={page === "placer"} onClick={function () { setPage("placer"); }}>
          Camera placer
        </TabButton>
        <TabButton active={page === "calibrator"} onClick={function () { setPage("calibrator"); }}>
          Court calibrator
        </TabButton>
        <TabButton active={page === "pnpResult"} onClick={function () { setPage("pnpResult"); }}>
          PnP result
        </TabButton>
        <TabButton active={page === "peopleMap"} onClick={function () { setPage("peopleMap"); }}>
          People map
        </TabButton>
        <TabButton active={page === "annotate"} onClick={function () { setPage("annotate"); }}>
          Annotate
        </TabButton>
        <TabButton active={page === "nerf"} onClick={function () { setPage("nerf"); }}>
          NeRF
        </TabButton>
      </nav>
      <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
        {page === "viewer" && (
          <>
            <SceneCanvas scene={scene} />
            <HUD scene={scene} />
          </>
        )}
        {page === "placer" && <CameraPlacer scene={scene} />}
        {page === "calibrator" && <CourtCalibrator scene={scene} />}
        {page === "pnpResult" && <PnPResult scene={scene} />}
        {page === "peopleMap" && <PeopleMap scene={scene} />}
        {page === "annotate" && <Annotate />}
        {page === "nerf" && <NeRFResults />}
      </div>
    </div>
  );
}

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}
/** One button in the top tab bar, highlighted while its tab is open. */
function TabButton({ active, onClick, children }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      style={{
        background: active ? "#fc6" : "#252526",
        color: active ? "#1f1f1f" : "#cccccc",
        border: `1px solid ${active ? "#fc6" : "#3c3c3c"}`,
        padding: "4px 12px", fontSize: 12, cursor: "pointer", borderRadius: 3,
        fontWeight: active ? 600 : 400,
      }}
    >
      {children}
    </button>
  );
}
