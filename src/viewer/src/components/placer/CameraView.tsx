/** Placer pane showing the thumbnail of the selected camera. */
import { useMemo } from "react";
import type { SceneCameraEntry } from "../../types";
import { usePlacer } from "../../state/placerStore";

interface Props {
  cameras: SceneCameraEntry[];
}

/** object-fit: contain keeps the camera's aspect ratio in any pane shape.
 *  https://developer.mozilla.org/en-US/docs/Web/CSS/object-fit */
export default function CameraView({ cameras }: Props) {
  const selected = usePlacer(function (s) { return s.selected; });
  const selectedCam = useMemo(
    function () {
      return cameras.find(function (c) { return c.name === selected; }) ?? null;
    },
    [cameras, selected],
  );

  return (
    <div style={{
      background: "#181818",
      border: "1px solid #2b2b2b",
      borderRadius: 4,
      padding: 8,
      minHeight: 0,
      display: "flex",
      flexDirection: "column",
      gap: 6,
    }}>
      <div style={{ fontSize: 12, color: "#9d9d9d" }}>
        Camera view{selectedCam ? ` — ${selectedCam.name}${selectedCam.is_stereo ? " (stereo eye)" : ""}` : ""}
      </div>
      <div style={{
        flex: 1, minHeight: 0,
        display: "flex", alignItems: "center", justifyContent: "center",
        background: "#141414",
        border: "1px solid #2b2b2b",
        borderRadius: 3,
        overflow: "hidden",
      }}>
        {selectedCam?.thumb ? (
          <img
            src={`/${selectedCam.thumb}`}
            alt={selectedCam.name}
            style={{
              maxWidth: "100%", maxHeight: "100%",
              objectFit: "contain", display: "block",
            }}
          />
        ) : (
          <div style={{ color: "#6d6d6d", fontSize: 12, padding: 12, textAlign: "center" }}>
            {selectedCam
              ? "No thumbnail available for this camera."
              : "Click a camera on the map to select it."}
          </div>
        )}
      </div>
    </div>
  );
}
