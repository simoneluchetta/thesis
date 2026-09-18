/**
 * One camera in the 3-D scene: a frustum with the thumbnail on the far face.
 */
// https://github.com/pmndrs/drei#html
// https://github.com/pmndrs/drei#texture--usetexture
// https://threejs.org/docs/#api/en/math/Color
import { useMemo } from "react";
import { Html, useTexture } from "@react-three/drei";
import * as THREE from "three";
import type { SceneCameraEntry } from "../types";
import { useViewer } from "../state/store";
import { buildFrustumGeometry, colmapCamToWorldMatrix } from "../utils/coords";

interface Props {
  cam: SceneCameraEntry;
  showLabel: boolean;
  showThumb: boolean;
  selected: boolean;
  /** Multiplier on near/far, to shrink or grow the whole frustum. */
  scale: number;
  /** Draw at this pose instead of the camera's own, for the PnP overlay. */
  poseOverride?: {
    rotation_c2w: number[][];
    position: [number, number, number];
  };
  /** Replaces the reproj-error colour. */
  colorOverride?: THREE.Color;
  /** Replaces cam.name in the label. */
  labelOverride?: string;
}

interface ThumbProps {
  url: string;
  width: number;
  height: number;
  z: number;
  selected: boolean;
  onSelect: () => void;
}

/** Textured plane on the far face of a frustum, showing that camera's frame. */
function ThumbnailPlane({ url, width, height, z, selected, onSelect }: ThumbProps) {
  // useTexture suspends while the image loads and caches it across frustums.
  const tex = useTexture(url);
  return (
    <mesh
      position={[0, 0, z]}
      onClick={function (e) {
        e.stopPropagation();
        onSelect();
      }}
    >
      <planeGeometry args={[width, height]} />
      {/* toneMapped off so the thumbnail keeps the colours of the source frame.
          https://threejs.org/docs/#api/en/materials/MeshBasicMaterial
          https://threejs.org/docs/#manual/en/introduction/Color-management */}
      <meshBasicMaterial
        map={tex}
        side={THREE.DoubleSide}
        transparent
        opacity={selected ? 1.0 : 0.85}
        toneMapped={false}
      />
    </mesh>
  );
}

// Amber, used for whichever camera is selected.
const COLOR_SELECTED = new THREE.Color("#ffd44f");
// Reprojection-error bands, in px at full image resolution (3840x2160).
const COLOR_GOOD = new THREE.Color("#5fff8a"); // green  <= 1.5
const COLOR_OK = new THREE.Color("#5fa8ff");   // blue   <= 2.5
const COLOR_WARN = new THREE.Color("#ffc04f"); // yellow <= 4.0
const COLOR_BAD = new THREE.Color("#ff5a5a");  // red    > 4.0 or unknown

/** Colour for a camera: amber when selected, otherwise its reprojection-error band. */
function colorForCam(cam: SceneCameraEntry, selected: boolean): THREE.Color {
  if (selected) return COLOR_SELECTED;
  const r = cam.diag?.reproj_error_px;
  if (r === null || r === undefined) return COLOR_BAD;
  if (r <= 1.5) return COLOR_GOOD;
  if (r <= 2.5) return COLOR_OK;
  if (r <= 4.0) return COLOR_WARN;
  return COLOR_BAD;
}

/** One camera in the 3-D scene: frustum lines, name label and thumbnail plane. */
export default function CameraFrustum({
  cam,
  showLabel,
  showThumb,
  selected,
  scale,
  poseOverride,
  colorOverride,
  labelOverride,
}: Props) {
  const setSelectedCam = useViewer(function (s) { return s.setSelectedCam; });
  const labelScale = useViewer(function (s) { return s.labelScale; });

  const rotation = poseOverride?.rotation_c2w ?? cam.rotation_c2w;
  const position = poseOverride?.position ?? cam.position;

  // matrixAutoUpdate is off on the group below, so this matrix is used as given.
  // https://threejs.org/docs/#api/en/core/Object3D.matrixAutoUpdate
  const matrix = useMemo(
    function () { return colmapCamToWorldMatrix(rotation, position); },
    [rotation, position],
  );

  const far = cam.far * scale;

  const lines = useMemo(function () {
    const g = buildFrustumGeometry(cam.fovYDeg, cam.aspect, far);
    const color = colorOverride ?? colorForCam(cam, selected);
    // linewidth is ignored by most WebGL drivers; the label border carries the hint.
    // https://threejs.org/docs/#api/en/materials/LineBasicMaterial.linewidth
    const m = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: selected ? 1.0 : 0.75,
      depthWrite: false,
      linewidth: cam.is_stereo ? 2 : 1, // stereo eyes drawn thicker
    });
    return new THREE.LineSegments(g, m);
  }, [cam, far, selected, colorOverride]);

  // Thumbnail plane at the camera's aspect, just inside the far plane at -far.
  const thumbDims = useMemo(function () {
    const planeH = 0.7 * far;
    const planeW = planeH * cam.aspect;
    const z = -0.95 * far;
    return { w: planeW, h: planeH, z };
  }, [far, cam.aspect]);

  function handleSelect() {
    setSelectedCam(selected ? null : cam.id);
  }

  return (
    <group matrix={matrix} matrixAutoUpdate={false}>
      <primitive object={lines} />

      {showLabel && (
        <Html
          distanceFactor={Math.max(0.5 * far, 1) * labelScale}
          position={[0, 0.05 * far, 0]}
          center
        >
          <div
            onClick={function (e) {
              e.stopPropagation();
              handleSelect();
            }}
            style={{
              cursor: "pointer",
              padding: "4px 10px",
              borderRadius: 4,
              background: selected ? "#ffd44fcc" : "#0008",
              color: selected ? "#000" : "#fff",
              fontFamily: "sans-serif",
              fontSize: 20,
              fontWeight: cam.is_stereo ? 700 : 400,
              userSelect: "none",
              whiteSpace: "nowrap",
              border: cam.is_stereo ? "1px solid #ff5a5a" : "none",
            }}
            title={
              cam.diag?.reproj_error_px === null ||
              cam.diag?.reproj_error_px === undefined
                ? "no SfM observations"
                : `reproj=${cam.diag.reproj_error_px.toFixed(2)}px, ` +
                  `${cam.diag.n_visible_points} pts`
            }
          >
            {labelOverride ?? cam.name}
            {cam.is_stereo ? " ⌬" : ""}
          </div>
        </Html>
      )}

      {showThumb && cam.thumb && (
        <ThumbnailPlane
          url={cam.thumb}
          width={thumbDims.w}
          height={thumbDims.h}
          z={thumbDims.z}
          selected={selected}
          onSelect={handleSelect}
        />
      )}
    </group>
  );
}
