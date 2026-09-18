/**
 * The 3-D scene: court, cameras and point clouds in one three.js canvas.
 * Holds no state of its own, everything visible comes from the viewer store.
 */
// https://docs.pmnd.rs/react-three-fiber/api/canvas
// https://docs.pmnd.rs/react-three-fiber/api/hooks (useThree)
// https://threejs.org/docs/#examples/en/controls/OrbitControls
// https://github.com/pmndrs/drei#grid
// https://github.com/pmndrs/drei#text
import { useLayoutEffect, useMemo } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { Grid, OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { SceneJson } from "../types";
import { useViewer } from "../state/store";
import { alignUpToY, resolveUp } from "../utils/coords";
import CameraFrustum from "./CameraFrustum";
import DensePly from "./DensePly";
import CourtOverlay3D from "./CourtOverlay3D";
import PeopleLayer from "./PeopleLayer";
import TiePoints3D from "./TiePoints3D";
import { OrbitAxisSync, WorldZOrbitDrag } from "./OrbitAxisControls";

interface Props {
  scene: SceneJson;
  /** Draw the manual-placement frustums. The PnP result tab can hide them. */
  showManualCams?: boolean;
  /** Also draw each cam at its `pnp_*` pose, in magenta. */
  showPnpOverlay?: boolean;
  /** Hide the tie points and the PLY layers whatever the store says. */
  hidePoints?: boolean;
}

/** Re-targets OrbitControls on reset and on camera select. Must live inside <Canvas>.
 *  useLayoutEffect, so the camera never paints at the old spot.
 *  https://react.dev/reference/react/useLayoutEffect */
function CameraFocus({
  scene,
  centroidOffset,
  worldQuat,
  initialPosition,
}: {
  scene: SceneJson;
  centroidOffset: THREE.Vector3;
  worldQuat: THREE.Quaternion;
  initialPosition: THREE.Vector3;
}) {
  const selectedCamId = useViewer(function (s) { return s.selectedCamId; });
  const resetViewKey = useViewer(function (s) { return s.resetViewKey; });
  const camera = useThree(function (s) { return s.camera; });
  const controls = useThree(function (s) { return s.controls; }) as OrbitControlsImpl | null;

  // Reset view: first mount and every reset click.
  useLayoutEffect(function () {
    if (!controls) return;
    controls.target.set(0, 0, 0);
    camera.position.copy(initialPosition);
    camera.up.set(0, 1, 0);
    controls.update();
  }, [resetViewKey, controls, camera, initialPosition]);

  // Focus on selected camera.
  useLayoutEffect(function () {
    if (selectedCamId === null || !controls) return;
    const target = scene.cameras.find(function (c) { return c.id === selectedCamId; });
    if (!target) return;
    // Same transform as the world group, so the target lands where the user sees the cam.
    // https://threejs.org/docs/#api/en/math/Vector3.applyQuaternion
    const p = new THREE.Vector3(...target.position)
      .add(centroidOffset)
      .applyQuaternion(worldQuat);
    controls.target.copy(p);
    const dir = new THREE.Vector3().subVectors(camera.position, p).normalize();
    if (dir.lengthSq() < 1e-6) dir.set(0, 0, 1);
    camera.position.copy(p).addScaledVector(dir, Math.max(scene.world.scale_hint, 1));
    controls.update();
  }, [selectedCamId, scene, controls, camera, centroidOffset, worldQuat]);
  return null;
}

// Magenta, so the PnP frustums read apart from the manual ones.
const PNP_OVERLAY_COLOR = new THREE.Color("#ff45d6");

/** The whole 3-D view: grid, court, cameras and point clouds in one canvas. */
export default function SceneCanvas({
  scene,
  showManualCams = true,
  showPnpOverlay = false,
  hidePoints = false,
}: Props) {
  const showFrustums = useViewer(function (s) { return s.showFrustums; });
  const showLabels = useViewer(function (s) { return s.showLabels; });
  const showThumbs = useViewer(function (s) { return s.showThumbs; });
  const showPeople = useViewer(function (s) { return s.showPeople; });
  const showBasketballCourt = useViewer(function (s) { return s.showBasketballCourt; });
  const showVolleyballCourt = useViewer(function (s) { return s.showVolleyballCourt; });
  const courtAnchorX = useViewer(function (s) { return s.courtAnchorX; });
  const courtAnchorY = useViewer(function (s) { return s.courtAnchorY; });
  const courtAnchorZ = useViewer(function (s) { return s.courtAnchorZ; });
  const courtYawDeg = useViewer(function (s) { return s.courtYawDeg; });
  const plyLayersOn = useViewer(function (s) { return s.plyLayersOn; });
  const selectedCamId = useViewer(function (s) { return s.selectedCamId; });
  const setSelectedCam = useViewer(function (s) { return s.setSelectedCam; });
  const frustumScale = useViewer(function (s) { return s.frustumScale; });
  const upAxis = useViewer(function (s) { return s.upAxis; });
  const orbitAxis = useViewer(function (s) { return s.orbitAxis; });

  // Up axis (auto or manual), then the rotation that maps it onto three.js +Y.
  // useMemo keeps the same objects across renders.
  // https://react.dev/reference/react/useMemo
  const upVec = useMemo(function () { return resolveUp(upAxis, scene.world.up); }, [upAxis, scene.world.up]);
  const worldQuat = useMemo(function () { return alignUpToY(upVec); }, [upVec]);

  // Put the camera centroid at the origin, where OrbitControls targets by default.
  const centroidOffset = useMemo(
    function () { return new THREE.Vector3(...scene.world.centroid).multiplyScalar(-1); },
    [scene.world.centroid],
  );

  // Starting eye position: a comfortable orbit radius above and away.
  const initialPosition = useMemo(function () {
    const r = Math.max(2.5 * scene.world.scale_hint, 2);
    return new THREE.Vector3(r, r * 0.7, r);
  }, [scene.world.scale_hint]);

  const gridSize = Math.max(8 * scene.world.scale_hint, 8);
  const sectionSize = Math.max(0.5 * scene.world.scale_hint, 0.5);

  return (
    <Canvas
      camera={{ position: initialPosition.toArray(), fov: 50, near: 0.05, far: 5000 }}
      gl={{ antialias: true }}
      // Fires on a click that hit nothing, which is how a camera gets deselected.
      // https://docs.pmnd.rs/react-three-fiber/api/events
      onPointerMissed={function () { setSelectedCam(null); }}
    >
      {/* attach writes this object into scene.background rather than adding a child.
          https://docs.pmnd.rs/react-three-fiber/api/objects */}
      <color attach="background" args={["#1f1f1f"]} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 20, 5]} intensity={0.8} />
      <OrbitControls
        makeDefault
        enableDamping={orbitAxis !== "z"}
        enableRotate={orbitAxis !== "z"}
        dampingFactor={0.08}
      />
      <CameraFocus
        scene={scene}
        centroidOffset={centroidOffset}
        worldQuat={worldQuat}
        initialPosition={initialPosition}
      />
      <OrbitAxisSync />
      <WorldZOrbitDrag />

      <group quaternion={worldQuat}>
        <group position={centroidOffset}>
          {/* https://threejs.org/docs/#api/en/helpers/AxesHelper */}
          <axesHelper args={[Math.max(scene.world.scale_hint, 1)]} />
          {(function () {
            // Axis labels at the tips of the axesHelper: X red, Y green, Z blue.
            const len = Math.max(scene.world.scale_hint, 1);
            const tip = len * 1.08;
            const size = len * 0.18;
            return (
              <>
                <Text position={[tip, 0, 0]} fontSize={size} color="#ff6464"
                  anchorX="left" anchorY="middle" outlineWidth={size * 0.06}
                  outlineColor="#000">X</Text>
                <Text position={[0, tip, 0]} fontSize={size} color="#64ff80"
                  anchorX="center" anchorY="bottom" outlineWidth={size * 0.06}
                  outlineColor="#000">Y</Text>
                <Text position={[0, 0, tip]} fontSize={size} color="#7aa2ff"
                  anchorX="center" anchorY="middle" outlineWidth={size * 0.06}
                  outlineColor="#000">Z</Text>
              </>
            );
          })()}
          <Grid
            args={[gridSize, gridSize]}
            cellSize={sectionSize / 5}
            sectionSize={sectionSize}
            cellColor="#2b2b2b"
            sectionColor="#3c3c3c"
            fadeDistance={10 * scene.world.scale_hint}
            fadeStrength={1}
            infiniteGrid={false}
            position={[
              scene.world.centroid[0],
              scene.world.centroid[1],
              scene.world.centroid[2],
            ]}
          />

          {showPeople && scene.people && <PeopleLayer data={scene.people} />}

          {!hidePoints && scene.tie_points_3d && (
            <TiePoints3D data={scene.tie_points_3d} />
          )}

          {/* Court lines at Z=0. Anchor and yaw come from the HUD. */}
          <CourtOverlay3D
            basketball={showBasketballCourt}
            volleyball={showVolleyballCourt}
            anchorX={courtAnchorX}
            anchorY={courtAnchorY}
            anchorZ={courtAnchorZ}
            yawRad={courtYawDeg * Math.PI / 180}
          />

          {showFrustums && showManualCams &&
            scene.cameras.map(function (cam) {
              return (
                <CameraFrustum
                  key={`m-${cam.id}`}
                  cam={cam}
                  showLabel={showLabels}
                  showThumb={showThumbs}
                  selected={selectedCamId === cam.id}
                  scale={frustumScale}
                />
              );
            })}

          {showFrustums && showPnpOverlay &&
            scene.cameras
              .filter(function (cam) { return cam.pnp_position && cam.pnp_rotation_c2w; })
              .map(function (cam) {
                return (
                  <CameraFrustum
                    key={`p-${cam.id}`}
                    cam={cam}
                    showLabel={showLabels}
                    showThumb={false}
                    selected={selectedCamId === cam.id}
                    scale={frustumScale}
                    poseOverride={{
                      rotation_c2w: cam.pnp_rotation_c2w!,
                      position: cam.pnp_position!,
                    }}
                    colorOverride={PNP_OVERLAY_COLOR}
                    labelOverride={`${cam.name} (PnP)`}
                  />
                );
              })}

          {!hidePoints && scene.ply_layers
            .filter(function (l) { return plyLayersOn[l.id]; })
            .map(function (l) {
              return (
                <DensePly
                  key={l.id}
                  url={l.url}
                  pointSize={Math.max(0.012 * scene.world.scale_hint, 0.006)}
                />
              );
            })}
        </group>
      </group>
    </Canvas>
  );
}
