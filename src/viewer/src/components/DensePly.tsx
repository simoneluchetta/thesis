/**
 * Loads a PLY and adds it to the scene: with faces a mesh, without them points.
 * useLoader caches by url and suspends while loading.
 */
// https://github.com/mrdoob/three.js/blob/dev/examples/jsm/loaders/PLYLoader.js
// https://en.wikipedia.org/wiki/PLY_(file_format)
// https://docs.pmnd.rs/react-three-fiber/api/hooks
import { Suspense, useMemo } from "react";
import { useLoader } from "@react-three/fiber";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

interface Props {
  url: string;
  /** Multiplier for point size when the PLY has no faces. */
  pointSize?: number;
}

// Required<Props>: the wrapper below fills in the defaults.
// https://www.typescriptlang.org/docs/handbook/utility-types.html#requiredtype
function PlyContent({ url, pointSize }: Required<Props>) {
  const geometry = useLoader(PLYLoader, url) as THREE.BufferGeometry;

  const { material, isMesh } = useMemo(function () {
    geometry.computeBoundingBox();
    const hasIndex = !!geometry.index;
    const hasColor = !!geometry.getAttribute("color");
    if (hasIndex) {
      // PLY files carry no normals, so lighting needs them computed here.
      // https://threejs.org/docs/#api/en/core/BufferGeometry.computeVertexNormals
      geometry.computeVertexNormals();
      return {
        material: new THREE.MeshStandardMaterial({
          vertexColors: hasColor,
          color: hasColor ? 0xffffff : 0xb0b0b0,
          flatShading: true,
          side: THREE.DoubleSide,
        }),
        isMesh: true,
      };
    }
    // https://threejs.org/docs/#api/en/materials/PointsMaterial
    return {
      material: new THREE.PointsMaterial({
        size: pointSize,
        vertexColors: hasColor,
        sizeAttenuation: true,
      }),
      isMesh: false,
    };
  }, [geometry, pointSize]);

  if (isMesh) {
    return <mesh geometry={geometry} material={material} />;
  }
  return <points geometry={geometry} material={material} />;
}

/** <Suspense> so toggling a layer in the HUD doesn't crash the canvas while the PLY loads.
 *  https://react.dev/reference/react/Suspense */
export default function DensePly({ url, pointSize = 0.015 }: Props) {
  return (
    <Suspense fallback={null}>
      <PlyContent url={url} pointSize={pointSize} />
    </Suspense>
  );
}
