/**
 * The tracked people, drawn as markers on the floor of the 3-D scene.
 * Colours match the People map so the same person looks the same in both.
 */
// https://threejs.org/docs/#api/en/geometries/CylinderGeometry
// https://threejs.org/docs/#api/en/geometries/SphereGeometry
// https://threejs.org/docs/#api/en/geometries/RingGeometry
import { useMemo } from "react";
import { useViewer } from "../state/store";
import type { ScenePeople } from "../types";
import { PALETTE } from "../utils/palette";

interface Props {
  data: ScenePeople;
}

const BODY_H = 1.8;   // metres - a rough standing person
// metres - radius of the body cylinder
const BODY_R = 0.16;
// metres - radius of the head sphere
const HEAD_R = 0.22;

/**
 * Markers for the people at the selected timeline index. Positions are raw
 * placer-world XYZ (Z up, floor z=0), inside SceneCanvas' world group.
 */
export default function PeopleLayer({ data }: Props) {
  const idx = useViewer(function (s) { return s.peopleTimeIndex; });
  const people = useMemo(function () {
    if (data.timestamps.length === 0) return [];
    const i = Math.max(0, Math.min(idx, data.timestamps.length - 1));
    return data.frames[String(data.timestamps[i])] ?? [];
  }, [data, idx]);

  return (
    <group>
      {people.map(function (p) {
        const pid = p.track_id ?? p.id;
        const color = PALETTE[pid % PALETTE.length];
        const [x, y] = p.pos; // z is 0 (floor)
        // More cameras agreeing = more solid.
        const opacity = Math.min(1, 0.45 + 0.12 * p.n_cams);
        return (
          <group key={pid} position={[x, y, 0]}>
            {/* cylinderGeometry stands along +Y, so rotate it onto world +Z */}
            <mesh position={[0, 0, BODY_H / 2]} rotation={[Math.PI / 2, 0, 0]}>
              <cylinderGeometry args={[BODY_R, BODY_R, BODY_H, 12]} />
              <meshStandardMaterial color={color} transparent opacity={opacity} />
            </mesh>
            {/* head */}
            <mesh position={[0, 0, BODY_H + HEAD_R * 0.5]}>
              <sphereGeometry args={[HEAD_R, 16, 12]} />
              <meshStandardMaterial color={color} transparent opacity={opacity} />
            </mesh>
            {/* Ground ring so the foot position reads from above. side=2 is DoubleSide.
                https://threejs.org/docs/#api/en/constants/Materials */}
            <mesh position={[0, 0, 0.01]} rotation={[Math.PI / 2, 0, 0]}>
              <ringGeometry args={[BODY_R * 1.4, BODY_R * 2.2, 24]} />
              <meshBasicMaterial color={color} transparent opacity={0.85} side={2} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}
