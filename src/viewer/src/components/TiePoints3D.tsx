/**
 * The off-floor tie points (net, bench) and the fitted court lines, in 3-D.
 */
// https://github.com/pmndrs/drei#line
// https://github.com/pmndrs/drei#text
import { useMemo } from "react";
import { Line, Text } from "@react-three/drei";
import type { SceneTiePoints3D, SceneTiePoint } from "../types";

interface Props {
  data: SceneTiePoints3D;
}

/** A world position in metres. */
type Vec3 = [number, number, number];

/** Net cyan, bench green. */
const GROUP_COLOR: Record<string, string> = {
  vbnet: "#28d7ff",
  bench: "#7cff45",
};
// Any group that is not in GROUP_COLOR.
const DEFAULT_COLOR = "#dddddd";

/** The hoops were dropped from the app; skip their tie points if a scene still has them. */
function isHoop(group: string) { return group.startsWith("hoop"); }

const SPHERE_R = 0.11; // metres

/**
 * Markers, droplines to the floor, per-group connectors and the court lines.
 * Positions are raw placer-world XYZ (Z up, floor z=0), inside SceneCanvas' world group.
 */
export default function TiePoints3D({ data }: Props) {
  const points = useMemo(function () {
    return data.points.filter(function (p) { return !isHoop(p.group); });
  }, [data]);

  const byId = useMemo(function () {
    const m = new Map<string, SceneTiePoint>();
    for (const p of points) m.set(p.id, p);
    return m;
  }, [points]);

  const groups = useMemo(function () {
    const g = new Map<string, SceneTiePoint[]>();
    for (const p of points) {
      const arr = g.get(p.group) ?? [];
      arr.push(p);
      g.set(p.group, arr);
    }
    return g;
  }, [points]);

  // Connector segments that redraw the shape of each group.
  const segments = useMemo(function () {
    const out: { pts: [Vec3, Vec3]; color: string }[] = [];
    function P(group: string, sub: string): Vec3 | null {
      return byId.get(`${group}/${sub}`)?.pos ?? null;
    }
    function push(a: Vec3 | null, b: Vec3 | null, color: string) {
      if (a && b) out.push({ pts: [a, b], color });
    }
    for (const group of groups.keys()) {
      const color = GROUP_COLOR[group] ?? DEFAULT_COLOR;
      if (group === "vbnet") {
        const chain = ["p1", "p2", "p3", "p4"].map(function (s) { return P(group, s); });
        for (let i = 0; i < chain.length - 1; i++) push(chain[i], chain[i + 1], color);
      } else if (group === "bench") {
        const fL = P(group, "fL");
        const fR = P(group, "fR");
        const bR = P(group, "bR");
        const bL = P(group, "bL");
        push(fL, fR, color);
        push(fR, bR, color);
        push(bR, bL, color);
        push(bL, fL, color);
      }
    }
    return out;
  }, [byId, groups]);

  // Lifted 12 mm off the floor to avoid z-fighting. Basketball yellow, volleyball cyan.
  // https://en.wikipedia.org/wiki/Z-fighting
  const courtLines = useMemo(function () {
    const out: { pts: Vec3[]; color: string; key: string }[] = [];
    const lines = data.court_lines ?? [];
    lines.forEach(function (l, i) {
      if (!l.pts3d || l.pts3d.length < 2) return;
      out.push({
        pts: l.pts3d.map(function ([x, y, z]) { return [x, y, z + 0.012] as Vec3; }),
        color: l.court === "vb" ? "#28d7ff" : "#ffe04a",
        key: `${l.court}-${i}`,
      });
    });
    return out;
  }, [data.court_lines]);

  // One small label per group at its top-most marker.
  const labels = useMemo(function () {
    const out: { text: string; pos: Vec3; color: string }[] = [];
    for (const [group, pts] of groups) {
      const top = pts.reduce(function (a, b) { return (b.pos[2] > a.pos[2] ? b : a); });
      out.push({
        text: group,
        pos: [top.pos[0], top.pos[1], top.pos[2] + 0.35],
        color: GROUP_COLOR[group] ?? DEFAULT_COLOR,
      });
    }
    return out;
  }, [groups]);

  if (points.length === 0 && courtLines.length === 0) return null;

  return (
    <group>
      {/* connector lines */}
      {segments.map(function (s, i) {
        return (
          <Line key={`seg-${i}`} points={s.pts} color={s.color} lineWidth={2} />
        );
      })}

      {/* court lines */}
      {courtLines.map(function (c) {
        return (
          <Line key={`court-${c.key}`} points={c.pts} color={c.color} lineWidth={1.5} transparent opacity={0.85} />
        );
      })}

      {/* markers + droplines to the floor */}
      {points.map(function (p) {
        const color = GROUP_COLOR[p.group] ?? DEFAULT_COLOR;
        const [x, y, z] = p.pos;
        return (
          <group key={p.id}>
            <mesh position={[x, y, z]}>
              <sphereGeometry args={[SPHERE_R, 16, 12]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.35}
              />
            </mesh>
            {z > 0.05 && (
              <Line
                points={[
                  [x, y, z],
                  [x, y, 0],
                ]}
                color={color}
                lineWidth={1}
                transparent
                opacity={0.35}
                dashed
                dashSize={0.1}
                gapSize={0.08}
              />
            )}
          </group>
        );
      })}

      {/* group labels */}
      {labels.map(function (l) {
        return (
          <Text
            key={`lbl-${l.text}`}
            position={l.pos}
            fontSize={0.32}
            color={l.color}
            anchorX="center"
            anchorY="bottom"
            outlineWidth={0.02}
            outlineColor="#000"
          >
            {l.text}
          </Text>
        );
      })}
    </group>
  );
}
