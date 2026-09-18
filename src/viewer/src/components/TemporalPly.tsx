import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

/**
 * One point cloud per instant, swapped as the timeline is scrubbed.
 * LRU of recent frames, prefetch ahead, previous frame stays up while the next loads.
 */
// https://github.com/mrdoob/three.js/blob/dev/examples/jsm/loaders/PLYLoader.js
// https://en.wikipedia.org/wiki/Cache_replacement_policies
// https://developer.mozilla.org/en-US/docs/Web/API/Response/arrayBuffer

// How many parsed frames stay in memory before the oldest is dropped.
const MAX_CACHED = 24;
// How many frames ahead to warm while the timeline moves.
const PREFETCH = 3;

/** Least-recently-used cache of parsed PLY frames, keyed by url. */
class PlyLru {
  private map = new Map<string, THREE.BufferGeometry>();
  private inflight = new Map<string, Promise<THREE.BufferGeometry>>();
  private loader = new PLYLoader();

  /** Cache hit, refreshing recency, or null. */
  peek(url: string): THREE.BufferGeometry | null {
    const g = this.map.get(url);
    if (g) {
      this.map.delete(url);
      this.map.set(url, g);
    }
    return g ?? null;
  }

  /** Cached geometry, or fetches and parses it. One request per url at a time. */
  load(url: string): Promise<THREE.BufferGeometry> {
    const hit = this.peek(url);
    if (hit) return Promise.resolve(hit);
    let p = this.inflight.get(url);
    if (!p) {
      p = fetch(url)
        .then(function (r) {
          if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
          return r.arrayBuffer();
        })
        .then((buf) => {
          // parse() takes the raw buffer, so the fetch above is reused.
          const g = this.loader.parse(buf);
          this.map.set(url, g);
          while (this.map.size > MAX_CACHED) {
            // Map iterates in insertion order, so the first entry is the oldest.
            // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map
            const oldest = this.map.entries().next().value as [string, THREE.BufferGeometry];
            this.map.delete(oldest[0]);
            oldest[1].dispose();
          }
          return g;
        })
        .finally(() => this.inflight.delete(url));
      this.inflight.set(url, p);
    }
    return p;
  }

  /** Frees every cached geometry, for unmount. */
  disposeAll(): void {
    for (const g of this.map.values()) g.dispose();
    this.map.clear();
  }
}

/** The point cloud for the frame at `index`, swapped as the timeline moves. */
export default function TemporalPly({
  urls, index, pointSize, onShownIndex,
}: {
  urls: string[];
  index: number;
  pointSize: number;
  /** Which frame is actually on screen. Lags `index` while loading. */
  onShownIndex?: (i: number) => void;
}) {
  // One cache for the life of the component.
  // https://react.dev/reference/react/useRef
  const cache = useRef<PlyLru | null>(null);
  if (!cache.current) cache.current = new PlyLru();
  const [shown, setShown] = useState<THREE.BufferGeometry | null>(null);

  useEffect(function () {
    return function () { return cache.current?.disposeAll(); };
  }, []);

  useEffect(function () {
    const c = cache.current!;
    const url = urls[index];
    if (!url) return;
    let live = true;
    c.load(url)
      .then(function (g) {
        if (!live) return;
        setShown(g);
        onShownIndex?.(index);
      })
      .catch(function () {});
    // Warm the next few frames so playback doesn't stall. Wraps at the end.
    for (let k = 1; k <= PREFETCH; k++) {
      const nxt = urls[(index + k) % urls.length];
      if (nxt) c.load(nxt).catch(function () {});
    }
    return function () { live = false; };
    // onShownIndex left out on purpose: a new identity would restart the load.
  }, [urls, index]);

  const material = useMemo(
    function () {
      return new THREE.PointsMaterial({ size: pointSize, vertexColors: true, sizeAttenuation: true });
    },
    [pointSize],
  );
  useEffect(function () {
    return function () { return material.dispose(); };
  }, [material]);

  if (!shown) return null;
  return <points geometry={shown} material={material} />;
}
