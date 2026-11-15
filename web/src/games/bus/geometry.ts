import { useEffect, useState } from "react";
import { api } from "../../api/client";

export type Pt = [number, number];

export interface BusGeometry {
  map: { width: number; height: number };
  nodes: { id: number; x: number; y: number; station: boolean; roundabout: boolean }[];
  streets: { id: number; a: number; b: number; via: Pt[] }[];
  sites: { id: number; tier: string; node: number; x: number; y: number }[];
  layout: {
    actions: Record<string, Record<string, Pt>>;
    bus_grid: Pt[];
    clock: { x: number; y: number; hand: number; radius: number; angles: Record<string, number> };
    stones: Pt[];
    score_track: Pt[];
  };
}

/** Validate an API geometry payload; null when it does not look like Bus geometry. */
export function parseGeometry(raw: unknown): BusGeometry | null {
  if (!raw || typeof raw !== "object") return null;
  const g = raw as Partial<BusGeometry>;
  if (!g.map || !Array.isArray(g.nodes) || !Array.isArray(g.streets) || !Array.isArray(g.sites))
    return null;
  const lay = g.layout;
  if (!lay || !lay.actions || !Array.isArray(lay.bus_grid) || !lay.clock) return null;
  if (!Array.isArray(lay.stones) || !Array.isArray(lay.score_track)) return null;
  return g as BusGeometry;
}

let cached: Promise<BusGeometry | null> | null = null;

/** Fetch the board geometry once per page load. */
export function loadGeometry(): Promise<BusGeometry | null> {
  if (!cached) {
    cached = Promise.resolve(api.GET("/api/boards/{game}", { params: { path: { game: "bus" } } }))
      .then((r) => parseGeometry(r?.data?.geometry))
      .catch(() => null);
    cached.then((g) => {
      if (!g) cached = null; // retry next mount after a failure
    });
  }
  return cached;
}

export function resetGeometryCache(): void {
  cached = null;
}

export function useBusGeometry(): BusGeometry | null {
  const [geo, setGeo] = useState<BusGeometry | null>(null);
  useEffect(() => {
    let alive = true;
    loadGeometry().then((g) => {
      if (alive) setGeo(g);
    });
    return () => {
      alive = false;
    };
  }, []);
  return geo;
}
