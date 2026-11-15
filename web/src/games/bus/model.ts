import type { BusGeometry, Pt } from "./geometry";

/** Seat colours: the Lines stripes of the art (D21), P0..P4 = red, green, orange, blue, white. */
export const SEAT_FILL = ["#FE0000", "#007F00", "#FE6500", "#0065FE", "#FEFEFE"] as const;
export const SEAT_INK = ["#FFFFFF", "#FFFFFF", "#1A1A1A", "#FFFFFF", "#1A1A1A"] as const;
export const SEAT_NAME = ["red", "green", "orange", "blue", "white"] as const;
export const INK = "#1A1A1A";
export const PAPER = "#FFF4D6";
export const RED = "#B00A0A";
export const STONE = "#8C1D1D";
export const PASSENGER = "#3A3A3A";
export const LINE_GAP = 13;

/** Indexed read that fails loudly (ids in the geometry are array indices). */
export function at<T>(a: readonly T[], i: number): T {
  const v = a[i];
  if (v === undefined) throw new Error(`index ${i} out of range`);
  return v;
}

export const seatFill = (s: number): string => at(SEAT_FILL, s % 5);
export const seatInk = (s: number): string => at(SEAT_INK, s % 5);

export interface SeatState {
  points: number;
  stones: number;
  buses: number;
  markers_left: number;
  line: number[];
}

export interface BusState {
  round: number;
  phase: string;
  current_player?: number;
  start_player: number;
  clock: string;
  stones_on_clock: number;
  seats: SeatState[];
  buildings: { site: number; type: string; occupied: boolean }[];
  waiting: Record<string, number>;
  supply: number;
  action_board: Record<string, (number | null)[]>;
}

export interface BusDerived {
  seats: { line_markers_left: number; line_nodes: number[]; ends: [number, number] | null }[];
  street_lines: Record<string, number[]>;
  open_tier: string | null;
}

export interface BusViewData {
  state: BusState;
  derived: BusDerived;
  prompt?: { phase?: string; actor?: number | null };
}

/** The adapter view (UI_DESIGN §6) when it has the expected shape, else null. */
export function parseBusView(raw: unknown): BusViewData | null {
  if (!raw || typeof raw !== "object") return null;
  const v = raw as Partial<BusViewData>;
  const s = v.state;
  if (!s || !Array.isArray(s.seats) || !s.action_board || !s.waiting || !Array.isArray(s.buildings))
    return null;
  if (!v.derived || !v.derived.street_lines || !Array.isArray(v.derived.seats)) return null;
  return v as BusViewData;
}

/** Shift a polyline sideways by d (positive = left of travel), averaging normals at bends. */
export function offsetPolyline(pts: Pt[], d: number): Pt[] {
  if (d === 0 || pts.length < 2) return pts;
  const normals: Pt[] = [];
  for (let i = 0; i + 1 < pts.length; i++) {
    const [x0, y0] = at(pts, i);
    const [x1, y1] = at(pts, i + 1);
    const len = Math.hypot(x1 - x0, y1 - y0) || 1;
    normals.push([(y0 - y1) / len, (x1 - x0) / len]);
  }
  return pts.map(([x, y], i) => {
    const ns = normals.slice(Math.max(i - 1, 0), i + 1);
    const nx = ns.reduce((a, n) => a + n[0], 0) / ns.length;
    const ny = ns.reduce((a, n) => a + n[1], 0) / ns.length;
    const norm = Math.hypot(nx, ny) || 1;
    return [x + (d * nx) / norm, y + (d * ny) / norm] as Pt;
  });
}

export function streetPoints(geo: BusGeometry, streetId: number): Pt[] {
  const st = at(geo.streets, streetId);
  const a = at(geo.nodes, st.a);
  const b = at(geo.nodes, st.b);
  return [[a.x, a.y], ...st.via.map((p) => [p[0], p[1]] as Pt), [b.x, b.y]];
}

export const pathD = (pts: Pt[]): string =>
  "M " + pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(" L ");

/** Bus-grid cell for a seat's bus number (2..5); null when not drawn (D19, Layout.bus_cell). */
export function busCell(geo: BusGeometry, seat: number, bus: number): Pt | null {
  if (seat < 0 || seat > 4 || bus < 2 || bus > 5) return null;
  return geo.layout.bus_grid[(bus - 2) * 5 + seat] ?? null;
}

/** Score-track position for a score (1..20, larger scores clamp to the last cell). */
export function scoreCell(geo: BusGeometry, points: number): Pt | null {
  if (points <= 0) return null;
  return geo.layout.score_track[Math.min(points, geo.layout.score_track.length) - 1] ?? null;
}
