import type { GameEvent } from "../../shell/playback";
import type { BusGeometry, Pt } from "./geometry";
import { streetPoints, type BusViewData } from "./model";

/*
 * Board motion (UI_DESIGN §3 motion, unit 9): which pieces are new after a shown change (they ease
 * in), and the passenger paths and "+1" floats drawn from the change's automatic-step events
 * (UI_DESIGN §6.6). Pure functions; the components in BusMotion.tsx render them.
 */

/** Durations at normal speed (ms); mirrors tokens.css --t-place / --t-move / --t-float. */
export const T_PLACE = 250;
export const T_MOVE = 700;
export const T_FLOAT = 950;

/** Keys of every placed piece whose appearance (or new position) should ease in. */
export function pieceKeys(data: BusViewData): Set<string> {
  const st = data.state;
  const out = new Set<string>();
  st.seats.forEach((seat, s) => {
    for (const sid of seat.line) out.add(`l${s}-${sid}`);
    for (let b = 2; b <= seat.buses; b++) out.add(`bus${s}-${b}`);
    if (seat.points > 0) out.add(`sc${s}:${seat.points}`);
  });
  for (const b of st.buildings) {
    out.add(`b${b.site}:${b.type}`);
    if (b.occupied) out.add(`o${b.site}`);
  }
  for (const [nid, count] of Object.entries(st.waiting)) if (count > 0) out.add(`w${nid}:${count}`);
  for (const [action, spaces] of Object.entries(st.action_board))
    spaces.forEach((seat, k) => {
      if (seat !== null && seat !== undefined) out.add(`m${action}-${k}:${seat}`);
    });
  out.add(`clock:${st.clock}`);
  return out;
}

/** Pieces in `cur` that were not in `prev` (none without a previous view). */
export function freshKeys(prev: BusViewData | null, cur: BusViewData): Set<string> {
  if (!prev) return new Set();
  const before = pieceKeys(prev);
  const out = new Set<string>();
  for (const k of pieceKeys(cur)) if (!before.has(k)) out.add(k);
  return out;
}

/** A passenger figure moving along `pts` after `delay` ms for `duration` ms. */
export interface Walk {
  key: string;
  kind: "deliver" | "exit" | "enter";
  pts: Pt[];
  delay: number;
  duration: number;
  seat?: number;
  site: number;
}

/** A "+1" floating from a delivery site to the scorer's panel. */
export interface Float {
  key: string;
  seat: number;
  site: number;
  points: number;
  delay: number;
  duration: number;
}

const ptOf = (geo: BusGeometry, node: number): Pt | null => {
  const n = geo.nodes[node];
  return n ? [n.x, n.y] : null;
};

/**
 * Street path of a delivery: along the scorer's line from the picked node to the destination's
 * node (line order from `derived.seats[seat].line_nodes`), then into the building. Falls back to
 * a straight line when the nodes are not on the line.
 */
export function deliveryPath(geo: BusGeometry, data: BusViewData, seat: number, node: number, site: number): Pt[] {
  const dest = geo.sites[site];
  const start = ptOf(geo, node);
  if (!dest || !start) return [];
  const end: Pt = [dest.x, dest.y];
  const nodes = data.derived.seats[seat]?.line_nodes ?? [];
  const line = data.state.seats[seat]?.line ?? [];
  const i = nodes.indexOf(node);
  const j = nodes.indexOf(dest.node);
  if (i < 0 || j < 0) return [start, end];
  const step = i <= j ? 1 : -1;
  const pts: Pt[] = [start];
  for (let k = i; k !== j; k += step) {
    const u = nodes[k]!;
    const v = nodes[k + step]!;
    const sid = line.find((id) => {
      const st = geo.streets[id];
      return st !== undefined && ((st.a === u && st.b === v) || (st.a === v && st.b === u));
    });
    const seg = sid === undefined ? null : streetPoints(geo, sid);
    const v_ = ptOf(geo, v);
    if (!seg || !v_) {
      if (v_) pts.push(v_);
      continue;
    }
    const oriented = geo.streets[sid!]!.a === u ? seg : [...seg].reverse();
    pts.push(...oriented.slice(1));
  }
  pts.push(end);
  return pts;
}

/** Walks and floats for one shown change's events, timed by `scale` (1 = normal speed). */
export function motionPlan(
  geo: BusGeometry,
  data: BusViewData,
  events: readonly GameEvent[],
  scale: number,
  seq: number,
): { walks: Walk[]; floats: Float[] } {
  const walks: Walk[] = [];
  const floats: Float[] = [];
  const move = T_MOVE * scale;
  events.forEach((e, n) => {
    if (e.type === "scored") {
      const seat = Number(e.seat);
      const node = Number(e.node);
      const site = Number(e.site);
      const pts = deliveryPath(geo, data, seat, node, site);
      if (pts.length >= 2)
        walks.push({ key: `${seq}:${n}:d`, kind: "deliver", pts, delay: 0, duration: move, seat, site });
      floats.push({ key: `${seq}:${n}:f`, seat, site, points: Number(e.points), delay: move * 0.8, duration: T_FLOAT * scale });
    } else if (e.type === "passengers_moved") {
      const list = (v: unknown) => (Array.isArray(v) ? (v as { site: number; node: number }[]) : []);
      const exited = list(e.exited);
      const entered = list(e.entered);
      const short = move * 0.6;
      for (const p of exited) {
        const s = geo.sites[p.site];
        const nd = ptOf(geo, p.node);
        if (s && nd)
          walks.push({ key: `${seq}:${n}:x${p.site}`, kind: "exit", pts: [[s.x, s.y], nd], delay: 0, duration: short, site: p.site });
      }
      for (const p of entered) {
        const s = geo.sites[p.site];
        const nd = ptOf(geo, p.node);
        if (s && nd)
          walks.push({
            key: `${seq}:${n}:e${p.site}`,
            kind: "enter",
            pts: [nd, [s.x, s.y]],
            delay: exited.length ? short : 0,
            duration: short,
            site: p.site,
          });
      }
    }
  });
  return { walks, floats };
}

/** Point at fraction `t` (0..1) of the polyline's length. */
export function pointAlong(pts: readonly Pt[], t: number): Pt {
  if (pts.length === 0) return [0, 0];
  if (pts.length === 1 || t <= 0) return pts[0]!;
  const lens: number[] = [];
  let total = 0;
  for (let i = 0; i + 1 < pts.length; i++) {
    const l = Math.hypot(pts[i + 1]![0] - pts[i]![0], pts[i + 1]![1] - pts[i]![1]);
    lens.push(l);
    total += l;
  }
  if (t >= 1 || total === 0) return pts[pts.length - 1]!;
  let d = t * total;
  for (let i = 0; i < lens.length; i++) {
    const l = lens[i]!;
    if (d <= l) {
      const f = l === 0 ? 0 : d / l;
      const a = pts[i]!;
      const b = pts[i + 1]!;
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
    }
    d -= l;
  }
  return pts[pts.length - 1]!;
}

/** Ease-in-out used for walks (matches --ease-move closely enough). */
export const easeInOut = (t: number): number => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

/** How long the change's motion lasts (ms): the latest end of a walk, float or placement. */
export function motionLength(plan: { walks: Walk[]; floats: Float[] }, scale: number): number {
  let end = T_PLACE * scale;
  for (const w of plan.walks) end = Math.max(end, w.delay + w.duration);
  for (const f of plan.floats) end = Math.max(end, f.delay + f.duration);
  return end;
}
