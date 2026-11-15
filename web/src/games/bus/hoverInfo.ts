import type { BusGeometry } from "./geometry";
import type { BusViewData } from "./model";

const ACTION_TITLE: Record<string, string> = {
  lines: "Lines",
  buses: "Buses",
  passengers: "Passengers",
  buildings: "Buildings",
  clock: "Clock",
  vrrooomm: "Vrrooomm!",
  starting_player: "Starting Player",
};

const seats = (list: number[]): string => list.map((s) => `P${s}`).join(", ");

/** Seats whose line passes through a node (from the derived ordered line nodes). */
function linesAtNode(data: BusViewData, node: number): number[] {
  const out: number[] = [];
  data.derived.seats.forEach((s, i) => {
    if (s.line_nodes.includes(node)) out.push(i);
  });
  return out;
}

export function nodeInfo(geo: BusGeometry, data: BusViewData, id: number): string[] {
  const n = geo.nodes[id];
  if (!n) return [];
  const out = [`Intersection ${id}${n.station ? " (station)" : ""}`];
  const waiting = data.state.waiting[String(id)] ?? 0;
  out.push(`Waiting passengers: ${waiting}`);
  const sites = geo.sites.filter((s) => s.node === id);
  for (const s of sites) {
    const b = data.state.buildings.find((x) => x.site === s.id);
    out.push(
      `Site ${s.id} (${s.tier}): ${b ? `${b.type}${b.occupied ? ", occupied" : ""}` : "empty"}`,
    );
  }
  const lines = linesAtNode(data, id);
  out.push(lines.length ? `Lines: ${seats(lines)}` : "Lines: none");
  return out;
}

export function siteInfo(geo: BusGeometry, data: BusViewData, id: number): string[] {
  const s = geo.sites[id];
  if (!s) return [];
  const b = data.state.buildings.find((x) => x.site === id);
  return [
    `Site ${id} · tier ${s.tier}`,
    b ? `${b.type}${b.occupied ? " (occupied)" : " (empty)"}` : "No building",
  ];
}

export function streetInfo(data: BusViewData, id: number): string[] {
  const users = data.derived.street_lines[String(id)] ?? [];
  return [`Street ${id}`, users.length ? `Lines: ${seats(users)}` : "No line"];
}

export function markerInfo(action: string, letter: string, seat: number): string[] {
  return [`P${seat}'s marker`, `${ACTION_TITLE[action] ?? action}, space ${letter}`];
}

export function scoreInfo(seat: number, points: number, stones: number): string[] {
  return [`P${seat}: ${points} point${points === 1 ? "" : "s"}`, `Time stones held: ${stones}`];
}

export function busInfo(seat: number, buses: number): string[] {
  return [`P${seat}: ${buses} bus${buses === 1 ? "" : "es"} (starting bus not shown)`];
}
