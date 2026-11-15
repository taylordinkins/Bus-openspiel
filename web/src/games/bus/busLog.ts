import type { components } from "../../api/schema";
import type { BusHighlight } from "./hoverStore";
import { ACTION_TITLE, BUILDING_NAME, END_TYPE_TEXT } from "./promptText";

type HistoryEntry = components["schemas"]["HistoryEntry"];
type Event = components["schemas"]["Event"];

export type LogRow =
  | { kind: "heading"; key: string; text: string }
  | { kind: "action"; key: string; seat: number | null; text: string; hl: Partial<BusHighlight> }
  | { kind: "event"; key: string; text: string; hl: Partial<BusHighlight> };

const LETTERS = "ABCDEF";
const n = (v: unknown): number | null => (typeof v === "number" ? v : null);
const ids = (v: unknown, key: string): number[] =>
  Array.isArray(v)
    ? v.flatMap((x) => {
        const id = x && typeof x === "object" ? n((x as Record<string, unknown>)[key]) : null;
        return id === null ? [] : [id];
      })
    : [];

/** One event as readable text plus the board things it names (UI_DESIGN 6.6). */
export function describeEvent(e: Event): { text: string; hl: Partial<BusHighlight> } {
  const seat = n(e.seat);
  switch (e.type) {
    case "scored":
      return {
        text: `P${seat} delivers a passenger: ${n(e.points)} pts`,
        hl: {
          seats: seat === null ? [] : [seat],
          nodes: n(e.node) === null ? [] : [n(e.node) as number],
          sites: n(e.site) === null ? [] : [n(e.site) as number],
        },
      };
    case "clock_advanced": {
      // the automatic Clock step of a round with no Clock marker (D22, R-050)
      const name = (v: unknown) => (typeof v === "string" ? (BUILDING_NAME[v] ?? v) : "?");
      return { text: `Clock advances ${name(e.from)} → ${name(e.to)} (no Clock marker)`, hl: {} };
    }
    case "passengers_moved": {
      const out = ids(e.exited, "site");
      const inn = ids(e.entered, "site");
      return {
        text: `Passengers move: ${out.length} leave buildings, ${inn.length} enter`,
        hl: { sites: [...out, ...inn], nodes: [...ids(e.exited, "node"), ...ids(e.entered, "node")] },
      };
    }
    case "selection_ended":
      return { text: "Selection ends; actions resolve in order", hl: {} };
    case "bus_added":
      return { text: `P${seat} adds a bus (now ${n(e.buses)})`, hl: { seats: seat === null ? [] : [seat] } };
    case "marker_skipped": {
      const a = typeof e.action === "string" ? (ACTION_TITLE[e.action] ?? e.action) : "?";
      const l = n(e.letter);
      const owner = n(e.owner);
      return {
        text: `P${owner}'s ${a} marker ${l === null ? "" : (LETTERS[l] ?? l)} resolves with nothing to do`,
        hl: { seats: owner === null ? [] : [owner] },
      };
    }
    case "round_ended": {
      const sp = n(e.start_player);
      return {
        text: `Round ${n(e.round)} ends${sp === null ? "" : `; P${sp} starts next`}`,
        hl: { seats: sp === null ? [] : [sp] },
      };
    }
    case "game_ended": {
      const types = Array.isArray(e.end_types) ? (e.end_types as string[]) : [];
      return { text: `Game ends: ${types.map((t) => END_TYPE_TEXT[t] ?? t).join("; ")}`, hl: {} };
    }
    default:
      return { text: `Unhandled event "${e.type}": ${JSON.stringify(e)}`, hl: {} };
  }
}

/** Flatten history into headings (setup, rounds), decisions and indented automatic steps. */
export function logRows(history: HistoryEntry[]): LogRow[] {
  const rows: LogRow[] = [{ kind: "heading", key: "h-setup", text: "Setup" }];
  for (const h of history) {
    rows.push({
      kind: "action",
      key: `a${h.index}`,
      seat: h.actor,
      // the server's labels already start with the seat ("P3: select Clock")
      text: h.actor === null || h.label.startsWith(`P${h.actor}`) ? h.label : `P${h.actor}: ${h.label}`,
      hl: h.actor === null ? {} : { seats: [h.actor] },
    });
    (h.events ?? []).forEach((e, k) => {
      if (e.type === "round_started") {
        rows.push({ kind: "heading", key: `h${h.index}-${k}`, text: `Round ${n(e.round)}` });
        return;
      }
      const d = describeEvent(e);
      rows.push({ kind: "event", key: `e${h.index}-${k}`, ...d });
    });
  }
  return rows;
}
