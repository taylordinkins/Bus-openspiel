import { createElement } from "react";
import type { GamePlugin } from "../registry";
import BusPanel from "./BusPanel";
import { seatFill, seatInk } from "./model";

/**
 * Bus: the shell's action list still does the playing (unit 8 adds board clicks); the board is the
 * SVG over the map art (unit 6).
 */
export const busPlugin: GamePlugin = {
  id: "bus",
  title: "Bus",
  phaseLabel: (g) => {
    const inner = (g.view.view ?? g.view) as Record<string, unknown>;
    const prompt = inner.prompt as { phase?: unknown } | undefined;
    return typeof prompt?.phase === "string" ? prompt.phase : undefined;
  },
  seatStyle: (seat) => ({ fill: seatFill(seat), ink: seatInk(seat) }),
  renderBoard: (g) => createElement(BusPanel, { game: g }),
};
