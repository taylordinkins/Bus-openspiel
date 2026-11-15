import { createElement } from "react";
import type { GamePlugin } from "../registry";
import BusLog from "./BusLog";
import BusPanel, { BusSide } from "./BusPanel";
import { busInner, seatFill, seatInk } from "./model";

/**
 * Bus: the shell's action list still does the playing (unit 8 adds board clicks); the board is the
 * SVG over the map art (unit 6); prompt bar, seat panels, log and results are unit 7.
 */
export const busPlugin: GamePlugin = {
  id: "bus",
  title: "Bus",
  boardDriven: true,
  phaseLabel: (g) => {
    const inner = busInner(g);
    const prompt = inner.prompt as { phase?: unknown } | undefined;
    return typeof prompt?.phase === "string" ? prompt.phase : undefined;
  },
  seatStyle: (seat) => ({ fill: seatFill(seat), ink: seatInk(seat) }),
  renderBoard: (g, ui) => createElement(BusPanel, { game: g, ui }),
  renderSide: (g) => createElement(BusSide, { game: g }),
  renderLog: (g) => createElement(BusLog, { game: g }),
};
