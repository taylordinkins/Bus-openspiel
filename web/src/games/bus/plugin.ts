import { createElement } from "react";
import type { GamePlugin } from "../registry";
import TextBoard from "./TextBoard";

/**
 * Bus on the action-list fallback (Phase 2 unit 5): the shell's action list does the playing and
 * the "board" is the adapter view as text. The SVG board replaces it in unit 6.
 */
export const busPlugin: GamePlugin = {
  id: "bus",
  title: "Bus",
  phaseLabel: (g) => {
    const prompt = g.view.prompt as { phase?: unknown } | undefined;
    return typeof prompt?.phase === "string" ? prompt.phase : undefined;
  },
  renderBoard: (g) => createElement(TextBoard, { game: g }),
};
