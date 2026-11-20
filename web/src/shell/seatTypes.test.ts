import { describe, expect, it } from "vitest";
import {
  SEAT_PRESETS,
  isMctsSeat,
  seatTypeName,
  thinkingText,
} from "./seatTypes";

describe("seat presets", () => {
  it("offers human, random, heuristic, MCTS strength and planner seats", () => {
    expect(SEAT_PRESETS.map((p) => p.value)).toEqual([
      "human",
      "random",
      "heuristic",
      "mcts:50",
      "mcts:200",
      "mcts:1000",
      "tplanner",
    ]);
  });
  it("names seat types", () => {
    expect(seatTypeName("random")).toBe("Random");
    expect(seatTypeName("human")).toBe("Human");
    expect(seatTypeName("mcts:50")).toBe("MCTS (50 sims)");
    expect(seatTypeName("mcts")).toBe("MCTS");
    expect(seatTypeName("heuristic")).toBe("Heuristic");
    expect(seatTypeName("net:ppo-smoke")).toBe("Trained net (ppo-smoke)");
    expect(seatTypeName("mcts:200-heuristic")).toBe(
      "MCTS (200 sims, heuristic rollouts)",
    );
  });
  it("builds the thinking text", () => {
    expect(thinkingText("random", 1)).toBe("Random (P1) is thinking…");
  });
});

describe("planner seats", () => {
  it("is named and counts as a search seat", () => {
    expect(seatTypeName("tplanner")).toBe("Planner");
    expect(seatTypeName("tplanner:192")).toBe("Planner (192 sims)");
    expect(isMctsSeat("tplanner")).toBe(true);
    expect(isMctsSeat("tplanner:192")).toBe(true);
  });
});

describe("net-guided MCTS seats", () => {
  it("is named and counts as a search seat", () => {
    expect(seatTypeName("mcts_net:50-ppo-smoke")).toBe(
      "Trained net + MCTS (ppo-smoke, 50 sims)",
    );
    expect(isMctsSeat("mcts_net:50-ppo-smoke")).toBe(true);
  });
});
