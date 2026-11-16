import type { SeatType } from "./useGame";

/** Seat-type choices offered in the lobby (values are validated server-side by pattern). */
export const SEAT_PRESETS: { value: SeatType; label: string }[] = [
  { value: "human", label: "Human" },
  { value: "random", label: "Random bot" },
  { value: "heuristic", label: "Heuristic bot" },
  { value: "mcts:50", label: "MCTS · Quick (50 sims)" },
  { value: "mcts:200", label: "MCTS · Normal (200 sims)" },
  { value: "mcts:1000", label: "MCTS · Strong (1000 sims)" },
];

/** Think-time cap (seconds per move) for MCTS seats; mirrors the server's 0.1..60 range. */
export const DEFAULT_THINK_TIME_S = 5;
export const MIN_THINK_TIME_S = 0.1;
export const MAX_THINK_TIME_S = 60;

export const isMctsSeat = (t: string): boolean => t === "mcts" || t.startsWith("mcts:");

/** Display name for a seat type, e.g. "Heuristic", "MCTS Medium", "MCTS (50 sims)". */
export function seatTypeName(t: string): string {
  if (t === "human") return "Human";
  if (t === "mcts-fast") return "MCTS (fast)";
  const m = /^mcts(?::(\d+)(?:-(\w+))?)?$/.exec(t);
  if (m) return m[1] ? `MCTS (${m[1]} sims${m[2] ? `, ${m[2]} rollouts` : ""})` : "MCTS";
  if (t.startsWith("trained:")) return `Trained (${t.slice(8)})`;
  return t.charAt(0).toUpperCase() + t.slice(1);
}

export function thinkingText(seatType: string, seat: number): string {
  return `${seatTypeName(seatType)} (P${seat}) is thinking…`;
}
