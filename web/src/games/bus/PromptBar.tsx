import type { GameView } from "../../shell/useGame";
import { seatTypeName } from "../../shell/seatTypes";
import type { BusViewData } from "./model";
import { seatFill } from "./model";
import { promptParts } from "./promptText";
import SeatBadge from "./SeatBadge";

/** BGA-style status bar: who acts, the phase, what to do now and the units/power/context chips. */
export default function PromptBar({ data, game }: { data: BusViewData; game: GameView }) {
  const parts = promptParts(data);
  if (!parts) return null;
  const { actor } = parts;
  const kind = actor !== null ? (game.seats[actor] ?? "human") : "human";
  const bot = actor !== null && kind !== "human";
  const over = parts.phase === "END";
  return (
    <div
      className="bus-prompt"
      data-testid="bus-prompt"
      data-phase={parts.phase}
      data-actor={actor ?? ""}
      role="status"
      aria-live="polite"
      style={{ borderLeftColor: actor !== null ? seatFill(actor) : "#1A1A1A" }}
    >
      {actor !== null && <SeatBadge seat={actor} />}
      <span className="bus-prompt-text">
        {bot && <span className="bus-prompt-wait">Waiting for {seatTypeName(kind)} bot{game.bot_thinking ? " (thinking…)" : ""} · </span>}
        <strong>{parts.title}</strong>
        {over ? " · " : ": "}
        {parts.instruction}
      </span>
      {parts.details.map((d) => (
        <span key={d} className="bus-chip">
          {d}
        </span>
      ))}
    </div>
  );
}
