import type { CSSProperties } from "react";
import { IconUndo } from "../../theme/icons";
import type { GameView } from "../../shell/useGame";
import { seatTypeName } from "../../shell/seatTypes";
import type { BoardUi } from "../registry";
import type { BoardAct } from "./interaction";
import type { BusViewData } from "./model";
import { seatFill, seatInk } from "./model";
import { BUILDING_NAME, promptParts } from "./promptText";
import SeatBadge from "./SeatBadge";

/** BGA-style status bar: who acts, the phase, what to do now and the units/power/context chips. */
export default function PromptBar({
  data,
  game,
  act = null,
  ui,
}: {
  data: BusViewData;
  game: GameView;
  /** Null when nothing is legal for the viewer (bot seat, request in flight): buttons are disabled. */
  act?: BoardAct | null;
  ui?: BoardUi;
}) {
  const parts = promptParts(data);
  if (!parts) return null;
  const { actor } = parts;
  const kind = actor !== null ? (game.seats[actor] ?? "human") : "human";
  const bot = actor !== null && kind !== "human";
  const over = parts.phase === "END";
  const ctx = data.prompt?.context ?? {};
  const t = data.targets;
  const dropOut = t?.buttons?.find((b) => b.kind === "drop_out");
  const advance = t?.clock?.advance ?? null;
  const pause = t?.clock?.pause ?? null;
  const next = typeof ctx.next_clock === "string" ? ctx.next_clock : null;
  const stones = typeof ctx.stones_on_clock === "number" ? ctx.stones_on_clock : null;
  return (
    <div
      className="bus-prompt"
      data-testid="bus-prompt"
      data-phase={parts.phase}
      data-actor={actor ?? ""}
      role="status"
      aria-live="polite"
      data-bot={bot || undefined}
      style={actor !== null ? ({ "--seat": seatFill(actor), "--seat-ink": seatInk(actor) } as CSSProperties) : undefined}
    >
      {actor !== null && <SeatBadge seat={actor} />}
      <span className="bus-prompt-text">
        {bot && (
          <span className="bus-prompt-wait">
            {game.bot_thinking && <span className="spinner" aria-hidden="true" />}
            Waiting for {seatTypeName(kind)} bot{game.bot_thinking ? " (thinking…)" : ""} ·{" "}
          </span>
        )}
        <strong>{parts.title}</strong>
        {over ? " · " : ": "}
        {parts.instruction}
      </span>
      {parts.details.length > 0 && (
        <span className="bus-chips">
          {parts.details.map((d) => (
            <span key={d} className="bus-chip">
              {d}
            </span>
          ))}
        </span>
      )}
      <span className="bus-prompt-actions">
        {dropOut && (
          <button type="button" className="btn-primary" disabled={!act} onClick={() => act?.post(dropOut.id)}>
            Drop out
          </button>
        )}
        {advance !== null && (
          <button type="button" className="btn-primary" disabled={!act} onClick={() => act?.post(advance)}>
            Advance{next ? ` → ${BUILDING_NAME[next] ?? next}` : ""}
          </button>
        )}
        {pause !== null && (
          <button type="button" className="btn-primary" disabled={!act} onClick={() => act?.pause()}>
            Take stone{stones !== null ? ` (${stones} left)` : ""}
          </button>
        )}
        {ui?.onBack && (
          <button type="button" className="bus-back" disabled={!ui.canBack} onClick={() => void ui.onBack?.()}>
            <IconUndo />
            Back
          </button>
        )}
      </span>
    </div>
  );
}
