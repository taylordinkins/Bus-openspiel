import { useEffect, useMemo, useState } from "react";
import type { BoardMotion, BoardUi } from "../registry";
import type { GameView } from "../../shell/useGame";
import BusBoard from "./BusBoard";
import EndResults from "./EndResults";
import { useBusGeometry } from "./geometry";
import type { BoardAct } from "./interaction";
import { busInner, parseBusView, type BusViewData } from "./model";
import PromptBar from "./PromptBar";
import SeatPanels from "./SeatPanels";
import TurnFrame from "./TurnFrame";
import "./bus.css";

/** Fetches the geometry once, only for views that carry board data. */
function LoadedBoard({ data, act, motion }: { data: BusViewData; act: BoardAct | null; motion: BoardMotion | null }) {
  const geo = useBusGeometry();
  return geo ? (
    <BusBoard geo={geo} data={data} act={act} motion={motion} />
  ) : (
    <div className="bus-board-loading">
      <span className="spinner" aria-hidden="true" />
      <p>Loading board…</p>
    </div>
  );
}

/** R-053 confirm: taking the last stone ends the game, so it asks first. */
function LastStoneConfirm({ onConfirm, onCancel }: { onConfirm: () => void; onCancel: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onCancel]);
  return (
    <div className="dlg-backdrop">
      <div role="alertdialog" aria-modal="true" aria-labelledby="bus-confirm-text" className="dlg bus-confirm">
        <p className="eyebrow">Clock · last time stone</p>
        <p id="bus-confirm-text" className="bus-confirm-text">
          Taking the last stone ends the game now.
        </p>
        <p className="bus-confirm-sub">Final scores are counted at once: points minus time stones.</p>
        <div className="dlg-actions">
          <button type="button" autoFocus onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="btn-primary" onClick={onConfirm}>
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}

interface RawPrompt {
  phase?: string;
  actor?: number | null;
  units_left?: number | null;
}

/** One-line summary for views without board data (hand-made fixtures). */
function PlainPrompt({ prompt }: { prompt: RawPrompt }) {
  const parts: string[] = [];
  if (prompt.phase) parts.push(`Phase ${prompt.phase}`);
  if (typeof prompt.actor === "number") parts.push(`P${prompt.actor} to act`);
  if (typeof prompt.units_left === "number") parts.push(`${prompt.units_left} left`);
  return parts.length > 0 ? (
    <p data-testid="bus-prompt" className="bus-prompt bus-prompt-plain">
      {parts.join(" · ")}
    </p>
  ) : null;
}

/** Prompt bar, the Bus board (SVG over the map art) in the hotseat turn frame, and the raw view in a details box. */
export default function BusPanel({ game, ui }: { game: GameView; ui?: BoardUi }) {
  const data = parseBusView(busInner(game));
  const [confirm, setConfirm] = useState(false);
  const sig = JSON.stringify([data?.prompt?.phase, data?.targets?.clock]);
  useEffect(() => setConfirm(false), [sig]);
  const onAct = ui?.canAct ? ui.onAct : undefined;
  const lastStone = data?.prompt?.context?.last_stone === true;
  const pauseId = data?.targets?.clock?.pause ?? null;
  const act = useMemo<BoardAct | null>(() => {
    if (!onAct || !data) return null;
    const post = (id: number) => void onAct(id);
    return {
      post,
      pause: () => {
        if (pauseId === null) return;
        if (lastStone) setConfirm(true);
        else post(pauseId);
      },
      legal: new Map(game.legal_actions.map((a) => [a.id, a])),
    };
  }, [onAct, data, game.legal_actions, lastStone, pauseId]);
  return (
    <section aria-label="Bus state" className="bus-main">
      {data ? (
        <PromptBar data={data} game={game} act={act} ui={ui} />
      ) : (
        <PlainPrompt prompt={(busInner(game).prompt ?? {}) as RawPrompt} />
      )}
      <TurnFrame game={game} actor={data?.prompt?.actor ?? null}>
        {data && <LoadedBoard data={data} act={act} motion={ui?.motion ?? null} />}
      </TurnFrame>
      {confirm && pauseId !== null && (
        <LastStoneConfirm
          onCancel={() => setConfirm(false)}
          onConfirm={() => {
            setConfirm(false);
            act?.post(pauseId);
          }}
        />
      )}
      <details className="bus-raw">
        <summary>Raw view</summary>
        <pre aria-label="Game state">{JSON.stringify(game.view, null, 2)}</pre>
      </details>
    </section>
  );
}

/** Seat panels and, at END, the results table. */
export function BusSide({ game }: { game: GameView }) {
  const data = parseBusView(busInner(game));
  if (!data) return null;
  // at END the results lead; the seat panels follow
  return (
    <>
      <EndResults data={data} />
      <SeatPanels data={data} game={game} />
    </>
  );
}
