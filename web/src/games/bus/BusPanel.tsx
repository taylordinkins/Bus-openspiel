import type { BoardUi } from "../registry";
import type { GameView } from "../../shell/useGame";
import BusBoard from "./BusBoard";
import EndResults from "./EndResults";
import { useBusGeometry } from "./geometry";
import { busInner, parseBusView, type BusViewData } from "./model";
import PromptBar from "./PromptBar";
import SeatPanels from "./SeatPanels";
import TurnFrame from "./TurnFrame";
import "./bus.css";

/** Fetches the geometry once, only for views that carry board data. */
function LoadedBoard({ data }: { data: BusViewData }) {
  const geo = useBusGeometry();
  return geo ? <BusBoard geo={geo} data={data} /> : <p>Loading board…</p>;
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
  return parts.length > 0 ? <p data-testid="bus-prompt">{parts.join(" · ")}</p> : null;
}

/** Prompt bar, the Bus board (SVG over the map art) in the hotseat turn frame, and the raw view in a details box. */
export default function BusPanel({ game }: { game: GameView; ui?: BoardUi }) {
  const data = parseBusView(busInner(game));
  return (
    <section aria-label="Bus state" className="bus-main">
      {data ? (
        <PromptBar data={data} game={game} />
      ) : (
        <PlainPrompt prompt={(busInner(game).prompt ?? {}) as RawPrompt} />
      )}
      <TurnFrame game={game} actor={data?.prompt?.actor ?? null}>
        {data && <LoadedBoard data={data} />}
      </TurnFrame>
      <details>
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
  return (
    <>
      <SeatPanels data={data} game={game} />
      <EndResults data={data} />
    </>
  );
}
