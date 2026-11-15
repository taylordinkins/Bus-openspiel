import type { GameView } from "../../shell/useGame";
import BusBoard from "./BusBoard";
import { useBusGeometry } from "./geometry";
import { parseBusView, type BusViewData } from "./model";

/** Fetches the geometry once, only for views that carry board data. */
function LoadedBoard({ data }: { data: BusViewData }) {
  const geo = useBusGeometry();
  return geo ? <BusBoard geo={geo} data={data} /> : <p>Loading board…</p>;
}

interface Prompt {
  phase?: string;
  actor?: number | null;
  units_left?: number | null;
}

/** The Bus board (SVG over the map art), a one-line prompt summary and the raw view in a details box. */
export default function BusPanel({ game }: { game: GameView }) {
  // the server nests the adapter's four keys one level down (GameView.view.view); older
  // fixtures put them directly in GameView.view
  const inner = (game.view.view ?? game.view) as Record<string, unknown>;
  const prompt = (inner.prompt ?? {}) as Prompt;
  const data = parseBusView(inner);
  const parts: string[] = [];
  if (prompt.phase) parts.push(`Phase ${prompt.phase}`);
  if (typeof prompt.actor === "number") parts.push(`P${prompt.actor} to act`);
  if (typeof prompt.units_left === "number") parts.push(`${prompt.units_left} left`);
  return (
    <section aria-label="Bus state">
      {parts.length > 0 && <p data-testid="bus-prompt">{parts.join(" · ")}</p>}
      {data && <LoadedBoard data={data} />}
      <details>
        <summary>Raw view</summary>
        <pre aria-label="Game state">{JSON.stringify(game.view, null, 2)}</pre>
      </details>
    </section>
  );
}
