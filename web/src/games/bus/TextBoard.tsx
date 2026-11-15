import type { GameView } from "../../shell/useGame";

interface Prompt {
  phase?: string;
  actor?: number | null;
  units_left?: number | null;
}

/** Readable stand-in for the board: a one-line prompt summary and the raw adapter view. */
export default function TextBoard({ game }: { game: GameView }) {
  const prompt = (game.view.prompt ?? {}) as Prompt;
  const parts: string[] = [];
  if (prompt.phase) parts.push(`Phase ${prompt.phase}`);
  if (typeof prompt.actor === "number") parts.push(`P${prompt.actor} to act`);
  if (typeof prompt.units_left === "number") parts.push(`${prompt.units_left} left`);
  return (
    <section aria-label="Bus state">
      {parts.length > 0 && <p data-testid="bus-prompt">{parts.join(" · ")}</p>}
      <pre aria-label="Game state">{JSON.stringify(game.view, null, 2)}</pre>
    </section>
  );
}
