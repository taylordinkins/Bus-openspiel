import type { GameView } from "../../shell/useGame";
import { logRows, type LogRow } from "./busLog";
import { highlightProps } from "./hoverStore";

/** Decision log with automatic steps indented under it; hovering a line lights what it names. */
export default function BusLog({ game }: { game: GameView }) {
  const rows = logRows(game.history);
  // headings sit inside the list item of the row below them, so the list holds one item per line
  const items: { row: LogRow; headings: string[] }[] = [];
  let pending: string[] = [];
  for (const r of rows) {
    if (r.kind === "heading") pending.push(r.text);
    else {
      items.push({ row: r, headings: pending });
      pending = [];
    }
  }
  const trailing = pending;
  return (
    <ol
      aria-label="Action log"
      className="bus-log"
      ref={(el) => {
        if (el) el.scrollTop = el.scrollHeight;
      }}
    >
      {items.map(({ row: r, headings }) => (
        <li
          key={r.key}
          className={r.kind === "event" ? "bus-log-event" : "bus-log-action"}
          tabIndex={0}
          {...(r.kind === "heading" ? {} : highlightProps(r.hl))}
        >
          {headings.map((h) => (
            <h4 key={h}>{h}</h4>
          ))}
          {r.text}
        </li>
      ))}
      {trailing.length > 0 && (
        <li className="bus-log-action">
          {trailing.map((h) => (
            <h4 key={h}>{h}</h4>
          ))}
        </li>
      )}
    </ol>
  );
}
