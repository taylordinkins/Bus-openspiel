import { highlightProps } from "./hoverStore";
import type { BusViewData } from "./model";
import { END_TYPE_TEXT } from "./promptText";
import SeatBadge from "./SeatBadge";

export interface Ranked {
  seat: number;
  rank: number;
  points: number;
  stones: number;
  final: number;
  lastScore: number | null;
  winner: boolean;
}

/** Ranking by R-068..R-070: final score, then stones, then earlier last scoring move; equal keys share a rank. */
export function rankSeats(data: BusViewData): Ranked[] {
  const res = data.derived.result;
  const seats = data.state.seats;
  const rows = seats.map((s, seat) => ({
    seat,
    points: s.points,
    stones: s.stones,
    final: res?.final_scores[seat] ?? s.points - s.stones,
    lastScore: s.last_score_move ?? null,
  }));
  const last = (r: { lastScore: number | null }) => r.lastScore ?? Number.POSITIVE_INFINITY;
  const cmp = (a: (typeof rows)[number], b: (typeof rows)[number]) =>
    b.final - a.final || b.stones - a.stones || Math.sign(last(a) - last(b)) || 0;
  rows.sort((a, b) => cmp(a, b) || a.seat - b.seat);
  return rows.map((r, _i, all) => {
    const first = all.findIndex((o) => cmp(o, r) === 0);
    return { ...r, rank: first + 1, winner: res ? res.winners.includes(r.seat) : first === 0 };
  });
}

/** How the winner was decided, from the ranked rows (R-069/R-070). */
export function tiebreakText(rows: Ranked[]): string {
  const top = rows.filter((r) => r.final === (rows[0]?.final ?? 0));
  if (top.length < 2) return "Highest final score wins.";
  const names = top.map((r) => `P${r.seat}`).join(", ");
  const mostStones = Math.max(...top.map((r) => r.stones));
  const withStones = top.filter((r) => r.stones === mostStones);
  if (withStones.length === 1) return `${names} tied on final score ${rows[0]?.final}: most time stones wins.`;
  const key = (r: Ranked) => r.lastScore ?? Number.POSITIVE_INFINITY;
  const earliest = Math.min(...withStones.map(key));
  const first = withStones.filter((r) => key(r) === earliest);
  if (first.length === 1)
    return `${names} tied on final score ${rows[0]?.final}; stones also tied: reached that total first wins.`;
  return `${names} tied on final score, stones and scoring move: shared win.`;
}

/** Final results table (UI_DESIGN 4, END). */
export default function EndResults({ data }: { data: BusViewData }) {
  const res = data.derived.result;
  if (!res) return null;
  const rows = rankSeats(data);
  const winners = res.winners;
  return (
    <section className="bus-end" aria-label="Final results" data-testid="end-results">
      <h3>Final results</h3>
      <p className="bus-end-headline">
        {winners.length === 1 ? `P${winners[0]} wins` : `Shared win: ${winners.map((s) => `P${s}`).join(", ")}`}
      </p>
      <p>
        Game ended because{" "}
        {res.end_types.map((t) => END_TYPE_TEXT[t] ?? t).join(" and ") || "the game is over"}.
      </p>
      <table className="bus-end-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Seat</th>
            <th>Pts</th>
            <th>Stones</th>
            <th>Final</th>
            <th>Last score</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.seat} data-seat={r.seat} data-winner={r.winner} {...highlightProps({ seats: [r.seat] })}>
              <td>{r.rank}</td>
              <td>
                <SeatBadge seat={r.seat} />
                {r.winner && <strong> winner</strong>}
              </td>
              <td>{r.points}</td>
              <td>{r.stones}</td>
              <td>{r.final}</td>
              <td>{r.lastScore ?? "none"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="bus-small">Final = points minus time stones. {tiebreakText(rows)}</p>
    </section>
  );
}
