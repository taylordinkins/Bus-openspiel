import type { CSSProperties } from "react";
import type { GameView } from "../../shell/useGame";
import { seatTypeName } from "../../shell/seatTypes";
import { highlightProps } from "./hoverStore";
import { SEAT_NAME, at, seatFill, seatInk, type BusViewData } from "./model";
import SeatBadge from "./SeatBadge";

const BUSES = 5;

/** Front-view bus, after the map's Buses grid: filled when owned, a dashed outline when not. */
function BusIcons({ owned }: { owned: number }) {
  return (
    <span className="bus-icons" aria-label={`${owned} of ${BUSES} buses`} role="img">
      {Array.from({ length: BUSES }, (_, i) => (
        <svg key={i} width="17" height="18" viewBox="0 0 17 18" data-owned={i < owned} className="bus-icon">
          <rect className="bus-icon-body" x="1.5" y="1.5" width="14" height="12.5" rx="2.5" />
          <rect className="bus-icon-glass" x="3.5" y="3.5" width="10" height="5" rx="1" />
          <rect className="bus-icon-wheel" x="3" y="14" width="3" height="3" rx="1" />
          <rect className="bus-icon-wheel" x="11" y="14" width="3" height="3" rx="1" />
        </svg>
      ))}
    </span>
  );
}

/** A taken time stone: the map's wax-seal stone in miniature. */
function StoneIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" className="bus-stone-icon">
      <circle cx="8" cy="8" r="7" />
      <circle cx="8" cy="8" r="4" />
    </svg>
  );
}

/** One panel per seat (UI_DESIGN 2); hovering lights the seat's line on the board. */
export default function SeatPanels({ data, game }: { data: BusViewData; game: GameView }) {
  const st = data.state;
  const actor = data.prompt?.actor ?? null;
  return (
    <div className="bus-panels" aria-label="Players">
      {st.seats.map((seat, i) => {
        const d = data.derived.seats[i];
        const acting = actor === i;
        const dropped = d?.dropped === true;
        const kind = game.seats[i] ?? "human";
        const lineLeft = d?.line_markers_left ?? 25 - seat.line.length;
        return (
          <section
            key={i}
            className="bus-panel"
            aria-label={`P${i} panel`}
            data-seat={i}
            data-acting={acting}
            data-dropped={dropped}
            tabIndex={0}
            {...highlightProps({ seats: [i] })}
            style={{ "--seat": seatFill(i), "--seat-ink": seatInk(i) } as CSSProperties}
          >
            <header>
              <SeatBadge seat={i} />
              <span className="bus-panel-kind">
                {seatTypeName(kind)} · {at(SEAT_NAME, i % 5)}
              </span>
              {st.start_player === i && (
                <span className="bus-flag" title="Start player" aria-label="Start player">
                  ⚑ start
                </span>
              )}
              {acting && <span className="bus-acting">to act</span>}
              {dropped && <span className="bus-dropped">dropped out</span>}
              <span className="bus-score" aria-label="Score">
                <b>{seat.points}</b> pts
              </span>
            </header>
            <div className="bus-panel-row">
              <BusIcons owned={seat.buses} />
              {seat.stones > 0 && (
                <span className="bus-stones">
                  <StoneIcon />
                  {seat.stones} {seat.stones === 1 ? "stone" : "stones"}
                </span>
              )}
            </div>
            <div className="bus-panel-row bus-small">
              <span>
                Action markers: <b>{seat.markers_left}</b>
              </span>
              <span>
                Line markers: <b>{lineLeft}</b>
              </span>
            </div>
          </section>
        );
      })}
      <p className="bus-round" aria-label="Round info">
        Round {st.round}
        {st.round === 0 ? " (setup)" : ""}
        {data.prompt?.phase === "SEL" && typeof data.prompt.context?.pass === "number"
          ? ` · pass ${data.prompt.context.pass + 1}`
          : ""}{" "}
        · clock {st.clock} · {st.stones_on_clock} stones on clock · {st.supply} in supply
      </p>
    </div>
  );
}
