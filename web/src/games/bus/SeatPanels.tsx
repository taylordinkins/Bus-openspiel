import type { GameView } from "../../shell/useGame";
import { seatTypeName } from "../../shell/seatTypes";
import { highlightProps } from "./hoverStore";
import { SEAT_NAME, at, seatFill, type BusViewData } from "./model";
import SeatBadge from "./SeatBadge";

const BUSES = 5;

function BusIcons({ owned }: { owned: number }) {
  return (
    <span className="bus-icons" aria-label={`${owned} of ${BUSES} buses`} role="img">
      {Array.from({ length: BUSES }, (_, i) => (
        <svg key={i} width="18" height="12" viewBox="0 0 18 12" data-owned={i < owned}>
          <rect
            x="1"
            y="1"
            width="16"
            height="9"
            rx="2"
            fill={i < owned ? "#1A1A1A" : "none"}
            stroke="#1A1A1A"
            strokeWidth="1.5"
          />
          <circle cx="5" cy="10.5" r="1.4" fill="#1A1A1A" />
          <circle cx="13" cy="10.5" r="1.4" fill="#1A1A1A" />
        </svg>
      ))}
    </span>
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
            style={{ borderLeftColor: seatFill(i) }}
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
                {seat.points} pts
              </span>
            </header>
            <div className="bus-panel-row">
              <BusIcons owned={seat.buses} />
              {seat.stones > 0 && <span>{seat.stones} {seat.stones === 1 ? "stone" : "stones"}</span>}
            </div>
            <div className="bus-panel-row bus-small">
              <span>Action markers: {seat.markers_left}</span>
              <span>Line markers: {lineLeft}</span>
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
