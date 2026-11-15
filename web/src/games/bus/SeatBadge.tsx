import { INK, SEAT_NAME, at, seatFill, seatInk } from "./model";

/** "P0" in the seat colour; the label is always text so identity never rests on colour (UI_DESIGN 8). */
export default function SeatBadge({ seat }: { seat: number }) {
  return (
    <span
      className="bus-seat-badge"
      data-seat={seat}
      title={`P${seat} (${at(SEAT_NAME, seat % 5)})`}
      style={{
        background: seatFill(seat),
        color: seatInk(seat),
        border: `2px solid ${INK}`,
      }}
    >
      P{seat}
    </span>
  );
}
