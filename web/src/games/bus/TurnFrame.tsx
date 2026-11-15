import { useEffect, useRef, useState, type ReactNode } from "react";
import type { GameView } from "../../shell/useGame";
import { seatFill } from "./model";
import SeatBadge from "./SeatBadge";

/**
 * Hotseat turn frame (UI_DESIGN 2): with more than one human seat the board gets a frame in the
 * acting seat's colour, and a "P3's turn" toast shows when the acting human seat changes.
 */
export default function TurnFrame({
  game,
  actor,
  children,
}: {
  game: GameView;
  actor: number | null;
  children: ReactNode;
}) {
  const hotseat = game.seats.filter((s) => s === "human").length > 1;
  const humanActor = actor !== null && game.seats[actor] === "human" ? actor : null;
  const prev = useRef<number | null>(null);
  const [toast, setToast] = useState<number | null>(null);
  useEffect(() => {
    if (!hotseat || humanActor === null) return;
    if (prev.current !== humanActor) {
      const first = prev.current === null;
      prev.current = humanActor;
      if (!first) {
        setToast(humanActor);
        const t = setTimeout(() => setToast(null), 2500);
        return () => clearTimeout(t);
      }
    }
  }, [hotseat, humanActor]);
  if (!hotseat) return <>{children}</>;
  return (
    <div
      className="bus-turn-frame"
      data-testid="turn-frame"
      data-seat={humanActor ?? ""}
      style={{ borderColor: humanActor !== null ? seatFill(humanActor) : "#1A1A1A" }}
    >
      {toast !== null && (
        <div className="bus-toast" role="status">
          <SeatBadge seat={toast} />
          <span>&rsquo;s turn</span>
        </div>
      )}
      {children}
    </div>
  );
}
