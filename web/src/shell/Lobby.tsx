import { useState, type CSSProperties, type FormEvent } from "react";
import { api } from "../api/client";
import { getPlugin } from "../games/registry";
import { SEAT_PRESETS } from "./seatTypes";
import { errorText, type SeatType } from "./useGame";

interface Props {
  onCreated: (gameId: string) => void;
}

export default function Lobby({ onCreated }: Props) {
  const [players, setPlayers] = useState(4);
  const [seats, setSeats] = useState<SeatType[]>(["human", "random", "random", "random", "random"]);
  const [returns, setReturns] = useState<"win_loss" | "margin">("win_loss");
  const [seed, setSeed] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const seatStyle = getPlugin("bus").seatStyle;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const trimmed = seed.trim();
      const { data, error: err } = await api.POST("/api/games", {
        body: {
          game: "bus",
          params: { num_players: players, returns },
          seats: seats.slice(0, players),
          ...(trimmed !== "" ? { seed: Number(trimmed) } : {}),
        },
      });
      if (data) onCreated(data.id);
      else setError(errorText(err));
    } catch (ex) {
      setError(ex instanceof Error ? ex.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  const seedBad = seed.trim() !== "" && !Number.isInteger(Number(seed.trim()));
  const humans = seats.slice(0, players).filter((s) => s === "human").length;

  return (
    <form className="card-framed lobby-form" onSubmit={(e) => void submit(e)} aria-busy={busy}>
      <h2 className="section-title">
        <span className="num-tile is-red">1</span> New game
      </h2>
      <div className="lobby-row">
        <label className="field">
          <span className="field-label">Players</span>{" "}
          <select value={players} onChange={(e) => setPlayers(Number(e.target.value))}>
            <option value={3}>3</option>
            <option value={4}>4</option>
            <option value={5}>5</option>
          </select>
        </label>
        <label className="field">
          <span className="field-label">Returns</span>{" "}
          <select
            value={returns}
            onChange={(e) => setReturns(e.target.value as "win_loss" | "margin")}
          >
            <option value="win_loss">Win / loss</option>
            <option value="margin">Margin</option>
          </select>
        </label>
      </div>
      <fieldset className="lobby-seats">
        <legend className="field-label">Seats</legend>
        {seats.slice(0, players).map((s, i) => {
          const st = seatStyle?.(i);
          return (
            <div
              key={i}
              className="lobby-seat"
              data-seat={i}
              style={st ? ({ "--seat": st.fill, "--seat-ink": st.ink } as CSSProperties) : undefined}
            >
              <label>
                <span className="lobby-seat-word">Seat</span>{" "}
                <span className="lobby-seat-tile">P{i}</span>{" "}
                <select
                  value={s}
                  onChange={(e) =>
                    setSeats(seats.map((x, j) => (j === i ? (e.target.value as SeatType) : x)))
                  }
                >
                  {SEAT_PRESETS.map((p) => (
                    <option key={p.value} value={p.value}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          );
        })}
        <p className="lobby-hint">
          {humans > 1
            ? `Hotseat: ${humans} people share this screen; the board is covered between their turns.`
            : humans === 1
              ? "One person against bots."
              : "Bots only: the game plays itself to the end."}
        </p>
      </fieldset>
      <label className="field lobby-seed">
        <span className="field-label">Seed (optional)</span>{" "}
        <input
          value={seed}
          onChange={(e) => setSeed(e.target.value)}
          inputMode="numeric"
          placeholder="random"
          aria-invalid={seedBad || undefined}
        />
        <small> The seed only affects random bots; leave it empty for a random game.</small>
      </label>
      <div className="lobby-submit">
        <button type="submit" className="btn-primary btn-lg" disabled={busy || seedBad}>
          Create game
        </button>
        {busy && <span className="spinner" aria-hidden="true" />}
      </div>
      {error && (
        <p role="alert" className="alert">
          Error: {error}
        </p>
      )}
    </form>
  );
}
