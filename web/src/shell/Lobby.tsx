import { useEffect, useRef, useState, type ChangeEvent, type CSSProperties, type FormEvent } from "react";
import { api } from "../api/client";
import type { components } from "../api/schema";
import { getPlugin } from "../games/registry";
import { IconUpload } from "../theme/icons";
import {
  DEFAULT_THINK_TIME_S,
  MAX_THINK_TIME_S,
  MIN_THINK_TIME_S,
  SEAT_PRESETS,
  isMctsSeat,
} from "./seatTypes";
import { parseStartDoc, type StartDoc } from "./startDoc";
import { errorText, type SeatType } from "./useGame";

type ScenarioSummary = components["schemas"]["ScenarioSummary"];
type GameExport = components["schemas"]["GameExport"];
type BotModel = components["schemas"]["BotModel"];

interface Props {
  onCreated: (gameId: string) => void;
}

type Mode = "new" | "scenario";

const RETURNS_LABEL = { win_loss: "Win / loss", margin: "Margin" } as const;

function docSummary(p: StartDoc): string {
  if (p.kind === "scenario") {
    const bits = [
      "Scenario",
      p.players !== null ? `${p.players} players` : "player count not 3–5",
      RETURNS_LABEL[p.returns],
      ...(p.phase ? [`phase ${p.phase}`] : []),
    ];
    return bits.join(" · ") + (p.description ? ` — ${p.description}` : "");
  }
  if (p.kind === "replay") {
    return [
      p.fromScenario ? "Replay from a scenario" : "Replay",
      p.players !== null ? `${p.players} players` : "player count not 3–5",
      RETURNS_LABEL[p.returns],
      `${p.actions} actions`,
    ].join(" · ");
  }
  if (p.kind === "error") return p.message;
  return "Paste a scenario or an exported replay, upload a .json file, or pick a named scenario.";
}

export default function Lobby({ onCreated }: Props) {
  const [mode, setMode] = useState<Mode>("new");
  const [players, setPlayers] = useState(4);
  const [seats, setSeats] = useState<SeatType[]>(["human", "random", "random", "random", "random"]);
  const [returns, setReturns] = useState<"win_loss" | "margin">("win_loss");
  const [seed, setSeed] = useState("");
  const [thinkTime, setThinkTime] = useState(String(DEFAULT_THINK_TIME_S));
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [text, setTextRaw] = useState("");
  const [named, setNamed] = useState<ScenarioSummary[] | null>(null);
  const [namedId, setNamedId] = useState("");
  const [models, setModels] = useState<BotModel[]>([]);
  const replaySeats = useRef("");
  const alertRef = useRef<HTMLParagraphElement>(null);
  // The form can run past a short screen in Scenario mode: bring a new error into view.
  useEffect(() => {
    if (error) alertRef.current?.scrollIntoView?.({ block: "nearest" });
  }, [error]);
  // Trained nets offered by the server (models/manifest.yaml); none if the call fails.
  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const res = await api.GET("/api/bot-models");
        if (live && Array.isArray(res?.data)) setModels(res.data);
      } catch {
        /* no trained seats offered */
      }
    })();
    return () => {
      live = false;
    };
  }, []);
  const seatStyle = getPlugin("bus").seatStyle;

  const parsed = mode === "scenario" ? parseStartDoc(text) : null;
  const docPlayers =
    parsed && (parsed.kind === "scenario" || parsed.kind === "replay") ? parsed.players : null;
  const n = mode === "scenario" ? (docPlayers ?? 4) : players;
  const netPresets = models
    .filter((m) => m.players.includes(n))
    .map((m) => ({ value: m.seat_type as SeatType, label: m.label }));
  const seatOptions = [...SEAT_PRESETS, ...netPresets];

  /** Set the document text; a newly loaded replay brings its own seat types. */
  function setText(t: string) {
    setTextRaw(t);
    const p = parseStartDoc(t);
    const key = p.kind === "replay" ? JSON.stringify(p.seats) : "";
    if (p.kind === "replay" && key !== replaySeats.current) {
      setSeats((old) => old.map((s, i) => (p.seats[i] as SeatType | undefined) ?? s));
    }
    replaySeats.current = key;
  }

  async function loadNamed() {
    if (named !== null) return;
    try {
      const res = await api.GET("/api/scenarios", { params: { query: { game: "bus" } } });
      if (res?.data) setNamed(res.data);
      else setError(`Could not list scenarios: ${errorText(res?.error)}`);
    } catch (ex) {
      setError(`Could not list scenarios: ${ex instanceof Error ? ex.message : "Network error"}`);
    }
  }

  function chooseMode(m: Mode) {
    setMode(m);
    setError(null);
    if (m === "scenario") void loadNamed();
  }

  async function pickNamed(id: string) {
    setNamedId(id);
    setError(null);
    if (id === "") return;
    try {
      const res = await api.GET("/api/scenarios/{scenario_id}", {
        params: { path: { scenario_id: id }, query: { game: "bus" } },
      });
      if (res?.data) setText(JSON.stringify(res.data.scenario, null, 2));
      else setError(errorText(res?.error));
    } catch (ex) {
      setError(ex instanceof Error ? ex.message : "Network error");
    }
  }

  async function upload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setError(null);
    setNamedId("");
    try {
      setText(await file.text());
    } catch (ex) {
      setError(`Could not read ${file.name}: ${ex instanceof Error ? ex.message : "read failed"}`);
    }
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    if (parsed && (parsed.kind === "empty" || parsed.kind === "error")) {
      setError(
        parsed.kind === "empty"
          ? "No scenario yet: paste one, upload a file or pick a named scenario."
          : parsed.message,
      );
      return;
    }
    setBusy(true);
    try {
      const trimmed = seed.trim();
      const seedPart = trimmed !== "" ? { seed: Number(trimmed) } : {};
      const chosen = seats.slice(0, n);
      const thinkPart = chosen.some(isMctsSeat) ? { bot_think_time_s: Number(thinkTime) } : {};
      const res =
        parsed?.kind === "replay"
          ? await api.POST("/api/games/import", {
              body: { ...(parsed.doc as unknown as GameExport), seats: chosen, ...thinkPart },
            })
          : await api.POST("/api/games", {
              body:
                parsed?.kind === "scenario"
                  ? { game: "bus", seats: chosen, scenario: parsed.doc, ...seedPart, ...thinkPart }
                  : { game: "bus", params: { num_players: players, returns }, seats: chosen, ...seedPart, ...thinkPart },
            });
      if (res.data) onCreated(res.data.id);
      else setError(errorText(res.error));
    } catch (ex) {
      setError(ex instanceof Error ? ex.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  const seedBad = seed.trim() !== "" && !Number.isInteger(Number(seed.trim()));
  const thinkNum = Number(thinkTime);
  const thinkBad =
    thinkTime.trim() === "" || !(thinkNum >= MIN_THINK_TIME_S && thinkNum <= MAX_THINK_TIME_S);
  const hasMcts = seats.slice(0, n).some(isMctsSeat);
  const humans = seats.slice(0, n).filter((s) => s === "human").length;
  const isReplay = parsed?.kind === "replay";

  return (
    <form className="card-framed lobby-form" data-mode={mode} onSubmit={(e) => void submit(e)} aria-busy={busy}>
      <h2 className="section-title">
        <span className="num-tile is-red">1</span> New game
      </h2>
      <div role="radiogroup" aria-label="Start from" className="lg-seg lobby-mode">
        {(
          [
            ["new", "Fresh setup"],
            ["scenario", "Scenario"],
          ] as const
        ).map(([m, label]) => (
          <label key={m} className="lg-seg-opt" data-checked={mode === m}>
            <input
              type="radio"
              name="lobby-mode"
              value={m}
              checked={mode === m}
              onChange={() => chooseMode(m)}
            />
            <span>{label}</span>
          </label>
        ))}
      </div>
      {mode === "new" ? (
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
      ) : (
        <div className="lobby-scenario">
          <div className="lobby-scenario-pick">
            <label className="field">
              <span className="field-label">Named scenario</span>{" "}
              <select
                value={namedId}
                onChange={(e) => void pickNamed(e.target.value)}
                disabled={named === null}
              >
                <option value="">{named === null ? "Loading…" : "Choose one…"}</option>
                {named?.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.title} · {s.players}p
                  </option>
                ))}
              </select>
            </label>
            <label className="btn import-btn lobby-upload" title="Upload a scenario or replay (.json)">
              <IconUpload />
              <span>Upload .json</span>
              <input
                className="sr-only"
                type="file"
                accept="application/json,.json"
                aria-label="Upload scenario file"
                onChange={(e) => void upload(e)}
              />
            </label>
          </div>
          <label className="field">
            <span className="field-label">Scenario JSON</span>{" "}
            <textarea
              className="lobby-json"
              rows={4}
              spellCheck={false}
              value={text}
              placeholder='{"format": 1, …}  or an exported replay'
              aria-invalid={parsed?.kind === "error" || undefined}
              aria-describedby="lobby-doc-summary"
              onChange={(e) => {
                setNamedId("");
                setText(e.target.value);
              }}
            />
          </label>
          <p id="lobby-doc-summary" className="lobby-hint lobby-doc-summary" role="status" data-kind={parsed?.kind}>
            {parsed && docSummary(parsed)}
          </p>
        </div>
      )}
      <fieldset className="lobby-seats">
        <legend className="field-label">Seats</legend>
        {seats.slice(0, n).map((s, i) => {
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
                  {seatOptions.some((p) => p.value === s) ? null : <option value={s}>{s}</option>}
                  {seatOptions.map((p) => (
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
        {hasMcts && (
          <label className="field lobby-think">
            <span className="field-label">Think time (seconds per move)</span>{" "}
            <input
              type="number"
              min={MIN_THINK_TIME_S}
              max={MAX_THINK_TIME_S}
              step="any"
              value={thinkTime}
              onChange={(e) => setThinkTime(e.target.value)}
              aria-invalid={thinkBad || undefined}
            />
            <small> MCTS bots stop searching at this limit, even before their simulation count.</small>
          </label>
        )}
      </fieldset>
      {!isReplay && (
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
      )}
      <div className="lobby-submit">
        <button type="submit" className="btn-primary btn-lg" disabled={busy || (seedBad && !isReplay) || (hasMcts && thinkBad)}>
          Create game
        </button>
        {busy && <span className="spinner" aria-hidden="true" />}
      </div>
      {error && (
        <p role="alert" className="alert" ref={alertRef}>
          Error: {error}
        </p>
      )}
    </form>
  );
}
