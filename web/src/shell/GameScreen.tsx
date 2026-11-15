import { useEffect, useState } from "react";
import "./layout.css";
import { api } from "../api/client";
import Curtain from "./Curtain";
import ActionLog from "./ActionLog";
import ImportButton from "./ImportButton";
import { getPlugin, type BoardUi } from "../games/registry";
import ReportDialog from "./ReportDialog";
import TabBar, { type ScreenTab } from "./TabBar";
import YourMove from "./YourMove";
import { thinkingText } from "./seatTypes";
import Tip from "./Tip";
import {
  IconArrowLeft,
  IconDownload,
  IconFlag,
  IconList,
  IconRewind,
  IconSkip,
  IconSliders,
  IconUndo,
} from "../theme/icons";
import { usePlayback } from "./playback";
import SettingsDialog from "./SettingsDialog";
import { SPEED_SCALE, useSettings } from "./settings";
import { deriveViewSeat, SeatViewContext } from "./viewSeat";
import {
  errorText,
  currentPlayer,
  DEFAULT_POLL,
  isTerminal,
  legalActions,
  useGame,
  type GameView,
  type PollIntervals,
} from "./useGame";

interface Props {
  gameId: string;
  pollMs?: PollIntervals;
  onExit?: () => void;
  /** Called with the new game id after a replay import. */
  onImported?: (gameId: string) => void;
  /**
   * Hotseat hand-off hook (unit 8's curtain). `onSeatChange` fires whenever the viewed seat
   * changes; while `revealed` is false the board and action list are hidden. Defaults to true.
   */
  onSeatChange?: (seat: number) => void;
  revealed?: boolean;
}

async function downloadExport(gameId: string): Promise<void> {
  const { data, error } = await api.GET("/api/games/{game_id}/export", {
    params: { path: { game_id: gameId } },
  });
  if (!data) throw new Error(errorText(error));
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `bus-${gameId}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

const isTerminalGame = (g: GameView | null): boolean => g !== null && isTerminal(g);

const dollars = (n: number): string => (n < 0 ? `-$${-n}` : `$${n}`);

function GameOver({ game }: { game: GameView }) {
  const winners = game.winners ?? [];
  const final = game.view.final;
  return (
    <div aria-label="Game over" className="lg-gameover">
      <h2>Game over</h2>
      {winners.length === 1 && <p className="lg-gameover-win">P{winners[0]} wins</p>}
      {winners.length > 1 && (
        <p className="lg-gameover-win">Tie: {winners.map((p) => `P${p}`).join(", ")} win</p>
      )}
      {game.returns && (
        <p>Returns: {game.returns.map((r, i) => `P${i} ${r}`).join(", ")}</p>
      )}
      {Array.isArray(final) && (
        <p>
          Final dollars:{" "}
          {(final as number[]).map((d, i) => `P${i} ${dollars(d)}`).join(", ")}
        </p>
      )}
    </div>
  );
}

export default function GameScreen({
  gameId,
  pollMs = DEFAULT_POLL,
  onExit,
  onImported,
  onSeatChange,
  revealed = true,
}: Props) {
  const [gameName, setGameName] = useState("");
  // Seat whose view is requested (null = server default, i.e. the seat to move).
  const [viewSeat, setViewSeat] = useState<number | null>(null);
  const [settings] = useSettings();
  // Forced-move auto-advance waits until playback has caught up (no input while behind).
  const [holdAuto, setHoldAuto] = useState(false);
  const { game: latest, error, busy, act, actBatch, goBack, rewind, retry, autoNote, fetchFrame } = useGame(
    gameId,
    pollMs,
    { autoAdvance: holdAuto ? undefined : getPlugin(gameName).autoAdvance, seat: viewSeat },
  );
  // Bot moves are shown one at a time at the bot pace (UI_DESIGN §7); `raw` is the shown view.
  const pb = usePlayback(latest, { pace: settings.botPace, fetchFrame });
  const behind = pb.behind;
  useEffect(() => setHoldAuto(behind), [behind]);
  const raw = pb.shown;
  // Data rendered for another seat than the viewed one is treated as loading (no private leak).
  const game = raw !== null && viewSeat !== null && raw.seat !== viewSeat ? null : raw;
  const mismatch = raw !== null && game === null;
  if (raw && raw.game !== gameName) setGameName(raw.game);
  const derivedSeat = raw ? deriveViewSeat(raw, viewSeat) : null;
  if (derivedSeat !== null && derivedSeat !== viewSeat) setViewSeat(derivedSeat);
  // Hotseat curtain: covered whenever the viewed human seat differs from the last acknowledged one.
  const [ackSeat, setAckSeat] = useState<number | null>(null);
  if (viewSeat !== null && ackSeat === null) setAckSeat(viewSeat);
  const hotseat = raw !== null && raw.seats.filter((s) => s === "human").length > 1;
  const covered =
    hotseat && !isTerminalGame(raw) && viewSeat !== null && ackSeat !== null && viewSeat !== ackSeat;
  const showBoard = revealed && !covered;
  useEffect(() => {
    if (viewSeat !== null) onSeatChange?.(viewSeat);
  }, [viewSeat, onSeatChange]);

  const [tab, setTab] = useState<ScreenTab>("play");
  const [hoverAction, setHoverAction] = useState<number | null>(null);
  const [reporting, setReporting] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [listOpen, setListOpen] = useState(false);
  const [rewindTo, setRewindTo] = useState("0");
  const [exportError, setExportError] = useState<string | null>(null);
  const rewindValid = /^\d+$/.test(rewindTo.trim());
  const plugin = getPlugin(game?.game ?? "");
  const over = game !== null && isTerminal(game);
  const cp = game ? currentPlayer(game) : null;
  const humanTurn =
    game !== null && cp !== null && game.seats[cp] === "human" && !game.bot_thinking && !behind;
  const botThinking = game !== null && !over && (game.bot_thinking || behind);
  const canSkip =
    game !== null && !over && (behind || latest?.bot_thinking === true) && game.seats.includes("human");

  const phase = game ? plugin.phaseLabel?.(game) : undefined;
  const ui: BoardUi | undefined = game
    ? {
        hoverAction,
        canAct: !over && humanTurn && !busy && showBoard,
        onAct: (id) => act(id),
        onActBatch: (ids) => actBatch(ids),
        onBack: () => goBack(),
        canBack: game.history_length > 0 && !over && !busy,
        viewSeat,
        motion:
          pb.motion && settings.animation !== "off"
            ? { ...pb.motion, speed: settings.animation, scale: SPEED_SCALE[settings.animation] }
            : null,
      }
    : undefined;

  const boardDriven = plugin.boardDriven === true;
  const moveArea = over ? (
    <GameOver game={game} />
  ) : (
    cp !== null &&
    game && (
      <>
        <YourMove
          humanTurn={humanTurn && showBoard}
          waitingFor={humanTurn ? null : `P${cp} (${game.seats[cp]})`}
          phase={phase}
          actions={legalActions(game)}
          busy={busy}
          onAct={(id) => void act(id)}
          labelFor={(a) => plugin.actionLabel?.(game, a) ?? a.label}
          onHover={setHoverAction}
          describe={(a) => plugin.actionHelp?.(game, a)}
        />
        <p className="lg-current">
          Current player: P{cp} ({game.seats[cp]})
        </p>
      </>
    )
  );

  return (
    <SeatViewContext.Provider value={{ viewSeat, revealed: showBoard }}>
      <section className="lg-screen" data-tab={tab}>
        <div className="lg-topbar">
          <div className="lg-topbar-row">
            {onExit && (
              <button type="button" className="btn-sm btn-ghost lg-lobby" onClick={onExit}>
                <IconArrowLeft />
                Lobby
              </button>
            )}
            {game && (
              <div role="toolbar" aria-label="Game tools" className="lg-tools">
                <Tip text="Back: undo the last decision">
                  <button
                    type="button"
                    className="btn-sm btn-ghost btn-icon"
                    aria-label="Back"
                    disabled={busy || game.history_length === 0}
                    onClick={() => void goBack()}
                  >
                    <IconUndo />
                  </button>
                </Tip>
                <Tip text="Export replay (JSON file)">
                  <button
                    type="button"
                    className="btn-sm btn-ghost btn-icon"
                    aria-label="Export replay"
                    onClick={() => {
                      setExportError(null);
                      downloadExport(gameId).catch((e: unknown) =>
                        setExportError(e instanceof Error ? e.message : "Export failed"),
                      );
                    }}
                  >
                    <IconDownload />
                  </button>
                </Tip>
                {onImported && <ImportButton onImported={onImported} compact />}
                <Tip text="Settings: animation speed, bot pace">
                  <button
                    type="button"
                    className="btn-sm btn-ghost btn-icon"
                    aria-label="Settings"
                    onClick={() => setSettingsOpen(true)}
                  >
                    <IconSliders />
                  </button>
                </Tip>
                <Tip text="Report a problem with this position">
                  <button
                    type="button"
                    className="btn-sm btn-ghost btn-icon"
                    aria-label="Report a problem"
                    onClick={() => setReporting(true)}
                  >
                    <IconFlag />
                  </button>
                </Tip>
              </div>
            )}
          </div>
          {game && (
            <p className="lg-status-line" title={`Game ${game.id}`}>
              Game {game.id} · seats: {game.seats.map((s, i) => `P${i} ${s}`).join(", ")}
            </p>
          )}
          {game?.dev_mode && (
            <div className="lg-rewind">
              <label>
                Rewind to player action{" "}
                <input
                  value={rewindTo}
                  onChange={(e) => setRewindTo(e.target.value)}
                  inputMode="numeric"
                  size={4}
                />
              </label>{" "}
              <button
                type="button"
                className="btn-sm"
                disabled={busy || !rewindValid}
                onClick={() => void rewind(Number(rewindTo.trim()))}
              >
                <IconRewind />
                Rewind
              </button>
            </div>
          )}
        </div>
        {exportError && (
          <p role="alert" className="alert lg-toast">
            Export error: {exportError}
          </p>
        )}
        {error && (
          <p role="alert" className="alert lg-toast">
            Error: {error}
          </p>
        )}
        {!game && !error && !covered && (
          <div className="lg-loading">
            <span className="spinner" aria-hidden="true" />
            <p>Loading…</p>
          </div>
        )}
        {settingsOpen && <SettingsDialog onClose={() => setSettingsOpen(false)} />}
        {raw && covered && viewSeat !== null && (
          <Curtain seat={viewSeat} accent={plugin.seatStyle?.(viewSeat)} disabled={mismatch} onReveal={() => setAckSeat(viewSeat)} />
        )}
        {game && ui && !covered && (
          <>
            {reporting && <ReportDialog gameId={gameId} onClose={() => setReporting(false)} />}
            <TabBar value={tab} onChange={setTab} />
            <div className="lg-grid">
              <div className="lg-pane-board">
                {showBoard ? plugin.renderBoard(game, ui) : <p className="lg-board-hidden">Board hidden.</p>}
              </div>
              <div className="lg-side">
                <div className="lg-pane-play">
                  {plugin.renderSide?.(game, ui)}
                  {botThinking && cp !== null && (
                    <div className="lg-thinking-row">
                      <p role="status" className="note lg-thinking">
                        <span className="spinner" aria-hidden="true" />
                        {game.seats[cp] === "human" ? "Showing moves…" : thinkingText(game.seats[cp] ?? "", cp)}
                      </p>
                      {canSkip && (
                        <button
                          type="button"
                          className="btn-sm lg-skip"
                          onClick={pb.skip}
                          disabled={pb.skipping}
                        >
                          <IconSkip />
                          Skip to my turn
                        </button>
                      )}
                    </div>
                  )}
                  {game.bot_error && !over && (
                    <div role="alert" className="alert">
                      Bot error: {game.bot_error}{" "}
                      <button type="button" className="btn-sm" onClick={() => void retry()}>
                        Retry
                      </button>
                    </div>
                  )}
                  {autoNote && !over && (
                    <p role="status" className="note">
                      {autoNote} (only legal action)
                    </p>
                  )}
                  {boardDriven ? (
                    <div className="lg-debug" data-open={listOpen}>
                      <button
                        type="button"
                        className="btn-sm btn-ghost lg-debug-toggle"
                        aria-expanded={listOpen}
                        onClick={() => setListOpen((o) => !o)}
                      >
                        <IconList />
                        {over ? "Server result" : `Action list (${legalActions(game).length})`}
                      </button>
                      <div className="lg-debug-body">{moveArea}</div>
                    </div>
                  ) : (
                    moveArea
                  )}
                </div>
                <div className="lg-log lg-pane-log">
                  <h3>Action log</h3>
                  {plugin.renderLog ? (
                    plugin.renderLog(game)
                  ) : (
                    <ActionLog history={game.history} actorStyle={plugin.seatStyle} />
                  )}
                </div>
              </div>
            </div>
          </>
        )}
      </section>
    </SeatViewContext.Provider>
  );
}
