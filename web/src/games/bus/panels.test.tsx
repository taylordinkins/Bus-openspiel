import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GameView } from "../../shell/useGame";
import { getPlugin } from "../registry";
import geometry from "./fixtures/geometry.json";
import five from "./fixtures/five.json";
import logGame from "./fixtures/log.json";
import BD from "./fixtures/phases/BD.json";
import CK from "./fixtures/phases/CK.json";
import END from "./fixtures/phases/END.json";
import L1 from "./fixtures/phases/L1.json";
import L2 from "./fixtures/phases/L2.json";
import LN from "./fixtures/phases/LN.json";
import PX from "./fixtures/phases/PX.json";
import SB from "./fixtures/phases/SB.json";
import SEL from "./fixtures/phases/SEL.json";
import VD from "./fixtures/phases/VD.json";
import VP from "./fixtures/phases/VP.json";
import { resetGeometryCache } from "./geometry";
import { setHighlight } from "./hoverStore";
import { logRows, describeEvent } from "./busLog";
import { rankSeats, tiebreakText } from "./EndResults";
import { busInner, parseBusView } from "./model";

const GET = vi.fn();
vi.mock("../../api/client", () => ({ api: { GET: (...a: unknown[]) => GET(...a) } }));

beforeEach(() => {
  GET.mockReset();
  GET.mockReturnValue(new Promise(() => {})); // geometry never arrives unless a test calls withBoard()
  resetGeometryCache();
  setHighlight(null);
});

const withBoard = () => GET.mockResolvedValue({ data: { game: "bus", geometry: geometry.geometry } });
/** Wait for the board and let its passive effects (store subscription) run before hovering. */
async function boardReady() {
  await screen.findByTestId("bus-svg");
  await act(async () => {});
}
const asGame = (f: unknown): GameView => f as GameView;

/** Everything the plugin draws for a game: board column, side column, log. */
function renderAll(g: GameView) {
  const plugin = getPlugin("bus");
  return render(
    <>
      {plugin.renderBoard(g)}
      {plugin.renderSide?.(g)}
      {plugin.renderLog?.(g)}
    </>,
  );
}

const PHASES: [string, unknown, RegExp, string[]][] = [
  ["SB", SB, /Setup: buildings: place a building on an A site/, ["building 2 of 2"]],
  ["L1", L1, /Setup: first line marker: place your first Bus Line marker on any street/, []],
  ["L2", L2, /Setup: second line marker: extend your line from either end/, []],
  ["SEL", SEL, /Selection: place an Action marker, or drop out for this round/, ["pass 3", "18 markers in hand"]],
  ["LN", LN, /Lines: extend your line along a street/, ["2 streets left", "power 0"]],
  ["PX", PX, /Passengers: add a passenger to a station/, ["3 passengers left", "power 1", "11 in supply"]],
  ["BD", BD, /Buildings: build on an A site/, ["2 buildings left", "power 1"]],
  ["CK", CK, /Clock: advance the clock to Office or take a time stone/, ["5 stones left"]],
  ["VP", VP, /Vrrooomm!: pick a waiting passenger on your line/, ["1 move left", "deliver to Houses"]],
  ["VD", VD, /Vrrooomm!: deliver the passenger to a House on your line/, ["1 move left"]],
];

describe("prompt bar per phase", () => {
  it.each(PHASES)("%s", (code, fixture, text, chips) => {
    const g = asGame(fixture);
    renderAll(g);
    const bar = screen.getByTestId("bus-prompt");
    expect(bar).toHaveAttribute("data-phase", code);
    expect(bar).toHaveTextContent(text);
    const actor = g.current_player as number;
    expect(within(bar).getByText(`P${actor}`)).toBeInTheDocument();
    for (const c of chips) expect(within(bar).getByText(c)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/Player \d/);
  });

  it("SEL pass 1 says the marker must be placed", () => {
    renderAll(asGame(five));
    const bar = screen.getByTestId("bus-prompt");
    expect(bar).toHaveTextContent(/Selection: place an Action marker on a free action space/);
    expect(within(bar).getByText("must place")).toBeInTheDocument();
    expect(within(bar).getByText("pass 1")).toBeInTheDocument();
  });

  it("END names the winner and has no actor", () => {
    renderAll(asGame(END));
    const bar = screen.getByTestId("bus-prompt");
    expect(bar).toHaveAttribute("data-actor", "");
    expect(bar).toHaveTextContent(/Game over · P\d wins|share the win/);
  });

  it("a bot actor is shown as waiting", () => {
    const g = asGame({ ...logGame, seats: ["random", "random", "random", "random"] });
    const inner = busInner(g) as { prompt: { actor: number | null } };
    const patched = {
      ...g,
      view: { ...g.view, view: { ...inner, prompt: { phase: "LN", actor: 1, units_left: 1, power: 1, context: {} } } },
      bot_thinking: true,
    } as unknown as GameView;
    renderAll(patched);
    expect(screen.getByTestId("bus-prompt")).toHaveTextContent(/Waiting for Random bot \(thinking…\)/);
  });

  it("keeps the bot-wait note on its own line, apart from the phase title", () => {
    const g = asGame({ ...logGame, seats: ["random", "random", "random", "random"] });
    const inner = busInner(g) as { prompt: { actor: number | null } };
    const patched = {
      ...g,
      view: { ...g.view, view: { ...inner, prompt: { phase: "LN", actor: 1, units_left: 1, power: 1, context: {} } } },
      bot_thinking: true,
    } as unknown as GameView;
    renderAll(patched);
    const bar = screen.getByTestId("bus-prompt");
    const wait = bar.querySelector(".bus-prompt-wait");
    const title = bar.querySelector(".bus-prompt-text strong");
    expect(wait?.querySelector(".bus-prompt-wait-text")).toHaveTextContent("Waiting for Random bot (thinking…)");
    // the title is a sibling after the note, not inside it (the note is a block row; see bus.css)
    expect(wait?.contains(title ?? null)).toBe(false);
    expect(wait?.nextElementSibling).toBe(title);
  });
});

describe("seat panels", () => {
  it("shows kind, score, start flag, buses, markers and the acting seat", () => {
    const g = asGame(SEL);
    renderAll(g);
    const data = parseBusView(busInner(g));
    expect(data).not.toBeNull();
    const panels = screen.getAllByLabelText(/^P\d panel$/);
    expect(panels).toHaveLength(4);
    const p0 = screen.getByLabelText("P0 panel");
    expect(p0).toHaveAttribute("data-acting", "true");
    expect(screen.getByLabelText("P1 panel")).toHaveAttribute("data-acting", "false");
    expect(within(p0).getByText("P0")).toBeInTheDocument();
    expect(p0).toHaveTextContent(/Human · red/);
    expect(p0).toHaveTextContent(`Action markers: ${data!.state.seats[0]!.markers_left}`);
    expect(p0).toHaveTextContent(`Line markers: ${data!.derived.seats[0]!.line_markers_left}`);
    expect(within(p0).getByLabelText("Score")).toHaveTextContent(`${data!.state.seats[0]!.points} pts`);
    const start = data!.state.start_player;
    panels.forEach((p, i) => {
      expect(within(p).queryByText(/⚑ start/) !== null).toBe(i === start);
    });
    const owned = data!.state.seats[0]!.buses;
    expect(within(p0).getByLabelText(`${owned} of 5 buses`).querySelectorAll('[data-owned="true"]')).toHaveLength(owned);
    expect(screen.getByLabelText("Round info")).toHaveTextContent(/Round \d+ · pass 3/);
  });

  it("marks a dropped-out seat", () => {
    renderAll(asGame(SEL));
    const p3 = screen.getByLabelText("P3 panel");
    expect(p3).toHaveAttribute("data-dropped", "true");
    expect(p3).toHaveTextContent("dropped out");
    expect(screen.getByLabelText("P0 panel")).toHaveAttribute("data-dropped", "false");
  });

  it("labels bot seats", () => {
    renderAll(asGame(logGame));
    expect(screen.getByLabelText("P2 panel")).toHaveTextContent(/Random · orange/);
  });

  it("five players: P4 white panel and badge are labelled and outlined", () => {
    renderAll(asGame(five));
    const p4 = screen.getByLabelText("P4 panel");
    expect(p4).toHaveTextContent(/P4/);
    expect(p4).toHaveTextContent(/white/);
    const badge = within(p4).getByText("P4");
    expect(badge).toHaveStyle({ background: "#FEFEFE", color: "#1A1A1A" });
    expect(badge.style.border).toContain("2px solid");
    expect(screen.getAllByLabelText(/^P\d panel$/)).toHaveLength(5);
  });

  it("hovering a panel highlights that seat's line on the board", async () => {
    withBoard();
    const g = asGame(LN);
    const { container } = renderAll(g);
    await boardReady();
    const line = parseBusView(busInner(g))!.state.seats[1]!.line;
    expect(line.length).toBeGreaterThan(0);
    expect(container.querySelectorAll(".hl-street")).toHaveLength(0);
    fireEvent.mouseEnter(screen.getByLabelText("P1 panel"));
    const lit = [...container.querySelectorAll(".hl-street")].map((e) => Number(e.getAttribute("data-street")));
    expect(lit.sort()).toEqual([...new Set(line)].sort());
    fireEvent.mouseLeave(screen.getByLabelText("P1 panel"));
    expect(container.querySelectorAll(".hl-street")).toHaveLength(0);
  });
});

describe("hotseat turn frame", () => {
  it("frames the board in the acting seat's colour and toasts a change of seat", () => {
    vi.useFakeTimers();
    try {
      const plugin = getPlugin("bus");
      const { rerender } = render(<>{plugin.renderBoard(asGame(PX))}</>);
      const frame = screen.getByTestId("turn-frame");
      expect(frame).toHaveAttribute("data-seat", "2");
      expect(frame).toHaveStyle({ borderColor: "#FE6500" });
      expect(screen.queryByText(/’s turn/)).toBeNull();
      rerender(<>{plugin.renderBoard(asGame(BD))}</>);
      expect(screen.getByTestId("turn-frame")).toHaveAttribute("data-seat", "1");
      expect(screen.getByText(/’s turn/).parentElement).toHaveTextContent("P1’s turn");
      act(() => {
        vi.advanceTimersByTime(3000);
      });
      expect(screen.queryByText(/’s turn/)).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("is absent with one human seat", () => {
    const g = asGame({ ...PX, seats: ["human", "random", "random", "random"] });
    renderAll(g);
    expect(screen.queryByTestId("turn-frame")).toBeNull();
  });
});

describe("log", () => {
  it("renders P-labelled decisions, indented automatic steps and round headings", () => {
    renderAll(asGame(logGame));
    const log = screen.getByRole("list", { name: "Action log" });
    expect(log).toHaveTextContent("Setup");
    expect(log).toHaveTextContent("Round 1");
    expect(log).toHaveTextContent("Round 4");
    expect(log).toHaveTextContent(/P\d: /);
    expect(log).toHaveTextContent("Selection ends; actions resolve in order");
    expect(log).toHaveTextContent(/P\d adds a bus \(now \d\)/);
    expect(log).toHaveTextContent(/P\d's (Lines|Buses|Passengers|Buildings|Clock|Vrrooomm!|Starting Player) marker [A-F] resolves with nothing to do/);
    expect(log).toHaveTextContent(/Round \d ends; P\d starts next/);
    expect(log).toHaveTextContent(/Game ends: at most one player has Action markers left/);
    expect(log).toHaveTextContent(/P2 delivers a passenger: 1 pts/);
    expect(log).toHaveTextContent(/Passengers move: \d+ leave buildings, \d+ enter/);
    expect(log.textContent).not.toMatch(/Player \d/);
    expect(log.textContent).not.toContain("Unhandled event");
    // one list item per line
    expect(log.querySelectorAll("li").length).toBe(logGame.history.length + logGame.history.flatMap((h) => h.events.filter((e) => e.type !== "round_started")).length);
  });

  it("names the automatic Clock step (D22): clock_advanced has no seat", () => {
    expect(describeEvent({ type: "clock_advanced", from: "house", to: "office" })).toEqual({
      text: "Clock advances House → Office (no Clock marker)",
      hl: {},
    });
  });

  it("falls back to a readable line for an unknown event type", () => {
    expect(describeEvent({ type: "sparkle", x: 1 }).text).toBe('Unhandled event "sparkle": {"type":"sparkle","x":1}');
    const rows = logRows([{ index: 0, actor: 1, id: 3, label: "Do", events: [{ type: "sparkle" }] }]);
    expect(rows.map((r) => r.kind)).toEqual(["heading", "action", "event"]);
  });

  it("hovering a delivery highlights its site, node and seat on the board", async () => {
    withBoard();
    const { container } = renderAll(asGame(logGame));
    await boardReady();
    const line = screen.getByText(/P2 delivers a passenger/);
    fireEvent.mouseEnter(line);
    expect(container.querySelectorAll(".hl-site")).toHaveLength(1);
    expect(container.querySelectorAll(".hl-node")).toHaveLength(1);
    expect(container.querySelectorAll(".hl-street").length).toBeGreaterThan(0); // seat 2's line
    fireEvent.mouseLeave(line);
    expect(container.querySelectorAll(".hl-site")).toHaveLength(0);
  });

  it("hovering a passenger move highlights the buildings involved", async () => {
    withBoard();
    const { container } = renderAll(asGame(logGame));
    await boardReady();
    const row = screen.getAllByText(/Passengers move:/)[0]!;
    fireEvent.mouseEnter(row);
    expect(container.querySelectorAll(".hl-site").length).toBeGreaterThan(0);
  });
});

describe("END results", () => {
  it("shows ranking, end type, tiebreak note and highlights the winner", () => {
    renderAll(asGame(logGame));
    const end = screen.getByTestId("end-results");
    expect(end).toHaveTextContent("P2 wins");
    expect(end).toHaveTextContent(/at most one player has Action markers left/);
    const rows = within(end).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(4);
    expect(rows[0]).toHaveAttribute("data-seat", "2");
    expect(rows[0]).toHaveAttribute("data-winner", "true");
    expect(rows[0]).toHaveTextContent(/1\s*P2\s*winner1\s*0\s*1\s*136/);
    // P0 and P3 tie on final score -1 with 1 stone each, and share rank 2 or later
    const ranks = rows.map((r) => r.querySelector("td")!.textContent);
    expect(ranks[0]).toBe("1");
    expect(rows.filter((r) => r.getAttribute("data-winner") === "true")).toHaveLength(1);
    expect(end).toHaveTextContent(/Final = points minus time stones/);
  });

  it("END scenario fixture renders the results table", () => {
    renderAll(asGame(END));
    expect(screen.getByTestId("end-results")).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(5);
  });

  it("is absent before END", () => {
    renderAll(asGame(SEL));
    expect(screen.queryByTestId("end-results")).toBeNull();
  });

  it("ranks ties by stones, then earlier last scoring move, and shares equal rows", () => {
    const base = parseBusView(busInner(asGame(logGame)))!;
    const mk = (seats: [number, number, number | null][], winners: number[]) =>
      ({
        ...base,
        state: { ...base.state, seats: seats.map(([points, stones, last]) => ({ ...base.state.seats[0]!, points, stones, last_score_move: last })) },
        derived: { ...base.derived, result: { end_types: ["markers"], winners, final_scores: seats.map(([p, s]) => p - s), returns: [] } },
      }) as typeof base;
    const stones = rankSeats(mk([[5, 2, 10], [3, 0, 4], [4, 1, 7]], [0, 1, 2]));
    expect(stones.map((r) => [r.seat, r.rank])).toEqual([[0, 1], [2, 2], [1, 3]]);
    // seats that never scored have no move number and tie (R-070)
    const none = rankSeats(mk([[0, 0, null], [0, 0, null], [2, 0, 3]], [2]));
    expect(none.map((r) => [r.seat, r.rank])).toEqual([[2, 1], [0, 2], [1, 2]]);
    const t = rankSeats(mk([[3, 0, 9], [3, 1, 5], [5, 0, 2]], [2]));
    expect(t.map((r) => r.seat)).toEqual([2, 0, 1]);
    expect(tiebreakText(rankSeats(mk([[4, 1, 5], [3, 0, 2]], [0, 1])))).toMatch(/shared win|reached that total first|time stones/);
    expect(tiebreakText(rankSeats(mk([[4, 0, 5], [3, 0, 2]], [0])))).toBe("Highest final score wins.");
    expect(tiebreakText(rankSeats(mk([[4, 2, 5], [3, 1, 2]], [0])))).toMatch(/most time stones wins/);
    expect(tiebreakText(rankSeats(mk([[3, 1, 5], [2, 0, 2], [3, 1, 9]], [0])))).toMatch(/reached that total first/);
  });
});
