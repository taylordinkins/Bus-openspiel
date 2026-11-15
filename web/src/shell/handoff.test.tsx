import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { createElement } from "react";
import { beforeEach, vi } from "vitest";
import type { BoardUi, GamePlugin } from "../games/registry";
import GameScreen from "./GameScreen";
import { updateSettings } from "./settings";
import type { GameView } from "./useGame";

const GET = vi.fn();
const POST = vi.fn();
vi.mock("../api/client", () => ({ api: { GET: (...a: unknown[]) => GET(...a), POST: (...a: unknown[]) => POST(...a) } }));

const MOTION_MS = 800;
const motionMs = vi.fn((_m: unknown) => MOTION_MS);

/** A board that shows the motion it was given; the plugin reports a fixed animation length. */
const testPlugin: GamePlugin = {
  id: "handoff-test",
  title: "Handoff test",
  renderBoard: (g, ui?: BoardUi) =>
    createElement("pre", { "aria-label": "Game state", "data-motion": ui?.motion?.seq ?? "none" }, JSON.stringify(g.view)),
  motionMs: (m) => motionMs(m),
};
vi.mock("../games/registry", () => ({ getPlugin: () => testPlugin }));

const POLL = { fast: 20, slow: 20 };

/** Two human seats; `n` entries alternating P0, P1; rendered for `seat` (default: the mover). */
function view(n: number, seat?: number): GameView {
  const cp = n % 2;
  return {
    id: "g1",
    game: "handoff-test",
    seats: ["human", "human"],
    seat_types: ["human", "human"],
    bot_thinking: false,
    seat: seat ?? cp,
    history_length: n,
    history: Array.from({ length: n }, (_, i) => ({ index: i, actor: i % 2, id: 3, label: `m${i}`, events: [] })),
    dev_mode: false,
    current_player: cp,
    is_terminal: false,
    legal_actions: [{ id: 3, label: "Play A" }],
    returns: null,
    winners: null,
    view: { n },
  };
}

/** GET serves the live game for the requested seat; POST (act) advances one entry. */
function serve() {
  let n = 0;
  GET.mockImplementation(async (_p: string, o: { params?: { query?: { seat?: number } } }) => ({
    data: view(n, o?.params?.query?.seat),
  }));
  POST.mockImplementation(async (_p: string, o: { params?: { query?: { seat?: number } } }) => {
    n += 1;
    return { data: view(n, o?.params?.query?.seat) };
  });
}

const stateText = () => screen.getByLabelText("Game state").textContent ?? "";

beforeEach(() => {
  GET.mockReset();
  POST.mockReset();
  motionMs.mockClear();
});

describe("hotseat hand-off waits for the mover's animation (UI_DESIGN 7, 6.8)", () => {
  it("the board stays up while the move animates, then the curtain covers it", async () => {
    updateSettings({ animation: "normal", botPace: 0.8 });
    serve();
    render(<GameScreen gameId="g1" pollMs={POLL} />);
    await waitFor(() => expect(stateText()).toContain('"n":0'));
    fireEvent.click(screen.getByRole("button", { name: "Play A" }));
    // the mover's change is shown with its motion, and no curtain yet
    await waitFor(() => expect(stateText()).toContain('"n":1'));
    expect(screen.getByLabelText("Game state")).not.toHaveAttribute("data-motion", "none");
    expect(motionMs).toHaveBeenCalled();
    expect(screen.queryByText(/Pass to/)).toBeNull();
    // no input for the next seat while the animation plays
    expect(screen.queryByRole("button", { name: "Play A" })).toBeNull();
    await new Promise((r) => setTimeout(r, MOTION_MS / 4));
    expect(screen.queryByText(/Pass to/)).toBeNull();
    expect(stateText()).toContain('"n":1');
    // then the hand-off
    await screen.findByText(/Pass to P1/, undefined, { timeout: 3000 });
    expect(screen.queryByLabelText("Game state")).toBeNull();
    // the button is enabled once the view for P1 has arrived
    await waitFor(() => expect(screen.getByRole("button", { name: /click when ready/i })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: /click when ready/i }));
    await waitFor(() => expect(stateText()).toContain('"n":1'));
    expect(screen.getByRole("button", { name: "Play A" })).toBeEnabled();
  });

  it.each([
    ["animation off", { animation: "off" as const, botPace: 0.8 }],
    ["pace 0", { animation: "normal" as const, botPace: 0 }],
  ])("%s: the curtain comes at once", async (_name, s) => {
    updateSettings(s);
    serve();
    render(<GameScreen gameId="g1" pollMs={POLL} />);
    await waitFor(() => expect(stateText()).toContain('"n":0'));
    fireEvent.click(screen.getByRole("button", { name: "Play A" }));
    await screen.findByText(/Pass to P1/, undefined, { timeout: MOTION_MS / 2 });
  });
});
