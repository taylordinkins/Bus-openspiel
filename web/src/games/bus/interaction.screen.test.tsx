import { render, screen, waitFor, within, fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GameScreen from "../../shell/GameScreen";
import { getPlugin } from "../registry";
import { resetGeometryCache } from "./geometry";
import { HAND, PHASES, gameOf, geometryResponse, type Fx } from "./interactionHarness";

const GET = vi.fn();
const POST = vi.fn();
vi.mock("../../api/client", () => ({ api: { GET: (...a: unknown[]) => GET(...a), POST: (...a: unknown[]) => POST(...a) } }));

const POLL = { fast: 20, slow: 5000 };
const served = (f: Fx, extra: Record<string, unknown> = {}) => ({ ...gameOf(f), ...extra });

function serve(f: Fx, extra: Record<string, unknown> = {}) {
  GET.mockImplementation(async (path: string) =>
    path === "/api/boards/{game}" ? geometryResponse() : { data: served(f, extra) },
  );
}

beforeEach(() => {
  GET.mockReset();
  POST.mockReset();
  resetGeometryCache();
});

const bar = () => screen.findByTestId("bus-prompt");

describe("unit8 (4d) Back through the shell (UI_DESIGN 7; 6.7e)", () => {
  it("unit8: the bar's Back posts /back once and shows the returned view", async () => {
    serve(PHASES.VP!, { history_length: 4 });
    POST.mockResolvedValue({ data: served(PHASES.LN!, { history_length: 3 }) });
    render(<GameScreen gameId="g" pollMs={POLL} />);
    const back = within(await bar()).getByRole("button", { name: /^back$/i });
    expect(back).toBeEnabled();
    fireEvent.click(back);
    await waitFor(() => expect(POST).toHaveBeenCalledTimes(1));
    expect(POST.mock.calls[0]![0]).toBe("/api/games/{game_id}/back");
    expect(POST.mock.calls[0]![1]).toMatchObject({ params: { path: { game_id: "g" } } });
    await waitFor(() => expect(screen.getByTestId("bus-prompt")).toHaveAttribute("data-phase", "LN"));
  });

  it("unit8: Back is disabled at the start of the game (history_length 0)", async () => {
    serve(PHASES.L1!, { history_length: 0 });
    render(<GameScreen gameId="g" pollMs={POLL} />);
    expect(within(await bar()).getByRole("button", { name: /^back$/i })).toBeDisabled();
  });

  it("unit8: Back is disabled at END", async () => {
    serve(PHASES.END!, { history_length: 120, is_terminal: true });
    render(<GameScreen gameId="g" pollMs={POLL} />);
    expect(within(await bar()).getByRole("button", { name: /^back$/i })).toBeDisabled();
  });
});

describe("unit8 (5) human seats always click, even with one legal action (UI_DESIGN 4, 9 Q2)", () => {
  it("unit8: the Bus plugin has no auto-advance", () => {
    expect(getPlugin("bus").autoAdvance).toBeUndefined();
  });

  it("unit8: a lone pick is not auto-posted; the node glows and a click posts it", async () => {
    // Hand-edited from interaction/VP.multi, trimmed to the single pick at n16.
    const f = HAND["VP.lone"]!;
    expect(f.legal_actions).toHaveLength(1);
    serve(f);
    POST.mockResolvedValue({ data: served(HAND["VD.multi"]!) });
    const { container } = render(<GameScreen gameId="g" pollMs={POLL} />);
    await bar();
    await new Promise((r) => setTimeout(r, 700));
    expect(POST).not.toHaveBeenCalled();
    const node = await waitFor(() => {
      const e = container.querySelector("#node-16");
      if (!e) throw new Error("board not drawn");
      return e;
    });
    expect(node.getAttribute("data-legal")).toBe("true");
    fireEvent.click(node);
    await waitFor(() => expect(POST).toHaveBeenCalledTimes(1));
    expect(POST.mock.calls[0]![1]).toMatchObject({ body: { action: 309 } });
  });

  it("unit8: board-full SEL waits for a click on Drop out", async () => {
    serve(HAND["SEL.dropout_only"]!);
    POST.mockResolvedValue({ data: served(PHASES.SEL!) });
    render(<GameScreen gameId="g" pollMs={POLL} />);
    const btn = await within(await bar()).findByRole("button", { name: /drop out/i });
    await new Promise((r) => setTimeout(r, 700));
    expect(POST).not.toHaveBeenCalled();
    fireEvent.click(btn);
    await waitFor(() => expect(POST).toHaveBeenCalledTimes(1));
    expect(POST.mock.calls[0]![1]).toMatchObject({ body: { action: 7 } });
  });

  it("unit8: a bot seat's turn offers no clickable targets", async () => {
    const f = PHASES.SB!;
    serve(f, { seats: ["human", "random", "human", "human"], current_player: 1, seat: 0 });
    const { container } = render(<GameScreen gameId="g" pollMs={POLL} />);
    await bar();
    await waitFor(() => expect(container.querySelector("#site-3")).not.toBeNull());
    expect(container.querySelectorAll('[data-legal="true"]')).toHaveLength(0);
  });
});
