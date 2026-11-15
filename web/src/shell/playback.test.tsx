import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import { extendsView, planSteps, usePlayback, type HistoryEntry } from "./playback";
import type { GameView } from "./useGame";

/** P0 human, P1 and P2 bots; `actors` are the actors of the history entries. */
function view(actors: (number | null)[], over: Partial<GameView> = {}): GameView {
  const history: HistoryEntry[] = actors.map((a, i) => ({
    index: i,
    actor: a,
    id: 100 + i,
    label: `move ${i}`,
    events: [{ type: "marker", n: i }],
  }));
  const next = actors.length % 3;
  return {
    id: "g1",
    game: "bus",
    seats: ["human", "random", "random"],
    seat_types: ["human", "random", "random"],
    bot_thinking: false,
    seat: 0,
    history_length: actors.length,
    history,
    dev_mode: false,
    current_player: next,
    is_terminal: false,
    legal_actions: [],
    returns: null,
    winners: null,
    view: {},
    ...over,
  };
}

describe("unit9 planSteps / extendsView", () => {
  it("each bot entry is a step; human runs merge; chance joins the step before", () => {
    const g = view([0, 0, 1, null, 2, 0]);
    expect(planSteps(g, 0)).toEqual([
      { end: 2, bot: false },
      { end: 4, bot: true },
      { end: 5, bot: true },
      { end: 6, bot: false },
    ]);
    expect(planSteps(g, 4)).toEqual([
      { end: 5, bot: true },
      { end: 6, bot: false },
    ]);
  });

  it("detects a history that is no longer a prefix (Back, rewind, another game)", () => {
    const a = view([0, 1]);
    expect(extendsView(a, view([0, 1, 2]))).toBe(true);
    expect(extendsView(a, view([0]))).toBe(false);
    const other = view([0, 1, 2]);
    other.history[1] = { ...other.history[1]!, label: "different" };
    expect(extendsView(a, other)).toBe(false);
    expect(extendsView(a, view([0, 1, 2], { id: "g2" }))).toBe(false);
  });
});

describe("unit9 usePlayback (UI_DESIGN 7)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  const flush = async () => {
    await act(async () => {
      await Promise.resolve();
    });
  };

  function setup(pace: number) {
    const frames = vi.fn(async (at: number) => view([0, 1, 2, 0, 1, 2].slice(0, at)));
    const hook = renderHook(({ g, p }: { g: GameView | null; p: number }) => usePlayback(g, { pace: p, fetchFrame: frames }), {
      initialProps: { g: view([]) as GameView | null, p: pace },
    });
    return { frames, hook };
  }

  it("the first view shows at once, without motion", () => {
    const { hook } = setup(0.8);
    expect(hook.result.current.shown?.history_length).toBe(0);
    expect(hook.result.current.motion).toBeNull();
    expect(hook.result.current.behind).toBe(false);
  });

  it("a human move shows immediately with its events", () => {
    const { hook, frames } = setup(0.8);
    hook.rerender({ g: view([0]), p: 0.8 });
    expect(hook.result.current.shown?.history_length).toBe(1);
    expect(hook.result.current.motion).toMatchObject({ from: 0, to: 1, events: [{ type: "marker", n: 0 }] });
    expect(frames).not.toHaveBeenCalled();
  });

  it("bot moves play one at a time at the pace, fetching frames, and block input while behind", async () => {
    const { hook, frames } = setup(0.8);
    hook.rerender({ g: view([0]), p: 0.8 });
    hook.rerender({ g: view([0, 1, 2]), p: 0.8 });
    expect(hook.result.current.shown?.history_length).toBe(1);
    expect(hook.result.current.behind).toBe(true);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(799);
    });
    expect(hook.result.current.shown?.history_length).toBe(1);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    await flush();
    expect(frames).toHaveBeenCalledWith(2, 0);
    expect(hook.result.current.shown?.history_length).toBe(2);
    expect(hook.result.current.motion).toMatchObject({ from: 1, to: 2, events: [{ type: "marker", n: 1 }] });
    expect(hook.result.current.behind).toBe(true);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(800);
    });
    await flush();
    // the last step is the latest view itself: no frame request
    expect(frames).toHaveBeenCalledTimes(1);
    expect(hook.result.current.shown?.history_length).toBe(3);
    expect(hook.result.current.motion).toMatchObject({ from: 2, to: 3 });
    expect(hook.result.current.behind).toBe(false);
  });

  it("pace 0 shows the latest state at once with every new event", () => {
    const { hook, frames } = setup(0);
    hook.rerender({ g: view([0, 1, 2]), p: 0 });
    expect(hook.result.current.shown?.history_length).toBe(3);
    expect(hook.result.current.motion?.events).toHaveLength(3);
    expect(frames).not.toHaveBeenCalled();
  });

  it("Skip to my turn fast-forwards to the latest state, without motion", async () => {
    const { hook } = setup(2);
    hook.rerender({ g: view([0]), p: 2 });
    const seq = hook.result.current.motion?.seq;
    hook.rerender({ g: view([0, 1, 2]), p: 2 });
    expect(hook.result.current.behind).toBe(true);
    act(() => hook.result.current.skip());
    await flush();
    expect(hook.result.current.shown?.history_length).toBe(3);
    expect(hook.result.current.motion?.seq).toBe(seq);
    expect(hook.result.current.behind).toBe(false);
    expect(hook.result.current.skipping).toBe(false); // P0 (human) is to move: skip is done
  });

  it("skip keeps fast-forwarding while bots still play, until a human turn", async () => {
    const { hook } = setup(2);
    hook.rerender({ g: view([0]), p: 2 });
    hook.rerender({ g: view([0, 1], { bot_thinking: true }), p: 2 });
    act(() => hook.result.current.skip());
    await flush();
    expect(hook.result.current.shown?.history_length).toBe(2);
    expect(hook.result.current.skipping).toBe(true);
    hook.rerender({ g: view([0, 1, 2]), p: 2 });
    await flush();
    expect(hook.result.current.shown?.history_length).toBe(3);
    expect(hook.result.current.skipping).toBe(false);
    // the next bot run is paced again
    hook.rerender({ g: view([0, 1, 2, 0, 1, 2]), p: 2 });
    await flush(); // the human step needs a frame (not the last step)
    expect(hook.result.current.shown?.history_length).toBe(4);
    expect(hook.result.current.behind).toBe(true);
  });

  it("Back (a shorter history) cancels playback and shows the result at once", async () => {
    const { hook, frames } = setup(1);
    hook.rerender({ g: view([0]), p: 1 });
    hook.rerender({ g: view([0, 1, 2]), p: 1 });
    hook.rerender({ g: view([]), p: 1 });
    expect(hook.result.current.shown?.history_length).toBe(0);
    expect(hook.result.current.behind).toBe(false);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(frames).not.toHaveBeenCalled();
    expect(hook.result.current.shown?.history_length).toBe(0);
  });

  it("a failed frame request jumps to the latest state", async () => {
    const { hook, frames } = setup(0.5);
    frames.mockRejectedValueOnce(new Error("network"));
    hook.rerender({ g: view([0]), p: 0.5 });
    hook.rerender({ g: view([0, 1, 2]), p: 0.5 });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    await flush();
    expect(hook.result.current.shown?.history_length).toBe(3);
    expect(hook.result.current.behind).toBe(false);
  });

  it("a poll of the same position updates the shown view without new motion", () => {
    const { hook } = setup(0.8);
    hook.rerender({ g: view([0]), p: 0.8 });
    const seq = hook.result.current.motion?.seq;
    const again = view([0], { bot_thinking: true });
    hook.rerender({ g: again, p: 0.8 });
    expect(hook.result.current.shown).toBe(again);
    expect(hook.result.current.motion?.seq).toBe(seq);
  });
});
