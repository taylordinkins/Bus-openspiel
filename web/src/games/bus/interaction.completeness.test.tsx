import { beforeEach, describe, expect, it, vi } from "vitest";
import { resetGeometryCache } from "./geometry";
import {
  ALL_FIXTURES, HAND, PHASES, clickPathFor, geometryResponse, inner, legalEls, mountReady, idsOf, barButton,
  fireEvent, type Fx,
} from "./interactionHarness";
import { cleanup } from "@testing-library/react";

const GET = vi.fn();
vi.mock("../../api/client", () => ({ api: { GET: (...a: unknown[]) => GET(...a) } }));

beforeEach(() => {
  GET.mockReset();
  GET.mockResolvedValue(geometryResponse());
  resetGeometryCache();
});

const playable = ALL_FIXTURES.filter(([k]) => k !== "phases/END");

/** Every id in view.legal_actions, with a fixture label, for it.each. */
const cases: [string, Fx, number][] = playable.flatMap(([name, f]) =>
  f.legal_actions.map((a): [string, Fx, number] => [`${name} #${a.id} ${a.kind}`, f, a.id]),
);

describe("unit8 (6) completeness: every legal action is reachable by clicking (UI_DESIGN 8, ACTIONS 5)", () => {
  it("unit8: the generator-recorded fixtures give each non-END phase code a multi-action view", () => {
    for (const [code, f] of Object.entries(PHASES)) {
      if (code === "END") continue;
      expect(f.legal_actions.length, `${code} fixture needs >= 2 legal actions`).toBeGreaterThan(1);
    }
  });

  it("unit8: each phase code (SB L1 L2 SEL LN PX BD CK VP VD) has a view in the fixture set, in the phase's own prompt", () => {
    for (const code of ["SB", "L1", "L2", "SEL", "LN", "PX", "BD", "CK", "VP", "VD"]) {
      expect(inner(PHASES[code]!).prompt.phase).toBe(code);
    }
  });

  it("unit8: the fixture set has more than 40 legal actions to click through (guard against empty parametrisation)", () => {
    expect(cases.length).toBeGreaterThan(40);
  });

  it.each(playable)("unit8: %s - legal_actions is non-empty and covered exactly once by targets (6.4)", (_n, f) => {
    const t = inner(f).targets;
    const ids = f.legal_actions.map((a) => a.id);
    expect(ids.length).toBeGreaterThan(0);
    const seen = [
      ...Object.values(t.streets).flat(), ...Object.values(t.sites).flat(), ...Object.values(t.stations),
      ...Object.values(t.nodes), ...Object.values(t.actions),
      ...(t.clock.advance !== null ? [t.clock.advance] : []), ...(t.clock.pause !== null ? [t.clock.pause] : []),
      ...t.buttons.map((b) => b.id),
    ];
    expect([...seen].sort((a, b) => a - b)).toEqual([...ids].sort((a, b) => a - b));
  });

  it.each(cases)("unit8: %s posts exactly its own id by a click path", async (_n, f, id) => {
    const a = f.legal_actions.find((x) => x.id === id)!;
    const m = await mountReady(f);
    await clickPathFor(f, a, m.container);
    expect(m.onAct.mock.calls).toEqual([[id]]);
    cleanup();
  }, 15000);

  it.each(playable)("unit8: %s - every legal board target lists only legal ids and each legal id sits on some legal target or bar button", async (_n, f) => {
    const m = await mountReady(f);
    const onBoard = new Set(legalEls(m.container).flatMap(idsOf));
    const legalIds = new Set(f.legal_actions.map((a) => a.id));
    for (const id of onBoard) expect(legalIds.has(id)).toBe(true);
    const barIds = new Set(inner(f).targets.buttons.map((b) => b.id));
    for (const id of legalIds) expect(onBoard.has(id) || barIds.has(id), `id ${id} unreachable`).toBe(true);
    cleanup();
  });

  it("unit8: the same CK ids are also reachable from bar buttons", async () => {
    const f = HAND["CK.notlast"]!;
    const m = await mountReady(f);
    fireEvent.click(barButton(/advance/i));
    fireEvent.click(barButton(/take stone/i));
    expect(m.onAct.mock.calls).toEqual([[291], [292]]);
  });
});

describe("unit8 (6b) fixture contract: UI_DESIGN 6.7 fields exist in the recorded fixtures", () => {
  it("unit8: every CK view carries prompt.context.last_stone (boolean) equal to the pause action's ends_game", () => {
    for (const [name, f] of ALL_FIXTURES.filter(([, x]) => inner(x).prompt.phase === "CK")) {
      const last = inner(f).prompt.context.last_stone;
      expect(typeof last, `${name}: last_stone`).toBe("boolean");
      const pause = f.legal_actions.find((a) => a.kind === "clock_pause")!;
      expect(last, name).toBe(pause.params.ends_game);
    }
  });

  it("unit8: every LN view has targets.street_ends parallel to targets.streets with end/node per extend id", () => {
    for (const [name, f] of ALL_FIXTURES.filter(([, x]) => ["LN", "L2", "L1"].includes(inner(x).prompt.phase))) {
      const t = inner(f).targets;
      expect(t.street_ends, `${name}: street_ends`).toBeDefined();
      for (const [sid, ids] of Object.entries(t.streets)) {
        const ends = t.street_ends![sid] ?? [];
        expect(ends.map((e) => e.id), `${name} s${sid}`).toEqual(ids);
        for (const e of ends) {
          const a = f.legal_actions.find((x) => x.id === e.id)!;
          expect(e.end).toBe(a.params.end);
          expect(e.node).toBe(a.params.from_node);
        }
      }
    }
  });

  it("unit8: every extend action has params.end of 0 or 1", () => {
    for (const [, f] of ALL_FIXTURES)
      for (const a of f.legal_actions.filter((x) => x.kind === "extend")) expect([0, 1]).toContain(a.params.end);
  });

  it("unit8: SEL - a drop_out button target exists iff drop_out is legal, and mandatory is false iff it is legal", () => {
    for (const [name, f] of ALL_FIXTURES.filter(([, x]) => inner(x).prompt.phase === "SEL")) {
      const legal = f.legal_actions.some((a) => a.kind === "drop_out");
      const buttons = inner(f).targets.buttons.filter((b) => b.kind === "drop_out");
      expect(buttons.length, name).toBe(legal ? 1 : 0);
      expect(inner(f).prompt.context.mandatory, name).toBe(!legal);
    }
  });

  it("unit8: board-full SEL (only Drop out legal) still reports mandatory=false", () => {
    const f = HAND["SEL.dropout_only"]!;
    expect(f.legal_actions.map((a) => a.kind)).toEqual(["drop_out"]);
    expect(inner(f).prompt.context.mandatory).toBe(false);
  });

  it("unit8: VD views carry picked_node in prompt.context and in state.pending", () => {
    for (const [name, f] of ALL_FIXTURES.filter(([, x]) => inner(x).prompt.phase === "VD")) {
      const pc = inner(f).prompt.context.picked_node;
      expect(typeof pc, name).toBe("number");
      expect((inner(f).state.pending as { picked_node?: number }).picked_node, name).toBe(pc);
    }
  });

  it("unit8: VP views have picked_node null", () => {
    for (const [name, f] of ALL_FIXTURES.filter(([, x]) => inner(x).prompt.phase === "VP"))
      expect(inner(f).prompt.context.picked_node, name).toBeNull();
  });
});
