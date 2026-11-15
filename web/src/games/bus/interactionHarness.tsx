/** Shared helpers for the unit 8 interaction tests (UI_DESIGN 6.7). Not a test file. */
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { vi } from "vitest";
import type { GameView } from "../../shell/useGame";
import { getPlugin } from "../registry";
import geometry from "./fixtures/geometry.json";
import five from "./fixtures/five.json";
import mid from "./fixtures/mid.json";
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
import CKlast from "./fixtures/interaction/CK.last.json";
import CKnotlast from "./fixtures/interaction/CK.notlast.json";
import LNends from "./fixtures/interaction/LN.ends.json";
import LNloop from "./fixtures/interaction/LN.loop.json";
import SELdrop from "./fixtures/interaction/SEL.dropout_only.json";
import VDmulti from "./fixtures/interaction/VD.multi.json";
import VPmulti from "./fixtures/interaction/VP.multi.json";
import VPlone from "./fixtures/interaction/VP.lone.json";

export interface LegalFx {
  id: number;
  label: string;
  kind: string;
  params: Record<string, unknown>;
}
export interface Fx {
  seats: string[];
  history_length: number;
  current_player: number | null;
  legal_actions: LegalFx[];
  view: {
    legal_actions: LegalFx[];
    view: {
      state: Record<string, unknown>;
      derived: Record<string, unknown>;
      prompt: { phase: string; actor: number | null; context: Record<string, unknown> };
      targets: {
        streets: Record<string, number[]>;
        street_ends?: Record<string, { id: number; end: number; node: number }[]>;
        sites: Record<string, number[]>;
        stations: Record<string, number>;
        nodes: Record<string, number>;
        actions: Record<string, number>;
        clock: { advance: number | null; pause: number | null };
        buttons: { id: number; kind: string }[];
      };
    };
  };
}
const fx = (x: unknown): Fx => x as Fx;

/** Recorded by tools/gen_ui_fixtures.py, one per phase code. */
export const PHASES: Record<string, Fx> = {
  SB: fx(SB), L1: fx(L1), L2: fx(L2), SEL: fx(SEL), LN: fx(LN), PX: fx(PX), BD: fx(BD),
  CK: fx(CK), VP: fx(VP), VD: fx(VD), END: fx(END),
};
export const MID = fx(mid);
export const FIVE = fx(five);
/** Hand-edited copies (fixtures/interaction/) until the generator emits them. */
export const HAND: Record<string, Fx> = {
  "LN.ends": fx(LNends), "LN.loop": fx(LNloop), "CK.last": fx(CKlast), "CK.notlast": fx(CKnotlast),
  "SEL.dropout_only": fx(SELdrop), "VD.multi": fx(VDmulti), "VP.multi": fx(VPmulti), "VP.lone": fx(VPlone),
};
export const ALL_FIXTURES: [string, Fx][] = [
  ...Object.entries(PHASES).map(([k, v]): [string, Fx] => [`phases/${k}`, v]),
  ["mid", MID], ["five", FIVE],
  ...Object.entries(HAND).map(([k, v]): [string, Fx] => [`interaction/${k}`, v]),
];

export const GEOMETRY = geometry.geometry;
export const MAP_W = geometry.geometry.map.width;
/** Phase fixtures nest the adapter keys at view.view; mid/five (hand-made) put them in view. */
export const inner = (f: Fx): Fx["view"]["view"] => ((f.view as { view?: Fx["view"]["view"] }).view ?? (f.view as unknown as Fx["view"]["view"]));
/** mid.json is a bare view without the server envelope; seats default to humans. */
const HUMANS = ["human", "human", "human", "human", "human"];
export const gameOf = (f: Fx, extra: Partial<Record<string, unknown>> = {}): GameView =>
  ({ ...({ id: "g", game: "bus", seats: HUMANS, seat_types: HUMANS, bot_thinking: false, bot_error: null, dev_mode: false, history_length: 3 } as Record<string, unknown>), ...(f as unknown as Record<string, unknown>), ...extra }) as unknown as GameView;

/** Network mock shared by every interaction test file (see the vi.mock in each). */
export function geometryResponse() {
  return { data: { game: "bus", geometry: GEOMETRY } };
}

export function mount(f: Fx, opts: { canAct?: boolean; canBack?: boolean; onBack?: () => void } = {}) {
  const onAct = vi.fn();
  const onBack = opts.onBack ?? vi.fn();
  const game = gameOf(f);
  const ui = { hoverAction: null, canAct: opts.canAct ?? true, onAct, onBack, canBack: opts.canBack ?? true };
  const view = render(<>{getPlugin("bus").renderBoard(game, ui as never)}</>);
  return { ...view, onAct, onBack, game, ui, rerenderFx: (g: Fx) => view.rerender(<>{getPlugin("bus").renderBoard(gameOf(g), ui as never)}</>) };
}
export async function mountReady(f: Fx, opts: Parameters<typeof mount>[1] = {}) {
  const m = mount(f, opts);
  await screen.findByTestId("bus-svg");
  return m;
}

export const q = (c: ParentNode, sel: string): Element[] => Array.from(c.querySelectorAll(sel));
export const byId = (c: ParentNode, sel: string): Element => {
  const e = c.querySelector(sel);
  if (!e) throw new Error(`no element for ${sel}`);
  return e;
};
export const legalEls = (c: ParentNode): Element[] => q(c, '[data-legal="true"]');
export const idsOf = (e: Element): number[] =>
  (e.getAttribute("data-action-ids") ?? "").split(",").filter(Boolean).map(Number);

const actionKey = (name: string) => name.toLowerCase().replace(/[^a-z ]/g, "").replace(/ /g, "_");
export const actionTestId = (a: LegalFx) => `target-action-${actionKey(String(a.params.action_name))}`;

/** Selectors of the hit shapes the targets of a view must light, from view.targets only. */
export function expectedLegalSelectors(f: Fx): string[] {
  const t = inner(f).targets;
  return [
    ...Object.keys(t.streets).map((s) => `#street-${s}`),
    ...Object.keys(t.sites).map((s) => `#site-${s}`),
    ...Object.keys(t.stations).map((s) => `#node-${s}`),
    ...Object.keys(t.nodes).map((s) => `#node-${s}`),
    ...Object.keys(t.actions).map((n) => `[data-testid="target-action-${n}"]`),
    ...(t.clock.advance !== null ? ['[data-testid="target-clock-advance"]'] : []),
    ...(t.clock.pause !== null ? ['[data-testid="target-clock-pause"]'] : []),
  ];
}

export const promptBar = () => screen.getByTestId("bus-prompt");
export const barButton = (name: RegExp) => within(promptBar()).getByRole("button", { name });

/** Smallest dimension of a hit shape in map units (circle: 2r, rect: min(w, h), path: stroke-width). */
export function hitUnits(e: Element): number {
  const n = (a: string) => Number(e.getAttribute(a));
  switch (e.tagName.toLowerCase()) {
    case "circle": return 2 * n("r");
    case "rect": return Math.min(n("width"), n("height"));
    case "path": return n("stroke-width");
    default: throw new Error(`hit shape must be circle, rect or path, got ${e.tagName}`);
  }
}
export const REF_BOARD_PX = 640;
export const MIN_HIT_PX = 24;
export const unitsToPx = (u: number) => (u * REF_BOARD_PX) / MAP_W;

/**
 * Click path for one legal action: board target, then popover / handle / confirm as the contract
 * requires. Returns after the last click; the caller checks what was posted.
 */
export async function clickPathFor(f: Fx, a: LegalFx, container: HTMLElement): Promise<void> {
  const p = a.params;
  const t = inner(f).targets;
  const click = (e: Element) => fireEvent.click(e);
  switch (a.kind) {
    case "select": click(byId(container, `[data-testid="${actionTestId(a)}"]`)); return;
    case "drop_out": click(barButton(/drop out/i)); return;
    case "extend": {
      const street = byId(container, `#street-${String(p.street)}`);
      click(street);
      if ((t.streets[String(p.street)] ?? []).length > 1) click(byId(container, `[data-testid="handle-${String(p.end)}"]`));
      return;
    }
    case "build": {
      click(byId(container, `#site-${String(p.site)}`));
      if ((t.sites[String(p.site)] ?? []).length > 1) {
        const pop = await screen.findByTestId("type-popover", {}, { timeout: 5000 });
        click(within(pop).getByRole("menuitem", { name: new RegExp(String(p.building_type), "i") }));
      }
      return;
    }
    case "passenger_to": case "pick_passenger": click(byId(container, `#node-${String(p.node)}`)); return;
    case "deliver_to": click(byId(container, `#site-${String(p.site)}`)); return;
    case "clock_advance": click(byId(container, '[data-testid="target-clock-advance"]')); return;
    case "clock_pause": {
      click(byId(container, '[data-testid="target-clock-pause"]'));
      if (inner(f).prompt.context.last_stone === true) {
        const dlg = await screen.findByRole("alertdialog");
        click(within(dlg).getByRole("button", { name: /confirm/i }));
      }
      return;
    }
    default: throw new Error(`no click path known for kind ${a.kind}`);
  }
}

export { fireEvent, screen, waitFor, within };
