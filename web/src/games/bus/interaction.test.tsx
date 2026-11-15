import { beforeEach, describe, expect, it, vi } from "vitest";
import { resetGeometryCache } from "./geometry";
import {
  HAND, MID, PHASES, barButton, byId, fireEvent, geometryResponse, idsOf, inner, legalEls, mountReady, q,
  expectedLegalSelectors, screen, within,
} from "./interactionHarness";
import { seatFill } from "./model";

const GET = vi.fn();
vi.mock("../../api/client", () => ({ api: { GET: (...a: unknown[]) => GET(...a) } }));

beforeEach(() => {
  GET.mockReset();
  GET.mockResolvedValue(geometryResponse());
  resetGeometryCache();
});

const SEL_ = (id: string) => `[data-testid="${id}"]`;

describe("unit8 (1) only legal targets glow and respond (UI_DESIGN 1.2, 6.7e)", () => {
  it("unit8: LN lights exactly the streets in targets.streets and nothing else", async () => {
    const f = PHASES.LN!;
    const { container } = await mountReady(f);
    const lit = legalEls(container);
    const want = expectedLegalSelectors(f);
    expect(lit.map((e) => e.id).filter(Boolean).sort()).toEqual(want.map((s) => s.slice(1)).sort());
    for (const e of lit) {
      expect(e.getAttribute("class")).toMatch(/\bglow\b/);
      expect(idsOf(e).length).toBeGreaterThan(0);
    }
    expect(q(container, ".street.hit")).toHaveLength(70);
  });

  it("unit8: a click on an illegal street, site or node posts nothing", async () => {
    const f = PHASES.LN!;
    const { container, onAct } = await mountReady(f);
    const legal = new Set(Object.keys(inner(f).targets.streets));
    const dead = q(container, ".street.hit").find((e) => !legal.has(e.getAttribute("data-street") ?? ""))!;
    expect(dead.getAttribute("data-legal")).not.toBe("true");
    expect(dead.getAttribute("class")).not.toMatch(/\bglow\b/);
    fireEvent.click(dead);
    fireEvent.click(byId(container, "#site-0"));
    fireEvent.click(byId(container, "#node-0"));
    expect(onAct).not.toHaveBeenCalled();
  });

  it.each(Object.entries(PHASES).filter(([k]) => k !== "END"))("unit8: %s glow set equals the view's targets", async (_k, f) => {
    const { container } = await mountReady(f);
    const lit = legalEls(container).map((e) => e.id || e.getAttribute("data-testid"));
    const want = expectedLegalSelectors(f).map((s) => (s.startsWith("#") ? s.slice(1) : s.slice(14, -2)));
    expect(lit.sort()).toEqual(want.sort());
  });

  it("unit8: SEL lights the action boxes in targets.actions only (mid-game pass 1: no Starting Player gap, no Clock)", async () => {
    const { container, onAct } = await mountReady(MID);
    const legalNames = Object.keys(inner(MID).targets.actions);
    expect(legalNames).not.toContain("clock");
    expect(legalEls(container)).toHaveLength(legalNames.length);
    const clockBox = byId(container, SEL_("target-action-clock"));
    expect(clockBox.getAttribute("data-legal")).not.toBe("true");
    fireEvent.click(clockBox);
    expect(onAct).not.toHaveBeenCalled();
  });

  it("unit8: a bot seat or a request in flight (canAct=false) makes nothing legal or clickable", async () => {
    const { container, onAct } = await mountReady(PHASES.SB!, { canAct: false });
    expect(legalEls(container)).toHaveLength(0);
    fireEvent.click(byId(container, "#site-3"));
    expect(screen.queryByTestId("type-popover")).toBeNull();
    expect(onAct).not.toHaveBeenCalled();
  });

  it("unit8: canAct=false disables the bar buttons (SEL Drop out)", async () => {
    const { onAct } = await mountReady(PHASES.SEL!, { canAct: false });
    const btn = within(screen.getByTestId("bus-prompt")).queryByRole("button", { name: /drop out/i });
    if (btn) {
      expect(btn).toBeDisabled();
      fireEvent.click(btn);
    }
    expect(onAct).not.toHaveBeenCalled();
  });

  it("unit8: LN loop - handles are legal and listed only for ends with legal ids (R-041)", async () => {
    const f = HAND["LN.loop"]!;
    const { container } = await mountReady(f);
    const h0 = byId(container, SEL_("handle-0"));
    const h1 = byId(container, SEL_("handle-1"));
    expect(h0.getAttribute("data-legal")).toBe("true");
    expect(h1.getAttribute("data-legal")).toBe("true");
    expect(idsOf(h0).sort()).toEqual([52, 53]);
    expect(idsOf(h1).sort()).toEqual([117, 122, 123]);
  });
});

describe("unit8 (2) hover ghosts and previews (UI_DESIGN 1.5)", () => {
  const ghost = () => screen.queryByTestId("ghost");

  it("unit8: L1 hover on a street shows a street ghost in the actor's colour; leaving removes it", async () => {
    const f = PHASES.L1!;
    const { container } = await mountReady(f);
    const st = byId(container, "#street-12");
    fireEvent.mouseEnter(st);
    const g = ghost()!;
    expect(g).not.toBeNull();
    expect(g.getAttribute("data-kind")).toBe("street");
    expect(g.getAttribute("data-street")).toBe("12");
    expect(g.getAttribute("data-seat")).toBe(String(inner(f).prompt.actor));
    expect(g.outerHTML.toLowerCase()).toContain(seatFill(inner(f).prompt.actor!).toLowerCase());
    fireEvent.mouseLeave(st);
    expect(ghost()).toBeNull();
  });

  it("unit8: hovering an illegal street (LN) shows no ghost", async () => {
    const { container } = await mountReady(PHASES.LN!);
    fireEvent.mouseEnter(byId(container, "#street-0"));
    expect(ghost()).toBeNull();
  });

  it("unit8: SB hover on a site shows a building ghost", async () => {
    const { container } = await mountReady(PHASES.SB!);
    fireEvent.mouseEnter(byId(container, "#site-3"));
    expect(ghost()!.getAttribute("data-kind")).toBe("building");
    expect(ghost()!.getAttribute("data-site")).toBe("3");
  });

  it("unit8: PX hover on a station shows a passenger ghost", async () => {
    const { container } = await mountReady(PHASES.PX!);
    fireEvent.mouseEnter(byId(container, "#node-5"));
    expect(ghost()!.getAttribute("data-kind")).toBe("passenger");
    expect(ghost()!.getAttribute("data-node")).toBe("5");
  });

  it("unit8: SEL hover on an action box ghosts the marker on the letter it would take (params.space)", async () => {
    const f = PHASES.SEL!;
    const { container } = await mountReady(f);
    const lines = f.legal_actions.find((a) => a.id === 0)!;
    fireEvent.mouseEnter(byId(container, SEL_("target-action-lines")));
    const g = ghost()!;
    expect(g.getAttribute("data-kind")).toBe("marker");
    expect(g.getAttribute("data-action")).toBe("lines");
    expect(g.getAttribute("data-letter")).toBe("abcdef"[lines.params.space as number]);
  });

  it("unit8: CK hover on Advance shows the clock hand at next_clock", async () => {
    const f = PHASES.CK!;
    const { container } = await mountReady(f);
    fireEvent.mouseEnter(byId(container, SEL_("target-clock-advance")));
    expect(ghost()!.getAttribute("data-kind")).toBe("clock-hand");
    expect(ghost()!.getAttribute("data-clock")).toBe(inner(f).prompt.context.next_clock);
  });

  it("unit8: VD hover on a deliver site draws a dashed path from the picked node", async () => {
    const f = HAND["VD.multi"]!;
    const { container } = await mountReady(f);
    fireEvent.mouseEnter(byId(container, "#site-8"));
    const g = ghost()!;
    expect(g.getAttribute("data-kind")).toBe("path");
    expect(g.getAttribute("data-from")).toBe("16");
    expect(g.getAttribute("data-to")).toBe("8");
    expect(g.outerHTML).toMatch(/stroke-dasharray/);
  });

  it("unit8: hovering does not post anything", async () => {
    const { container, onAct } = await mountReady(PHASES.SB!);
    fireEvent.mouseEnter(byId(container, "#site-3"));
    fireEvent.mouseLeave(byId(container, "#site-3"));
    expect(onAct).not.toHaveBeenCalled();
  });
});

describe("unit8 (3) building type popover (UI_DESIGN 4: SB, BD)", () => {
  it.each(["SB", "BD"])("unit8: %s site click opens a popover with house, office, pub and posts nothing yet", async (k) => {
    const f = PHASES[k]!;
    const { container, onAct } = await mountReady(f);
    const site = Object.keys(inner(f).targets.sites)[0]!;
    fireEvent.click(byId(container, `#site-${site}`));
    const pop = await screen.findByTestId("type-popover");
    expect(pop).toHaveAttribute("role", "menu");
    const items = within(pop).getAllByRole("menuitem");
    expect(items.map((i) => i.textContent?.toLowerCase())).toEqual([
      expect.stringContaining("house"), expect.stringContaining("office"), expect.stringContaining("pub"),
    ]);
    expect(items.map((i) => Number(i.getAttribute("data-action-id")))).toEqual(inner(f).targets.sites[site]);
    expect(onAct).not.toHaveBeenCalled();
  });

  it("unit8: choosing a type posts that building's id exactly once", async () => {
    const { container, onAct } = await mountReady(PHASES.BD!); // site 8 -> [172 house, 173 office, 174 pub]
    fireEvent.click(byId(container, "#site-8"));
    fireEvent.click(within(await screen.findByTestId("type-popover")).getByRole("menuitem", { name: /office/i }));
    expect(onAct.mock.calls).toEqual([[173]]);
  });

  it("unit8: Escape closes the popover without posting", async () => {
    const { container, onAct } = await mountReady(PHASES.BD!);
    fireEvent.click(byId(container, "#site-8"));
    await screen.findByTestId("type-popover");
    fireEvent.keyDown(document.activeElement ?? document.body, { key: "Escape" });
    expect(screen.queryByTestId("type-popover")).toBeNull();
    expect(onAct).not.toHaveBeenCalled();
  });

  it("unit8: clicking another legal site moves the popover there (one popover at a time)", async () => {
    const { container, onAct } = await mountReady(PHASES.BD!);
    fireEvent.click(byId(container, "#site-8"));
    await screen.findByTestId("type-popover");
    fireEvent.click(byId(container, "#site-9"));
    const pops = screen.getAllByTestId("type-popover");
    expect(pops).toHaveLength(1);
    expect(within(pops[0]!).getAllByRole("menuitem").map((i) => Number(i.getAttribute("data-action-id")))).toEqual([175, 176, 177]);
    expect(onAct).not.toHaveBeenCalled();
  });

  it("unit8: a click on an illegal site opens no popover", async () => {
    const { container } = await mountReady(PHASES.BD!);
    fireEvent.click(byId(container, "#site-0"));
    expect(screen.queryByTestId("type-popover")).toBeNull();
  });

  it("unit8: deliver_to sites (VD) post at once and never open a popover", async () => {
    const { container, onAct } = await mountReady(HAND["VD.multi"]!);
    fireEvent.click(byId(container, "#site-17"));
    expect(screen.queryByTestId("type-popover")).toBeNull();
    expect(onAct.mock.calls).toEqual([[346]]);
  });
});

describe("unit8 (3b) LN loop end choice (UI_DESIGN 4 LN, R-041; 6.7a)", () => {
  it("unit8: a one-id street posts on the first click", async () => {
    const { container, onAct } = await mountReady(HAND["LN.loop"]!);
    fireEvent.click(byId(container, "#street-39"));
    expect(onAct.mock.calls).toEqual([[117]]);
  });

  it("unit8: street first then handle posts that end's id; the street click alone posts nothing", async () => {
    const { container, onAct } = await mountReady(HAND["LN.loop"]!);
    fireEvent.click(byId(container, "#street-45"));
    expect(onAct).not.toHaveBeenCalled();
    expect(byId(container, SEL_("handle-0")).getAttribute("data-pending")).toBe("true");
    expect(byId(container, SEL_("handle-1")).getAttribute("data-pending")).toBe("true");
    fireEvent.click(byId(container, SEL_("handle-1")));
    expect(onAct.mock.calls).toEqual([[123]]);
  });

  it("unit8: handle first then street posts that end's id", async () => {
    const { container, onAct } = await mountReady(HAND["LN.loop"]!);
    fireEvent.click(byId(container, SEL_("handle-0")));
    expect(onAct).not.toHaveBeenCalled();
    fireEvent.click(byId(container, "#street-44"));
    expect(onAct.mock.calls).toEqual([[52]]);
  });

  it("unit8: after arming end 0, a street legal only from end 1 is not legal and does not post", async () => {
    const { container, onAct } = await mountReady(HAND["LN.loop"]!);
    fireEvent.click(byId(container, SEL_("handle-0")));
    const s39 = byId(container, "#street-39");
    expect(s39.getAttribute("data-legal")).not.toBe("true");
    fireEvent.click(s39);
    expect(onAct).not.toHaveBeenCalled();
  });
});

describe("unit8 (4) VP to VD (6.7d)", () => {
  it("unit8: a VP node click posts the pick id at once - no popover, no confirm, no local picked marker", async () => {
    const { container, onAct } = await mountReady(HAND["VP.multi"]!);
    fireEvent.click(byId(container, "#node-20"));
    expect(onAct.mock.calls).toEqual([[313]]);
    expect(screen.queryByTestId("type-popover")).toBeNull();
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(screen.queryByTestId("picked-node")).toBeNull();
  });

  it("unit8: VP lights pickable nodes only, not sites", async () => {
    const { container } = await mountReady(HAND["VP.multi"]!);
    expect(legalEls(container).map((e) => e.id).sort()).toEqual(["node-16", "node-19", "node-20"]);
  });

  it("unit8: the VD view shows the picked node marker from prompt.context.picked_node and lights deliver sites only", async () => {
    const f = HAND["VD.multi"]!;
    const { container } = await mountReady(f);
    const m = screen.getByTestId("picked-node");
    expect(m.getAttribute("data-node")).toBe(String(inner(f).prompt.context.picked_node));
    expect(legalEls(container).map((e) => e.id).sort()).toEqual(["site-17", "site-39", "site-8"]);
    fireEvent.click(byId(container, "#node-16")); // the picked node is not a legal target in VD
    expect(screen.queryByTestId("picked-node")).not.toBeNull();
  });

  it("unit8: the picked marker follows the next view, not the click (VP view then VD view)", async () => {
    const m = await mountReady(HAND["VP.multi"]!);
    expect(screen.queryByTestId("picked-node")).toBeNull();
    fireEvent.click(byId(m.container, "#node-16"));
    expect(screen.queryByTestId("picked-node")).toBeNull();
    m.rerenderFx(HAND["VD.multi"]!);
    expect(screen.getByTestId("picked-node").getAttribute("data-node")).toBe("16");
  });

  it("unit8: VD deliver click posts the id of that site", async () => {
    const { container, onAct } = await mountReady(HAND["VD.multi"]!);
    fireEvent.click(byId(container, "#site-39"));
    expect(onAct.mock.calls).toEqual([[368]]);
  });
});

describe("unit8 (4b) bar buttons and the last-stone confirm (UI_DESIGN 4 SEL, CK; 6.7b, 6.7c)", () => {
  it("unit8: SEL pass >= 2 has a Drop out button that posts id 7 once", async () => {
    const { onAct } = await mountReady(PHASES.SEL!);
    fireEvent.click(barButton(/drop out/i));
    expect(onAct.mock.calls).toEqual([[7]]);
  });

  it("unit8: SEL mandatory pass has no Drop out button (mid.json: targets.buttons empty)", async () => {
    await mountReady(MID);
    expect(within(screen.getByTestId("bus-prompt")).queryByRole("button", { name: /drop out/i })).toBeNull();
  });

  it("unit8: board-full SEL shows Drop out as the only way forward, enabled, never auto-posted", async () => {
    const { onAct } = await mountReady(HAND["SEL.dropout_only"]!);
    await new Promise((r) => setTimeout(r, 400));
    expect(onAct).not.toHaveBeenCalled();
    const b = barButton(/drop out/i);
    expect(b).toBeEnabled();
    fireEvent.click(b);
    expect(onAct.mock.calls).toEqual([[7]]);
  });

  it("unit8: CK Advance bar button posts 291 and names the next clock type", async () => {
    const { onAct } = await mountReady(PHASES.CK!);
    const b = barButton(/advance/i);
    expect(b.textContent).toMatch(/office/i);
    fireEvent.click(b);
    expect(onAct.mock.calls).toEqual([[291]]);
  });

  it("unit8: CK Take stone (not last) posts 292 at once, from the bar and from the stones", async () => {
    const f = HAND["CK.notlast"]!;
    const a = await mountReady(f);
    expect(barButton(/take stone/i).textContent).toMatch(/5/); // "(5 left)" from stones_on_clock
    fireEvent.click(barButton(/take stone/i));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(a.onAct.mock.calls).toEqual([[292]]);
    a.onAct.mockClear();
    fireEvent.click(byId(a.container, SEL_("target-clock-pause")));
    expect(a.onAct.mock.calls).toEqual([[292]]);
  });

  it.each(["bar", "board"])("unit8: taking the last stone (%s) asks first and posts only on Confirm", async (via) => {
    const f = HAND["CK.last"]!;
    const { container, onAct } = await mountReady(f);
    const open = () =>
      via === "bar" ? fireEvent.click(barButton(/take stone/i)) : fireEvent.click(byId(container, SEL_("target-clock-pause")));
    open();
    const dlg = await screen.findByRole("alertdialog");
    expect(dlg.textContent).toMatch(/last stone ends the game/i);
    expect(onAct).not.toHaveBeenCalled();
    fireEvent.click(within(dlg).getByRole("button", { name: /cancel/i }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(onAct).not.toHaveBeenCalled();
    open();
    fireEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: /confirm/i }));
    expect(onAct.mock.calls).toEqual([[292]]);
  });

  it("unit8: Advance never asks for confirmation, even when the last stone is on the clock", async () => {
    const { container, onAct } = await mountReady(HAND["CK.last"]!);
    fireEvent.click(byId(container, SEL_("target-clock-advance")));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(onAct.mock.calls).toEqual([[291]]);
  });
});

describe("unit8 (4c) Back in the status bar (UI_DESIGN 7; 6.7e)", () => {
  it("unit8: the bar has a Back button that calls onBack and posts no action", async () => {
    const { onAct, onBack } = await mountReady(PHASES.LN!);
    fireEvent.click(barButton(/^back$/i));
    expect(onBack).toHaveBeenCalledTimes(1);
    expect(onAct).not.toHaveBeenCalled();
  });

  it("unit8: Back is disabled when canBack is false (start of game, END, request in flight)", async () => {
    const { onBack } = await mountReady(PHASES.LN!, { canBack: false });
    const b = barButton(/^back$/i);
    expect(b).toBeDisabled();
    fireEvent.click(b);
    expect(onBack).not.toHaveBeenCalled();
  });
});
