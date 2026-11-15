import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import type { BoardMotion } from "../registry";
import BusBoard from "./BusBoard";
import geometry from "./fixtures/geometry.json";
import midView from "./fixtures/mid.json";
import { parseGeometry } from "./geometry";
import { parseBusView, type BusViewData } from "./model";
import { deliveryPath, freshKeys, motionLength, motionPlan, pointAlong, T_MOVE } from "./motion";

const geo = parseGeometry(geometry.geometry)!;
const mid = parseBusView(midView.view) as BusViewData;
const clone = (d: BusViewData): BusViewData => JSON.parse(JSON.stringify(d)) as BusViewData;

/** Seat 0's line: a delivery from its first node to a site on its last node. */
const line0 = mid.derived.seats[0]!.line_nodes;
const from = line0[0]!;
const dest = geo.sites.find((s) => s.node === line0[line0.length - 1])!;

describe("unit9 motion plan (UI_DESIGN 3 motion, 6.6 events)", () => {
  it("fresh keys are the pieces new since the previous view", () => {
    const next = clone(mid);
    const empty = Object.entries(next.state.action_board).find(([, sp]) => sp.includes(null))!;
    const k = empty[1].indexOf(null);
    empty[1][k] = 1;
    next.state.seats[1]!.buses += 1;
    const fresh = freshKeys(mid, next);
    expect([...fresh].sort()).toEqual(
      [`bus1-${next.state.seats[1]!.buses}`, `m${empty[0]}-${k}:1`].sort(),
    );
    expect(freshKeys(null, next).size).toBe(0);
    expect(freshKeys(mid, mid).size).toBe(0);
  });

  it("a delivery walks the scorer's line street by street into the building", () => {
    const pts = deliveryPath(geo, mid, 0, from, dest.id);
    const node = (id: number) => [geo.nodes[id]!.x, geo.nodes[id]!.y];
    expect(pts[0]).toEqual(node(from));
    expect(pts[pts.length - 1]).toEqual([dest.x, dest.y]);
    for (const n of line0) expect(pts).toContainEqual(node(n)); // every node of the route, in order
    const idx = line0.map((n) => pts.findIndex((p) => p[0] === node(n)[0] && p[1] === node(n)[1]));
    expect([...idx].sort((a, b) => a - b)).toEqual(idx);
  });

  it("a delivery off the line falls back to a straight path", () => {
    const off = geo.sites.find((s) => !line0.includes(s.node))!;
    expect(deliveryPath(geo, mid, 0, from, off.id)).toHaveLength(2);
  });

  it("scored -> a walk and a +1 float; passengers_moved -> exits then entries", () => {
    const plan = motionPlan(
      geo,
      mid,
      [
        { type: "scored", seat: 0, points: 3, node: from, site: dest.id },
        {
          type: "passengers_moved",
          exited: [{ site: 3, node: geo.sites[3]!.node }],
          entered: [{ site: 0, node: geo.sites[0]!.node }],
        },
        { type: "round_started", round: 2 },
      ],
      1,
      7,
    );
    expect(plan.walks.map((w) => w.kind)).toEqual(["deliver", "exit", "enter"]);
    expect(plan.walks[0]).toMatchObject({ seat: 0, site: dest.id, delay: 0, duration: T_MOVE });
    expect(plan.walks[2]!.delay).toBe(plan.walks[1]!.duration); // entries follow the exits
    expect(plan.floats).toEqual([expect.objectContaining({ seat: 0, site: dest.id, points: 3 })]);
    expect(motionLength(plan, 1)).toBeGreaterThan(T_MOVE);
    const fast = motionPlan(geo, mid, [{ type: "scored", seat: 0, points: 3, node: from, site: dest.id }], 0.4, 8);
    expect(fast.walks[0]!.duration).toBeCloseTo(T_MOVE * 0.4);
  });

  it("pointAlong interpolates by length", () => {
    const pts: [number, number][] = [
      [0, 0],
      [10, 0],
      [10, 30],
    ];
    expect(pointAlong(pts, 0)).toEqual([0, 0]);
    expect(pointAlong(pts, 0.25)).toEqual([10, 0]);
    expect(pointAlong(pts, 0.625)).toEqual([10, 15]);
    expect(pointAlong(pts, 1)).toEqual([10, 30]);
  });
});

describe("unit9 BusBoard motion", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  const motion = (seq: number, events: BoardMotion["events"] = []): BoardMotion => ({
    seq,
    from: 0,
    to: 1,
    events,
    speed: "normal",
    scale: 1,
  });

  /** mid with seat 0's delivery done: a passenger gone from `from`, the destination occupied. */
  function delivered(): BusViewData {
    const next = clone(mid);
    next.state.seats[0]!.points += 1;
    const b = next.state.buildings.find((x) => x.site === dest.id);
    if (b) b.occupied = true;
    else next.state.buildings.push({ site: dest.id, type: "house", occupied: true });
    return next;
  }

  it("new pieces ease in only after an animated change", () => {
    const next = clone(mid);
    next.state.seats[1]!.buses += 1;
    const { container, rerender } = render(<BusBoard geo={geo} data={mid} motion={null} />);
    expect(container.querySelectorAll('[data-anim="in"]')).toHaveLength(0);
    rerender(<BusBoard geo={geo} data={next} motion={motion(1)} />);
    const anim = Array.from(container.querySelectorAll('[data-anim="in"]'));
    expect(anim.map((e) => e.getAttribute("class"))).toEqual(["bus-piece"]);
    expect(container.querySelector(".bus-board")?.getAttribute("data-motion")).toBe("normal");
  });

  it("animation off: the change is drawn at once, no walkers or floats", () => {
    const { container, rerender } = render(<BusBoard geo={geo} data={mid} motion={null} />);
    rerender(<BusBoard geo={geo} data={delivered()} motion={null} />);
    expect(container.querySelectorAll('[data-anim="in"]')).toHaveLength(0);
    expect(container.querySelectorAll('[data-testid="walker"]')).toHaveLength(0);
    expect(document.querySelectorAll('[data-testid="score-float"]')).toHaveLength(0);
    expect(container.querySelector(".bus-board")?.getAttribute("data-motion")).toBe("off");
  });

  it("a delivery: a walker on the street path, the occupant waits for it, a +1 floats, then all clear", () => {
    const { container, rerender, unmount } = render(<BusBoard geo={geo} data={mid} />);
    const ev = [{ type: "scored", seat: 0, points: mid.state.seats[0]!.points + 1, node: from, site: dest.id }];
    rerender(<BusBoard geo={geo} data={delivered()} motion={motion(1, ev)} />);
    const walker = container.querySelector('[data-testid="walker"]');
    expect(walker?.getAttribute("data-kind")).toBe("deliver");
    expect(walker?.getAttribute("data-seat")).toBe("0");
    expect(walker?.closest("[pointer-events]")?.getAttribute("pointer-events")).toBe("none");
    const occ = container.querySelector(`.building[data-site="${dest.id}"] .occupant`);
    expect(occ?.getAttribute("data-anim")).toBe("in");
    expect((occ as SVGGElement).style.animationDelay).toBe(`${T_MOVE}ms`);
    const float = document.querySelector('[data-testid="score-float"]');
    expect(float?.getAttribute("data-seat")).toBe("0");
    expect(float?.textContent).toBe("+1");
    act(() => {
      vi.advanceTimersByTime(3000);
    });
    expect(container.querySelectorAll('[data-testid="walker"]')).toHaveLength(0);
    expect(document.querySelectorAll('[data-testid="score-float"]')).toHaveLength(0);
    unmount();
  });

  it("legal targets stay clickable during motion (the motion layer takes no pointer events)", () => {
    const { container, rerender } = render(<BusBoard geo={geo} data={mid} />);
    const ev = [{ type: "scored", seat: 0, points: 1, node: from, site: dest.id }];
    rerender(<BusBoard geo={geo} data={delivered()} motion={motion(1, ev)} />);
    expect(container.querySelector('[data-testid="motion-layer"]')?.getAttribute("pointer-events")).toBe("none");
  });
});
