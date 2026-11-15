import { fireEvent, render, screen, within } from "@testing-library/react";
import BusBoard from "./BusBoard";
import endView from "./fixtures/end.json";
import geometry from "./fixtures/geometry.json";
import midView from "./fixtures/mid.json";
import { parseGeometry } from "./geometry";
import { nodeInfo, siteInfo, streetInfo } from "./hoverInfo";
import { busCell, parseBusView, seatFill, type BusViewData } from "./model";
import { clampViewport, fitViewport, panBy, zoomAt } from "./viewport";

const geo = parseGeometry(geometry.geometry);
if (!geo) throw new Error("bad geometry fixture");
const mid = parseBusView(midView.view) as BusViewData;
const end = parseBusView(endView.view) as BusViewData;

const q = (c: HTMLElement, sel: string) => Array.from(c.querySelectorAll(sel));

describe("fixtures", () => {
  it("parse as Bus views", () => {
    expect(mid.state.phase).toBe("SEL");
    expect(end.state.phase).toBe("END");
    expect(geo.layout.bus_grid).toHaveLength(20);
  });
});

describe("BusBoard pieces (mid-game, 4 players)", () => {
  it("draws buildings, passengers, lines, markers and the clock from the view", () => {
    const { container } = render(<BusBoard geo={geo} data={mid} />);
    expect(q(container, ".building")).toHaveLength(mid.state.buildings.length);
    const kinds = (k: string) =>
      q(container, `.building[data-kind="${k}"]`).length;
    expect(kinds("house") + kinds("office") + kinds("pub")).toBe(mid.state.buildings.length);
    const occupied = mid.state.buildings.filter((b) => b.occupied).length;
    expect(q(container, '.building[data-occupied="true"] .passenger')).toHaveLength(occupied);
    // waiting passengers: one group per node, count kept; >= 4 collapses to one badged figure
    const groups = q(container, ".waiting");
    expect(groups).toHaveLength(Object.keys(mid.state.waiting).length);
    for (const g of groups) {
      const n = Number(g.getAttribute("data-count"));
      expect(n).toBe(mid.state.waiting[g.getAttribute("data-node") ?? ""]);
      expect(g.querySelectorAll(".passenger")).toHaveLength(n >= 4 ? 1 : n);
      expect(g.querySelectorAll(".count-badge")).toHaveLength(n >= 4 ? 1 : 0);
    }
    // line markers: one per street per seat, in the seat's colour
    mid.state.seats.forEach((seat, s) => {
      const lines = q(container, `.bus-line[data-seat="${s}"]`);
      expect(lines).toHaveLength(seat.line.length);
      for (const l of lines) expect(l.querySelectorAll("path")[1]?.getAttribute("stroke")).toBe(seatFill(s));
      expect(q(container, `.line-end[data-seat="${s}"], .line-loop[data-seat="${s}"]`).length).toBeGreaterThan(0);
    });
    // action markers
    const placed = Object.values(mid.state.action_board).flat().filter((x) => x !== null).length;
    expect(q(container, ".action-marker")).toHaveLength(placed);
    expect(container.querySelector(".clock-hand")?.getAttribute("data-clock")).toBe(mid.state.clock);
    // time stones: present rings plus taken ones
    expect(q(container, ".stone-present")).toHaveLength(mid.state.stones_on_clock);
    const held = mid.state.seats.reduce((a, s) => a + s.stones, 0);
    expect(q(container, ".stone-taken")).toHaveLength(held);
    expect(container.querySelector(".start-flag")?.textContent).toContain(`P${mid.state.start_player}`);
  });

  it("puts buses on the grid by seat column, starting bus not drawn", () => {
    const { container } = render(<BusBoard geo={geo} data={mid} />);
    const expected = mid.state.seats.reduce((a, s) => a + (s.buses - 1), 0);
    const pieces = q(container, ".bus-piece");
    expect(pieces).toHaveLength(expected);
    for (const el of pieces) {
      const seat = Number(el.getAttribute("data-seat"));
      const bus = Number(el.getAttribute("data-bus"));
      expect(bus).toBeGreaterThanOrEqual(2);
      const cell = busCell(geo, seat, bus);
      expect(cell).toEqual(geo.layout.bus_grid[(bus - 2) * 5 + seat]);
      const rect = el.querySelector("rect");
      expect(Number(rect?.getAttribute("x"))).toBe((cell?.[0] ?? 0) - 27);
      expect(el.querySelector(".bus-glyph")?.getAttribute("fill")).toBe(seatFill(seat));
    }
  });

  it("uses the seat colours red, green, orange, blue and white", () => {
    expect([0, 1, 2, 3, 4].map(seatFill)).toEqual(["#FE0000", "#007F00", "#FE6500", "#0065FE", "#FEFEFE"]);
  });

  it("labels seats P0.. and never Player 1", () => {
    const { container } = render(<BusBoard geo={geo} data={mid} />);
    expect(container.textContent).not.toMatch(/Player \d/);
    expect(container.textContent).toContain("P0");
  });
});

describe("BusBoard END state", () => {
  it("renders the final position with scores on the track", () => {
    const { container } = render(<BusBoard geo={geo} data={end} />);
    expect(q(container, ".building")).toHaveLength(end.state.buildings.length);
    const scoring = end.state.seats.filter((s) => s.points > 0).length;
    expect(q(container, ".score-disc")).toHaveLength(scoring);
    expect(q(container, ".start-flag")).toHaveLength(1);
  });

  it("stacks tied score discs side by side", () => {
    const data: BusViewData = JSON.parse(JSON.stringify(end));
    data.state.seats.forEach((s) => (s.points = 3));
    const { container } = render(<BusBoard geo={geo} data={data} />);
    const xs = q(container, ".score-disc circle").map((c) => Number(c.getAttribute("cx")));
    expect(new Set(xs).size).toBe(data.state.seats.length);
  });
});

describe("zoom and pan", () => {
  it("pure viewport maths stays in bounds", () => {
    const size = 1991;
    let v = fitViewport(size);
    v = zoomAt(v, size, 10);
    expect(v.scale).toBe(2.5);
    v = panBy(v, size, 5, 5);
    const half = size / 2.5 / 2;
    expect(v.cx).toBe(half);
    expect(v.cy).toBe(half);
    expect(zoomAt(v, size, 0.01).scale).toBe(1);
    expect(clampViewport({ scale: 1, cx: 0, cy: 0 }, size)).toEqual({ scale: 1, cx: size / 2, cy: size / 2 });
  });

  it("buttons change the viewBox and Fit resets", () => {
    render(<BusBoard geo={geo} data={mid} />);
    const svg = screen.getByTestId("bus-svg");
    expect(svg.getAttribute("data-scale")).toBe("1");
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(Number(svg.getAttribute("data-scale"))).toBeGreaterThan(1);
    expect(svg.getAttribute("viewBox")).not.toBe(`0 0 ${geo.map.width} ${geo.map.height}`);
    expect(screen.getByRole("button", { name: "Zoom out" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Reset zoom" }));
    expect(svg.getAttribute("data-scale")).toBe("1");
    expect(svg.getAttribute("viewBox")).toBe(`0 0 ${geo.map.width} ${geo.map.height}`);
    expect(screen.getByRole("button", { name: "Zoom out" })).toBeDisabled();
  });

  it("wheel zooms and drag pans", () => {
    // jsdom has no PointerEvent; mouse events carry the coordinates the handlers read
    vi.stubGlobal("PointerEvent", class extends MouseEvent {});
    render(<BusBoard geo={geo} data={mid} />);
    const svg = screen.getByTestId("bus-svg");
    fireEvent.wheel(svg, { deltaY: -100 });
    expect(Number(svg.getAttribute("data-scale"))).toBeGreaterThan(1);
    const rectSpy = vi.spyOn(svg, "getBoundingClientRect").mockReturnValue({
      x: 0, y: 0, left: 0, top: 0, right: 500, bottom: 500, width: 500, height: 500, toJSON: () => ({}),
    });
    const before = Number(svg.getAttribute("data-cx"));
    fireEvent.pointerDown(svg, { clientX: 250, clientY: 250, button: 0 });
    fireEvent.pointerMove(svg, { clientX: 150, clientY: 250 });
    fireEvent.pointerUp(svg);
    expect(Number(svg.getAttribute("data-cx"))).toBeGreaterThan(before);
    rectSpy.mockRestore();
    vi.unstubAllGlobals();
  });
});

describe("hover info", () => {
  it("describes nodes, sites and streets from the view", () => {
    const waitingNode = Number(Object.keys(mid.state.waiting)[0]);
    const text = nodeInfo(geo, mid, waitingNode).join("\n");
    expect(text).toContain(`Intersection ${waitingNode}`);
    expect(text).toContain(`Waiting passengers: ${mid.state.waiting[String(waitingNode)]}`);
    const b = mid.state.buildings.find((x) => x.occupied) ?? mid.state.buildings[0];
    expect(b).toBeDefined();
    expect(siteInfo(geo, mid, b?.site ?? 0).join(" ")).toContain(b?.type ?? "");
    expect(siteInfo(geo, mid, 12).join(" ")).toMatch(/No building|tier/);
    const [sid, users] = Object.entries(mid.derived.street_lines)[0] ?? ["0", []];
    expect(streetInfo(mid, Number(sid)).join(" ")).toContain(`P${users[0]}`);
    expect(streetInfo(mid, 9999).join(" ")).toContain("No line");
  });

  it("shows a tooltip while hovering a street, site and marker", () => {
    const { container } = render(<BusBoard geo={geo} data={mid} />);
    expect(screen.queryByTestId("bus-tooltip")).toBeNull();
    const sid = Object.keys(mid.derived.street_lines)[0];
    fireEvent.mouseEnter(container.querySelector(`#street-${sid}`) as Element);
    const tip = screen.getByTestId("bus-tooltip");
    expect(within(tip).getByText(`Street ${sid}`)).toBeInTheDocument();
    fireEvent.mouseLeave(container.querySelector(`#street-${sid}`) as Element);
    expect(screen.queryByTestId("bus-tooltip")).toBeNull();
    fireEvent.mouseEnter(container.querySelector("#site-0") as Element);
    expect(screen.getByTestId("bus-tooltip")).toHaveTextContent("Site 0");
    fireEvent.mouseEnter(container.querySelector(".action-marker") as Element);
    expect(screen.getByTestId("bus-tooltip")).toHaveTextContent(/marker/);
  });
});
