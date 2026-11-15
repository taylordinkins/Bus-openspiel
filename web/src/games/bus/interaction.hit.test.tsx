import { beforeEach, describe, expect, it, vi } from "vitest";
import { resetGeometryCache } from "./geometry";
import {
  ALL_FIXTURES, MIN_HIT_PX, geometryResponse, hitUnits, legalEls, mountReady, unitsToPx, REF_BOARD_PX, MAP_W, HAND,
  byId,
} from "./interactionHarness";
import { cleanup, screen, within } from "@testing-library/react";

const GET = vi.fn();
vi.mock("../../api/client", () => ({ api: { GET: (...a: unknown[]) => GET(...a) } }));

beforeEach(() => {
  GET.mockReset();
  GET.mockResolvedValue(geometryResponse());
  resetGeometryCache();
});

describe("unit8 (7) hit areas >= 24 px at fit zoom (UI_DESIGN 8, 6.7e)", () => {
  it("unit8: the scale helper maps the reference board to pixels (640 px for the whole map)", () => {
    expect(unitsToPx(MAP_W)).toBeCloseTo(REF_BOARD_PX);
    expect(unitsToPx(MAP_W * (MIN_HIT_PX / REF_BOARD_PX))).toBeCloseTo(MIN_HIT_PX);
  });

  it.each(ALL_FIXTURES.filter(([k]) => k !== "phases/END"))("unit8: %s - every legal target is at least 24 px", async (_n, f) => {
    const { container } = await mountReady(f);
    const lit = legalEls(container);
    if (f === HAND["SEL.dropout_only"]) {
      // Board-full SEL: the only legal action is the prompt bar's Drop out button, which has no board shape.
      expect(lit).toHaveLength(0);
      const btn = within(await screen.findByTestId("bus-prompt")).getByRole("button", { name: /drop out/i });
      expect(btn.tagName).toBe("BUTTON");
      expect(btn).toBeEnabled();
      cleanup();
      return;
    }
    expect(lit.length).toBeGreaterThan(0);
    for (const e of lit) {
      const px = unitsToPx(hitUnits(e));
      expect(px, `${e.id || e.getAttribute("data-testid")} is ${px.toFixed(1)} px`).toBeGreaterThanOrEqual(MIN_HIT_PX);
    }
    cleanup();
  });

  it("unit8: LN loop handles are at least 24 px", async () => {
    const { container } = await mountReady(HAND["LN.loop"]!);
    for (const end of [0, 1]) {
      const h = byId(container, `[data-testid="handle-${end}"]`);
      expect(unitsToPx(hitUnits(h))).toBeGreaterThanOrEqual(MIN_HIT_PX);
    }
  });
});
