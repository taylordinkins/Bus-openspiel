import { expect, test, type Page } from "@playwright/test";
import { boardReady, newGame, scenarioGame, useSettings } from "./helpers";

// UI_DESIGN 2 / 3a: on desktop the game screen is exactly one viewport, board left, side right.
const SIZES: [number, number][] = [
  [1440, 900],
  [1280, 720],
];

async function pageScroll(page: Page) {
  return page.evaluate(() => ({
    sw: document.documentElement.scrollWidth,
    sh: document.documentElement.scrollHeight,
  }));
}

async function expectOneViewport(page: Page, w: number, h: number, minBoard = 0) {
  const d = await pageScroll(page);
  expect(d.sw, "page scroll width").toBeLessThanOrEqual(w);
  expect(d.sh, "page scroll height").toBeLessThanOrEqual(h);
  const board = page.getByTestId("bus-svg");
  const side = page.locator(".lg-side");
  await expect(board).toBeVisible();
  await expect(side).toBeVisible();
  const b = (await board.boundingBox())!;
  const s = (await side.boundingBox())!;
  for (const [name, r] of [["board", b], ["side column", s]] as const) {
    expect(r.width, `${name} width`).toBeGreaterThan(0);
    expect(r.x, `${name} left`).toBeGreaterThanOrEqual(0);
    expect(r.y, `${name} top`).toBeGreaterThanOrEqual(0);
    expect(r.x + r.width, `${name} right`).toBeLessThanOrEqual(w + 0.5);
    expect(r.y + r.height, `${name} bottom`).toBeLessThanOrEqual(h + 0.5);
  }
  expect(b.x + b.width, "board ends before the side column").toBeLessThanOrEqual(s.x + 0.5);
  expect(Math.min(b.width, b.height), "board size").toBeGreaterThanOrEqual(minBoard);
}

for (const [w, h] of SIZES) {
  test.describe(`${w}x${h}`, () => {
    test.use({ viewport: { width: w, height: h } });
    test.beforeEach(async ({ page }) => {
      await useSettings(page, "off", 0);
    });

    test("lobby has no horizontal page scroll", async ({ page }) => {
      await page.goto("/");
      await expect(page.getByRole("button", { name: "Create game" })).toBeVisible();
      expect((await pageScroll(page)).sw).toBeLessThanOrEqual(w);
    });

    test("fresh game fits one viewport", async ({ page }) => {
      await newGame(page, 4, ["human", "random", "random", "random"]);
      await boardReady(page);
      // UI_DESIGN 3a: the board is at least 640 px square at 1280 x 720.
      await expectOneViewport(page, w, h, 640);
    });

    test("5-player hotseat line expansion fits one viewport", async ({ page }) => {
      await scenarioGame(page, "lines_five_player_shared", ["human", "human", "human", "human", "human"]);
      await boardReady(page);
      await expect(page.getByTestId("turn-frame")).toBeVisible();
      // UI_DESIGN 3a: the hotseat turn frame costs the board no size (>= 640 px at 1280 x 720).
      await expectOneViewport(page, w, h, 640);
    });

    test("hotseat resolution phase fits one viewport", async ({ page }) => {
      await scenarioGame(page, "resolve_ln", ["human", "human", "human", "human"]);
      await boardReady(page);
      await expect(page.getByTestId("turn-frame")).toBeVisible();
      await expectOneViewport(page, w, h, 640);
    });

    test("END results fit one viewport", async ({ page }) => {
      await scenarioGame(page, "end_returns_margin_3p", ["human", "random", "random"]);
      await expect(page.getByRole("region", { name: "Final results" })).toBeVisible();
      await expectOneViewport(page, w, h);
    });
  });
}
