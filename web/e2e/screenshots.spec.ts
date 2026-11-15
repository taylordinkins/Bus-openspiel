import { expect, test, type Page } from "@playwright/test";
import { boardReady, curtain, results, scenarioGame, useSettings } from "./helpers";

/*
 * Screenshot baselines of fixed positions (named scenarios, all seats human so nothing moves on
 * its own). Pixels depend on the machine's font stack and rasteriser (the "⚑" and "→" glyphs come
 * from system fonts), so these run only with BUS_SCREENSHOTS=1: `npm run e2e:shots`, and
 * `npm run e2e:shots -- --update-snapshots` to re-baseline. Baselines are Linux chromium
 * (`*-chromium-linux.png`); CI runs them as an advisory step.
 */
test.skip(process.env.BUS_SCREENSHOTS !== "1", "screenshot baselines run with BUS_SCREENSHOTS=1");
test.use({ viewport: { width: 1440, height: 900 } });

const SHOT = { maxDiffPixelRatio: 0.01, animations: "disabled", caret: "hide" } as const;

async function settle(page: Page) {
  await page.mouse.move(0, 0);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForFunction(() => Array.from(document.images).every((i) => i.complete));
  await page.waitForTimeout(300);
}

/** The game id in the status line differs every run. */
const masks = (page: Page) => [page.locator(".lg-status-line")];

const humans = (n: number) => Array.from({ length: n }, () => "human" as const);

test.beforeEach(async ({ page }) => {
  await useSettings(page, "off", 0);
});

test("lobby", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Create game" })).toBeVisible();
  await settle(page);
  await expect(page).toHaveScreenshot("lobby.png", SHOT);
});

const POSITIONS: [string, string, number][] = [
  ["setup", "setup_sb_midway", 4],
  ["selection", "sel_pass1_mid", 4],
  ["resolution-bd", "resolve_bd", 4],
];

for (const [name, id, n] of POSITIONS) {
  test(name, async ({ page }) => {
    await scenarioGame(page, id, humans(n));
    await boardReady(page);
    await settle(page);
    await expect(page).toHaveScreenshot(`${name}.png`, { ...SHOT, mask: masks(page) });
  });
}

test("end", async ({ page }) => {
  await scenarioGame(page, "end_returns_margin_3p", humans(3));
  await expect(results(page)).toBeVisible();
  await settle(page);
  await expect(page).toHaveScreenshot("end.png", { ...SHOT, mask: masks(page) });
});

test("hotseat curtain", async ({ page }) => {
  await scenarioGame(page, "sel_pass1_mid", humans(4));
  await boardReady(page);
  await page.getByTestId("bus-svg").locator('[data-legal="true"]').first().click({ force: true });
  await expect(curtain(page)).toBeVisible();
  await settle(page);
  await expect(page).toHaveScreenshot("hotseat-curtain.png", { ...SHOT, mask: masks(page) });
});
