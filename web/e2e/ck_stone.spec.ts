import { expect, test, type Page } from "@playwright/test";
import { boardReady, curtain, promptBar, realClick, results, scenarioGame, useSettings } from "./helpers";

/*
 * Clock resolution by mouse (UI_DESIGN 4 `CK`, 6.7b; R-052, R-053): the Take-stone path from the
 * board's stones target, and the last-stone confirm (Cancel posts nothing, Confirm ends the game).
 * All seats are human, so nothing moves on its own; P3 holds the Clock marker in both scenarios.
 */

const humans = ["human", "human", "human", "human"] as const;
const stonesTarget = (page: Page) => page.getByTestId("target-clock-pause");
const roundInfo = (page: Page) => page.getByLabel("Round info");
const p3Panel = (page: Page) => page.getByLabel("P3 panel");
const confirmDialog = (page: Page) => page.getByRole("alertdialog");

async function atClock(page: Page, stonesOnClock: number) {
  await boardReady(page);
  await expect(promptBar(page)).toHaveAttribute("data-phase", "CK");
  await expect(p3Panel(page)).toHaveAttribute("data-acting", "true");
  await expect(roundInfo(page)).toContainText(`${stonesOnClock} stones on clock`);
  await expect(stonesTarget(page)).toHaveAttribute("data-legal", "true");
  await expect(promptBar(page).getByRole("button", { name: /take stone/i })).toBeEnabled();
}

test.beforeEach(async ({ page }) => {
  await useSettings(page, "off", 0);
});

test("taking a time stone from the board's stones target", async ({ page }) => {
  await scenarioGame(page, "resolve_ck", [...humans]);
  await atClock(page, 5);
  await expect(p3Panel(page)).not.toContainText(/stone/);

  await realClick(stonesTarget(page));

  // Not the last stone: no confirm, the pause is posted by the first click.
  await expect(confirmDialog(page)).toHaveCount(0);
  // The round ends and the next seat is handed the board (the curtain shows unless that is P3).
  const ready = curtain(page).getByRole("button", { name: "Click when ready" });
  await expect(ready.or(promptBar(page).and(page.locator(':not([data-phase="CK"])')))).toBeVisible();
  if (await ready.isVisible()) await ready.click();
  await expect(promptBar(page)).not.toHaveAttribute("data-phase", "CK");
  await expect(p3Panel(page)).toContainText("1 stone");
  await expect(roundInfo(page)).toContainText("4 stones on clock");
  await expect(page.getByTestId("bus-svg").locator('.stone-taken[data-seat="3"]')).toHaveCount(1);
  await expect(results(page)).toHaveCount(0);
});

test("the last stone asks first: Cancel posts nothing, Confirm ends the game", async ({ page }) => {
  await scenarioGame(page, "ck_last_stone_4p", [...humans]);
  await atClock(page, 1);
  await expect(p3Panel(page)).not.toContainText(/stone/);
  const logBefore = await page.locator(".lg-log").textContent();
  const takeStone = promptBar(page).getByRole("button", { name: /take stone/i });
  await expect(takeStone).toHaveText(/1 left/);

  // Prompt bar button: the confirm opens; Cancel closes it and nothing is posted.
  await takeStone.click();
  const dlg = confirmDialog(page);
  await expect(dlg).toBeVisible();
  await expect(dlg).toContainText("Taking the last stone ends the game now");
  await dlg.getByRole("button", { name: "Cancel" }).click();
  await expect(dlg).toHaveCount(0);
  await expect(promptBar(page)).toHaveAttribute("data-phase", "CK");
  await expect(roundInfo(page)).toContainText("1 stones on clock");
  await expect(p3Panel(page)).not.toContainText(/stone/);
  expect(await page.locator(".lg-log").textContent(), "log unchanged").toBe(logBefore);
  await expect(results(page)).toHaveCount(0);

  // Board stones target: the confirm opens again; Confirm takes the stone and ends the game (R-053).
  await realClick(stonesTarget(page));
  await expect(dlg).toBeVisible();
  await expect(dlg).toContainText("Taking the last stone ends the game now");
  await dlg.getByRole("button", { name: "Confirm" }).click();
  await expect(dlg).toHaveCount(0);
  await expect(results(page)).toBeVisible();
  await expect(roundInfo(page)).toContainText("0 stones on clock");
  await expect(p3Panel(page)).toContainText("1 stone");
  await expect(page.getByTestId("bus-svg").locator('.stone-taken[data-seat="3"]')).toHaveCount(1);
});
