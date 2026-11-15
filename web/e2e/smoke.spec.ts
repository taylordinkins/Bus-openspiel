import { expect, test } from "@playwright/test";
import { boardReady, newGame, promptBar, useSettings } from "./helpers";

test.beforeEach(async ({ page }) => {
  await useSettings(page, "off", 0);
});

test("lobby loads with the new-game form", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Bus", level: 1 })).toBeVisible();
  await expect(page.getByRole("radiogroup", { name: "Start from" })).toBeVisible();
  await expect(page.getByLabel("Players")).toHaveValue("4");
  await expect(page.getByRole("button", { name: "Create game" })).toBeEnabled();
});

test("create a game: the board renders with legal targets for P0", async ({ page }) => {
  await newGame(page, 4, ["human", "random", "random", "random"]);
  await boardReady(page);
  await expect(page.getByTestId("bus-svg").locator("image").first()).toBeAttached();
  await expect(promptBar(page)).toHaveAttribute("data-phase", "SB");
  await expect(promptBar(page)).toHaveAttribute("data-actor", "0");
  await expect(page.getByText(/^Game \S+ · seats: P0 human, P1 random, P2 random, P3 random$/)).toBeVisible();
  expect(await page.locator("body").innerText()).not.toMatch(/Player [0-9]/);
});

test("named scenario starts at its phase", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("radio", { name: "Scenario" }).check();
  await page.getByLabel("Named scenario").selectOption("resolve_bd");
  await expect(page.locator("#lobby-doc-summary")).toContainText("phase BD");
  for (let i = 0; i < 4; i++) await page.getByLabel(`Seat P${i}`).selectOption("human");
  await page.getByRole("button", { name: "Create game" }).click();
  await boardReady(page);
  await expect(promptBar(page)).toHaveAttribute("data-phase", "BD");
});
