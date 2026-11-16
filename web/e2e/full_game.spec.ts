import { expect, test } from "@playwright/test";
import { newGame, playToEnd, results, useSettings } from "./helpers";

test("a 3-player game against random bots is finished by clicking the board", async ({ page }) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await useSettings(page, "fast", 0);
  await newGame(page, 3, ["human", "random", "random"], "7");
  const started = Date.now();
  const { clicks, phases } = await playToEnd(page);
  const secs = Math.round((Date.now() - started) / 1000);
  console.log(`full game: ${clicks} clicks in ${[...phases].join(" ")}, ${secs}s`);
  const end = results(page);
  await expect(end).toBeVisible();
  await expect(end.getByRole("table")).toBeVisible();
  // Setup, selection and resolution were clicked through on the board (phase read before each click).
  for (const p of ["SB", "SEL", "LN"]) expect(phases, `clicked in ${p}`).toContain(p);
  const body = await page.locator("body").innerText();
  expect(body).toMatch(/\bP0\b/);
  expect(body).not.toMatch(/Player [0-9]/);
  expect(errors, "console errors").toEqual([]);
});

test("a game with a heuristic bot seat is finished by clicking the board", async ({ page }) => {
  test.setTimeout(300_000);
  await useSettings(page, "off", 0);
  await newGame(page, 3, ["human", "heuristic", "random"], "5");
  await expect(page.getByText(/P1 heuristic/)).toBeVisible();
  await playToEnd(page);
  await expect(results(page)).toBeVisible();
});
