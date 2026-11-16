import { expect, type Locator, type Page } from "@playwright/test";

export type Seat = "human" | "random" | "heuristic";
export type Anim = "normal" | "fast" | "off";

/** Stored settings (UI_DESIGN 6.8), applied before the app reads them. */
export async function useSettings(page: Page, animation: Anim, botPace: number): Promise<void> {
  await page.addInitScript(
    ([a, p]) => window.localStorage.setItem("bus.settings.v1", JSON.stringify({ animation: a, botPace: p })),
    [animation, botPace] as const,
  );
}

const board = (page: Page): Locator => page.getByTestId("bus-svg");
export const promptBar = (page: Page): Locator => page.getByTestId("bus-prompt");
export const results = (page: Page): Locator => page.getByRole("region", { name: "Final results" });
export const curtain = (page: Page): Locator => page.getByRole("dialog", { name: "Hand-off" });

async function waitForGame(page: Page): Promise<void> {
  await expect(page).toHaveURL(/#\/game\//);
  await expect(page.getByText(/^Game \S+ · seats/)).toBeVisible();
}

/** Lobby, Fresh setup: players, seats, seed; then the game screen. */
export async function newGame(page: Page, players: number, seats: Seat[], seed = "1"): Promise<void> {
  await page.goto("/");
  await page.getByLabel("Players").selectOption(String(players));
  for (let i = 0; i < players; i++) await page.getByLabel(`Seat P${i}`).selectOption(seats[i]!);
  await page.getByLabel("Seed (optional)").fill(seed);
  await page.getByRole("button", { name: "Create game" }).click();
  await waitForGame(page);
}

/** Lobby, Scenario mode: a named scenario from data/scenarios.yaml, with the given seats. */
export async function scenarioGame(page: Page, id: string, seats: Seat[], seed = "1"): Promise<void> {
  await page.goto("/");
  await page.getByRole("radio", { name: "Scenario" }).check();
  const pick = page.getByLabel("Named scenario");
  await expect(pick).toBeEnabled();
  await pick.selectOption(id);
  await expect(page.locator("#lobby-doc-summary")).toHaveText(/^Scenario · \d players/);
  for (let i = 0; i < seats.length; i++) await page.getByLabel(`Seat P${i}`).selectOption(seats[i]!);
  await page.getByLabel("Seed (optional)").fill(seed);
  await page.getByRole("button", { name: "Create game" }).click();
  await waitForGame(page);
}

/** The board has drawn its legal targets (or the results) for the seat to move. */
export async function boardReady(page: Page): Promise<void> {
  await expect(board(page)).toBeVisible();
  await expect(board(page).locator('[data-legal="true"]').first().or(results(page))).toBeVisible();
}

/**
 * A mouse click on a board hit shape, at a point inside it: the middle of a street's path (a bent
 * street's bounding-box centre can miss it), else the shape's centre. Neighbouring hit shapes may
 * overlap there; the board sends such a click to the nearest centre (UI_DESIGN 6.7).
 */
export async function realClick(target: Locator): Promise<void> {
  await target.scrollIntoViewIfNeeded({ timeout: 1_500 });
  const pt = await target.evaluate((el) => {
    const g = el as SVGGraphicsElement;
    if (el instanceof SVGPathElement) {
      const m = g.getScreenCTM();
      const p = el.getPointAtLength(el.getTotalLength() / 2);
      if (m) return { x: m.a * p.x + m.c * p.y + m.e, y: m.b * p.x + m.d * p.y + m.f };
    }
    const r = g.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  });
  await target.page().mouse.click(pt.x, pt.y);
}

/**
 * One human decision by clicking only the board and the prompt bar (UI_DESIGN 6.7): a pending
 * confirm, type popover or loop handle first, then "Take stone" (it shortens the game), then the
 * first legal board target, then any enabled bar action. Returns false when nothing was clickable.
 */
export async function clickOneMove(page: Page): Promise<boolean> {
  const confirm = page.getByRole("alertdialog");
  if (await confirm.isVisible()) {
    await confirm.getByRole("button", { name: /confirm/i }).click();
    return true;
  }
  const pop = page.getByTestId("type-popover");
  if (await pop.isVisible()) {
    await pop.getByRole("menuitem").first().click();
    return true;
  }
  const svg = board(page);
  const pending = svg.locator('[data-pending="true"]');
  if ((await pending.count()) > 0) {
    await realClick(pending.first());
    return true;
  }
  const bar = promptBar(page);
  const stone = bar.getByRole("button", { name: /take stone/i });
  if ((await stone.count()) > 0 && (await stone.isEnabled())) {
    await stone.click();
    return true;
  }
  const legal = svg.locator('[data-legal="true"]');
  if ((await legal.count()) > 0) {
    await realClick(legal.first());
    return true;
  }
  const barAct = bar.locator("button.btn-primary:enabled");
  if ((await barAct.count()) > 0) {
    await barAct.first().click();
    return true;
  }
  return false;
}

/** What changes when a move lands: prompt, log length, lit targets, open dialogs (runs in the page). */
function pageSignature(): string {
  const bar = document.querySelector('[data-testid="bus-prompt"]')?.textContent ?? "";
  const log = document.querySelector(".lg-log")?.textContent?.length ?? 0;
  const lit = Array.from(document.querySelectorAll('[data-legal="true"], [data-pending="true"]'))
    .map((e) => `${e.id || e.getAttribute("data-testid")}:${e.getAttribute("data-action-ids")}`)
    .join(" ");
  const open = ['[role="alertdialog"]', '[data-testid="type-popover"]', '[role="dialog"]']
    .map((q) => (document.querySelector(q) ? 1 : 0))
    .join("");
  return `${bar}|${log}|${lit}|${open}`;
}

async function signature(page: Page): Promise<string> {
  return page.evaluate(pageSignature).catch(() => "");
}

/** The lit targets, for a stall report. */
async function litTargets(page: Page): Promise<string> {
  return page
    .locator('[data-legal="true"], [data-pending="true"]')
    .evaluateAll((es) => es.map((e) => e.outerHTML.slice(0, 240)).join("\n"))
    .catch(() => "");
}

/**
 * Click moves for the human seat(s) until the results panel shows. Returns the click count and
 * the phase codes (prompt bar `data-phase`) a click was made in. Fails with the lit targets when
 * clicks stop changing anything.
 */
export async function playToEnd(page: Page, cap = 2000): Promise<{ clicks: number; phases: Set<string> }> {
  const end = results(page);
  const phases = new Set<string>();
  const ready = curtain(page).getByRole("button", { name: "Click when ready" });
  let clicks = 0;
  let last = "";
  let idle = 0;
  for (let i = 0; i < cap * 4 && clicks < cap; i++) {
    if (await end.isVisible()) return { clicks, phases };
    if (await ready.isVisible()) {
      if (await ready.isEnabled()) await ready.click();
      continue;
    }
    const sig = await signature(page);
    idle = sig === last ? idle + 1 : 0;
    last = sig;
    if (idle > 40) throw new Error(`no progress after clicking; prompt|log|lit = ${sig}\n${await litTargets(page)}`);
    const phase = await promptBar(page).getAttribute("data-phase", { timeout: 1_000 }).catch(() => null);
    if (await clickOneMove(page).catch(() => false)) {
      clicks++;
      if (phase) phases.add(phase);
      // Let the click land (new view, armed street, popover or confirm) before the next one.
      await expect
        .poll(() => signature(page), { timeout: 5_000, intervals: [25, 50, 100] })
        .not.toBe(sig)
        .catch(() => {});
    } else await page.waitForTimeout(50);
  }
  return { clicks, phases };
}
