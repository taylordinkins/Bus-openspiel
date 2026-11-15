import type { BusViewData } from "./model";

export const ACTION_TITLE: Record<string, string> = {
  lines: "Lines",
  buses: "Buses",
  passengers: "Passengers",
  buildings: "Buildings",
  clock: "Clock",
  vrrooomm: "Vrrooomm!",
  starting_player: "Starting Player",
};

export const PHASE_TITLE: Record<string, string> = {
  SB: "Setup: buildings",
  L1: "Setup: first line marker",
  L2: "Setup: second line marker",
  SEL: "Selection",
  LN: "Lines",
  PX: "Passengers",
  BD: "Buildings",
  CK: "Clock",
  VP: "Vrrooomm!",
  VD: "Vrrooomm!",
  END: "Game over",
};

export const END_TYPE_TEXT: Record<string, string> = {
  sites_full: "every site has a building",
  last_stone: "the last time stone was taken",
  markers: "at most one player has Action markers left",
};

export const BUILDING_NAME: Record<string, string> = { house: "House", office: "Office", pub: "Pub" };
const cap = (s: string): string => BUILDING_NAME[s] ?? s;
const plural = (n: number, one: string, many = `${one}s`): string => `${n} ${n === 1 ? one : many}`;

/** "an A site" / "a B site" (by the letter's spoken name); `fallback` when there is no tier. */
function tierSite(tier: string | null, fallback: string): string {
  if (!tier) return `${fallback} site`;
  const L = tier.toUpperCase();
  return `${/^[AEFHILMNORSX]/.test(L) ? "an" : "a"} ${L} site`;
}

export interface PromptParts {
  /** Acting seat, null at END. */
  actor: number | null;
  phase: string;
  /** Phase heading, e.g. "Lines". */
  title: string;
  /** What to do now (imperative sentence without the seat). */
  instruction: string;
  /** Units left, power, context, as short chips. */
  details: string[];
}

const num = (v: unknown): number | null => (typeof v === "number" ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);

/** The status-bar wording for every phase code, from the adapter prompt (UI_DESIGN 1, 4, 6.3). */
export function promptParts(data: BusViewData): PromptParts | null {
  const p = data.prompt;
  if (!p?.phase) return null;
  const ctx = p.context ?? {};
  const left = p.units_left ?? null;
  const power = p.power ?? null;
  const seats = data.state.seats.length;
  const tier = str((ctx as { open_tier?: unknown }).open_tier) ?? data.derived.open_tier;
  const details: string[] = [];
  const unitChip = (what: string) => {
    if (left !== null) details.push(`${plural(left, what)} left`);
    if (power !== null) details.push(`power ${power}`);
  };
  let instruction: string;
  switch (p.phase) {
    case "SB": {
      const done = num(ctx.done) ?? 0;
      instruction = `place a building on ${tierSite(tier, "a")}`;
      details.push(`building ${(done % 2) + 1} of 2`, `${done + 1} of ${seats * 2} placed in setup`);
      break;
    }
    case "L1":
      instruction = "place your first Bus Line marker on any street";
      break;
    case "L2":
      instruction = "extend your line from either end with a second marker";
      break;
    case "SEL": {
      const pass = num(ctx.pass) ?? 0;
      const mandatory = ctx.mandatory === true;
      const rest = num(ctx.markers_left);
      instruction = mandatory
        ? "place an Action marker on a free action space"
        : "place an Action marker, or drop out for this round";
      details.push(`pass ${pass + 1}`);
      if (mandatory) details.push("must place");
      if (rest !== null) details.push(`${plural(rest, "marker")} in hand`);
      break;
    }
    case "LN":
      instruction = "extend your line along a street";
      unitChip("street");
      break;
    case "PX": {
      instruction = "add a passenger to a station";
      unitChip("passenger");
      const sup = num(ctx.supply);
      if (sup !== null) details.push(`${sup} in supply`);
      break;
    }
    case "BD":
      instruction = `build on ${tierSite(tier, "an open")}`;
      unitChip("building");
      break;
    case "CK": {
      const next = str(ctx.next_clock);
      const stones = num(ctx.stones_on_clock);
      const t = data.targets?.clock;
      const canAdvance = t ? t.advance !== null : true;
      const canStone = t ? t.pause !== null : true;
      const advance = `advance the clock${next ? ` to ${cap(next)}` : ""}`;
      instruction = canAdvance && canStone ? `${advance} or take a time stone` : canAdvance ? advance : "take a time stone";
      if (stones !== null) details.push(`${plural(stones, "stone")} left`);
      break;
    }
    case "VP": {
      const clock = str(ctx.clock);
      instruction = "pick a waiting passenger on your line";
      if (left !== null) details.push(`${plural(left, "move")} left`);
      if (clock) details.push(`deliver to ${cap(clock)}s`);
      break;
    }
    case "VD": {
      const clock = str(ctx.clock);
      instruction = `deliver the passenger to ${clock ? `a ${cap(clock)}` : "a building"} on your line`;
      if (left !== null) details.push(`${plural(left, "move")} left`);
      break;
    }
    case "END": {
      const w = data.derived.result?.winners ?? [];
      instruction =
        w.length === 0 ? "the game has ended" : w.length === 1 ? `P${w[0]} wins` : `${w.map((s) => `P${s}`).join(", ")} share the win`;
      break;
    }
    default:
      instruction = `choose an action (phase ${p.phase})`;
  }
  return {
    actor: typeof p.actor === "number" ? p.actor : null,
    phase: p.phase,
    title: PHASE_TITLE[p.phase] ?? p.phase,
    instruction,
    details,
  };
}
