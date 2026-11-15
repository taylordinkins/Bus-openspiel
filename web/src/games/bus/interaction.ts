import type { LegalAction } from "../../shell/useGame";
import type { BusTargets } from "./model";

/**
 * What the board and the status bar may do on a human turn (UI_DESIGN 6.7). `null` where used
 * means nothing is legal (bot seat, request in flight, other seat's view).
 */
export interface BoardAct {
  /** Post one legal id through BoardUi.onAct. */
  post: (id: number) => void;
  /** Take a time stone: asks for confirmation first when `prompt.context.last_stone` is set. */
  pause: () => void;
  /** Legal actions by id (for `params`: building type, marker space, ...). */
  legal: Map<number, LegalAction>;
}

/** Smallest hit size in map units: 24 px on the 640 px reference board (UI_DESIGN 6.7e). */
export const hitMin = (mapWidth: number): number => Math.ceil((24 * mapWidth) / 640) + 2;

export const BUILDING_ORDER = ["house", "office", "pub"] as const;

/** LN street-then-handle state (R-041): a street with two ids is armed, or a line end is chosen. */
export interface Armed {
  street: string | null;
  end: number | null;
}
export const NOT_ARMED: Armed = { street: null, end: null };

/** Ids a click on the street may post, after the end chosen by a handle (if any). */
export function streetIds(t: BusTargets, street: string, armedEnd: number | null): number[] {
  const ids = t.streets?.[street] ?? [];
  if (armedEnd === null || !t.street_ends) return ids;
  const ends = t.street_ends[street] ?? [];
  return ids.filter((id) => ends.find((e) => e.id === id)?.end === armedEnd);
}

/** True when some street has several ids (the loop case): only then are the handles used. */
export function hasAmbiguousStreet(t: BusTargets): boolean {
  return !!t.street_ends && Object.values(t.streets ?? {}).some((ids) => ids.length > 1);
}

/** All extend ids that extend the given line end. */
export function handleIds(t: BusTargets, end: number): number[] {
  return Object.values(t.street_ends ?? {}).flatMap((es) => es.filter((e) => e.end === end).map((e) => e.id));
}

/** The id on `street` that extends line end `end`, if any. */
export function idForEnd(t: BusTargets, street: string, end: number): number | undefined {
  return (t.street_ends?.[street] ?? []).find((e) => e.end === end)?.id;
}
