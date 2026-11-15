import { useSyncExternalStore } from "react";

/** Things a hovered panel or log line asks the board to highlight (ids are board ids). */
export interface BusHighlight {
  seats: number[];
  streets: number[];
  sites: number[];
  nodes: number[];
}

export const NO_HIGHLIGHT: BusHighlight = { seats: [], streets: [], sites: [], nodes: [] };

let current: BusHighlight = NO_HIGHLIGHT;
const listeners = new Set<() => void>();

/** Shared between seat panels, the log and the board (they render in different shell panes). */
export function setHighlight(h: Partial<BusHighlight> | null): void {
  current = h ? { ...NO_HIGHLIGHT, ...h } : NO_HIGHLIGHT;
  listeners.forEach((l) => l());
}

export const getHighlight = (): BusHighlight => current;

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useHighlight(): BusHighlight {
  return useSyncExternalStore(subscribe, getHighlight, getHighlight);
}

/** Mouse/focus handlers that set the highlight while the pointer or focus is on the element. */
export function highlightProps(h: Partial<BusHighlight>) {
  return {
    onMouseEnter: () => setHighlight(h),
    onMouseLeave: () => setHighlight(null),
    onFocus: () => setHighlight(h),
    onBlur: () => setHighlight(null),
  };
}
