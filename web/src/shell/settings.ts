import { useSyncExternalStore } from "react";

/** Animation speed (UI_DESIGN §3 motion): normal, fast, or off (immediate). */
export type AnimSpeed = "normal" | "fast" | "off";

export interface Settings {
  animation: AnimSpeed;
  /** Seconds between paced bot moves (UI_DESIGN §7), 0..2. */
  botPace: number;
}

export const SETTINGS_KEY = "bus.settings.v1";
export const BOT_PACE_MAX = 2;
export const BOT_PACE_DEFAULT = 0.8;
const SPEEDS: readonly AnimSpeed[] = ["normal", "fast", "off"];

/** Duration multiplier per speed; 0 = no animation. */
export const SPEED_SCALE: Record<AnimSpeed, number> = { normal: 1, fast: 0.4, off: 0 };

export function prefersReducedMotion(): boolean {
  try {
    return typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false;
  } catch {
    return false;
  }
}

/** Defaults: normal speed (off under prefers-reduced-motion), 0.8 s bot pace. */
export function defaultSettings(): Settings {
  return { animation: prefersReducedMotion() ? "off" : "normal", botPace: BOT_PACE_DEFAULT };
}

export function clampPace(v: unknown): number {
  const n = typeof v === "number" ? v : Number(v);
  if (!Number.isFinite(n)) return BOT_PACE_DEFAULT;
  return Math.round(Math.min(BOT_PACE_MAX, Math.max(0, n)) * 10) / 10;
}

/** Validate a stored value; anything unusable falls back to the defaults field by field. */
export function parseSettings(raw: unknown): Settings {
  const d = defaultSettings();
  if (!raw || typeof raw !== "object") return d;
  const r = raw as Partial<Record<keyof Settings, unknown>>;
  return {
    animation: SPEEDS.includes(r.animation as AnimSpeed) ? (r.animation as AnimSpeed) : d.animation,
    botPace: r.botPace === undefined ? d.botPace : clampPace(r.botPace),
  };
}

export function loadSettings(): Settings {
  try {
    const text = window.localStorage.getItem(SETTINGS_KEY);
    return text === null ? defaultSettings() : parseSettings(JSON.parse(text));
  } catch {
    return defaultSettings();
  }
}

let current: Settings | null = null;
const listeners = new Set<() => void>();

export function getSettings(): Settings {
  if (current === null) current = loadSettings();
  return current;
}

/** Merge, persist (best effort) and notify. */
export function updateSettings(patch: Partial<Settings>): Settings {
  const next = parseSettings({ ...getSettings(), ...patch });
  current = next;
  try {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
  } catch {
    // storage full or blocked: the setting still applies for this page
  }
  for (const l of listeners) l();
  return next;
}

/** Forget the cached settings (tests; the next read goes back to localStorage). */
export function resetSettingsCache(): void {
  current = null;
  for (const l of listeners) l();
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useSettings(): [Settings, (patch: Partial<Settings>) => void] {
  const s = useSyncExternalStore(subscribe, getSettings, getSettings);
  return [s, updateSettings];
}
