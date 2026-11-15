import { afterEach, vi } from "vitest";
import {
  BOT_PACE_DEFAULT,
  SETTINGS_KEY,
  clampPace,
  defaultSettings,
  getSettings,
  loadSettings,
  parseSettings,
  resetSettingsCache,
  updateSettings,
} from "./settings";

function mockReducedMotion(reduce: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn((q: string) => ({ matches: reduce && q.includes("reduce"), media: q, addEventListener() {}, removeEventListener() {} })),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  window.localStorage.clear();
  resetSettingsCache();
});

describe("unit9 settings (UI_DESIGN 3 motion, 7 bots)", () => {
  it("defaults to normal speed and a 0.8 s bot pace", () => {
    mockReducedMotion(false);
    expect(defaultSettings()).toEqual({ animation: "normal", botPace: BOT_PACE_DEFAULT });
    expect(BOT_PACE_DEFAULT).toBe(0.8);
  });

  it("prefers-reduced-motion makes animation off by default", () => {
    mockReducedMotion(true);
    expect(defaultSettings().animation).toBe("off");
    window.localStorage.removeItem(SETTINGS_KEY);
    expect(loadSettings()).toEqual({ animation: "off", botPace: 0.8 });
  });

  it("an explicit choice wins over prefers-reduced-motion", () => {
    mockReducedMotion(true);
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify({ animation: "fast", botPace: 1.5 }));
    expect(loadSettings()).toEqual({ animation: "fast", botPace: 1.5 });
  });

  it("persists updates to localStorage and notifies readers", () => {
    mockReducedMotion(false);
    window.localStorage.removeItem(SETTINGS_KEY);
    resetSettingsCache();
    expect(getSettings().animation).toBe("normal");
    updateSettings({ animation: "fast" });
    updateSettings({ botPace: 1.2 });
    expect(JSON.parse(window.localStorage.getItem(SETTINGS_KEY) ?? "{}")).toEqual({ animation: "fast", botPace: 1.2 });
    resetSettingsCache();
    expect(getSettings()).toEqual({ animation: "fast", botPace: 1.2 });
  });

  it("clamps the pace to 0..2 s in 0.1 s steps; bad values fall back field by field", () => {
    mockReducedMotion(false);
    expect(clampPace(-1)).toBe(0);
    expect(clampPace(9)).toBe(2);
    expect(clampPace(0.84)).toBe(0.8);
    expect(clampPace("x")).toBe(0.8);
    expect(parseSettings({ animation: "warp", botPace: 1 })).toEqual({ animation: "normal", botPace: 1 });
    expect(parseSettings(null)).toEqual(defaultSettings());
    window.localStorage.setItem(SETTINGS_KEY, "{not json");
    expect(loadSettings()).toEqual(defaultSettings());
  });

  it("survives a storage that throws (private mode, quota)", () => {
    mockReducedMotion(false);
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota");
    });
    resetSettingsCache();
    expect(getSettings()).toEqual(defaultSettings());
    expect(updateSettings({ animation: "off" }).animation).toBe("off");
    expect(getSettings().animation).toBe("off");
  });
});
