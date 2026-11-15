import "@testing-library/jest-dom/vitest";
import { beforeEach } from "vitest";
import { resetSettingsCache, SETTINGS_KEY } from "./shell/settings";

// Tests run with animation off and no bot pacing (immediate, as before unit 9); tests of the
// motion and pacing logic set their own settings.
beforeEach(() => {
  try {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify({ animation: "off", botPace: 0 }));
  } catch {
    // no storage in this environment
  }
  resetSettingsCache();
});
