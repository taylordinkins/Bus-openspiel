import { useEffect } from "react";
import { BOT_PACE_MAX, useSettings, type AnimSpeed } from "./settings";

const SPEEDS: { value: AnimSpeed; label: string }[] = [
  { value: "normal", label: "Normal" },
  { value: "fast", label: "Fast" },
  { value: "off", label: "Off" },
];

/** Animation speed and bot pacing (UI_DESIGN §3 motion, §7); saved in this browser. */
export default function SettingsDialog({ onClose }: { onClose: () => void }) {
  const [s, update] = useSettings();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="dlg-backdrop" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div role="dialog" aria-modal="true" aria-label="Settings" className="dlg lg-settings">
        <p className="eyebrow">This browser</p>
        <h2>Settings</h2>
        <fieldset className="lg-settings-group">
          <legend className="field-label">Animation speed</legend>
          <div role="radiogroup" aria-label="Animation speed" className="lg-seg">
            {SPEEDS.map((o) => (
              <label key={o.value} className="lg-seg-opt" data-checked={s.animation === o.value}>
                <input
                  type="radio"
                  name="anim-speed"
                  value={o.value}
                  checked={s.animation === o.value}
                  autoFocus={s.animation === o.value}
                  onChange={() => update({ animation: o.value })}
                />
                <span>{o.label}</span>
              </label>
            ))}
          </div>
          <p className="lg-settings-hint">Placements, passenger moves and score floats. Off shows each change at once.</p>
        </fieldset>
        <label className="field lg-settings-group">
          <span className="field-label">
            Bot pace <output className="lg-settings-value">{s.botPace.toFixed(1)} s</output>
          </span>
          <input
            type="range"
            aria-label="Bot pace (seconds per move)"
            min={0}
            max={BOT_PACE_MAX}
            step={0.1}
            value={s.botPace}
            onChange={(e) => update({ botPace: Number(e.target.value) })}
          />
          <span className="lg-settings-hint">Pause between bot moves; 0 shows them all at once.</span>
        </label>
        <div className="dlg-actions">
          <button type="button" className="btn-primary" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
