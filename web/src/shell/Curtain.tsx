import type { CSSProperties } from "react";

/** Hotseat hand-off cover: shown instead of the board, actions and log. */
export default function Curtain({
  seat,
  onReveal,
  disabled = false,
  accent,
}: {
  /** Seat colour from the game plugin, drawn as a band around the cover. */
  accent?: { fill: string; ink: string };
  seat: number;
  onReveal: () => void;
  disabled?: boolean;
}) {
  return (
    <div
      className="lg-curtain"
      role="dialog"
      aria-modal="true"
      aria-label="Hand-off"
      style={accent ? ({ "--seat": accent.fill, "--seat-ink": accent.ink } as CSSProperties) : undefined}
    >
      <div className="lg-curtain-card">
        <p className="eyebrow">Hotseat · hand-off</p>
        <span className="lg-curtain-tile" aria-hidden="true">
          P{seat}
        </span>
        <p className="lg-curtain-title">Pass to P{seat}</p>
        <p className="lg-curtain-sub">The board stays covered until P{seat} is ready to play.</p>
        <button type="button" className="btn-primary btn-lg" onClick={onReveal} disabled={disabled} autoFocus>
          Click when ready
        </button>
      </div>
    </div>
  );
}
