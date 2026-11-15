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
      style={accent ? { boxShadow: `inset 0 0 0 14px ${accent.fill}` } : undefined}
    >
      <p>Pass to P{seat}</p>
      <button type="button" onClick={onReveal} disabled={disabled} autoFocus>
        Click when ready
      </button>
    </div>
  );
}
