import { cloneElement, useId, useState, type ReactElement } from "react";


/**
 * Hover/focus tooltip around a single element. The popover is position:fixed (so scrolling panels do not
 * clip it) and pointer-events:none (so it can never block a click). Escape dismisses it.
 */
export default function Tip({
  text,
  children,
}: {
  text?: string;
  children: ReactElement<Record<string, unknown>>;
}) {
  const id = useId();
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  if (!text) return children;
  const show = (el: HTMLElement) => {
    const r = el.getBoundingClientRect();
    const below = r.bottom + 6 + 90 < window.innerHeight;
    setPos({
      left: Math.max(8, Math.min(r.left, window.innerWidth - 296)),
      top: below ? r.bottom + 6 : Math.max(8, r.top - 6 - 80),
    });
  };
  return (
    <span
      className="lg-tip-wrap"
      onMouseEnter={(e) => show(e.currentTarget)}
      onMouseLeave={() => setPos(null)}
      onFocus={(e) => show(e.currentTarget)}
      onBlur={() => setPos(null)}
      onKeyDown={(e) => {
        if (e.key === "Escape") setPos(null);
      }}
    >
      {cloneElement(children, pos ? { "aria-describedby": id } : {})}
      {pos && (
        <span role="tooltip" id={id} className="lg-tip" style={{ left: pos.left, top: pos.top }}>
          {text}
        </span>
      )}
    </span>
  );
}
