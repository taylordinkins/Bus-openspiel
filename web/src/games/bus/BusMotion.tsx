import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type RefObject } from "react";
import { createPortal } from "react-dom";
import type { BusGeometry } from "./geometry";
import { INK, PAPER, seatFill, seatInk } from "./model";
import { easeInOut, pointAlong, type Float, type Walk } from "./motion";
import { Passenger } from "./pieces";

/** A passenger figure walking its path (rAF-driven, so it needs no SMIL or CSS offset-path). */
function Walker({ walk }: { walk: Walk }) {
  const ref = useRef<SVGGElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let raf = 0;
    const t0 = performance.now() + walk.delay;
    const place = (now: number) => {
      const t = walk.duration > 0 ? Math.min(1, Math.max(0, (now - t0) / walk.duration)) : 1;
      const [x, y] = pointAlong(walk.pts, easeInOut(t));
      el.setAttribute("transform", `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
      el.setAttribute("opacity", now < t0 ? "0" : "1");
      if (t < 1) raf = requestAnimationFrame(place);
    };
    place(performance.now());
    return () => cancelAnimationFrame(raf);
  }, [walk]);
  const [x, y] = walk.pts[0] ?? [0, 0];
  return (
    <g
      ref={ref}
      className="bus-walker"
      data-testid="walker"
      data-kind={walk.kind}
      data-site={walk.site}
      data-seat={walk.seat}
      transform={`translate(${x} ${y})`}
      opacity={walk.delay > 0 ? 0 : 1}
    >
      <circle
        r={30}
        cy={-4}
        fill={walk.seat !== undefined ? seatFill(walk.seat) : PAPER}
        fillOpacity={walk.seat !== undefined ? 0.85 : 0.9}
        stroke={INK}
        strokeWidth={4}
      />
      <circle r={22} cy={-4} fill={PAPER} />
      <Passenger x={0} y={4} scale={1.3} />
    </g>
  );
}

/** Transient walkers over the pieces (pointer-transparent). */
export function WalkLayer({ walks }: { walks: Walk[] }) {
  if (walks.length === 0) return null;
  return (
    <g className="bus-motion-layer" pointerEvents="none" data-testid="motion-layer">
      {walks.map((w) => (
        <Walker key={w.key} walk={w} />
      ))}
    </g>
  );
}

interface Placed extends Float {
  from: { x: number; y: number };
  to: { x: number; y: number };
}

/** Screen position of a map point, via the svg's CTM (null without layout, e.g. jsdom). */
function screenOf(svg: SVGSVGElement | null, x: number, y: number): { x: number; y: number } | null {
  const m = svg?.getScreenCTM?.();
  if (!svg || !m) return null;
  return { x: m.a * x + m.c * y + m.e, y: m.b * x + m.d * y + m.f };
}

function panelTarget(seat: number): { x: number; y: number } | null {
  const el =
    document.querySelector(`.bus-panel[data-seat="${seat}"] .bus-score`) ??
    document.querySelector(`.bus-panel[data-seat="${seat}"]`);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
}

/**
 * "+1" chips flying from the delivery site to the scorer's seat panel. Rendered in a body portal
 * with fixed positions, so they can cross from the board into the side column.
 */
export function FloatLayer({
  floats,
  geo,
  svgRef,
}: {
  floats: Float[];
  geo: BusGeometry;
  svgRef: RefObject<SVGSVGElement>;
}) {
  const [placed, setPlaced] = useState<Placed[]>([]);
  useLayoutEffect(() => {
    setPlaced(
      floats.map((f) => {
        const site = geo.sites[f.site];
        const from = (site && screenOf(svgRef.current, site.x, site.y)) || { x: 0, y: 0 };
        const to = panelTarget(f.seat) ?? from;
        return { ...f, from, to };
      }),
    );
  }, [floats, geo, svgRef]);
  if (placed.length === 0) return null;
  return createPortal(
    <div className="bus-float-layer" aria-hidden="true">
      {placed.map((f) => (
        <span
          key={f.key}
          className="bus-float"
          data-testid="score-float"
          data-seat={f.seat}
          data-site={f.site}
          style={
            {
              "--fx": `${f.from.x}px`,
              "--fy": `${f.from.y}px`,
              "--tx": `${f.to.x}px`,
              "--ty": `${f.to.y}px`,
              "--seat": seatFill(f.seat),
              "--seat-ink": seatInk(f.seat),
              animationDuration: `${f.duration}ms`,
              animationDelay: `${f.delay}ms`,
            } as CSSProperties
          }
        >
          +1
        </span>
      ))}
    </div>,
    document.body,
  );
}
