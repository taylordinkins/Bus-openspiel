import { useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";
import type { BusGeometry } from "./geometry";
import { busInfo, markerInfo, nodeInfo, scoreInfo, siteInfo, streetInfo } from "./hoverInfo";
import {
  INK,
  LINE_GAP,
  PAPER,
  at,
  busCell,
  offsetPolyline,
  pathD,
  scoreCell,
  seatFill,
  streetPoints,
  type BusViewData,
} from "./model";
import {
  BusGlyph,
  Icon,
  Passenger,
  SeatToken,
  StartFlag,
  TakenStone,
  TimeStoneRing,
} from "./pieces";
import {
  clampViewport,
  fitViewport,
  panBy,
  viewBoxOf,
  zoomAt,
  type Viewport,
} from "./viewport";

export const MAP_SRC = "/bus-map.png";

interface Hover {
  lines: string[];
  x: number;
  y: number;
}

/** Group action_board entries per action in letter order. */
const LETTERS = "abcdef";

export default function BusBoard({ geo, data }: { geo: BusGeometry; data: BusViewData }) {
  const size = geo.map.width;
  const [vp, setVp] = useState<Viewport>(() => fitViewport(size));
  const [hover, setHover] = useState<Hover | null>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const vpRef = useRef(vp);
  vpRef.current = vp;

  // wheel zoom needs a non-passive listener to stop the page scrolling
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const fx = r.width ? (e.clientX - r.left) / r.width : 0.5;
      const fy = r.height ? (e.clientY - r.top) / r.height : 0.5;
      setVp((v) => zoomAt(v, size, e.deltaY < 0 ? 1.2 : 1 / 1.2, fx, fy));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [size]);

  const show = (lines: string[]) => (e: MouseEvent) => {
    const r = wrap.current?.getBoundingClientRect();
    setHover({ lines, x: e.clientX - (r?.left ?? 0), y: e.clientY - (r?.top ?? 0) });
  };
  const hide = () => setHover(null);
  const hov = (lines: string[]) => ({ onMouseMove: show(lines), onMouseEnter: show(lines), onMouseLeave: hide });

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    drag.current = { x: e.clientX, y: e.clientY };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const r = svgRef.current?.getBoundingClientRect();
    if (!r || !r.width) return;
    const dx = (e.clientX - d.x) / r.width;
    const dy = (e.clientY - d.y) / r.height;
    drag.current = { x: e.clientX, y: e.clientY };
    if (dx || dy) {
      setHover(null);
      setVp((v) => panBy(v, size, dx, dy));
    }
  };
  const endDrag = () => {
    drag.current = null;
  };

  const st = data.state;
  const nSeats = st.seats.length;
  const lay = geo.layout;

  const layers = useMemo(() => {
    // ---- lines: offset per seat on shared streets
    const lineEls: ReactNode[] = [];
    st.seats.forEach((seat, s) => {
      for (const sid of seat.line) {
        const users = data.derived.street_lines[String(sid)] ?? [s];
        const idx = Math.max(0, users.indexOf(s));
        const d = (idx - (users.length - 1) / 2) * LINE_GAP;
        const path = pathD(offsetPolyline(streetPoints(geo, sid), d));
        lineEls.push(
          <g
            key={`l${s}-${sid}`}
            className={`bus-line seat-${s}`}
            data-seat={s}
            data-street={sid}
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d={path} stroke={INK} strokeWidth={14} />
            <path d={path} stroke={seatFill(s)} strokeWidth={9} />
          </g>,
        );
      }
      const ends = data.derived.seats[s]?.ends;
      if (ends) {
        if (ends[0] === ends[1]) {
          const n = at(geo.nodes, ends[0]);
          lineEls.push(
            <g key={`loop${s}`} className={`line-loop seat-${s}`} data-seat={s} fill="none">
              <circle cx={n.x} cy={n.y} r={24} stroke={INK} strokeWidth={11} />
              <circle cx={n.x} cy={n.y} r={24} stroke={seatFill(s)} strokeWidth={7} strokeDasharray="9 6" />
            </g>,
          );
        } else {
          ends.forEach((nid, k) => {
            const n = at(geo.nodes, nid);
            lineEls.push(
              <rect
                key={`end${s}-${k}`}
                className={`line-end seat-${s}`}
                data-seat={s}
                data-node={nid}
                x={n.x - 9}
                y={n.y - 9}
                width={18}
                height={18}
                fill={seatFill(s)}
                stroke={INK}
                strokeWidth={3}
              />,
            );
          });
        }
      }
    });

    // ---- buildings
    const buildingEls = st.buildings.map((b) => {
      const site = at(geo.sites, b.site);
      return (
        <g
          key={`b${b.site}`}
          className="building"
          data-site={b.site}
          data-kind={b.type}
          data-occupied={b.occupied}
        >
          <circle cx={site.x} cy={site.y} r={25} fill={PAPER} stroke={INK} strokeWidth={3} />
          <Icon kind={b.type} x={site.x} y={site.y} scale={0.8} />
          {b.occupied && <Passenger x={site.x + 20} y={site.y - 14} scale={0.8} />}
        </g>
      );
    });

    // ---- waiting passengers (>= 4 at one node: one figure with a count badge)
    const waitingEls = Object.entries(st.waiting).flatMap(([nid, count]) => {
      const n = geo.nodes[Number(nid)];
      if (!n || count < 1) return [];
      const cx = n.x;
      const cy = n.y - (n.station ? 62 : 30); // clear of the station icon
      const figs =
        count >= 4 ? (
          <Passenger x={cx} y={cy} scale={1.2} count={count} />
        ) : (
          Array.from({ length: count }, (_, i) => (
            <Passenger key={i} x={cx + (i - (count - 1) / 2) * 24} y={cy} scale={1.2} />
          ))
        );
      return [
        <g key={`w${nid}`} className="waiting" data-node={nid} data-count={count}>
          {figs}
        </g>,
      ];
    });

    // ---- buses in the grid (starting bus not drawn)
    const busEls: ReactNode[] = [];
    st.seats.forEach((seat, s) => {
      for (let bus = 2; bus <= seat.buses; bus++) {
        const cell = busCell(geo, s, bus);
        if (!cell) continue;
        busEls.push(
          <g key={`bus${s}-${bus}`} className="bus-piece" data-seat={s} data-bus={bus} {...hov(busInfo(s, seat.buses))}>
            <rect x={cell[0] - 27} y={cell[1] - 26} width={54} height={52} rx={6} fill={PAPER} opacity={0.9} />
            <BusGlyph x={cell[0]} y={cell[1]} seat={s} scale={0.8} />
          </g>,
        );
      }
    });

    // ---- action markers
    const markerEls: ReactNode[] = [];
    for (const [action, spaces] of Object.entries(st.action_board)) {
      spaces.forEach((seat, k) => {
        if (seat === null || seat === undefined) return;
        const p = lay.actions[action]?.[LETTERS[k] ?? ""];
        if (!p) return;
        markerEls.push(
          <g
            key={`m${action}-${k}`}
            className="action-marker"
            data-seat={seat}
            data-action={action}
            data-letter={LETTERS[k]}
            {...hov(markerInfo(action, LETTERS[k] ?? "", seat))}
          >
            <SeatToken x={p[0]} y={p[1]} seat={seat} r={19} />
          </g>,
        );
      });
    }

    // ---- scores on the track, stacked sideways on ties
    const scoreEls: ReactNode[] = [];
    const stack = new Map<number, number>();
    st.seats.forEach((seat, s) => {
      const cell = scoreCell(geo, seat.points);
      if (!cell) return;
      const key = Math.min(seat.points, lay.score_track.length);
      const k = stack.get(key) ?? 0;
      stack.set(key, k + 1);
      scoreEls.push(
        <g key={`sc${s}`} className="score-disc" data-seat={s} data-points={seat.points} {...hov(scoreInfo(s, seat.points, seat.stones))}>
          <SeatToken x={cell[0] - 40 - k * 36} y={cell[1]} seat={s} r={17} />
        </g>,
      );
    });

    // ---- clock hand
    const angle = ((lay.clock.angles[st.clock] ?? 0) * Math.PI) / 180;
    const hx = lay.clock.x + lay.clock.hand * Math.cos(angle);
    const hy = lay.clock.y + lay.clock.hand * Math.sin(angle);
    const clockEl = (
      <g className="clock-hand" data-clock={st.clock}>
        <line x1={lay.clock.x} y1={lay.clock.y} x2={hx} y2={hy} stroke={PAPER} strokeWidth={14} strokeLinecap="round" />
        <line x1={lay.clock.x} y1={lay.clock.y} x2={hx} y2={hy} stroke={INK} strokeWidth={8} strokeLinecap="round" />
        <circle cx={lay.clock.x} cy={lay.clock.y} r={11} fill={INK} stroke={PAPER} strokeWidth={3} />
      </g>
    );

    // ---- time stones: present ones on the clock side, taken ones show their holder
    const holders = st.seats.flatMap((seat, s) => Array.from({ length: seat.stones }, () => s));
    const stoneEls = lay.stones.map(([x, y], i) => {
      if (i < st.stones_on_clock) return <TimeStoneRing key={`st${i}`} x={x} y={y} />;
      const h = holders[i - st.stones_on_clock];
      return h === undefined ? null : <TakenStone key={`st${i}`} x={x} y={y} seat={h} />;
    });

    // ---- start player flag beside the top of the score track
    const t0 = at(lay.score_track, 0);
    const flag = <StartFlag x={t0[0] - 110} y={t0[1]} seat={st.start_player} scale={1.25} />;

    return { lineEls, buildingEls, waitingEls, busEls, markerEls, scoreEls, clockEl, stoneEls, flag };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geo, data]);

  const zoomButton = (factor: number) => () => setVp((v) => zoomAt(v, size, factor));

  return (
    <div className="bus-board" ref={wrap} style={{ position: "relative", maxWidth: 900 }}>
      <div role="group" aria-label="Board zoom" style={{ display: "flex", gap: 4, marginBottom: 4 }}>
        <button type="button" aria-label="Zoom in" onClick={zoomButton(1.25)} disabled={vp.scale >= 2.5}>+</button>
        <button type="button" aria-label="Zoom out" onClick={zoomButton(1 / 1.25)} disabled={vp.scale <= 1}>−</button>
        <button type="button" aria-label="Reset zoom" onClick={() => setVp(fitViewport(size))}>Fit</button>
        <span data-testid="zoom-level" style={{ alignSelf: "center", fontSize: 12 }}>
          {Math.round(vp.scale * 100)}%
        </span>
      </div>
      <svg
        ref={svgRef}
        role="img"
        aria-label={`Bus board, ${nSeats} players`}
        data-testid="bus-svg"
        data-scale={vp.scale}
        data-cx={vp.cx}
        data-cy={vp.cy}
        viewBox={viewBoxOf(clampViewport(vp, size), size)}
        style={{ width: "100%", height: "auto", aspectRatio: "1 / 1", touchAction: "none", cursor: "grab", background: "#2B2B2B" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
      >
        <image href={MAP_SRC} x={0} y={0} width={size} height={geo.map.height} />
        <g className="hit-layer" fill="none">
          {geo.streets.map((s) => (
            <path
              key={`s${s.id}`}
              id={`street-${s.id}`}
              className="street hit"
              data-street={s.id}
              d={pathD(streetPoints(geo, s.id))}
              stroke="#000"
              strokeOpacity={0}
              strokeWidth={26}
              strokeLinejoin="round"
              {...hov(streetInfo(data, s.id))}
            />
          ))}
          {geo.sites.map((s) => (
            <rect
              key={`site${s.id}`}
              id={`site-${s.id}`}
              className="site hit"
              data-site={s.id}
              x={s.x - 24}
              y={s.y - 24}
              width={48}
              height={48}
              rx={7}
              fill="#000"
              fillOpacity={0}
              {...hov(siteInfo(geo, data, s.id))}
            />
          ))}
          {geo.nodes.map((n) => (
            <circle
              key={`n${n.id}`}
              id={`node-${n.id}`}
              className="node hit"
              data-node={n.id}
              cx={n.x}
              cy={n.y}
              r={n.station ? 34 : 18}
              fill="#000"
              fillOpacity={0}
              {...hov(nodeInfo(geo, data, n.id))}
            />
          ))}
        </g>
        <g pointerEvents="none">
          {layers.lineEls}
          {layers.buildingEls}
          {layers.waitingEls}
          {layers.stoneEls}
          {layers.clockEl}
          {layers.flag}
        </g>
        {layers.busEls}
        {layers.markerEls}
        {layers.scoreEls}
      </svg>
      {hover && (
        <div
          role="tooltip"
          data-testid="bus-tooltip"
          style={{
            position: "absolute",
            left: hover.x + 14,
            top: hover.y + 14,
            background: "#1A1A1A",
            color: "#FFF4D6",
            padding: "4px 8px",
            borderRadius: 4,
            fontSize: 12,
            pointerEvents: "none",
            whiteSpace: "nowrap",
            zIndex: 5,
          }}
        >
          {hover.lines.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
      )}
    </div>
  );
}
