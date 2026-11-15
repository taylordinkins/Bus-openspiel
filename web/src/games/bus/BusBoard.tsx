import { useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";
import type { BusGeometry } from "./geometry";
import {
  BUILDING_ORDER,
  NOT_ARMED,
  hasAmbiguousStreet,
  handleIds,
  hitMin,
  idForEnd,
  streetIds,
  type Armed,
  type BoardAct,
} from "./interaction";
import { useHighlight, type BusHighlight } from "./hoverStore";
import { ACTION_TITLE, BUILDING_NAME } from "./promptText";
import { busInfo, markerInfo, nodeInfo, scoreInfo, siteInfo, streetInfo } from "./hoverInfo";
import {
  GLOW,
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
import { IconFit, IconMinus, IconPlus } from "../../theme/icons";

const HL = GLOW;

/** Streets to light: those named directly plus every street of a highlighted seat's line. */
function hlStreets(data: BusViewData, hl: BusHighlight): number[] {
  const out = new Set<number>(hl.streets);
  for (const s of hl.seats) for (const sid of data.state.seats[s]?.line ?? []) out.add(sid);
  return [...out];
}

export const MAP_SRC = "/bus-map.png";

interface Hover {
  lines: string[];
  x: number;
  y: number;
  /** Open the tooltip to the left / above the pointer when it is near the right / bottom edge. */
  flipX?: boolean;
  flipY?: boolean;
}

/** Group action_board entries per action in letter order. */
const LETTERS = "abcdef";

/** Hover state of a legal target (drives the ghost). */
type Hv =
  | { t: "street" | "site" | "node" | "handle"; id: number }
  | { t: "action"; name: string }
  | { t: "advance" };

/** A legal point-like target; overlapping hit shapes resolve to the nearest centre. */
interface Centre {
  key: string;
  t: "site" | "node" | "handle";
  id: number;
  x: number;
  y: number;
  half: number;
}

const LN_HANDLE_PHASES = ["LN", "L2"];

export default function BusBoard({
  geo,
  data,
  act = null,
}: {
  geo: BusGeometry;
  data: BusViewData;
  act?: BoardAct | null;
}) {
  const size = geo.map.width;
  const [vp, setVp] = useState<Viewport>(() => fitViewport(size));
  const [hover, setHover] = useState<Hover | null>(null);
  const [hv, setHv] = useState<Hv | null>(null);
  const [pop, setPop] = useState<{ site: number; x: number; y: number } | null>(null);
  const [armed, setArmed] = useState<Armed>(NOT_ARMED);
  const hl = useHighlight();
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
      setPop(null);
      setVp((v) => zoomAt(v, size, e.deltaY < 0 ? 1.2 : 1 / 1.2, fx, fy));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [size]);

  const show = (lines: string[]) => (e: MouseEvent) => {
    const r = wrap.current?.getBoundingClientRect();
    const x = e.clientX - (r?.left ?? 0);
    const y = e.clientY - (r?.top ?? 0);
    setHover({ lines, x, y, flipX: !!r && x > r.width * 0.62, flipY: !!r && y > r.height * 0.6 });
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
      setPop(null);
      setVp((v) => panBy(v, size, dx, dy));
    }
  };
  const endDrag = () => {
    drag.current = null;
  };

  const st = data.state;
  const nSeats = st.seats.length;
  const lay = geo.layout;

  // ---------------------------------------------------------------- interaction (UI_DESIGN 4, 6.7)
  const tg = data.targets ?? {};
  const phase = data.prompt?.phase ?? "";
  const actor = typeof data.prompt?.actor === "number" ? data.prompt.actor : null;
  const live = act !== null;
  const hm = hitMin(size) / 2; // half of the smallest legal hit shape
  const sig = JSON.stringify([phase, actor, tg]);
  // Reset transient UI when the prompt/targets change. Not on mount: the mount effect would run after
  // an early click and wipe the popover/armed state that click just set.
  const lastSig = useRef(sig);
  useEffect(() => {
    if (lastSig.current === sig) return;
    lastSig.current = sig;
    setPop(null);
    setArmed(NOT_ARMED);
    setHv(null);
  }, [sig]);
  useEffect(() => {
    if (!pop) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPop(null);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [pop]);

  const sIds = (id: number): number[] => (live ? streetIds(tg, String(id), armed.end) : []);
  const siteIdsOf = (id: number): number[] => (live ? (tg.sites?.[String(id)] ?? []) : []);
  const nodeIdsOf = (id: number): number[] => {
    if (!live) return [];
    const a = tg.stations?.[String(id)];
    const b = tg.nodes?.[String(id)];
    return [a, b].filter((x): x is number => x !== undefined);
  };
  const actionIdOf = (name: string): number | undefined => (live ? tg.actions?.[name] : undefined);
  const advanceId = live ? (tg.clock?.advance ?? null) : null;
  const pauseId = live ? (tg.clock?.pause ?? null) : null;
  const lineEnds =
    actor !== null && LN_HANDLE_PHASES.includes(phase) ? (data.derived.seats[actor]?.ends ?? null) : null;
  const handlesOn = live && lineEnds !== null && hasAmbiguousStreet(tg);
  const handlePos = (end: number): { x: number; y: number } => {
    const ends = lineEnds ?? [0, 0];
    const n = at(geo.nodes, ends[end] ?? 0);
    const same = ends[0] === ends[1];
    return { x: n.x + (same ? (end === 0 ? -1 : 1) * hm * 1.1 : 0), y: n.y };
  };

  const centres: Centre[] = [];
  if (live) {
    for (const s of geo.sites)
      if (siteIdsOf(s.id).length) centres.push({ key: `site:${s.id}`, t: "site", id: s.id, x: s.x, y: s.y, half: hm });
    for (const n of geo.nodes)
      if (nodeIdsOf(n.id).length)
        centres.push({ key: `node:${n.id}`, t: "node", id: n.id, x: n.x, y: n.y, half: Math.max(hm, n.station ? 34 : 18) });
    if (handlesOn)
      for (const end of [0, 1])
        if (handleIds(tg, end).length) {
          const p = handlePos(end);
          centres.push({ key: `handle:${end}`, t: "handle", id: end, x: p.x, y: p.y, half: hm });
        }
  }
  const centreOf = (t: Centre["t"], id: number): Centre | undefined => centres.find((c) => c.t === t && c.id === id);

  /** Screen frame of the svg for the current viewport (null without layout, e.g. in jsdom). */
  const frame = () => {
    const el = svgRef.current;
    if (!el) return null;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return null;
    const v = clampViewport(vp, size);
    const side = size / v.scale;
    const k = Math.min(r.width, r.height) / side;
    return { r, x0: v.cx - side / 2, y0: v.cy - side / 2, k, ox: (r.width - side * k) / 2, oy: (r.height - side * k) / 2 };
  };
  const toMap = (cx: number, cy: number) => {
    const f = frame();
    return f ? { x: f.x0 + (cx - f.r.left - f.ox) / f.k, y: f.y0 + (cy - f.r.top - f.oy) / f.k } : null;
  };
  const toWrap = (x: number, y: number) => {
    const f = frame();
    const w = wrap.current?.getBoundingClientRect();
    if (!f || !w) return { x: 0, y: 0 };
    return { x: f.r.left + f.ox + (x - f.x0) * f.k - w.left, y: f.r.top + f.oy + (y - f.y0) * f.k - w.top };
  };
  /** Overlapping legal hit shapes: the nearest centre under the pointer wins; else the element's own. */
  const pick = (e: { clientX: number; clientY: number }, own: Centre | undefined): Centre | undefined => {
    if (e.clientX || e.clientY) {
      const p = toMap(e.clientX, e.clientY);
      if (p) {
        let best: Centre | undefined;
        let bd = Infinity;
        for (const c of centres) {
          const dx = Math.abs(p.x - c.x);
          const dy = Math.abs(p.y - c.y);
          if (dx <= c.half && dy <= c.half && Math.hypot(dx, dy) < bd) {
            bd = Math.hypot(dx, dy);
            best = c;
          }
        }
        if (best) return best;
      }
    }
    return own;
  };

  const clickCentre = (c: Centre) => {
    if (!act) return;
    if (c.t === "site") {
      const ids = siteIdsOf(c.id);
      if (ids.length > 1) {
        setPop({ site: c.id, ...toWrap(c.x, c.y) });
        return;
      }
      setPop(null);
      if (ids[0] !== undefined) act.post(ids[0]);
    } else if (c.t === "node") {
      setPop(null);
      const id = nodeIdsOf(c.id)[0];
      if (id !== undefined) act.post(id);
    } else {
      setPop(null);
      if (armed.street !== null) {
        const id = idForEnd(tg, armed.street, c.id);
        if (id !== undefined) {
          setArmed(NOT_ARMED);
          act.post(id);
        }
        return;
      }
      setArmed(armed.end === c.id ? NOT_ARMED : { street: null, end: c.id });
    }
  };
  const onPoint = (e: MouseEvent, t: Centre["t"], id: number) => {
    const c = pick(e, centreOf(t, id));
    if (c) clickCentre(c);
    else setPop(null);
  };
  const clickStreet = (id: number) => {
    setPop(null);
    if (!act) return;
    const ids = sIds(id);
    if (ids.length === 0) return;
    if (ids.length === 1) {
      setArmed(NOT_ARMED);
      act.post(ids[0] as number);
      return;
    }
    setArmed({ street: String(id), end: armed.end }); // R-041: pick the end with a handle next
  };
  const clickAction = (name: string) => {
    setPop(null);
    const id = actionIdOf(name);
    if (act && id !== undefined) act.post(id);
  };
  const hoverPoint = (e: MouseEvent, t: Centre["t"], id: number) => {
    const c = pick(e, centreOf(t, id));
    setHv(c ? { t: c.t, id: c.id } : null);
  };
  const leave = () => {
    hide();
    setHv(null);
  };

  const pickedNode = typeof data.prompt?.context?.picked_node === "number" ? data.prompt.context.picked_node : null;
  const nextClock = typeof data.prompt?.context?.next_clock === "string" ? data.prompt.context.next_clock : null;

  const actionBox = (name: string) => {
    const pts = Object.values(lay.actions[name] ?? {});
    if (pts.length === 0) return null;
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const x = Math.min(...xs) - hm;
    const y = Math.min(...ys) - hm;
    return { x, y, width: Math.max(...xs) + hm - x, height: Math.max(...ys) + hm - y };
  };
  const stonesBox = (() => {
    if (lay.stones.length === 0) return null;
    const xs = lay.stones.map((p) => p[0]);
    const ys = lay.stones.map((p) => p[1]);
    const x = Math.min(...xs) - hm;
    const y = Math.min(...ys) - hm;
    return { x, y, width: Math.max(...xs) + hm - x, height: Math.max(...ys) + hm - y };
  })();
  const clockR = Math.max(lay.clock.radius, hm);

  /** Hover ghost for the legal target under the pointer (UI_DESIGN 6.7e); at most one. */
  const ghost = ((): ReactNode => {
    if (!live || !hv || actor === null) return null;
    const colour = seatFill(actor);
    const common = { "data-testid": "ghost", "data-seat": actor, pointerEvents: "none" as const };
    if (hv.t === "street") {
      if (sIds(hv.id).length === 0) return null;
      return (
        <path {...common} data-kind="street" data-street={hv.id} d={pathD(streetPoints(geo, hv.id))}
          fill="none" stroke={colour} strokeWidth={12} strokeOpacity={0.7} strokeLinecap="round" strokeLinejoin="round" />
      );
    }
    if (hv.t === "site") {
      if (siteIdsOf(hv.id).length === 0) return null;
      const site = at(geo.sites, hv.id);
      if (phase === "VD" && pickedNode !== null) {
        const from = at(geo.nodes, pickedNode);
        return (
          <path {...common} data-kind="path" data-from={pickedNode} data-to={hv.id}
            d={pathD([[from.x, from.y], [site.x, site.y]])} fill="none" stroke={colour}
            strokeWidth={9} strokeDasharray="16 12" strokeLinecap="round" />
        );
      }
      return (
        <g {...common} data-kind="building" data-site={hv.id} opacity={0.75}>
          <circle cx={site.x} cy={site.y} r={26} fill={PAPER} fillOpacity={0.6} stroke={colour} strokeWidth={6} strokeDasharray="8 5" />
        </g>
      );
    }
    if (hv.t === "node") {
      const n = geo.nodes[hv.id];
      if (!n || nodeIdsOf(hv.id).length === 0 || phase === "VP") return null;
      return (
        <g {...common} data-kind="passenger" data-node={hv.id} opacity={0.6}>
          <circle cx={n.x} cy={n.y - 62 - 10} r={22} fill="none" stroke={colour} strokeWidth={4} strokeDasharray="6 4" />
          <Passenger x={n.x} y={n.y - 62} scale={1.2} />
        </g>
      );
    }
    if (hv.t === "action") {
      const id = actionIdOf(hv.name);
      const space = id !== undefined ? act?.legal.get(id)?.params?.space : undefined;
      const letter = typeof space === "number" ? LETTERS[space] : undefined;
      const p = letter ? lay.actions[hv.name]?.[letter] : undefined;
      if (!p || !letter) return null;
      return (
        <g {...common} data-kind="marker" data-action={hv.name} data-letter={letter} opacity={0.6}>
          <SeatToken x={p[0]} y={p[1]} seat={actor} r={19} />
        </g>
      );
    }
    if (hv.t === "advance") {
      if (advanceId === null || !nextClock) return null;
      const a = ((lay.clock.angles[nextClock] ?? 0) * Math.PI) / 180;
      return (
        <line {...common} data-kind="clock-hand" data-clock={nextClock} x1={lay.clock.x} y1={lay.clock.y}
          x2={lay.clock.x + lay.clock.hand * Math.cos(a)} y2={lay.clock.y + lay.clock.hand * Math.sin(a)}
          stroke={colour} strokeWidth={9} strokeOpacity={0.75} strokeLinecap="round" strokeDasharray="12 8" />
      );
    }
    return null;
  })();

  const legalProps = (ids: number[]) =>
    ids.length ? { "data-legal": "true", "data-action-ids": ids.join(",") } : {};
  const popTypes = pop
    ? siteIdsOf(pop.site).map((id, i) => {
        const bt = act?.legal.get(id)?.params?.building_type;
        return { id, type: typeof bt === "string" ? bt : (BUILDING_ORDER[i] ?? "house") };
      })
    : [];

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

  const zoomButton = (factor: number) => () => {
    setPop(null);
    setVp((v) => zoomAt(v, size, factor));
  };

  return (
    <div className="bus-board" ref={wrap}>
      <div role="group" aria-label="Board zoom" className="bus-zoom">
        <button type="button" className="btn-icon" aria-label="Zoom in" title="Zoom in" onClick={zoomButton(1.25)} disabled={vp.scale >= 2.5}>
          <IconPlus />
        </button>
        <button type="button" className="btn-icon" aria-label="Zoom out" title="Zoom out" onClick={zoomButton(1 / 1.25)} disabled={vp.scale <= 1}>
          <IconMinus />
        </button>
        <button type="button" className="btn-icon" aria-label="Reset zoom" title="Fit the board" onClick={() => setVp(fitViewport(size))}>
          <IconFit />
        </button>
        <span data-testid="zoom-level" className="bus-zoom-level">
          {Math.round(vp.scale * 100)}%
        </span>
      </div>
      <div className="bus-svg-box">
      <svg
        ref={svgRef}
        role="img"
        aria-label={`Bus board, ${nSeats} players`}
        data-testid="bus-svg"
        data-scale={vp.scale}
        data-cx={vp.cx}
        data-cy={vp.cy}
        viewBox={viewBoxOf(clampViewport(vp, size), size)}
        className="bus-svg"
        style={{ touchAction: "none" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onClick={(e) => {
          if (e.target === e.currentTarget || (e.target as Element).tagName === "image") setPop(null);
        }}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
      >
        <image href={MAP_SRC} x={0} y={0} width={size} height={geo.map.height} />
        <g className="hover-layer" pointerEvents="none" fill="none" data-testid="hover-layer">
          {hlStreets(data, hl).map((sid) => (
            <path
              key={`hs${sid}`}
              className="hl-street"
              data-street={sid}
              d={pathD(streetPoints(geo, sid))}
              stroke={HL}
              strokeWidth={30}
              strokeOpacity={0.85}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}
          {hl.sites.flatMap((id) => {
            const t = geo.sites[id];
            return t
              ? [<circle key={`hst${id}`} className="hl-site" data-site={id} cx={t.x} cy={t.y} r={36} stroke={HL} strokeWidth={8} />]
              : [];
          })}
          {hl.nodes.flatMap((id) => {
            const n = geo.nodes[id];
            return n
              ? [<circle key={`hn${id}`} className="hl-node" data-node={id} cx={n.x} cy={n.y} r={30} stroke={HL} strokeWidth={8} />]
              : [];
          })}
        </g>
        <g className="glow-layer" pointerEvents="none" fill="none" data-testid="glow-layer">
          {geo.streets.map((st2) =>
            sIds(st2.id).length ? (
              <path key={`gs${st2.id}`} className="glow-fx" d={pathD(streetPoints(geo, st2.id))} stroke={HL}
                strokeWidth={30} strokeOpacity={0.6} strokeLinecap="round" strokeLinejoin="round" />
            ) : null,
          )}
          {centres.map((c) => (
            <circle key={`g${c.key}`} className="glow-fx" cx={c.x} cy={c.y}
              r={c.t === "site" ? 36 : c.t === "handle" ? hm : geo.nodes[c.id]?.station ? 44 : 26}
              fill={HL} fillOpacity={0.3} stroke={HL} strokeWidth={8} />
          ))}
          {Object.keys(lay.actions).map((name) => {
            const bx = actionBox(name);
            return bx && actionIdOf(name) !== undefined ? (
              <rect key={`ga${name}`} className="glow-fx" {...bx} rx={14} fill={HL} fillOpacity={0.2} stroke={HL} strokeWidth={8} />
            ) : null;
          })}
          {advanceId !== null && (
            <circle className="glow-fx" cx={lay.clock.x} cy={lay.clock.y} r={clockR} fill={HL} fillOpacity={0.2} stroke={HL} strokeWidth={8} />
          )}
          {pauseId !== null && stonesBox && (
            <rect className="glow-fx" {...stonesBox} rx={14} fill={HL} fillOpacity={0.2} stroke={HL} strokeWidth={8} />
          )}
        </g>
        <g className="hit-layer" fill="none">
          {geo.streets.map((s) => {
            const ids = sIds(s.id);
            return (
              <path
                key={`s${s.id}`}
                id={`street-${s.id}`}
                className={`street hit${ids.length ? " glow" : ""}`}
                data-street={s.id}
                {...legalProps(ids)}
                d={pathD(streetPoints(geo, s.id))}
                stroke="#000"
                strokeOpacity={0}
                strokeWidth={ids.length ? hm * 2 : 26}
                strokeLinejoin="round"
                onClick={() => clickStreet(s.id)}
                onMouseEnter={(e) => {
                  show(streetInfo(data, s.id))(e);
                  setHv({ t: "street", id: s.id });
                }}
                onMouseMove={show(streetInfo(data, s.id))}
                onMouseLeave={leave}
              />
            );
          })}
          {geo.sites.map((s) => {
            const ids = siteIdsOf(s.id);
            const half = ids.length ? hm : 24;
            return (
              <rect
                key={`site${s.id}`}
                id={`site-${s.id}`}
                className={`site hit${ids.length ? " glow" : ""}`}
                data-site={s.id}
                {...legalProps(ids)}
                x={s.x - half}
                y={s.y - half}
                width={half * 2}
                height={half * 2}
                rx={7}
                fill="#000"
                fillOpacity={0}
                onClick={(e) => onPoint(e, "site", s.id)}
                onMouseEnter={(e) => {
                  show(siteInfo(geo, data, s.id))(e);
                  hoverPoint(e, "site", s.id);
                }}
                onMouseMove={(e) => {
                  show(siteInfo(geo, data, s.id))(e);
                  hoverPoint(e, "site", s.id);
                }}
                onMouseLeave={leave}
              />
            );
          })}
          {geo.nodes.map((n) => {
            const ids = nodeIdsOf(n.id);
            const base = n.station ? 34 : 18;
            return (
              <circle
                key={`n${n.id}`}
                id={`node-${n.id}`}
                className={`node hit${ids.length ? " glow" : ""}`}
                data-node={n.id}
                {...legalProps(ids)}
                cx={n.x}
                cy={n.y}
                r={ids.length ? Math.max(base, hm) : base}
                fill="#000"
                fillOpacity={0}
                onClick={(e) => onPoint(e, "node", n.id)}
                onMouseEnter={(e) => {
                  show(nodeInfo(geo, data, n.id))(e);
                  hoverPoint(e, "node", n.id);
                }}
                onMouseMove={(e) => {
                  show(nodeInfo(geo, data, n.id))(e);
                  hoverPoint(e, "node", n.id);
                }}
                onMouseLeave={leave}
              />
            );
          })}
          {handlesOn &&
            lineEnds &&
            [0, 1].map((end) => {
              const p = handlePos(end);
              const ids = handleIds(tg, end);
              const pending = armed.street !== null && idForEnd(tg, armed.street, end) !== undefined;
              return (
                <circle
                  key={`h${end}`}
                  data-testid={`handle-${end}`}
                  className={`handle hit${ids.length ? " glow" : ""}`}
                  data-end={end}
                  data-node={lineEnds[end]}
                  data-pending={pending ? "true" : undefined}
                  data-armed={armed.end === end ? "true" : undefined}
                  {...legalProps(ids)}
                  cx={p.x}
                  cy={p.y}
                  r={hm}
                  fill="#000"
                  fillOpacity={0}
                  onClick={(e) => onPoint(e, "handle", end)}
                  onMouseEnter={() => setHv(null)}
                />
              );
            })}
          <circle
            data-testid="target-clock-advance"
            className={`clock-target hit${advanceId !== null ? " glow" : ""}`}
            {...legalProps(advanceId !== null ? [advanceId] : [])}
            cx={lay.clock.x}
            cy={lay.clock.y}
            r={clockR}
            fill="#000"
            fillOpacity={0}
            pointerEvents={advanceId !== null ? undefined : "none"}
            onClick={() => advanceId !== null && act?.post(advanceId)}
            onMouseEnter={() => setHv({ t: "advance" })}
            onMouseLeave={leave}
          />
          {stonesBox && (
            <rect
              data-testid="target-clock-pause"
              className={`stones-target hit${pauseId !== null ? " glow" : ""}`}
              {...legalProps(pauseId !== null ? [pauseId] : [])}
              {...stonesBox}
              rx={14}
              fill="#000"
              fillOpacity={0}
              pointerEvents={pauseId !== null ? undefined : "none"}
              onClick={() => pauseId !== null && act?.pause()}
              onMouseEnter={() => setHv(null)}
            />
          )}
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
        <g className="action-hit-layer" fill="none">
          {Object.keys(lay.actions).map((name) => {
            const bx = actionBox(name);
            if (!bx) return null;
            const id = actionIdOf(name);
            const legal = id !== undefined;
            const placed = (st.action_board[name] ?? []).flatMap((seat, k) =>
              seat === null || seat === undefined ? [] : markerInfo(name, LETTERS[k] ?? "", seat),
            );
            const lines = [ACTION_TITLE[name] ?? name, ...placed];
            return (
              <rect
                key={`act${name}`}
                data-testid={`target-action-${name}`}
                className={`action-target hit${legal ? " glow" : ""}`}
                data-action={name}
                {...legalProps(legal ? [id] : [])}
                {...bx}
                rx={14}
                fill="#000"
                fillOpacity={0}
                pointerEvents={legal ? undefined : "none"}
                onClick={() => clickAction(name)}
                onMouseEnter={(e) => {
                  show(lines)(e);
                  setHv({ t: "action", name });
                }}
                onMouseMove={show(lines)}
                onMouseLeave={leave}
              />
            );
          })}
        </g>
        {pickedNode !== null && phase === "VD" && geo.nodes[pickedNode] && (
          <g data-testid="picked-node" data-node={pickedNode} pointerEvents="none" fill="none">
            <circle cx={geo.nodes[pickedNode].x} cy={geo.nodes[pickedNode].y} r={30} stroke={INK} strokeWidth={10} />
            <circle cx={geo.nodes[pickedNode].x} cy={geo.nodes[pickedNode].y} r={30} stroke={HL} strokeWidth={6} strokeDasharray="10 6" />
          </g>
        )}
        {ghost}
      </svg>
      </div>
      {pop && (
        <div
          role="menu"
          aria-label="Choose building type"
          data-testid="type-popover"
          data-site={pop.site}
          className="bus-type-popover"
          style={{ left: pop.x, top: pop.y }}
        >
          {popTypes.map((o, i) => (
            <button
              key={o.id}
              type="button"
              role="menuitem"
              data-action-id={o.id}
              autoFocus={i === 0}
              onClick={() => {
                setPop(null);
                act?.post(o.id);
              }}
            >
              <svg width={28} height={28} viewBox="-32 -32 64 64" aria-hidden="true">
                <Icon kind={o.type} x={0} y={0} scale={1.2} />
              </svg>
              <span>{BUILDING_NAME[o.type] ?? o.type}</span>
            </button>
          ))}
        </div>
      )}
      {hover && (
        <div
          role="tooltip"
          data-testid="bus-tooltip"
          className="bus-tip"
          style={{
            left: hover.x + (hover.flipX ? -14 : 14),
            top: hover.y + (hover.flipY ? -14 : 14),
            transform: `translate(${hover.flipX ? "-100%" : "0"}, ${hover.flipY ? "-100%" : "0"})`,
          }}
        >
          {hover.lines.map((l, i) => (
            <div key={i} className={i === 0 ? "bus-tip-title" : undefined}>
              {l}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
