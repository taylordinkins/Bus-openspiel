import type { ReactNode } from "react";
import { INK, PAPER, PASSENGER, RED, STONE, seatFill, seatInk } from "./model";

const FONT = "Helvetica, Arial, sans-serif";

export function Label({
  x,
  y,
  children,
  size,
  fill = INK,
}: {
  x: number;
  y: number;
  children: ReactNode;
  size: number;
  fill?: string;
}) {
  return (
    <text
      x={x}
      y={y}
      fontFamily={FONT}
      fontSize={size}
      fontWeight="bold"
      fill={fill}
      textAnchor="middle"
      dominantBaseline="central"
    >
      {children}
    </text>
  );
}

/** Generic waiting-passenger figure: head and rounded body; `count` adds a badge. */
export function Passenger({
  x,
  y,
  scale = 1,
  count,
}: {
  x: number;
  y: number;
  scale?: number;
  count?: number;
}) {
  const r = 7 * scale;
  const h = 20 * scale;
  const w = 9 * scale;
  return (
    <g className="passenger" fill={PASSENGER} stroke={PAPER} strokeWidth={1.5}>
      <circle cx={x} cy={y - h * 0.55} r={r} />
      <path
        d={`M ${x - w} ${y + h * 0.5} Q ${x - w} ${y - h * 0.2} ${x} ${y - h * 0.2} Q ${x + w} ${y - h * 0.2} ${x + w} ${y + h * 0.5} Z`}
      />
      {count !== undefined && (
        <g className="count-badge">
          <circle cx={x + w + 6} cy={y - h * 0.55} r={11} fill={RED} stroke={INK} strokeWidth={2} />
          <Label x={x + w + 6} y={y - h * 0.55 + 1} size={14} fill={PAPER}>
            {count}
          </Label>
        </g>
      )}
    </g>
  );
}

export type BuildingKind = "house" | "office" | "pub";

/** Our own house / office / pub glyphs, about 30 px wide at scale 1. */
export function Icon({
  kind,
  x,
  y,
  scale = 1,
}: {
  kind: string;
  x: number;
  y: number;
  scale?: number;
}) {
  let body: ReactNode;
  if (kind === "house") {
    body = (
      <>
        <path d="M -15 -2 L 0 -16 L 15 -2 L 11 -2 L 11 14 L -11 14 L -11 -2 Z" />
        <rect x={-3} y={4} width={6} height={10} fill={PAPER} />
      </>
    );
  } else if (kind === "office") {
    body = (
      <>
        <rect x={-11} y={-16} width={22} height={31} />
        {[0, 1, 2, 3].flatMap((row) =>
          [0, 1].map((col) => (
            <rect
              key={`${row}-${col}`}
              x={-8 + col * 9}
              y={-12 + row * 7}
              width={5}
              height={4}
              fill={PAPER}
            />
          )),
        )}
      </>
    );
  } else {
    body = (
      <>
        <rect x={-12} y={-13} width={20} height={27} rx={3} />
        <path
          d="M 8 -8 h 6 a 3 3 0 0 1 3 3 v 9 a 3 3 0 0 1 -3 3 h -6"
          fill="none"
          stroke={INK}
          strokeWidth={3}
        />
        <rect x={-9} y={-9} width={14} height={5} fill={PAPER} />
      </>
    );
  }
  return (
    <g className={`icon-${kind}`} fill={INK} transform={`translate(${x} ${y}) scale(${scale})`}>
      {body}
    </g>
  );
}

export function BusGlyph({
  x,
  y,
  seat,
  scale = 1,
}: {
  x: number;
  y: number;
  seat: number;
  scale?: number;
}) {
  return (
    <g
      className="bus-glyph"
      transform={`translate(${x} ${y}) scale(${scale})`}
      fill={seatFill(seat)}
      stroke={INK}
      strokeWidth={2}
    >
      <rect x={-20} y={-14} width={40} height={26} rx={6} />
      <rect x={-15} y={-9} width={30} height={10} rx={2} fill={PAPER} stroke="none" />
      <circle cx={-11} cy={14} r={4} />
      <circle cx={11} cy={14} r={4} />
    </g>
  );
}

/** Seat-coloured token with the P-number (identity never by colour alone). */
export function SeatToken({
  x,
  y,
  seat,
  r = 17,
  className = "marker",
}: {
  x: number;
  y: number;
  seat: number;
  r?: number;
  className?: string;
}) {
  return (
    <g className={`${className} seat-${seat}`}>
      <circle cx={x} cy={y} r={r} fill={seatFill(seat)} stroke={INK} strokeWidth={2.5} />
      <Label x={x} y={y + 1} size={Math.round(r * 0.82)} fill={seatInk(seat)}>
        {`P${seat}`}
      </Label>
    </g>
  );
}

/** Small flag in the start player's colour. */
export function StartFlag({ x, y, seat, scale = 1 }: { x: number; y: number; seat: number; scale?: number }) {
  return (
    <g className="start-flag" transform={`translate(${x} ${y}) scale(${scale})`}>
      <rect x={-30} y={-26} width={60} height={52} rx={10} fill={PAPER} stroke={INK} strokeWidth={3} />
      <line x1={-12} y1={-16} x2={-12} y2={16} stroke={INK} strokeWidth={4} strokeLinecap="round" />
      <path d="M -12 -16 L 16 -9 L -12 0 Z" fill={seatFill(seat)} stroke={INK} strokeWidth={2.5} strokeLinejoin="round" />
      <Label x={10} y={11} size={15}>{`P${seat}`}</Label>
    </g>
  );
}

export function TimeStoneRing({ x, y }: { x: number; y: number }) {
  return (
    <circle
      className="stone-present"
      cx={x}
      cy={y}
      r={70}
      fill="none"
      stroke={INK}
      strokeWidth={4}
      strokeDasharray="3 7"
    />
  );
}

export function TakenStone({ x, y, seat }: { x: number; y: number; seat: number }) {
  return (
    <g className="stone-taken" data-seat={seat}>
      <circle cx={x} cy={y} r={76} fill={PAPER} opacity={0.92} />
      <circle cx={x} cy={y} r={38} fill={STONE} />
      <circle cx={x} cy={y} r={27} fill={PAPER} />
      <SeatToken x={x} y={y} seat={seat} r={22} />
    </g>
  );
}
