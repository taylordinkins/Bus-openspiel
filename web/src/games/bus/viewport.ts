/** Zoom/pan state in map units: the visible window is a square of side size/scale centred (cx, cy). */
export interface Viewport {
  scale: number;
  cx: number;
  cy: number;
}

export const MIN_SCALE = 1;
export const MAX_SCALE = 2.5;

export function fitViewport(size: number): Viewport {
  return { scale: 1, cx: size / 2, cy: size / 2 };
}

/** Keep the scale in range and the window inside the map. */
export function clampViewport(v: Viewport, size: number): Viewport {
  const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale));
  const half = size / scale / 2;
  const clampC = (c: number) => Math.min(size - half, Math.max(half, c));
  return { scale, cx: clampC(v.cx), cy: clampC(v.cy) };
}

/** SVG viewBox string for a viewport. */
export function viewBoxOf(v: Viewport, size: number): string {
  const side = size / v.scale;
  return `${round(v.cx - side / 2)} ${round(v.cy - side / 2)} ${round(side)} ${round(side)}`;
}

/** Zoom by `factor` keeping the map point under (fx, fy) (fractions 0..1 of the window) fixed. */
export function zoomAt(v: Viewport, size: number, factor: number, fx = 0.5, fy = 0.5): Viewport {
  const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale * factor));
  const side = size / v.scale;
  const px = v.cx - side / 2 + fx * side;
  const py = v.cy - side / 2 + fy * side;
  const nside = size / scale;
  return clampViewport({ scale, cx: px - fx * nside + nside / 2, cy: py - fy * nside + nside / 2 }, size);
}

/** Drag by (dx, dy) fractions of the window (positive = content follows the pointer). */
export function panBy(v: Viewport, size: number, dx: number, dy: number): Viewport {
  const side = size / v.scale;
  return clampViewport({ ...v, cx: v.cx - dx * side, cy: v.cy - dy * side }, size);
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}
