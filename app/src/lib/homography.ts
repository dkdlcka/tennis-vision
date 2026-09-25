import { COURT } from './court';
import type { Point2 } from './types';

export type Mat3 = number[]; // row-major 3x3

const HL = COURT.length / 2;
const HDW = COURT.doublesWidth / 2;

// Doubles corners in court meters: near-left, near-right, far-right, far-left.
export const COURT_CORNERS: Point2[] = [
  [-HDW, -HL],
  [HDW, -HL],
  [HDW, HL],
  [-HDW, HL],
];

/** Homography mapping four source points onto four destination points. */
export function homographyFrom(src: Point2[], dst: Point2[]): Mat3 {
  const a: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    a.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const h = solve(a, b);
  return [...h, 1];
}

export function apply(h: Mat3, [x, y]: Point2): Point2 {
  const w = h[6] * x + h[7] * y + h[8];
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
}

function solve(a: number[][], b: number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    const p = m[col][col] || 1e-12;
    for (let c = col; c <= n; c++) m[col][c] /= p;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r][col];
      for (let c = col; c <= n; c++) m[r][c] -= f * m[col][c];
    }
  }
  return m.map((row) => row[n]);
}
