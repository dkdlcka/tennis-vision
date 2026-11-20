import Svg, { Circle, Line, Rect } from 'react-native-svg';

import { COURT, COURT_LINES } from '../lib/court';
import type { Bounce, Point2 } from '../lib/types';

interface Props {
  bounces: Bounce[];
  width: number;
  highlight?: number | null;
  colorFor?: (b: Bounce) => string;
  /** Shot paths drawn under the bounces, from where the ball was struck to where it landed. */
  paths?: { from: Point2; to: Point2; color: string; bold?: boolean }[];
}

const PAD = 3; // meters of run-off drawn around the court

/** Top-down court with every bounce: filled when in, hollow when out. */
export function CourtDiagram({ bounces, width, highlight, colorFor, paths }: Props) {
  const wM = COURT.doublesWidth + 2 * PAD;
  const hM = COURT.length + 2 * PAD;
  const s = width / wM;
  const height = hM * s;
  // Far half at the top, like the camera view.
  const X = (x: number) => (x + wM / 2) * s;
  const Y = (y: number) => (hM / 2 - y) * s;
  const color = colorFor ?? ((b: Bounce) => (b.hitter === 'A' ? '#1e88e5' : '#e53935'));

  return (
    <Svg width={width} height={height}>
      <Rect x={0} y={0} width={width} height={height} fill="#2f6b4f" rx={8} />
      <Rect
        x={X(-COURT.doublesWidth / 2)}
        y={Y(COURT.length / 2)}
        width={COURT.doublesWidth * s}
        height={COURT.length * s}
        fill="#3a7ca5"
      />
      {COURT_LINES.map(([a, b], i) => (
        <Line key={i} x1={X(a[0])} y1={Y(a[1])} x2={X(b[0])} y2={Y(b[1])} stroke="#fff" strokeWidth={1.5} />
      ))}
      <Line x1={X(-wM / 2 + 1)} y1={Y(0)} x2={X(wM / 2 - 1)} y2={Y(0)} stroke="#ddd" strokeWidth={3} />
      {paths?.map((p, i) => (
        <Line
          key={`p${i}`}
          x1={X(p.from[0])}
          y1={Y(p.from[1])}
          x2={X(p.to[0])}
          y2={Y(p.to[1])}
          stroke={p.bold ? '#ffe600' : p.color}
          strokeWidth={p.bold ? 3 : 1.5}
          opacity={p.bold ? 1 : 0.55}
        />
      ))}
      {bounces.map((b) => {
        const c = color(b);
        const big = b.id === highlight;
        return (
          <Circle
            key={b.id}
            cx={X(b.court_xy[0])}
            cy={Y(b.court_xy[1])}
            r={big ? 7 : 4.5}
            fill={b.in ? c : 'transparent'}
            stroke={big ? '#ffe600' : c}
            strokeWidth={big ? 3 : 2}
          />
        );
      })}
    </Svg>
  );
}
