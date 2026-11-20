// Court geometry in meters, matching server/tennis_vision/court.py.
// Origin at the net center; +y is the far half (away from the camera).

export const COURT = {
  length: 23.77,
  doublesWidth: 10.97,
  singlesWidth: 8.23,
  serviceLine: 6.4,
};

const HL = COURT.length / 2;
const HDW = COURT.doublesWidth / 2;
const HSW = COURT.singlesWidth / 2;
const SL = COURT.serviceLine;

export const COURT_LINES: [[number, number], [number, number]][] = [
  [[-HDW, -HL], [HDW, -HL]],
  [[-HDW, HL], [HDW, HL]],
  [[-HDW, -HL], [-HDW, HL]],
  [[HDW, -HL], [HDW, HL]],
  [[-HSW, -HL], [-HSW, HL]],
  [[HSW, -HL], [HSW, HL]],
  [[-HSW, -SL], [HSW, -SL]],
  [[-HSW, SL], [HSW, SL]],
  [[0, -SL], [0, SL]],
];

export const REASON_LABEL: Record<string, string> = {
  ace: '서브 에이스',
  fault: '폴트',
  double_fault: '더블 폴트',
  out: '아웃',
  double_bounce: '투 바운드 / 네트',
  winner: '위너',
};

export const SHOT_LABEL: Record<string, string> = {
  serve: '서브',
  forehand: '포핸드',
  backhand: '백핸드',
  volley: '발리',
  overhead: '스매시',
  stroke: '스트로크',
};

export const SHOT_COLOR: Record<string, string> = {
  serve: '#8e24aa',
  forehand: '#1e88e5',
  backhand: '#fb8c00',
  volley: '#00897b',
  overhead: '#e53935',
  stroke: '#757575',
};

export const DEPTH_LABEL: Record<string, string> = { short: '짧게', mid: '중간', deep: '깊게' };
export const DIRECTION_LABEL: Record<string, string> = { cross: '크로스', line: '다운더라인', center: '센터' };
export const ZONE_LABEL: Record<string, string> = { wide: '와이드', body: '바디', T: 'T' };
