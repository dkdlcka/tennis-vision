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
