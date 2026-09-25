// Shapes of the analysis server's JSON (server/tennis_vision/pipeline.py).

export type Player = 'A' | 'B';
export type Side = 'near' | 'far';
export type Point2 = [number, number];

export interface ScoreState {
  sets: [number, number][];
  points: [string, string];
  server: Player;
  second_serve: boolean;
  tiebreak: boolean;
  winner: Player | null;
}

export type PointReason = 'ace' | 'fault' | 'double_fault' | 'out' | 'double_bounce' | 'winner';

export interface PointRecord {
  index: number;
  start_t: number;
  end_t: number;
  server: Player;
  server_side: Side;
  serve_box: 'deuce' | 'ad';
  second_serve: boolean;
  winner: Player | null;
  reason: PointReason;
  shots: number;
  bounce_ids: number[];
  serve_speed_kmh: number | null;
  hits: { t: number; player: Player; speed_kmh: number; height_m: number }[];
  net_clearance_m: number[];
  score_after: ScoreState;
}

export interface Bounce {
  id: number;
  frame: number;
  t: number;
  image_xy: Point2;
  court_xy: Point2;
  side: Side;
  in: boolean;
  margin_cm: number;
  kind: 'serve' | 'rally';
  hitter: Player;
  rally: number;
}

export interface PlayerStats {
  serves: number;
  first_serves: number;
  first_serves_in: number;
  second_serves: number;
  second_serves_in: number;
  aces: number;
  double_faults: number;
  rally_shots: number;
  rally_in: number;
  deep_shots: number;
  winners: number;
  errors_out: number;
  errors_net_or_missed: number;
  landing: Point2[];
  first_serve_pct: number | null;
  rally_in_pct: number | null;
  deep_pct: number | null;
  points_won: number;
  serve_speed_avg_kmh: number | null;
  serve_speed_max_kmh: number | null;
  shot_speed_avg_kmh: number | null;
}

export interface Report {
  players: Record<Player, string>;
  points: PointRecord[];
  bounces: Bounce[];
  score: ScoreState;
  stats: {
    players: Record<Player, PlayerStats>;
    rallies: number;
    avg_rally_shots: number;
    longest_rally_shots: number;
    close_calls: number[];
  };
  video: { fps: number; frames: number; width: number; height: number };
  court: { source: 'auto' | 'manual'; score: number | null; corners_px: Point2[] };
  ball_track: [number, number, number][]; // frame, x, y in source pixels
}

export interface AnalysisOptions {
  corners?: Point2[];
  doubles: boolean;
  best_of: number;
  no_ad: boolean;
  first_server: Player;
  a_starts_near: boolean;
  player_names: [string, string];
}

export type AnalysisStatus =
  | { id: string; status: 'queued' | 'running'; progress: number }
  | { id: string; status: 'done'; progress: number; report: Report }
  | { id: string; status: 'error'; progress: number; error: string; message: string };
