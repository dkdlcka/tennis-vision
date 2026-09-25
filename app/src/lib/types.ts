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
  zone?: 'wide' | 'body' | 'T'; // serves that landed in
  direction?: 'cross' | 'line' | 'center' | null; // rally shots
  miss?: 'long' | 'wide' | 'net'; // out balls
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
  serve_zones: { wide: number; body: number; T: number };
  directions: { cross: number; line: number; center: number };
  errors_long: number;
  errors_wide: number;
  errors_net: number;
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

// Rally highlights (server/tennis_vision/highlights.py).

export interface RallyBounce {
  clip_t: number; // seconds into the highlight video
  source_t: number; // seconds into the uploaded video
  court_xy: Point2;
  in: boolean;
  serve: boolean; // where the serve landed (judged against the service boxes)
  speed_kmh: number | null; // estimated speed off the racket of the shot that landed here
}

export interface RallyPoint {
  index: number;
  clip_start_t: number;
  clip_end_t: number;
  source_start_t: number;
  source_end_t: number;
  serve_seen: boolean;
  serve_speed_kmh: number | null;
  shots: number;
  bounces: RallyBounce[];
}

export interface RallyReport {
  video: { fps: number; width: number; height: number; court_shots: number; tracker: 'tracknet' | 'motion' };
  points: RallyPoint[];
  stats: {
    points: number;
    highlight_s: number;
    bounces: number;
    bounces_in: number;
    serve_speed_max_kmh: number | null;
    serve_speed_avg_kmh: number | null;
    shot_speed_avg_kmh: number | null;
  };
}

export type RallyJob =
  | { id: string; status: 'queued' | 'running'; progress: number; stage: string }
  | { id: string; status: 'done'; progress: number; stage: string; report: RallyReport }
  | { id: string; status: 'error'; progress: number; error: string; message: string };
