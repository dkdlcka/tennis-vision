import { useRef, useState } from 'react';
import { LayoutChangeEvent, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { REASON_LABEL } from '../lib/court';
import type { Player, PlayerStats, Report, ScoreState } from '../lib/types';
import { AnnotatedVideo, AnnotatedVideoHandle } from './AnnotatedVideo';
import { CourtDiagram } from './CourtDiagram';

type Tab = 'points' | 'map' | 'stats';
const PLAYER_COLOR: Record<Player, string> = { A: '#1e88e5', B: '#e53935' };

export function ResultView({ report, videoUri }: { report: Report; videoUri: string }) {
  const video = useRef<AnnotatedVideoHandle>(null);
  const [tab, setTab] = useState<Tab>('points');
  const [width, setWidth] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [mapFilter, setMapFilter] = useState<Player | 'all'>('all');
  const names = report.players;

  const selectedPoint = selected === null ? null : report.points[selected];
  const mapBounces = report.bounces.filter(
    (b) =>
      (mapFilter === 'all' || b.hitter === mapFilter) &&
      (selectedPoint === null || selectedPoint.bounce_ids.includes(b.id)),
  );

  return (
    <ScrollView
      contentContainerStyle={styles.page}
      onLayout={(e: LayoutChangeEvent) => setWidth(e.nativeEvent.layout.width - 32)}
    >
      <AnnotatedVideo ref={video} uri={videoUri} report={report} />
      <Scoreboard score={report.score} names={names} />

      <View style={styles.tabs}>
        {(['points', 'map', 'stats'] as Tab[]).map((t) => (
          <Pressable key={t} onPress={() => setTab(t)} style={[styles.tab, tab === t && styles.tabOn]}>
            <Text style={tab === t ? styles.tabOnText : styles.tabText}>
              {t === 'points' ? '포인트' : t === 'map' ? '바운스 맵' : '통계'}
            </Text>
          </Pressable>
        ))}
      </View>

      {tab === 'points' &&
        report.points.map((p) => (
          <Pressable
            key={p.index}
            style={[styles.pointRow, selected === p.index && styles.pointOn]}
            onPress={() => {
              setSelected(p.index);
              video.current?.seek(p.start_t - 0.5);
            }}
          >
            <View style={{ flex: 1 }}>
              <Text style={styles.pointTitle}>
                {p.winner ? `${names[p.winner]} 득점` : '폴트'} · {REASON_LABEL[p.reason] ?? p.reason}
              </Text>
              <Text style={styles.muted}>
                {names[p.server]} {p.second_serve ? '세컨드' : '퍼스트'} 서브 ({p.serve_box === 'deuce' ? '듀스' : '애드'})
                {p.serve_speed_kmh ? ` · ${Math.round(p.serve_speed_kmh)}km/h` : ''} · 바운스 {p.shots}회
              </Text>
            </View>
            <Text style={styles.pointScore}>{p.score_after.points.join(' - ')}</Text>
          </Pressable>
        ))}

      {tab === 'map' && width > 0 && (
        <View style={{ gap: 10 }}>
          <View style={styles.tabs}>
            {(['all', 'A', 'B'] as const).map((f) => (
              <Pressable key={f} onPress={() => setMapFilter(f)} style={[styles.tab, mapFilter === f && styles.tabOn]}>
                <Text style={mapFilter === f ? styles.tabOnText : styles.tabText}>{f === 'all' ? '전체' : names[f]}</Text>
              </Pressable>
            ))}
          </View>
          {selectedPoint && (
            <Pressable onPress={() => setSelected(null)}>
              <Text style={styles.muted}>포인트 {selectedPoint.index + 1}만 보는 중 · 눌러서 전체 보기</Text>
            </Pressable>
          )}
          <CourtDiagram bounces={mapBounces} width={width} colorFor={(b) => PLAYER_COLOR[b.hitter]} />
          <Text style={styles.muted}>● 인 · ○ 아웃 · 파랑 {names.A} · 빨강 {names.B}가 친 공</Text>
        </View>
      )}

      {tab === 'stats' && (
        <View style={{ gap: 12 }}>
          <StatsTable a={report.stats.players.A} b={report.stats.players.B} names={names} />
          <Text style={styles.muted}>
            랠리 {report.stats.rallies}회 · 평균 바운스 {report.stats.avg_rally_shots}회 · 최장 {report.stats.longest_rally_shots}회
            {report.stats.close_calls.length ? ` · 라인 5cm 이내 접전 ${report.stats.close_calls.length}회` : ''}
          </Text>
        </View>
      )}
    </ScrollView>
  );
}

function Scoreboard({ score, names }: { score: ScoreState; names: Record<Player, string> }) {
  return (
    <View style={styles.board}>
      {(['A', 'B'] as Player[]).map((p, i) => (
        <View key={p} style={styles.boardRow}>
          <Text style={styles.boardName}>
            {score.server === p ? '● ' : '   '}
            {names[p]}
          </Text>
          {score.sets.map((s, k) => (
            <Text key={k} style={styles.boardSet}>
              {s[i]}
            </Text>
          ))}
          <Text style={styles.boardPoints}>{score.points[i]}</Text>
        </View>
      ))}
    </View>
  );
}

function StatsTable({ a, b, names }: { a: PlayerStats; b: PlayerStats; names: Record<Player, string> }) {
  const pct = (v: number | null) => (v === null ? '-' : `${v}%`);
  const num = (v: number | null) => (v === null ? '-' : `${Math.round(v)}`);
  const rows: [string, string, string][] = [
    ['득점', `${a.points_won}`, `${b.points_won}`],
    ['퍼스트 서브 성공률', pct(a.first_serve_pct), pct(b.first_serve_pct)],
    ['에이스', `${a.aces}`, `${b.aces}`],
    ['더블 폴트', `${a.double_faults}`, `${b.double_faults}`],
    ['서브 속도 평균 (km/h)', num(a.serve_speed_avg_kmh), num(b.serve_speed_avg_kmh)],
    ['서브 속도 최고 (km/h)', num(a.serve_speed_max_kmh), num(b.serve_speed_max_kmh)],
    ['스트로크 속도 평균 (km/h)', num(a.shot_speed_avg_kmh), num(b.shot_speed_avg_kmh)],
    ['랠리 샷 인 비율', pct(a.rally_in_pct), pct(b.rally_in_pct)],
    ['깊은 샷 비율 (서비스라인 뒤)', pct(a.deep_pct), pct(b.deep_pct)],
    ['위너', `${a.winners}`, `${b.winners}`],
    ['아웃 에러', `${a.errors_out}`, `${b.errors_out}`],
    ['네트·투바운드 에러', `${a.errors_net_or_missed}`, `${b.errors_net_or_missed}`],
  ];
  return (
    <View style={styles.table}>
      <View style={styles.tableRow}>
        <Text style={[styles.cellLabel, styles.bold]} />
        <Text style={[styles.cell, styles.bold, { color: PLAYER_COLOR.A }]}>{names.A}</Text>
        <Text style={[styles.cell, styles.bold, { color: PLAYER_COLOR.B }]}>{names.B}</Text>
      </View>
      {rows.map(([label, x, y]) => (
        <View key={label} style={styles.tableRow}>
          <Text style={styles.cellLabel}>{label}</Text>
          <Text style={styles.cell}>{x}</Text>
          <Text style={styles.cell}>{y}</Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  page: { padding: 16, gap: 14, paddingBottom: 48 },
  muted: { fontSize: 13, color: '#666' },
  bold: { fontWeight: '700' },
  tabs: { flexDirection: 'row', gap: 8 },
  tab: { paddingVertical: 8, paddingHorizontal: 14, borderRadius: 20, backgroundColor: '#eef2ee' },
  tabOn: { backgroundColor: '#1b5e20' },
  tabText: { color: '#1b5e20', fontWeight: '600' },
  tabOnText: { color: '#fff', fontWeight: '700' },
  pointRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 10,
    backgroundColor: '#f6f8f6',
  },
  pointOn: { borderWidth: 2, borderColor: '#1b5e20' },
  pointTitle: { fontSize: 15, fontWeight: '700' },
  pointScore: { fontSize: 16, fontWeight: '700', marginLeft: 8 },
  board: { backgroundColor: '#12301f', borderRadius: 10, padding: 12, gap: 6 },
  boardRow: { flexDirection: 'row', alignItems: 'center' },
  boardName: { flex: 1, color: '#fff', fontSize: 16, fontWeight: '600' },
  boardSet: { color: '#cfe8d5', fontSize: 16, width: 26, textAlign: 'center' },
  boardPoints: { color: '#ffe600', fontSize: 18, fontWeight: '800', width: 44, textAlign: 'right' },
  table: { borderRadius: 10, backgroundColor: '#f6f8f6', padding: 8 },
  tableRow: { flexDirection: 'row', paddingVertical: 6 },
  cellLabel: { flex: 2, fontSize: 14, color: '#333' },
  cell: { flex: 1, fontSize: 14, textAlign: 'center' },
});
