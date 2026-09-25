import { useVideoPlayer, VideoPlayer, VideoView } from 'expo-video';
import { useState } from 'react';
import { LayoutChangeEvent, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import type { RallyPoint, RallyReport } from '../lib/types';
import { CourtDiagram } from './CourtDiagram';

/** The highlight video the server drew, with its points listed and their landings mapped. */
export function RallyResultView({ report, videoUrl }: { report: RallyReport; videoUrl: string }) {
  const player = useVideoPlayer(videoUrl, (p: VideoPlayer) => {
    p.play();
  });
  const [selected, setSelected] = useState<number | null>(null);
  const [width, setWidth] = useState(0);
  const { stats } = report;

  if (report.points.length === 0) {
    return (
      <View style={styles.empty}>
        <Text style={styles.title}>랠리를 찾지 못했어요</Text>
        <Text style={styles.body}>
          코트 전체가 보이는 화면(베이스라인 뒤 높은 곳에서 찍은 화면)이 있어야 해요. 서브가 보이지 않는 구간도 빠져요.
        </Text>
      </View>
    );
  }

  const shown: RallyPoint[] = selected === null ? report.points : [report.points[selected]];
  const marks = shown.flatMap((p) => p.bounces.map((b, i) => ({ id: p.index * 1000 + i, court_xy: b.court_xy, in: b.in })));

  return (
    <ScrollView
      contentContainerStyle={styles.page}
      onLayout={(e: LayoutChangeEvent) => setWidth(e.nativeEvent.layout.width - 32)}
    >
      <View style={styles.video}>
        <VideoView player={player} style={StyleSheet.absoluteFill} contentFit="contain" nativeControls />
      </View>

      <View style={styles.statsRow}>
        <Stat label="포인트" value={`${stats.points}`} />
        <Stat label="최고 서브" value={stats.serve_speed_max_kmh ? `${Math.round(stats.serve_speed_max_kmh)}km/h` : '-'} />
        <Stat label="평균 샷" value={stats.shot_speed_avg_kmh ? `${Math.round(stats.shot_speed_avg_kmh)}km/h` : '-'} />
        <Stat label="인" value={stats.bounces ? `${Math.round((100 * stats.bounces_in) / stats.bounces)}%` : '-'} />
      </View>

      <Text style={styles.h2}>포인트</Text>
      {report.points.map((p) => (
        <Pressable
          key={p.index}
          style={[styles.pointRow, selected === p.index && styles.pointOn]}
          onPress={() => {
            setSelected(selected === p.index ? null : p.index);
            player.currentTime = p.clip_start_t;
            player.play();
          }}
        >
          <Text style={styles.pointTitle}>
            포인트 {p.index + 1} · 원본 {clock(p.source_start_t)}
          </Text>
          <Text style={styles.muted}>
            {p.serve_speed_kmh ? `서브 ${Math.round(p.serve_speed_kmh)}km/h · ` : ''}
            바운스 {p.bounces.length}회{outNote(p)}
          </Text>
        </Pressable>
      ))}

      <Text style={styles.h2}>{selected === null ? '전체 착지 위치' : `포인트 ${selected + 1} 착지 위치`}</Text>
      {width > 0 && (
        <CourtDiagram bounces={marks} width={width} colorFor={(b) => (b.in ? '#43a047' : '#e53935')} />
      )}
      <Text style={styles.muted}>
        ● 인 · ○ 아웃 (서브는 서비스 박스 기준). 속도는 착지 위치와 비행 시간으로 추정한 라켓 직후 속도예요.
        {report.video.tracker === 'motion' ? ' 서버에 TrackNet이 설정되지 않아 공 추적 정확도가 낮을 수 있어요.' : ''}
      </Text>
    </ScrollView>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.muted}>{label}</Text>
    </View>
  );
}

function outNote(p: RallyPoint): string {
  const last = p.bounces[p.bounces.length - 1];
  if (!last) return '';
  if (last.serve && !last.in) return ' · 서브 폴트';
  return last.in ? '' : ' · 마지막 공 아웃';
}

function clock(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}

const styles = StyleSheet.create({
  page: { padding: 16, gap: 12, paddingBottom: 48 },
  video: { width: '100%', aspectRatio: 16 / 9, backgroundColor: '#000' },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24, gap: 12 },
  title: { fontSize: 18, fontWeight: '700' },
  h2: { fontSize: 16, fontWeight: '700', marginTop: 4 },
  body: { fontSize: 15, color: '#333', textAlign: 'center' },
  muted: { fontSize: 13, color: '#666' },
  statsRow: { flexDirection: 'row', gap: 8 },
  stat: { flex: 1, backgroundColor: '#f3f6f3', borderRadius: 10, paddingVertical: 10, alignItems: 'center' },
  statValue: { fontSize: 17, fontWeight: '700', color: '#1b5e20' },
  pointRow: { padding: 12, borderRadius: 10, backgroundColor: '#f7f7f7', gap: 2 },
  pointOn: { backgroundColor: '#e3f2e5' },
  pointTitle: { fontSize: 15, fontWeight: '600' },
});
