import { useVideoPlayer, VideoView } from 'expo-video';
import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { mediaUrl } from '../lib/api';
import { REASON_LABEL } from '../lib/court';
import type { Report } from '../lib/types';

type Source = { kind: 'condensed' } | { kind: 'highlights' } | { kind: 'point'; index: number };

interface Props {
  report: Report;
  server: string;
  analysisId: string;
}

/** Auto-edited videos: only the points, a highlight reel, and each highlighted point on its own. */
export function HighlightsPanel({ report, server, analysisId }: Props) {
  const media = report.media ?? { condensed: false, highlights: false };
  const [source, setSource] = useState<Source | null>(
    media.highlights ? { kind: 'highlights' } : media.condensed ? { kind: 'condensed' } : null,
  );
  const names = report.players;
  const url =
    source === null
      ? null
      : source.kind === 'point'
        ? mediaUrl.point(server, analysisId, source.index)
        : mediaUrl[source.kind](server, analysisId);

  const saved = report.condensed.original_s ? report.condensed.original_s - report.condensed.duration_s : 0;
  const isOn = (s: Source) =>
    source !== null && source.kind === s.kind && (s.kind !== 'point' || (source.kind === 'point' && source.index === s.index));

  if (!media.condensed && !media.highlights && report.highlights.length === 0) {
    return <Text style={styles.muted}>아직 하이라이트로 고를 만한 포인트가 없어요.</Text>;
  }

  return (
    <View style={{ gap: 12 }}>
      {url ? <ClipPlayer key={url} url={url} /> : null}
      {!media.condensed && !media.highlights && (
        <Text style={styles.muted}>서버에 ffmpeg가 없어서 편집 영상을 만들지 못했어요. 목록의 시간으로 원본에서 찾아볼 수 있어요.</Text>
      )}

      <View style={styles.row}>
        {media.highlights && (
          <Chip on={isOn({ kind: 'highlights' })} onPress={() => setSource({ kind: 'highlights' })}>
            하이라이트 모음 ({report.reel_points.length}개)
          </Chip>
        )}
        {media.condensed && (
          <Chip on={isOn({ kind: 'condensed' })} onPress={() => setSource({ kind: 'condensed' })}>
            경기만 보기 {fmt(report.condensed.duration_s)}
          </Chip>
        )}
      </View>
      {media.condensed && saved > 1 && (
        <Text style={styles.muted}>
          공이 오가지 않은 {fmt(saved)}를 잘라냈어요 (원본 {fmt(report.condensed.original_s ?? 0)}).
        </Text>
      )}

      {report.highlights.map((h) => {
        const p = report.points[h.point];
        return (
          <Pressable
            key={h.point}
            style={[styles.item, isOn({ kind: 'point', index: h.point }) && styles.itemOn]}
            onPress={() => setSource({ kind: 'point', index: h.point })}
          >
            <View style={{ flex: 1, gap: 4 }}>
              <View style={styles.row}>
                {h.labels.map((l) => (
                  <Text key={l} style={styles.tag}>
                    {l}
                  </Text>
                ))}
              </View>
              <Text style={styles.title}>
                포인트 {h.point + 1} · {p.winner ? `${names[p.winner]} 득점` : '폴트'} ({REASON_LABEL[p.reason] ?? p.reason})
              </Text>
              <Text style={styles.muted}>
                {fmt(h.start_t)} · 바운스 {p.shots}회
                {p.serve_speed_kmh ? ` · 서브 ${Math.round(p.serve_speed_kmh)}km/h` : ''}
                {h.close_call_cm !== null ? ` · 라인과 ${h.close_call_cm.toFixed(0)}cm` : ''}
              </Text>
            </View>
            <Text style={styles.play}>▶</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

function ClipPlayer({ url }: { url: string }) {
  const player = useVideoPlayer(url, (p) => p.play());
  return (
    <View style={styles.video}>
      <VideoView player={player} style={styles.fill} contentFit="contain" nativeControls />
    </View>
  );
}

function Chip({ on, onPress, children }: { on: boolean; onPress: () => void; children: React.ReactNode }) {
  return (
    <Pressable onPress={onPress} style={[styles.chip, on && styles.chipOn]}>
      <Text style={on ? styles.chipOnText : styles.chipText}>{children}</Text>
    </Pressable>
  );
}

function fmt(seconds: number) {
  const s = Math.round(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

const styles = StyleSheet.create({
  // Explicit size too: on web the <video> element ignores top/right/bottom/left stretching.
  fill: { position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' },
  video: { width: '100%', aspectRatio: 16 / 9, backgroundColor: '#000', borderRadius: 8, overflow: 'hidden' },
  row: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  muted: { fontSize: 13, color: '#666' },
  chip: { paddingVertical: 8, paddingHorizontal: 14, borderRadius: 20, backgroundColor: '#eef2ee' },
  chipOn: { backgroundColor: '#1b5e20' },
  chipText: { color: '#1b5e20', fontWeight: '600' },
  chipOnText: { color: '#fff', fontWeight: '700' },
  item: { flexDirection: 'row', alignItems: 'center', padding: 12, borderRadius: 10, backgroundColor: '#f6f8f6' },
  itemOn: { borderWidth: 2, borderColor: '#1b5e20' },
  tag: {
    fontSize: 12,
    fontWeight: '700',
    color: '#5d4100',
    backgroundColor: '#ffe600',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
    overflow: 'hidden',
  },
  title: { fontSize: 15, fontWeight: '700' },
  play: { fontSize: 20, color: '#1b5e20', marginLeft: 8 },
});
