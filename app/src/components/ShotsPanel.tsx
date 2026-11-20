import { useMemo, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { DEPTH_LABEL, DIRECTION_LABEL, SHOT_COLOR, SHOT_LABEL, ZONE_LABEL } from '../lib/court';
import type { Player, Report, Shot, ShotType } from '../lib/types';
import { CourtDiagram } from './CourtDiagram';

const TYPES: ShotType[] = ['serve', 'forehand', 'backhand', 'volley', 'overhead', 'stroke'];

interface Props {
  report: Report;
  width: number;
  onSeek: (seconds: number) => void;
}

/** Every shot: filter by player and stroke, see speed and where it landed. */
export function ShotsPanel({ report, width, onSeek }: Props) {
  const names = report.players;
  const [who, setWho] = useState<Player | 'all'>('all');
  const [type, setType] = useState<ShotType | 'all'>('all');
  const [selected, setSelected] = useState<number | null>(null);

  const present = useMemo(() => TYPES.filter((t) => report.shots.some((s) => s.type === t)), [report.shots]);
  const shots = report.shots.filter((s) => (who === 'all' || s.player === who) && (type === 'all' || s.type === type));
  const bouncesById = new Map(report.bounces.map((b) => [b.id, b]));
  const shown = shots.flatMap((s) => (s.bounce_id !== null && bouncesById.has(s.bounce_id) ? [bouncesById.get(s.bounce_id)!] : []));
  const shotByBounce = new Map(shots.filter((s) => s.bounce_id !== null).map((s) => [s.bounce_id as number, s]));
  const paths = shots.flatMap((s) => {
    const from = s.contact_xy ?? s.player_xy;
    return from && s.bounce_xy
      ? [{ from, to: s.bounce_xy, color: SHOT_COLOR[s.type], bold: s.id === selected }]
      : [];
  });
  const sel = selected === null ? null : report.shots[selected];

  const speeds = shots.map((s) => s.speed_kmh).filter((v): v is number => v !== null);
  const landed = shots.filter((s) => s.in !== null);
  const inCount = landed.filter((s) => s.in).length;
  const depths = shots.filter((s) => s.depth);
  const deep = depths.filter((s) => s.depth === 'deep').length;

  if (report.shots.length === 0) return <Text style={styles.muted}>찾은 샷이 없어요.</Text>;

  return (
    <View style={{ gap: 12 }}>
      <View style={styles.row}>
        {(['all', 'A', 'B'] as const).map((p) => (
          <Chip key={p} on={who === p} onPress={() => setWho(p)}>
            {p === 'all' ? '두 선수' : names[p]}
          </Chip>
        ))}
      </View>
      <View style={styles.row}>
        <Chip on={type === 'all'} onPress={() => setType('all')}>
          모든 샷
        </Chip>
        {present.map((t) => (
          <Chip key={t} on={type === t} onPress={() => setType(t)} color={SHOT_COLOR[t]}>
            {SHOT_LABEL[t]}
          </Chip>
        ))}
      </View>

      <View style={styles.cards}>
        <Card label="샷" value={`${shots.length}`} />
        <Card label="인" value={landed.length ? `${Math.round((100 * inCount) / landed.length)}%` : '-'} />
        <Card label="평균 속도" value={speeds.length ? `${Math.round(avg(speeds))}` : '-'} unit="km/h" />
        <Card label="최고 속도" value={speeds.length ? `${Math.round(Math.max(...speeds))}` : '-'} unit="km/h" />
        <Card label="깊은 샷" value={depths.length ? `${Math.round((100 * deep) / depths.length)}%` : '-'} />
      </View>

      {width > 0 && (
        <CourtDiagram
          bounces={shown}
          width={width}
          paths={paths}
          highlight={sel?.bounce_id ?? null}
          colorFor={(b) => SHOT_COLOR[shotByBounce.get(b.id)?.type ?? 'stroke']}
        />
      )}
      <Text style={styles.muted}>선: 친 곳 → 떨어진 곳 · ● 인 · ○ 아웃 · 색은 샷 종류</Text>

      {shots.map((s) => (
        <Pressable
          key={s.id}
          style={[styles.item, selected === s.id && styles.itemOn]}
          onPress={() => {
            setSelected(s.id);
            const t = s.t ?? report.points[s.point]?.start_t;
            if (t !== undefined && t !== null) onSeek(t - 1);
          }}
        >
          <View style={[styles.dot, { backgroundColor: SHOT_COLOR[s.type] }]} />
          <View style={{ flex: 1 }}>
            <Text style={styles.title}>
              {names[s.player]} {SHOT_LABEL[s.type]}
              {s.type === 'volley' && s.wing ? ` (${SHOT_LABEL[s.wing]})` : ''}
            </Text>
            <Text style={styles.muted}>{describe(s)}</Text>
          </View>
          <View style={{ alignItems: 'flex-end' }}>
            <Text style={styles.speed}>{s.speed_kmh !== null ? `${Math.round(s.speed_kmh)}` : '-'}</Text>
            <Text style={styles.unit}>km/h</Text>
          </View>
          <Text style={[styles.call, { color: s.in === null ? '#999' : s.in ? '#2e7d32' : '#c62828' }]}>
            {s.in === null ? '' : s.in ? 'IN' : 'OUT'}
          </Text>
        </Pressable>
      ))}
    </View>
  );
}

function describe(s: Shot) {
  const parts = [`포인트 ${s.point + 1}`];
  if (s.zone) parts.push(ZONE_LABEL[s.zone]);
  if (s.direction) parts.push(DIRECTION_LABEL[s.direction]);
  if (s.depth) parts.push(DEPTH_LABEL[s.depth]);
  if (s.miss) parts.push(s.miss === 'long' ? '길게 아웃' : s.miss === 'wide' ? '옆으로 아웃' : '네트');
  if (s.contact_height_m !== null && s.index > 0) parts.push(`타점 ${s.contact_height_m.toFixed(1)}m`);
  return parts.join(' · ');
}

function avg(v: number[]) {
  return v.reduce((a, b) => a + b, 0) / v.length;
}

function Card({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardValue}>
        {value}
      </Text>
      {unit && value !== '-' ? <Text style={styles.cardUnit}>{unit}</Text> : null}
      <Text style={styles.cardLabel}>{label}</Text>
    </View>
  );
}

function Chip({
  on,
  onPress,
  color = '#1b5e20',
  children,
}: {
  on: boolean;
  onPress: () => void;
  color?: string;
  children: React.ReactNode;
}) {
  return (
    <Pressable onPress={onPress} style={[styles.chip, on && { backgroundColor: color }]}>
      <Text style={[styles.chipText, { color: on ? '#fff' : color }]}>{children}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  muted: { fontSize: 13, color: '#666' },
  chip: { paddingVertical: 7, paddingHorizontal: 12, borderRadius: 18, backgroundColor: '#eef2ee' },
  chipText: { fontWeight: '700', fontSize: 14 },
  cards: { flexDirection: 'row', gap: 6 },
  card: { flex: 1, minWidth: 0, backgroundColor: '#12301f', borderRadius: 10, paddingVertical: 10, paddingHorizontal: 8 },
  cardValue: { color: '#fff', fontSize: 17, fontWeight: '800' },
  cardUnit: { color: '#9cc3a8', fontSize: 10 },
  cardLabel: { color: '#cfe8d5', fontSize: 12, marginTop: 2 },
  item: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 12, borderRadius: 10, backgroundColor: '#f6f8f6' },
  itemOn: { borderWidth: 2, borderColor: '#1b5e20' },
  dot: { width: 10, height: 10, borderRadius: 5 },
  title: { fontSize: 15, fontWeight: '700' },
  speed: { fontSize: 18, fontWeight: '800' },
  unit: { fontSize: 11, color: '#888' },
  call: { width: 38, textAlign: 'right', fontWeight: '800' },
});
