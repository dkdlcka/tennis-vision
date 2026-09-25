import * as VideoThumbnails from 'expo-video-thumbnails';
import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, Pressable, ScrollView, StyleSheet, Switch, Text, TextInput, View } from 'react-native';

import { CourtCornerEditor } from '../components/CourtCornerEditor';
import { detectCourt, startAnalysis } from '../lib/api';
import { addHistory, getServerUrl } from '../lib/storage';
import type { Player, Point2 } from '../lib/types';

type Params = { uri: string; width: string; height: string; mimeType: string };

export default function Setup() {
  const params = useLocalSearchParams<Params>();
  const [frame, setFrame] = useState<{ uri: string; width: number; height: number } | null>(null);
  const [corners, setCorners] = useState<Point2[]>([]);
  const [courtNote, setCourtNote] = useState('코트를 찾는 중...');
  const [names, setNames] = useState<[string, string]>(['나', '상대']);
  const [firstServer, setFirstServer] = useState<Player>('A');
  const [aNear, setANear] = useState(true);
  const [doubles, setDoubles] = useState(false);
  const [noAd, setNoAd] = useState(false);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    (async () => {
      const thumb = await VideoThumbnails.getThumbnailAsync(params.uri, { time: 500, quality: 0.9 });
      setFrame(thumb);
      // A starting guess the user can drag into place if detection fails.
      const w = thumb.width;
      const h = thumb.height;
      setCorners([
        [w * 0.08, h * 0.92],
        [w * 0.92, h * 0.92],
        [w * 0.66, h * 0.28],
        [w * 0.34, h * 0.28],
      ]);
      try {
        const res = await detectCourt(await getServerUrl(), thumb.uri);
        if (res.found) {
          setCorners(res.corners);
          setCourtNote('코트를 자동으로 찾았어요. 선이 맞지 않으면 노란 점을 끌어 맞춰 주세요.');
        } else {
          setCourtNote('코트를 자동으로 못 찾았어요. 노란 점 4개를 복식 코트 네 모서리에 맞춰 주세요.');
        }
      } catch (e) {
        setCourtNote(`서버에 연결하지 못했어요. 노란 점을 직접 맞춰 주세요. (${(e as Error).message})`);
      }
    })();
  }, [params.uri]);

  async function start() {
    if (!frame) return;
    setUploading(true);
    try {
      // Corners were placed on the thumbnail; send them in the video's own pixels.
      const vw = Number(params.width) || frame.width;
      const vh = Number(params.height) || frame.height;
      const sx = vw / frame.width;
      const sy = vh / frame.height;
      const server = await getServerUrl();
      const id = await startAnalysis(server, params.uri, params.mimeType || undefined, {
        corners: corners.map(([x, y]) => [x * sx, y * sy]),
        doubles,
        best_of: 3,
        no_ad: noAd,
        first_server: firstServer,
        a_starts_near: aNear,
        player_names: names,
      });
      await addHistory({
        id,
        videoUri: params.uri,
        createdAt: new Date().toISOString(),
        title: `${names[0]} vs ${names[1]}`,
      });
      router.replace({ pathname: '/analysis/[id]', params: { id, uri: params.uri } });
    } catch (e) {
      Alert.alert('업로드 실패', (e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.page}>
      <Text style={styles.h2}>1. 코트 맞추기</Text>
      {frame ? (
        <CourtCornerEditor
          imageUri={frame.uri}
          imageWidth={frame.width}
          imageHeight={frame.height}
          corners={corners}
          onChange={setCorners}
        />
      ) : (
        <ActivityIndicator />
      )}
      <Text style={styles.muted}>{courtNote}</Text>

      <Text style={styles.h2}>2. 선수</Text>
      <View style={styles.row}>
        <TextInput style={styles.input} value={names[0]} onChangeText={(t) => setNames([t, names[1]])} />
        <Text>vs</Text>
        <TextInput style={styles.input} value={names[1]} onChangeText={(t) => setNames([names[0], t])} />
      </View>
      <Choice
        label="첫 서브"
        options={[names[0], names[1]]}
        value={firstServer === 'A' ? 0 : 1}
        onChange={(i) => setFirstServer(i === 0 ? 'A' : 'B')}
      />
      <Choice
        label="카메라 쪽(화면 아래)에 있는 선수"
        options={[names[0], names[1]]}
        value={aNear ? 0 : 1}
        onChange={(i) => setANear(i === 0)}
      />

      <Text style={styles.h2}>3. 규칙</Text>
      <Toggle label="복식" value={doubles} onChange={setDoubles} />
      <Toggle label="노애드(40-40에서 한 포인트로 결정)" value={noAd} onChange={setNoAd} />

      <Pressable style={[styles.primary, uploading && { opacity: 0.6 }]} disabled={uploading || !frame} onPress={start}>
        <Text style={styles.primaryText}>{uploading ? '업로드 중...' : '분석 시작'}</Text>
      </Pressable>
    </ScrollView>
  );
}

function Choice(props: { label: string; options: string[]; value: number; onChange: (i: number) => void }) {
  return (
    <View style={{ gap: 6 }}>
      <Text style={styles.body}>{props.label}</Text>
      <View style={styles.row}>
        {props.options.map((o, i) => (
          <Pressable
            key={i}
            onPress={() => props.onChange(i)}
            style={[styles.chip, props.value === i && styles.chipOn]}
          >
            <Text style={props.value === i ? styles.chipOnText : styles.body}>{o}</Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

function Toggle(props: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <View style={[styles.row, { justifyContent: 'space-between' }]}>
      <Text style={styles.body}>{props.label}</Text>
      <Switch value={props.value} onValueChange={props.onChange} />
    </View>
  );
}

const styles = StyleSheet.create({
  page: { padding: 16, gap: 14, paddingBottom: 48 },
  h2: { fontSize: 17, fontWeight: '700', marginTop: 8 },
  body: { fontSize: 15, color: '#222' },
  muted: { fontSize: 13, color: '#666' },
  row: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  input: { flex: 1, backgroundColor: '#fff', borderRadius: 8, padding: 10, borderWidth: 1, borderColor: '#ccc' },
  chip: { paddingVertical: 8, paddingHorizontal: 14, borderRadius: 20, borderWidth: 1, borderColor: '#1b5e20' },
  chipOn: { backgroundColor: '#1b5e20' },
  chipOnText: { color: '#fff', fontSize: 15 },
  primary: { backgroundColor: '#1b5e20', padding: 16, borderRadius: 12, alignItems: 'center', marginTop: 12 },
  primaryText: { color: '#fff', fontSize: 17, fontWeight: '700' },
});
