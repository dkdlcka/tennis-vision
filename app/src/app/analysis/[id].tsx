import { useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { ResultView } from '../../components/ResultView';
import { getAnalysis } from '../../lib/api';
import { getServerUrl } from '../../lib/storage';
import type { AnalysisStatus } from '../../lib/types';

export default function Analysis() {
  const { id, uri } = useLocalSearchParams<{ id: string; uri: string }>();
  const [status, setStatus] = useState<AnalysisStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    (async () => {
      const server = await getServerUrl();
      const poll = async () => {
        try {
          const s = await getAnalysis(server, id);
          if (!alive) return;
          setStatus(s);
          setError(null);
          if (s.status === 'queued' || s.status === 'running') timer = setTimeout(poll, 1500);
        } catch (e) {
          if (!alive) return;
          setError((e as Error).message);
          timer = setTimeout(poll, 4000);
        }
      };
      poll();
    })();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [id]);

  if (status?.status === 'done') return <ResultView report={status.report} videoUri={uri} />;

  return (
    <View style={styles.center}>
      {status?.status === 'error' ? (
        <>
          <Text style={styles.title}>분석하지 못했어요</Text>
          <Text style={styles.body}>
            {status.error === 'court_not_found'
              ? '코트를 찾지 못했어요. 다시 올리면서 코트 모서리를 직접 맞춰 주세요.'
              : status.message}
          </Text>
        </>
      ) : (
        <>
          <ActivityIndicator size="large" />
          <Text style={styles.title}>
            {status?.status === 'running' ? `분석 중 ${Math.round(status.progress * 100)}%` : '대기 중...'}
          </Text>
          <Text style={styles.body}>공을 프레임마다 찾고, 궤적을 3D로 복원해 바운스를 판정하고 있어요.</Text>
          {error && <Text style={styles.muted}>{error}</Text>}
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24, gap: 12 },
  title: { fontSize: 18, fontWeight: '700' },
  body: { fontSize: 15, color: '#333', textAlign: 'center' },
  muted: { fontSize: 13, color: '#888' },
});
