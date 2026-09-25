import { useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { RallyResultView } from '../../components/RallyResultView';
import { getRallies, ralliesVideoUrl } from '../../lib/api';
import { getServerUrl } from '../../lib/storage';
import type { RallyJob } from '../../lib/types';

export default function Rallies() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [job, setJob] = useState<RallyJob | null>(null);
  const [server, setServer] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    (async () => {
      const url = await getServerUrl();
      setServer(url);
      const poll = async () => {
        try {
          const j = await getRallies(url, id);
          if (!alive) return;
          setJob(j);
          setError(null);
          if (j.status === 'queued' || j.status === 'running') timer = setTimeout(poll, 2000);
        } catch (e) {
          if (!alive) return;
          setError((e as Error).message);
          timer = setTimeout(poll, 5000);
        }
      };
      poll();
    })();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [id]);

  if (job?.status === 'done' && server) {
    return <RallyResultView report={job.report} videoUrl={ralliesVideoUrl(server, id)} />;
  }

  return (
    <View style={styles.center}>
      {job?.status === 'error' ? (
        <>
          <Text style={styles.title}>분석하지 못했어요</Text>
          <Text style={styles.body}>{job.message}</Text>
        </>
      ) : (
        <>
          <ActivityIndicator size="large" />
          <Text style={styles.title}>
            {job?.status === 'running' ? `${job.stage} ${Math.round(job.progress * 100)}%` : '대기 중...'}
          </Text>
          <Text style={styles.body}>
            코트가 보이는 화면에서 랠리만 골라, 서브부터 공을 따라가며 바운스와 샷 속도를 표시한 영상을 만들고 있어요.
            영상 길이에 따라 오래 걸릴 수 있어요.
          </Text>
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
