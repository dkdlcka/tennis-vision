import * as ImagePicker from 'expo-image-picker';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

import { getHistory, getServerUrl, HistoryItem, setServerUrl } from '../lib/storage';

export default function Home() {
  const [server, setServer] = useState('');
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useFocusEffect(
    useCallback(() => {
      getServerUrl().then(setServer);
      getHistory().then(setHistory);
    }, []),
  );

  async function pick(fromCamera: boolean) {
    const perm = fromCamera
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('권한이 필요해요', fromCamera ? '카메라 권한을 허용해 주세요.' : '사진 보관함 권한을 허용해 주세요.');
      return;
    }
    const options: ImagePicker.ImagePickerOptions = { mediaTypes: ['videos'], quality: 1 };
    const result = fromCamera
      ? await ImagePicker.launchCameraAsync(options)
      : await ImagePicker.launchImageLibraryAsync(options);
    if (result.canceled || !result.assets[0]) return;
    const a = result.assets[0];
    router.push({
      pathname: '/setup',
      params: { uri: a.uri, width: String(a.width), height: String(a.height), mimeType: a.mimeType ?? '' },
    });
  }

  return (
    <ScrollView contentContainerStyle={styles.page}>
      <Text style={styles.h1}>내 테니스 영상 분석</Text>
      <Text style={styles.body}>공 궤적을 따라가며 매 바운스마다 인/아웃을 판정하고, 점수를 자동으로 기록해요.</Text>

      <View style={styles.row}>
        <Pressable style={[styles.button, styles.primary]} onPress={() => pick(false)}>
          <Text style={styles.primaryText}>영상 선택</Text>
        </Pressable>
        <Pressable style={styles.button} onPress={() => pick(true)}>
          <Text style={styles.buttonText}>바로 촬영</Text>
        </Pressable>
      </View>

      <View style={styles.card}>
        <Text style={styles.h2}>촬영 가이드</Text>
        <Text style={styles.body}>
          {'• 한쪽 베이스라인 뒤, 가능한 높은 곳(펜스 위, 삼각대 2m 이상)에 폰을 고정하세요.\n' +
            '• 코트 네 모서리가 모두 화면에 들어오게 가로로 찍어 주세요.\n' +
            '• 60fps 이상, 1080p 이상이면 판정이 더 정확해져요.\n' +
            '• 촬영 중에는 폰을 움직이지 마세요.'}
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.h2}>분석 서버</Text>
        <TextInput
          style={styles.input}
          value={server}
          onChangeText={setServer}
          onEndEditing={() => setServerUrl(server)}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          placeholder="http://192.168.0.10:8000"
        />
      </View>

      {history.length > 0 && (
        <View style={styles.card}>
          <Text style={styles.h2}>최근 분석</Text>
          {history.map((h) => (
            <Pressable
              key={h.id}
              style={styles.historyRow}
              onPress={() => router.push({ pathname: '/analysis/[id]', params: { id: h.id, uri: h.videoUri } })}
            >
              <Text style={styles.body}>{h.title}</Text>
              <Text style={styles.muted}>{new Date(h.createdAt).toLocaleString()}</Text>
            </Pressable>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { padding: 20, gap: 16 },
  h1: { fontSize: 24, fontWeight: '700' },
  h2: { fontSize: 16, fontWeight: '700', marginBottom: 6 },
  body: { fontSize: 15, lineHeight: 22, color: '#222' },
  muted: { fontSize: 12, color: '#777' },
  row: { flexDirection: 'row', gap: 12 },
  button: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#1b5e20',
  },
  primary: { backgroundColor: '#1b5e20' },
  primaryText: { color: '#fff', fontWeight: '700', fontSize: 16 },
  buttonText: { color: '#1b5e20', fontWeight: '700', fontSize: 16 },
  card: { backgroundColor: '#f3f6f3', borderRadius: 12, padding: 14 },
  input: { backgroundColor: '#fff', borderRadius: 8, padding: 10, fontSize: 15, borderWidth: 1, borderColor: '#ccc' },
  historyRow: { paddingVertical: 8, borderBottomWidth: StyleSheet.hairlineWidth, borderColor: '#ccc' },
});
