import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

export default function RootLayout() {
  return (
    <>
      <StatusBar style="dark" />
      <Stack screenOptions={{ headerTintColor: '#1b5e20' }}>
        <Stack.Screen name="index" options={{ title: 'Tennis Vision' }} />
        <Stack.Screen name="setup" options={{ title: '분석 준비' }} />
        <Stack.Screen name="analysis/[id]" options={{ title: '분석 결과' }} />
      </Stack>
    </>
  );
}
