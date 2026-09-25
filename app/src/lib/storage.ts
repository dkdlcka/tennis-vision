import AsyncStorage from '@react-native-async-storage/async-storage';

const SERVER_KEY = 'serverUrl';
const HISTORY_KEY = 'history';

export const DEFAULT_SERVER = 'http://192.168.0.10:8000';

export interface HistoryItem {
  id: string;
  videoUri: string;
  createdAt: string;
  title: string;
}

export async function getServerUrl(): Promise<string> {
  return (await AsyncStorage.getItem(SERVER_KEY)) ?? DEFAULT_SERVER;
}

export async function setServerUrl(url: string): Promise<void> {
  await AsyncStorage.setItem(SERVER_KEY, url.replace(/\/+$/, ''));
}

export async function getHistory(): Promise<HistoryItem[]> {
  const raw = await AsyncStorage.getItem(HISTORY_KEY);
  return raw ? (JSON.parse(raw) as HistoryItem[]) : [];
}

export async function addHistory(item: HistoryItem): Promise<void> {
  const items = await getHistory();
  await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify([item, ...items.filter((i) => i.id !== item.id)].slice(0, 50)));
}
