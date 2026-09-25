import type { AnalysisOptions, AnalysisStatus, Point2 } from './types';

function fileField(uri: string, name: string, type: string) {
  // React Native's FormData accepts a file reference in this shape.
  return { uri, name, type } as unknown as Blob;
}

export async function detectCourt(
  server: string,
  imageUri: string,
): Promise<{ found: false } | { found: true; score: number; corners: Point2[] }> {
  const body = new FormData();
  body.append('image', fileField(imageUri, 'frame.jpg', 'image/jpeg'));
  const res = await fetch(`${server}/court/detect`, { method: 'POST', body });
  if (!res.ok) throw new Error(`코트 인식 요청 실패 (${res.status})`);
  return res.json();
}

export async function startAnalysis(
  server: string,
  videoUri: string,
  mimeType: string | undefined,
  options: AnalysisOptions,
): Promise<string> {
  const body = new FormData();
  const ext = mimeType?.includes('quicktime') ? 'mov' : 'mp4';
  body.append('video', fileField(videoUri, `match.${ext}`, mimeType ?? 'video/mp4'));
  body.append('options', JSON.stringify(options));
  const res = await fetch(`${server}/analyses`, { method: 'POST', body });
  if (!res.ok) throw new Error(`업로드 실패 (${res.status}): ${await res.text()}`);
  return (await res.json()).id as string;
}

export async function getAnalysis(server: string, id: string): Promise<AnalysisStatus> {
  const res = await fetch(`${server}/analyses/${id}`);
  if (!res.ok) throw new Error(`분석 상태 조회 실패 (${res.status})`);
  return res.json();
}
