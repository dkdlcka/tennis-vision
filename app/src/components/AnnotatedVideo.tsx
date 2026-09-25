import { useEvent } from 'expo';
import { useVideoPlayer, VideoPlayer, VideoView } from 'expo-video';
import { forwardRef, useImperativeHandle, useMemo, useState } from 'react';
import { LayoutChangeEvent, StyleSheet, View } from 'react-native';
import Svg, { Circle, G, Polyline, Rect, Text as SvgText } from 'react-native-svg';

import type { Report } from '../lib/types';

export interface AnnotatedVideoHandle {
  seek: (seconds: number) => void;
}

interface Props {
  uri: string;
  report: Report;
}

const TRAIL_FRAMES = 12;
const CALL_SHOW_S = 1.2;

/** The original video with the ball trail and each bounce's line call drawn on top. */
export const AnnotatedVideo = forwardRef<AnnotatedVideoHandle, Props>(function AnnotatedVideo({ uri, report }, ref) {
  const player = useVideoPlayer(uri, (p: VideoPlayer) => {
    p.timeUpdateEventInterval = 1 / 30;
  });
  const { currentTime } = useEvent(player, 'timeUpdate', {
    currentTime: 0,
    currentLiveTimestamp: null,
    currentOffsetFromLive: null,
    bufferedPosition: 0,
  });
  const [box, setBox] = useState({ w: 0, h: 0 });

  useImperativeHandle(ref, () => ({
    seek: (s: number) => {
      player.currentTime = Math.max(0, s);
      player.play();
    },
  }));

  const { fps, width: vw, height: vh } = report.video;
  // Where the video sits inside the view with contentFit="contain".
  const scale = box.w && box.h ? Math.min(box.w / vw, box.h / vh) : 0;
  const ox = (box.w - vw * scale) / 2;
  const oy = (box.h - vh * scale) / 2;

  const trackByFrame = useMemo(() => {
    const m = new Map<number, [number, number]>();
    for (const [f, x, y] of report.ball_track) m.set(f, [x, y]);
    return m;
  }, [report.ball_track]);

  const frame = Math.round(currentTime * fps);
  const trail: string[] = [];
  for (let f = frame - TRAIL_FRAMES; f <= frame; f++) {
    const p = trackByFrame.get(f);
    if (p) trail.push(`${ox + p[0] * scale},${oy + p[1] * scale}`);
  }
  const ball = trackByFrame.get(frame);
  const calls = report.bounces.filter((b) => currentTime >= b.t && currentTime - b.t < CALL_SHOW_S);

  return (
    <View
      style={styles.wrap}
      onLayout={(e: LayoutChangeEvent) => setBox({ w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height })}
    >
      <VideoView player={player} style={StyleSheet.absoluteFill} contentFit="contain" nativeControls />
      {scale > 0 && (
        <Svg style={StyleSheet.absoluteFill} pointerEvents="none">
          {trail.length > 1 && <Polyline points={trail.join(' ')} stroke="#ffe600" strokeWidth={2} fill="none" opacity={0.8} />}
          {ball && <Circle cx={ox + ball[0] * scale} cy={oy + ball[1] * scale} r={5} stroke="#ffe600" strokeWidth={2} fill="none" />}
          {calls.map((b) => {
            const x = ox + b.image_xy[0] * scale;
            const y = oy + b.image_xy[1] * scale;
            const color = b.in ? '#43a047' : '#e53935';
            const label = `${b.in ? 'IN' : 'OUT'} ${Math.abs(b.margin_cm) < 100 ? `${Math.abs(b.margin_cm).toFixed(0)}cm` : ''}`;
            return (
              <G key={b.id}>
                <Circle cx={x} cy={y} r={9} stroke={color} strokeWidth={3} fill="none" />
                <Rect x={x + 12} y={y - 22} width={label.length * 8 + 10} height={20} rx={4} fill={color} />
                <SvgText x={x + 17} y={y - 8} fill="#fff" fontSize={13} fontWeight="bold">
                  {label}
                </SvgText>
              </G>
            );
          })}
        </Svg>
      )}
    </View>
  );
});

const styles = StyleSheet.create({
  wrap: { width: '100%', aspectRatio: 16 / 9, backgroundColor: '#000' },
});
