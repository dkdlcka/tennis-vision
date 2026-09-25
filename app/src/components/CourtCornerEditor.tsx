import { useMemo, useRef, useState } from 'react';
import { Image, LayoutChangeEvent, PanResponder, StyleSheet, View } from 'react-native';
import Svg, { Circle, Line, Text as SvgText } from 'react-native-svg';

import { COURT_LINES } from '../lib/court';
import { apply, COURT_CORNERS, homographyFrom } from '../lib/homography';
import type { Point2 } from '../lib/types';

interface Props {
  imageUri: string;
  imageWidth: number;
  imageHeight: number;
  corners: Point2[]; // image pixels
  onChange: (corners: Point2[]) => void;
}

const LABELS = ['앞왼쪽', '앞오른쪽', '뒤오른쪽', '뒤왼쪽'];
const HANDLE = 14;

/** First frame with the court drawn on it; drag the four doubles corners to fit. */
export function CourtCornerEditor({ imageUri, imageWidth, imageHeight, corners, onChange }: Props) {
  const [viewWidth, setViewWidth] = useState(0);
  const scale = viewWidth / imageWidth;
  const viewHeight = imageHeight * scale;
  const latest = useRef({ corners, scale, onChange });
  latest.current = { corners, scale, onChange };

  const responders = useMemo(
    () =>
      [0, 1, 2, 3].map((i) => {
        let start: Point2 = [0, 0];
        return PanResponder.create({
          onStartShouldSetPanResponder: () => true,
          onPanResponderGrant: () => {
            start = latest.current.corners[i];
          },
          onPanResponderMove: (_, g) => {
            const { corners: cs, scale: s, onChange: cb } = latest.current;
            const next = cs.map((c) => [...c] as Point2);
            next[i] = [
              Math.min(imageWidth, Math.max(0, start[0] + g.dx / s)),
              Math.min(imageHeight, Math.max(0, start[1] + g.dy / s)),
            ];
            cb(next);
          },
        });
      }),
    [imageWidth, imageHeight],
  );

  const lines = useMemo(() => {
    if (corners.length !== 4) return [];
    const h = homographyFrom(COURT_CORNERS, corners);
    return COURT_LINES.map(([a, b]) => [apply(h, a), apply(h, b)]);
  }, [corners]);

  return (
    <View onLayout={(e: LayoutChangeEvent) => setViewWidth(e.nativeEvent.layout.width)} style={styles.wrap}>
      {viewWidth > 0 && (
        <View style={{ width: viewWidth, height: viewHeight }}>
          <Image source={{ uri: imageUri }} style={{ width: viewWidth, height: viewHeight }} />
          <Svg style={StyleSheet.absoluteFill} width={viewWidth} height={viewHeight}>
            {lines.map(([a, b], i) => (
              <Line
                key={i}
                x1={a[0] * scale}
                y1={a[1] * scale}
                x2={b[0] * scale}
                y2={b[1] * scale}
                stroke="#ffe600"
                strokeWidth={1.5}
              />
            ))}
            {corners.map((c, i) => (
              <SvgText key={i} x={c[0] * scale + 10} y={c[1] * scale - 10} fill="#ffe600" fontSize={11}>
                {LABELS[i]}
              </SvgText>
            ))}
            {corners.map((c, i) => (
              <Circle key={`c${i}`} cx={c[0] * scale} cy={c[1] * scale} r={6} fill="#ffe600" />
            ))}
          </Svg>
          {corners.map((c, i) => (
            <View
              key={i}
              {...responders[i].panHandlers}
              hitSlop={12}
              style={[
                styles.handle,
                { left: c[0] * scale - HANDLE, top: c[1] * scale - HANDLE, width: HANDLE * 2, height: HANDLE * 2 },
              ]}
            />
          ))}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { width: '100%' },
  handle: { position: 'absolute', borderRadius: HANDLE, borderWidth: 2, borderColor: '#ffe600' },
});
