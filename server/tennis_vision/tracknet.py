"""Learned ball detector: TrackNet (Huang et al., 2019).

Three consecutive frames go in, a heatmap of where the ball is in the last one
comes out. It finds the ball where the motion-and-color detector fails (a
blurred, blown-out ball, a ball over white lines or clothes) and ignores what
fooled it (players, ball kids, the crowd).

The network below is the paper's encoder-decoder. Weights are not bundled:
pass a PyTorch state dict trained for this layout at 640x360, for example the
broadcast-tennis weights published with github.com/yastrebksv/TrackNet (check
their terms before redistributing). Needs `pip install torch`, and is slow on
a CPU (about 2 s a frame on 4 cores); a GPU makes it real time.
"""

from __future__ import annotations

import cv2
import numpy as np

from .ball import Candidate

WIDTH, HEIGHT = 640, 360


def _build_network():
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.block = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(cout))

        def forward(self, x):
            return self.block(x)

    class TrackNet(nn.Module):
        # (name, in, out) in order; "pool" and "up" halve and double the size.
        LAYOUT = [
            ("conv1", 9, 64), ("conv2", 64, 64), "pool",
            ("conv3", 64, 128), ("conv4", 128, 128), "pool",
            ("conv5", 128, 256), ("conv6", 256, 256), ("conv7", 256, 256), "pool",
            ("conv8", 256, 512), ("conv9", 512, 512), ("conv10", 512, 512), "up",
            ("conv11", 512, 256), ("conv12", 256, 256), ("conv13", 256, 256), "up",
            ("conv14", 256, 128), ("conv15", 128, 128), "up",
            ("conv16", 128, 64), ("conv17", 64, 64), ("conv18", 64, 256),
        ]  # fmt: skip

        def __init__(self):
            super().__init__()
            self.steps = []
            for item in self.LAYOUT:
                if item == "pool":
                    self.steps.append(nn.MaxPool2d(2, 2))
                elif item == "up":
                    self.steps.append(nn.Upsample(scale_factor=2))
                else:
                    name, cin, cout = item
                    setattr(self, name, ConvBlock(cin, cout))
                    self.steps.append(name)

        def forward(self, x):
            for step in self.steps:
                x = getattr(self, step)(x) if isinstance(step, str) else step(x)
            return x  # (batch, 256 intensity classes, H, W)

    return TrackNet()


class TrackNetDetector:
    """`BallDetector` backed by TrackNet. Keeps the last two frames itself."""

    def __init__(self, weights: str, device: str = "cpu"):
        import torch

        self.torch = torch
        self.device = device
        self.net = _build_network()
        self.net.load_state_dict(torch.load(weights, map_location=device))
        self.net.to(device).eval()
        self.prev: list[np.ndarray] = []

    def reset(self) -> None:
        self.prev = []

    def detect(self, frame: np.ndarray, *_, **__) -> list[Candidate]:
        small = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        frames, self.prev = self.prev + [small], (self.prev + [small])[-2:]
        if len(frames) < 3:
            return []
        stacked = np.concatenate(frames[::-1], axis=2).astype(np.float32) / 255.0  # newest first
        with self.torch.inference_mode():
            out = self.net(self.torch.from_numpy(stacked.transpose(2, 0, 1)[None]).to(self.device))
        heat = out.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return heatmap_candidates(heat, frame.shape[1] / WIDTH, frame.shape[0] / HEIGHT)


def heatmap_candidates(heat: np.ndarray, sx: float, sy: float, threshold: int = 127) -> list[Candidate]:
    """Blobs of a TrackNet heatmap (intensity 0-255) as candidates in frame pixels, strongest first."""
    mask = (heat > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero(labels == i)
        w = heat[ys, xs].astype(np.float64)
        # Weighted center, then a score from how strongly and widely the net fired.
        cx, cy = (xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()
        score = min(1.0, 0.5 + 0.5 * float(w.max()) / 255.0 * min(1.0, len(xs) / 10.0))
        out.append(Candidate(float(cx * sx), float(cy * sy), score))
    return sorted(out, key=lambda c: -c.score)
