import numpy as np
import pytest

from tennis_vision.tracknet import HEIGHT, WIDTH, heatmap_candidates


def test_heatmap_blob_becomes_candidate_in_frame_pixels():
    heat = np.zeros((HEIGHT, WIDTH), np.uint8)
    heat[99:102, 199:202] = 250
    (c,) = heatmap_candidates(heat, 2.0, 2.0)
    assert abs(c.x - 400) < 1e-6 and abs(c.y - 200) < 1e-6
    assert heatmap_candidates(np.full((HEIGHT, WIDTH), 60, np.uint8), 2.0, 2.0) == []


def test_network_loads_weights_and_runs(tmp_path):
    torch = pytest.importorskip("torch")
    from tennis_vision.tracknet import TrackNetDetector, _build_network

    path = tmp_path / "w.pt"
    torch.save(_build_network().state_dict(), path)
    det = TrackNetDetector(str(path))
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert det.detect(frame) == [] and det.detect(frame) == []  # needs three frames
    assert isinstance(det.detect(frame), list)
