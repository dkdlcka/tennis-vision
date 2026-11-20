import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os  # noqa: E402

# Tests that want players pass a detector explicitly; don't load YOLO otherwise.
os.environ.setdefault("TENNIS_PLAYERS", "0")
