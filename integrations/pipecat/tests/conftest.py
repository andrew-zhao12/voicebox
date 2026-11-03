"""Import the standalone integration without adding it to backend dependencies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
