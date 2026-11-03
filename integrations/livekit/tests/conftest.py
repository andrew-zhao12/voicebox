"""Import paths for these tests: the example (``agent``) and the e2e scripts (``run_e2e``)."""

import sys
from pathlib import Path

INTEGRATION = Path(__file__).resolve().parent.parent
for path in (INTEGRATION, INTEGRATION / "e2e"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
