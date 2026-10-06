"""Let every test folder import the shared helpers that sit beside this file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
