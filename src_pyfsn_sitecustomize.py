"""Site customize to add pyfsn src to path."""

import sys
from pathlib import Path

# Add src to path using sys.path[0] manipulation instead of insert
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.append(str(src_path))
