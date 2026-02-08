#!/usr/bin/env python3
"""Wrapper script to run pyfsn, avoiding Python 3.14 runpy bug."""

import sys
from pathlib import Path

# Add src to path BEFORE any imports
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Now import and run
from pyfsn.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
