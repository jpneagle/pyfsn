"""Direct entry point for pyfsn command (avoids Python 3.14 runpy bug).

This file is used as the entry point for the pyfsn command line tool.
It imports and executes the main function from __main__.py.

Python 3.14 has a bug in runpy that causes RecursionError when using
python -m with packages that have src layout. This direct entry point
avoids that issue.
"""

import sys
from pathlib import Path


def main() -> int:
    """Entry point for pyfsn command.

    Returns:
        Exit code
    """
    # Import main function from __main__ module
    from pyfsn.__main__ import main as _main
    return _main()


if __name__ == "__main__":
    sys.exit(main())
