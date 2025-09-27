"""Setup script for Brokle SDK (fallback for older pip versions)."""

import sys
from pathlib import Path

# For older pip versions that don't support pyproject.toml
if sys.version_info < (3, 9):
    raise RuntimeError("Brokle SDK requires Python 3.9 or higher")

# Point to pyproject.toml
try:
    from setuptools import setup
    setup()
except ImportError:
    print(
        "Error: setuptools not found. Please install it with: pip install setuptools wheel"
    )
    sys.exit(1)