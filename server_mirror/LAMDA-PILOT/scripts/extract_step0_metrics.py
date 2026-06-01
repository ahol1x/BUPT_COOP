#!/usr/bin/env python
"""Compatibility wrapper for the general PILOT metrics extractor."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_metrics import main  # noqa: E402


if __name__ == "__main__":
    main()
