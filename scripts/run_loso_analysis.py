#!/usr/bin/env python3
"""Thin runner: regenerate loso_analysis result JSON from real on-disk data. No retrain of the
shipped oracle; no downloads. See cascade/src/cascade/loso_analysis.py."""
from cascade.loso_analysis import run

if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str)[:2000])
