#!/usr/bin/env python3
"""Thin runner: regenerate baselines_wbc result JSON from real on-disk data. No retrain of the
shipped oracle; no downloads. See cascade/src/cascade/baselines_wbc.py."""
from cascade.baselines_wbc import run

if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str)[:2000])
