#!/usr/bin/env python3
"""Thin runner: regenerate ablation result JSON from real on-disk data. No retrain of the
shipped oracle; no downloads. See cascade/src/cascade/ablation.py."""
from cascade.ablation import run

if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str)[:2000])
