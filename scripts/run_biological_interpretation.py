#!/usr/bin/env python3
"""Thin runner: regenerate biological_interpretation result JSON from real on-disk data. No retrain of the
shipped oracle; no downloads. See cascade/src/cascade/biological_interpretation.py."""
from cascade.biological_interpretation import run

if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str)[:2000])
