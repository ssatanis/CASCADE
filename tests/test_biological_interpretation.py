"""Biological interpretation result-shape + biology-direction tests."""
import json
from pathlib import Path
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _load():
    p = RESULTS / "biological_interpretation.json"
    if not p.exists():
        pytest.skip("biological_interpretation.json not generated")
    return json.loads(p.read_text())


def test_shape():
    r = _load()
    assert len(r["top20_high_p_replicate"]) == 20
    assert len(r["bottom20_low_p_replicate"]) == 20
    assert len(r["case_studies"]) == 5
    assert "essential_enrichment_top20" in r
    assert r["trained_on_real_data"] is True


def test_essentials_enriched_in_top_vs_bottom():
    """Biology check: essential genes more frequent among high-P than low-P genes."""
    r = _load()
    top_ess = sum(1 for d in r["top20_high_p_replicate"] if d["is_essential"])
    bot_ess = sum(1 for d in r["bottom20_low_p_replicate"] if d["is_essential"])
    assert top_ess >= bot_ess, f"top essentials {top_ess} should be >= bottom {bot_ess}"


def test_enrichment_test_present():
    r = _load()
    e = r["essential_enrichment_top20"]
    assert "odds_ratio" in e and "p_value" in e
