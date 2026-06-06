import json

import pytest

from cascade.cli import main, ARTIFACT
from cascade.corpus import RAW


def _run(capsys, argv):
    rc = main(argv)
    out = capsys.readouterr().out
    return rc, json.loads(out)


def test_cli_meta(capsys):
    payload = json.dumps({"studies": [{"beta": -1.0, "variance": 0.1}, {"beta": -1.2, "variance": 0.1}]})
    rc, data = _run(capsys, ["meta", payload])
    assert rc == 0
    assert data["effect"] < 0
    assert data["k"] == 2


def test_cli_dp(capsys):
    rc, data = _run(capsys, ["dp", json.dumps({"epsilon": 3.0, "delta": 1e-6, "steps": 10})])
    assert rc == 0
    assert data["sigma"] > 0
    assert data["achieved_delta_at_eps"] == pytest.approx(1e-6, rel=1e-2)
    assert data["composition"]["rdp_epsilon"] > 3.0


@pytest.mark.skipif(not ARTIFACT.exists(), reason="real Oracle artifact not present (run `cascade train`)")
def test_cli_replicate_real(capsys):
    payload = json.dumps(
        {"gene": "GENEX", "beta_a": -1.3, "var_a": 0.04, "lineage_a": "myeloid", "lineage_b": "neuron",
         "quality_a": 0.9, "quality_b": 0.85}
    )
    rc, data = _run(capsys, ["replicate", payload])
    assert rc == 0
    assert data["trained_on_real_data"] is True
    assert "abstained" in data
    assert "model" in data and "Replication Oracle" in data["model"]


@pytest.mark.skipif(not (RAW / "CRISPRGeneEffect.csv").exists(), reason="real DepMap data not downloaded")
def test_cli_collapse_real(capsys):
    rc, data = _run(capsys, ["collapse"])
    assert rc == 0
    assert data["r_raw_fitness"] > data["r_dlfc_deviation"]
    assert data["reproduces_documented_collapse_direction"] is True
