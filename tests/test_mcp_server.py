"""Tests for the splicr-mcp authority server (no corpus needed)."""

import json

from cascade import mcp_server as mcp


def _call(name, arguments):
    resp = mcp.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": name, "arguments": arguments}})
    return json.loads(resp["result"]["content"][0]["text"])


def test_initialize_advertises_server():
    r = mcp.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    info = r["result"]["serverInfo"]
    assert info["name"] == "splicr-mcp"
    assert r["result"]["protocolVersion"]


def test_tools_list_is_compact_five():
    r = mcp.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in r["result"]["tools"]}
    assert names == {"find_crispr_tools", "get_tool_info", "list_tools", "grep_tools", "execute_tool"}


def test_notifications_initialized_returns_none():
    assert mcp.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_find_ranks_replicate_for_replication_query():
    out = _call("find_crispr_tools", {"query": "will this hit replicate in another cell type"})
    assert out["matches"][0]["name"] == "cascade.replicate"


def test_meta_analyze_is_live_and_enveloped():
    out = _call("execute_tool", {"name": "cascade.meta_analyze", "arguments": {
        "studies": [{"beta": -1.2, "variance": 0.04}, {"beta": -1.0, "variance": 0.05},
                    {"beta": -1.3, "variance": 0.03}]}})
    # envelope shape
    for k in ("result", "provenance", "validation", "calibration", "citations"):
        assert k in out
    assert out["result"]["k_studies"] == 3
    assert -1.3 <= out["result"]["pooled_effect"] <= -1.0
    assert out["citations"]  # has a DOI
    assert out["provenance"]["local_first"] is True
    assert out["provenance"]["input_hash"]


def test_corpus_tool_honest_declines():
    out = _call("execute_tool", {"name": "cascade.redundancy_check", "arguments": {"library": "Brunello"}})
    assert out["result"]["abstained"] is True
    assert "corpus" in out["result"]["reason"]
    assert "fallback" in out["result"]


def test_unknown_tool_errors():
    out = _call("execute_tool", {"name": "cascade.does_not_exist", "arguments": {}})
    assert "error" in out


def test_get_tool_info_returns_schema():
    out = _call("get_tool_info", {"name": "cascade.replicate"})
    assert out["name"] == "cascade.replicate"
    assert "beta_a" in out["input_schema"]["properties"]


def test_list_tools_census_is_honest():
    out = _call("list_tools", {})
    assert out["count"] == len(mcp.TOOLS)
    # the census note must carry the honest disclaimer, not an affirmative breadth claim
    assert "NOT more tools than ToolUniverse" in out["census_note"]
    assert "depth" in out["census_note"].lower()
