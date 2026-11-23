"""Unit tests for the pure parts of scripts/docker_tour.py (no Docker needed).

The tour's promise is "the stack ends in the shape it started in", so the parts that decide that get tests:
which docs the tour must delete (writes_of), the shard-pinned doc ids, and the before/after comparison itself,
including that it really reports a difference when there is one.
"""
import json

from scripts.docker_tour import (compare, compress, parse_compose_ps, pinned_id, render_pins, shape_of, write_report,
                                 writes_of)
from src.index.sharding import shard_of


def _op(i, op, req, ok=True, **resp):
    return [{"type": "op_start", "id": i, "op": op, "req": req, "lane": "main"},
            {"type": "op_end", "id": i, "op": op, "ok": ok, "resp": resp}]


def test_compress_matches_the_summary_notation():
    assert compress(["minilm", "minilm", "bge-small", "minilm"]) == "minilm×2 → bge-small×1 → minilm×1"
    assert compress([]) == "-"


def test_pinned_ids_really_live_on_their_shard_and_are_stable():
    for shard in range(3):
        doc = pinned_id("tour-keep", shard, "20261121-170000", 3)
        assert shard_of(doc, 3) == shard
    text = '{"doc_id": "{on1:tour-keep}"} {"expect": ["{on1:tour-keep}", "{on0:other}"]}'
    out = render_pins(text, "tag", 3)
    keep = pinned_id("tour-keep", 1, "tag", 3)
    assert out.count(keep) == 2 and "{on" not in out


def test_writes_of_follows_the_last_applied_write():
    events = (_op(1, "upsert", {"doc_id": "a", "version": 1}, applied=True)
              + _op(2, "delete", {"doc_id": "a", "version": 2}, applied=True)
              + _op(3, "upsert", {"doc_id": "a", "version": 1}, applied=False, current_version=2)   # stale: no-op
              + _op(4, "upsert", {"doc_id": "b", "version": 1}, applied=True)
              + _op(5, "upsert", {"doc_id": "c", "version": 1}, ok=False))                         # failed (503)
    w = writes_of(events)
    assert w["a"] == {"version": 2, "live": False}     # deleted: nothing to clean up
    assert w["b"] == {"version": 1, "live": True}      # still live: the tour deletes it with version 2
    assert w["c"] == {"version": 1, "live": None}      # never applied


STATUS = {"active_model": "minilm", "previous_model": None, "shards": {
    "0": {"healthy": True, "n_docs": 100, "apply_mode": "sync", "models": {"minilm": {"ready": True}}},
    "1": {"healthy": True, "n_docs": 90, "apply_mode": "sync", "models": {"minilm": {"ready": True}}}}}


def _snap(status=STATUS, probes=None, compose=None):
    return {"shape": shape_of(status), "probes": probes or {"bm25 · q": ["1", "2"]},
            "compose": compose or {"shard0": {"state": "running", "health": "healthy", "image": "img"}}}


def test_compare_identical_snapshots_pass():
    assert all(c["ok"] for c in compare(_snap(), _snap()))


def test_compare_reports_every_kind_of_difference():
    changed = json.loads(json.dumps(STATUS))
    changed["shards"]["1"]["n_docs"] = 91                                    # a tour doc left behind
    changed["shards"]["0"]["apply_mode"] = "async"                           # mode not restored
    changed["shards"]["0"]["models"]["bge-small"] = {"ready": True}          # model not finalized
    changed["previous_model"] = "bge-small"
    after = _snap(changed, probes={"bm25 · q": ["2", "1"]},
                  compose={"shard0": {"state": "exited", "health": "", "image": "img"}})
    bad = {c["name"] for c in compare(_snap(), after) if not c["ok"]}
    assert bad == {"shard 1: live docs", "shard 0: apply mode", "shard 0: models",
                   "router: previous model (kept for rollback)", "answers: bm25 · q", "shard0: state, health, image"}


def test_parse_compose_ps_handles_both_output_formats():
    rows = [{"Service": "router", "State": "running", "Health": "healthy", "Image": "img"}]
    assert parse_compose_ps(json.dumps(rows)) == parse_compose_ps(json.dumps(rows[0]))
    assert parse_compose_ps("") == {}


def test_report_renders_records_and_the_shape_table(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps([
        {"key": "writes", "section": "3c", "title": "Upsert <b>", "what": "", "expected": "", "status": "pass",
         "seconds": 1, "t": 1, "got": "ok", "links": {"replay": "writes/replay.html"}, "details": [["x", False, "y"]]},
        {"key": "verify", "section": "end", "title": "Verify", "what": "", "expected": "", "status": "fail",
         "seconds": 1, "t": 2, "got": "1/2", "links": {}, "details": []}]))
    (tmp_path / "wrapper.jsonl").write_text('{"key": "stop_shard1", "section": "3g", "title": "stop", '
                                            '"status": "pass", "seconds": 2, "t": 1.5, "got": ""}\n')
    (tmp_path / "shape_checks.json").write_text(json.dumps(
        [{"name": "shard 0: live docs", "before": 1, "after": 2, "ok": False}]))
    page = write_report(tmp_path).read_text(encoding="utf-8")
    assert "Upsert &lt;b&gt;" in page and "writes/replay.html" in page and "stop" in page
    assert "NOT back in its original shape" in page and "shard 0: live docs" in page


def test_expected_model_sequences_follow_the_starting_model():
    from scripts.docker_tour import for_model

    assert for_model("minilm×3 → bge-small×2 → minilm×2", "minilm-ft") == "minilm-ft×3 → bge-small×2 → minilm-ft×2"
    assert for_model("minilm×7", "minilm") == "minilm×7"
