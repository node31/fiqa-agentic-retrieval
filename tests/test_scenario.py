"""Scenario runner end to end: real clusters (tiny corpus, hash embedder), both apply modes.

What this proves:
  * the example scenarios in scenarios/ pass in SYNC and ASYNC mode (freshness, idempotency,
    model migration with rollback, writes during a migration, a shard crash + WAL replay);
  * a failed hard expectation makes the run fail (so scenarios work as tests), a soft one does not;
  * golden files round-trip (--golden-save / --golden-check);
  * actors: an admin/ops op in a client (user/bot) actor is rejected at parse time; roles of an
    undeclared custom file are inferred (name, then ops);
  * the replay page is produced and embeds the recorded events.

Ports: 19500/19601+ for the shared per-mode cluster, 19520/19621+ for scenarios that need a cluster
of their own (migration changes the active model, 06 kills a shard), so nothing clashes with the
other integration tests (18000-18999) or a dev cluster (8000/8101+).
Runtime: ~2-3 min on Windows, dominated by starting 8 small clusters (~10 s each).
"""
import json
import re
from pathlib import Path

import pytest

from scripts.scenario import ScenarioError, parse_scenario, resolve_actors, run_scenario
from src.service.cluster import Cluster

ROOT = Path(__file__).resolve().parents[1]
SCN = ROOT / "scenarios"
FIX = Path(__file__).parent / "fixtures" / "tinyfiqa"


@pytest.fixture(scope="module", params=["sync", "async"])
def cluster(request, tmp_path_factory):
    """One tiny 3-shard cluster per apply mode, shared by the read/write scenarios."""
    root = tmp_path_factory.mktemp(f"scenario-{request.param}")
    env = {"FIQA_DATA_DIR": str(FIX), "EMBED_BACKEND": "hash", "ARTIFACTS_DIR": str(root / "artifacts"),
           "APPLY_MODE": request.param}
    c = Cluster(3, base_port=19601, router_port=19500, state_root=root, env=env, threads=1, timeout_s=2.0)
    try:
        c.start(wait_s=120)
    except BaseException:
        c.stop()  # a failed start leaves children running (cluster.py PITFALL)
        raise
    yield request.param, c
    c.stop()


def run(path: Path, tmp_path: Path, **kw):
    return run_scenario(path, out_dir=tmp_path / "out", quiet=True, color=False, print_fn=lambda *_: None, **kw)


def embedded(html: str) -> dict:
    m = re.search(r'<script id="scenario-data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1))


@pytest.mark.parametrize("name", ["freshness_demo.jsonl", "idempotency.jsonl"])
def test_example_scenarios_pass_and_replay_is_written(cluster, name, tmp_path):
    mode, c = cluster
    res = run(SCN / name, tmp_path, url=c.router_url)
    s = res.summary
    assert res.ok and s["errors"] == 0 and s["checks_failed"] == 0 and s["checks_passed"] >= 5, s
    data = embedded(res.replay_path.read_text(encoding="utf-8"))
    assert data["meta"]["mode"] == mode and data["summary"]["passed"]
    logged = [json.loads(ln) for ln in res.events_path.read_text(encoding="utf-8").splitlines()]
    # Every op_start/op_end/status event of events.jsonl is embedded in the page.
    assert len(data["events"]) == sum(e["type"] in ("op_start", "op_end", "status") for e in logged)
    ends = [e for e in data["events"] if e["type"] == "op_end"]
    assert ends and all(e["ok"] for e in ends)
    # Writes carry their owner shard (computed with the router's crc32 rule) for the animation.
    assert all(isinstance(e.get("shard"), int) for e in data["events"] if e.get("op") in ("upsert", "delete")
               and e["type"] == "op_start")


def test_failed_expectation_fails_the_run_soft_does_not(cluster, tmp_path):
    _, c = cluster
    f = tmp_path / "bad.jsonl"
    f.write_text('{"op": "search", "query": "short selling", "expect": ["no-such-doc"], "soft": true}\n'
                 '{"op": "search", "query": "short selling", "expect": ["no-such-doc"]}\n', encoding="utf-8")
    res = run(f, tmp_path, url=c.router_url)
    assert not res.ok
    assert res.summary["checks_failed"] == 1 and res.summary["soft_missed"] == 1


def test_golden_roundtrip(cluster, tmp_path):
    _, c = cluster
    golden = tmp_path / "golden.json"
    first = run(SCN / "02_multi_node_read.jsonl", tmp_path / "a", url=c.router_url, golden_save=golden)
    assert first.ok and set(json.loads(golden.read_text())["answers"]) == {"q1_hybrid", "q2_dense", "q3_bm25", "q4_hybrid"}
    second = run(SCN / "02_multi_node_read.jsonl", tmp_path / "b", url=c.router_url, golden_check=golden)
    assert second.ok and second.summary["golden"] == {"passed": 4, "total": 4}


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("name", ["04_multi_node_read_migrate.jsonl", "05_multi_node_read_write_migrate.jsonl",
                                  "06_shard_failure.jsonl"])
def test_migration_and_failure_scenarios(name, mode, tmp_path):
    """Each needs a fresh cluster: migration changes the active model, 06 kills a shard."""
    res = run(SCN / name, tmp_path, start=True, tiny=True, mode=mode, router_port=19520, base_port=19621)
    s = res.summary
    assert res.ok and s["errors"] == 0 and s["checks_failed"] == 0, s
    assert s["same_as"]["total"] > 0 and s["same_as"]["passed"] == s["same_as"]["total"]
    ops = [a["op"] for a in s["admin"]]
    if name.startswith("06"):
        assert ops == ["kill_shard", "restart_shard"] and s["partial_searches"] >= 2
    else:
        assert {"migrate", "wait_ready", "cutover", "rollback", "finalize"} <= set(ops)
        assert s["overlap"]["total"] > 0
    data = embedded(res.replay_path.read_text(encoding="utf-8"))
    # The replay needs the router's active model and per-shard model progress from the samples.
    assert any(e["type"] == "status" and e.get("active_model") == "bge-small" for e in data["events"]) \
        or name.startswith("06")
    assert (res.out_dir / "cluster_logs" / "router.log").exists()


@pytest.mark.parametrize("text, msg", [
    ('{"op": "serch", "query": "x"}', "unknown op"),
    ('{"op": "search", "query": "x", "expected": ["a"]}', "unknown field"),
    ('{"op": "sleep", "seconds": 1}\n{"op": "config", "shards": 1}', "config must be the first op"),
    ('{"op": "repeat", "until_done": "nope", "ops": [{"op": "flush"}]}', "does not exist"),
    ('{"op": "flush"}\n{"op": "search", "query": "x", "expect_overlap_with": "q1"}', "line 2"),
])
def test_parse_errors_name_the_problem(text, msg):
    with pytest.raises(ScenarioError, match=msg):
        parse_scenario(text)


@pytest.mark.parametrize("text, line", [
    # a declared user running a migration
    ('{"op": "config", "actors": {"fox": {"role": "user"}}}\n'
     '{"op": "parallel", "lanes": [{"name": "fox", "ops": [{"op": "search", "query": "x"}, {"op": "migrate", "model": "m"}]}]}',
     2),
    # an undeclared lane whose NAME says bot, killing a shard
    ('{"op": "parallel", "lanes": [{"name": "bot-1", "ops": [{"op": "kill_shard", "shard": 1}]}]}', 1),
    # a per-op actor that is a client
    ('{"op": "config", "actors": {"owl": {"role": "user"}}}\n{"op": "flush", "actor": "owl"}', 2),
])
def test_admin_op_in_a_client_actor_is_rejected(text, line):
    with pytest.raises(ScenarioError, match=rf"line {line}: .*admin ops must run in an admin/ops actor"):
        parse_scenario(text)


@pytest.mark.parametrize("text, line", [
    # an ops actor issuing a search (a health probe must be a bot)
    ('{"op": "config", "actors": {"ops": {"role": "ops"}}}\n'
     '{"op": "parallel", "lanes": [{"name": "ops", "ops": [\n'
     '  {"op": "kill_shard", "shard": 1},\n'
     '  {"op": "search", "query": "x"}]}]}', 4),
    # the admin writing a doc (per-op actor)
    ('{"op": "config", "actors": {"admin": {"role": "admin"}}}\n{"op": "upsert", "actor": "admin", "doc_id": "a", "text": "x"}', 2),
])
def test_client_op_in_an_admin_or_ops_actor_is_rejected(text, line):
    # The line is the op's OWN line, also inside a multi-line parallel block.
    with pytest.raises(ScenarioError, match=rf"line {line}: .*client ops .* must run in a user/bot actor"):
        parse_scenario(text)


def test_neutral_ops_are_allowed_for_every_role_and_signals_must_exist():
    ok = ('{"op": "config", "actors": {"ops": {"role": "ops"}, "fox": {"role": "user"}}}\n'
          '{"op": "parallel", "lanes": [\n'
          '  {"name": "ops", "ops": [{"op": "sleep", "seconds": 0}, {"op": "signal", "name": "go"}, {"op": "status"}]},\n'
          '  {"name": "fox", "ops": [{"op": "wait_signal", "name": "go"}, {"op": "search", "query": "x"}]}]}')
    ops = parse_scenario(ok)
    assert [o["_line"] for o in ops[1]["lanes"][1]["ops"]] == [4, 4]
    with pytest.raises(ScenarioError, match="no op signals 'nope'"):
        parse_scenario('{"op": "wait_signal", "name": "nope"}')


def test_roles_are_inferred_for_an_undeclared_custom_file():
    text = """
    {"op": "upsert", "doc_id": "a", "text": "x"}
    {"op": "status"}
    {"op": "parallel", "lanes": [
      {"name": "writer-1", "ops": [{"op": "upsert", "doc_id": "b", "text": "y"}]},
      {"name": "zebra", "ops": [{"op": "search", "query": "y"}]},
      {"name": "admin-x", "ops": [{"op": "migrate", "model": "m"}]},
      {"name": "night shift", "ops": [{"op": "restart_shard", "shard": 0}]}
    ]}
    """
    table = {a["name"]: a for a in resolve_actors(parse_scenario(text))}
    roles = {n: a["role"] for n, a in table.items()}
    assert roles == {"user": "user", "ops": "ops", "writer-1": "bot", "zebra": "user", "admin-x": "admin",
                     "night shift": "ops"}
    # Users get animal personas in order of appearance; a generic "user" is displayed as the animal.
    assert (table["user"]["display"], table["user"]["emoji"]) == ("fox", "🦊")
    assert table["zebra"]["emoji"] == "🐘" and table["writer-1"]["emoji"] == "🤖"
    assert table["night shift"]["how"] == "from its ops" and table["writer-1"]["how"] == "from its name"


def test_every_example_scenario_parses():
    files = sorted(SCN.glob("*.jsonl"))
    assert len(files) >= 9
    for f in files:
        ops = parse_scenario(f.read_text(encoding="utf-8"))
        # The catalog declares every actor, so nothing is left to guessing.
        assert all(a["how"] == "declared" for a in resolve_actors(ops)), f
