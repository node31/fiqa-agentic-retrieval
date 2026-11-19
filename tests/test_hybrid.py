"""Unit tests for fusion scoring and shard-result merging."""
import pytest

from src.retrieval.hybrid import fuse, linear, merge_shard_lists, minmax, rrf


def test_rrf_hand_computed():
    bm25 = [("a", 12.0), ("b", 7.0)]
    dense = [("b", 0.9), ("c", 0.8)]
    out = dict(rrf([bm25, dense], k=60))
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)  # rank 2 in bm25, rank 1 in dense
    assert out["a"] == pytest.approx(1 / 61)
    assert out["c"] == pytest.approx(1 / 62)
    assert [d for d, _ in rrf([bm25, dense])] == ["b", "a", "c"]


def test_rrf_ignores_score_scale():
    # multiplying one list's scores by 1000 must not change RRF
    a = [("x", 1.0), ("y", 0.5)]
    b = [("y", 0.2), ("x", 0.1)]
    big = [(d, s * 1000) for d, s in a]
    assert rrf([a, b]) == rrf([big, b])


def test_minmax_constant_list():
    assert minmax([("a", 3.0), ("b", 3.0)]) == {"a": 1.0, "b": 1.0}


def test_linear_alpha_extremes():
    bm25 = [("a", 10.0), ("b", 0.0)]
    dense = [("b", 0.9), ("a", 0.1)]
    assert linear(bm25, dense, alpha=0.0)[0][0] == "a"  # pure bm25
    assert linear(bm25, dense, alpha=1.0)[0][0] == "b"  # pure dense


def test_fuse_rejects_unknown_method():
    with pytest.raises(ValueError):
        fuse([], [], method="magic")


def test_merge_shard_lists_global_top_k():
    s0 = [("a", 9.0), ("b", 1.0)]
    s1 = [("c", 5.0), ("d", 4.0)]
    s2 = [("e", 4.5)]
    assert merge_shard_lists([s0, s1, s2], depth=3) == [("a", 9.0), ("c", 5.0), ("e", 4.5)]


def test_merge_then_fuse_differs_from_fuse_then_merge_for_linear():
    """Why fusion must happen AFTER the global merge (ADR 0003)."""
    shard0_bm25, shard0_dense = [("a", 10.0), ("b", 9.0)], [("a", 0.50), ("b", 0.49)]
    shard1_bm25, shard1_dense = [("c", 2.0), ("d", 1.0)], [("c", 0.90), ("d", 0.10)]
    g_bm25 = merge_shard_lists([shard0_bm25, shard1_bm25], 10)
    g_dense = merge_shard_lists([shard0_dense, shard1_dense], 10)
    correct = [d for d, _ in linear(g_bm25, g_dense, 0.5)]
    wrong = [d for d, _ in merge_shard_lists(
        [linear(shard0_bm25, shard0_dense, 0.5), linear(shard1_bm25, shard1_dense, 0.5)], 10)]
    assert correct != wrong
