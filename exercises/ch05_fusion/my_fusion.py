"""ch05: hybrid fusion, written by you. Standard library only.

Run the checker:   python -m pytest exercises/ch05_fusion
Run the demo:      python exercises/ch05_fusion/my_fusion.py
"""
from __future__ import annotations

from typing import Sequence

Ranked = list[tuple[str, float]]  # [(doc_id, score), ...] best first


def minmax(lst: Ranked) -> dict[str, float]:
    """Scale scores to [0, 1] within this one list: (s - min) / (max - min). All-equal scores → all 1.0.

    minmax([("a", 10.0), ("b", 5.0), ("c", 0.0)]) == {"a": 1.0, "b": 0.5, "c": 0.0}
    """
    raise NotImplementedError("TODO (ch05): implement minmax")


def linear(bm25: Ranked, dense: Ranked, alpha: float = 0.7, top: int | None = None) -> Ranked:
    """alpha · minmax(dense) + (1 - alpha) · minmax(bm25). Missing from a list = 0 for that list.
    Sorted by score desc, ties by doc_id asc; cut to `top` if given."""
    raise NotImplementedError("TODO (ch05): implement linear")


def rrf(lists: Sequence[Ranked], k: int = 60, weights: Sequence[float] | None = None,
        top: int | None = None) -> Ranked:
    """Reciprocal Rank Fusion: score(d) = Σ_i weights[i] / (k + rank of d in list i), ranks from 1.
    Sorted by score desc, ties by doc_id asc; cut to `top` if given."""
    raise NotImplementedError("TODO (ch05): implement rrf")


if __name__ == "__main__":
    bm25 = [("A", 18.2), ("B", 11.0), ("C", 10.7), ("D", 2.1)]
    dense = [("C", 0.83), ("E", 0.81), ("A", 0.52), ("B", 0.50)]
    print("minmax bm25 :", {d: round(s, 3) for d, s in minmax(bm25).items()})
    for a in (0.3, 0.5, 0.7):
        print(f"linear α={a}:", [(d, round(s, 3)) for d, s in linear(bm25, dense, a)])
    print("rrf        :", [(d, round(s, 5)) for d, s in rrf([bm25, dense])])
