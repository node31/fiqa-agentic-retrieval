"""Semantic cache: a question that means the same as an earlier one gets the earlier answer, with no LLM call.

WHY
    The disk cache in llm.py only hits on a byte-identical prompt. Real users rephrase: "how do I roll my
    401k into an IRA?" and "can I move my old 401(k) to an IRA?" are the same question to a person and two
    different prompts to the LLM. Here the key is the question's EMBEDDING (the same query encoder the
    search uses, so it costs nothing extra), and a hit is any stored question with cosine >= ``threshold``.
    It is only as good as the encoder: minilm-ft puts those two at 0.657 (a miss), "how do I roll over my
    401k to an IRA?" at 0.935 (a hit), and "...into a ROTH IRA?", a different question, at 0.928 (a hit,
    and a wrong one). docs/topics/19_llm_cost.md section 8 has the numbers.

    >>> from src.retrieval.dense import HashEmbedder
    >>> emb = HashEmbedder(seed="doc")
    >>> c = SemanticCache(lambda q: emb.encode_queries([q])[0], threshold=0.95)
    >>> c.store("what is a roth ira?", ["1", "2"], tenant="acme")
    >>> c.lookup("What is a Roth IRA?", tenant="acme")["value"]       # same words, different case
    ['1', '2']
    >>> c.lookup("what is a roth ira?", tenant="globex") is None      # never across tenants
    True
    >>> c.lookup("how do I pay off my mortgage early?", tenant="acme") is None
    True

THE RISK, AND HOW THE THRESHOLD IS CHOSEN
    A "false hit" serves the answer to a DIFFERENT question that happens to sound similar ("roth ira for
    my kid" vs "roth ira for my spouse"). scripts/semantic_cache_eval.py measures, for a range of
    thresholds, how many paraphrases hit (the benefit) and how many distinct FiQA questions would wrongly
    hit each other (the harm), and what the wrongly served results score. The default comes from there:
    with minilm-ft, 0.85 is the lowest threshold where at most 1% of dev questions hit a DIFFERENT question
    (0.8%, 4 of 500), and 65 of 100 Azure-written paraphrases still hit their original. The number belongs
    to the encoder: another model spreads its cosines differently, so re-run the eval after a migration.

TENANTS
    Entries are stored with the caller's tenant and only match the same tenant, so a cached answer built
    from acme's documents is never served to globex (the same rule as the tenant filter).

NOT HANDLED (on purpose, see docs/topics/19_llm_cost.md)
    Staleness: a cached answer does not notice that the corpus changed since. Production would expire
    entries by age or bump a corpus version on every write and key on it.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Callable

import numpy as np

# Calibrated by scripts/semantic_cache_eval.py (results/semantic_cache_eval.json); SEMANTIC_CACHE_MIN overrides.
DEFAULT_THRESHOLD = float(os.environ.get("SEMANTIC_CACHE_MIN", "0.85"))


class SemanticCache:
    """In-memory, thread-safe, first-in-first-out once ``max_entries`` is reached."""

    def __init__(self, embed: Callable[[str], np.ndarray], threshold: float = DEFAULT_THRESHOLD,
                 max_entries: int = 10_000):
        self.embed, self.threshold, self.max_entries = embed, threshold, max_entries
        self._entries: OrderedDict[int, dict] = OrderedDict()  # id -> {question, tenant, value, vec}
        self._next = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def _unit(v) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def lookup(self, question: str, tenant: str | None = None) -> dict | None:
        """The best stored entry of the same tenant with cosine >= threshold, as
        {"value", "question", "similarity"}; None on a miss."""
        q = self._unit(self.embed(question))
        with self._lock:
            same = [e for e in self._entries.values() if e["tenant"] == tenant]
            if same:
                sims = np.stack([e["vec"] for e in same]) @ q
                i = int(np.argmax(sims))
                if sims[i] >= self.threshold:
                    self.hits += 1
                    return {"value": same[i]["value"], "question": same[i]["question"],
                            "similarity": round(float(sims[i]), 4)}
            self.misses += 1
        return None

    def store(self, question: str, value, tenant: str | None = None) -> None:
        vec = self._unit(self.embed(question))
        with self._lock:
            self._entries[self._next] = {"question": question, "tenant": tenant, "value": value, "vec": vec}
            self._next += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"entries": len(self), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / n, 4) if n else 0.0, "threshold": self.threshold}
