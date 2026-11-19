# ch01 worksheet: answers

> Fill in `exercises/ch01_data/PAPER.md` yourself **before** reading this.

**1. The three files** (one line each, from `tests/fixtures/tinyfiqa/` or the real FiQA):
- corpus `corpus.jsonl`: `{"_id": "1", "title": "", "text": "Short selling means borrowing shares and selling them, ..."}`
- queries `queries.jsonl`: `{"_id": "q1", "text": "how does short selling work"}`
- qrels `qrels/dev.tsv` (tab-separated, with a header): `q1	1	1` = "query q1 is answered by doc 1, relevance 1"

**2. q1 "how does short selling work".** The qrels say **docs 1 and 2**:
- doc 1 "Short selling means borrowing shares and selling them ..." (the definition)
- doc 2 "When you short a stock your maximum loss is unlimited ..." (how it behaves)

Common disagreements:
- *Missing doc 2*: it never says "selling", but it is about shorting.
- *Adding doc 22* (margin call): related, but it doesn't explain shorting.
- *Adding doc 9* (capital gains "when you sell shares"): shares the word "sell" only.

That is exactly the lexical-vs-meaning tension of chapters 3 and 4. Judging relevance is subjective, so real qrels are incomplete and imperfect.

**3. Dev vs test.** Every choice we tune on a split (α, model, k) slowly fits that split's quirks, so its score gets
optimistic. Test stays untouched until the end, so its single score is an honest estimate of unseen queries
(our dev 0.419 → test 0.397 shows the normal, small drop).

**4. Recall.** 1 of 2 found → Recall@10 = **0.5**. Per-query recall moves in huge steps (0, 0.5, 1 with 2 gold docs),
so a single query tells you almost nothing. You need hundreds of queries to average out the noise (FiQA dev has 500).

**5. tinyfiqa dev:** **24** docs, **4** queries, **7** relevant pairs (q1: 2, q2: 2, q3: 1, q4: 2 → average 1.75).
Full FiQA (`--full`): 57,638 docs · dev 500 queries / 1,238 pairs · test 648 / 1,706 · train 5,500 / 14,166.

**Bonus: the `score` column.** It is a relevance grade: 0 = judged not relevant, 1 = relevant, 2+ = more relevant (graded sets).
FiQA has only 1s. It's an integer label, not a float (guide §1.2).
