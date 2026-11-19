# ch06 notes: a filled-in example

> Run `python exercises\ch06_pipeline\mini_pipeline.py` and fill in `NOTES.md` yourself **before** reading this.
> These are the answer key's real outputs (`python exercises\solutions\ch06_pipeline\mini_pipeline.py`).

## Query q1: "how does short selling work"
- **gold docs:** 1, 2
- **BM25 top-3:** `1, 9, 2`.
  - Doc 1 matches both stems, **short** and **sell** ("Short selling … selling them").
  - Doc 9 ("Capital gains tax applies when you **sell** shares") matches only "sell".
  - Doc 2 ("When you **short** a stock …") matches only "short".
  - Both terms have df = 2, so the same idf. Doc 9 is shorter than doc 2, so the length normalisation ranks it higher. A word-counting artefact.
- **dense top-3:** `1, 2, 9`. The model knows "short a stock … loss is unlimited" is *about* short selling,
  while doc 9 is about taxes. Meaning beats word overlap.
- **hybrid top-3:** `1, 2, 9`. **Fusion helped**: dense's confident #2 (doc 2) lifted it above BM25's spurious #2.

## Query q3: "how big should an emergency fund be"
- **gold docs:** 7
- **BM25 top-3:** `7, 14, 20`.
  - Doc 7 matches "emergency", "fund", "should".
  - Doc 14 matches only **should** ("Student loans … **should** usually be paid …"). Not a stopword in Lucene's list!
  - Doc 20 matches only **fund** ("An ETF is a **fund** …").
- **dense top-3:** `7, 4, 21`. Related *topics*: doc 4 investing extra money, doc 21 cash in savings accounts.
- **hybrid top-3:** `7, 4, 14`. Both agree on #1, so fusion keeps it. Below #1 the lists disagree and the fused list mixes them.
  No harm here, because only rank 1 is gold.

## Overall
- **Tiny set:**
  - nDCG@10 is BM25 0.934, dense **0.963**, hybrid 0.958 (Recall and MRR are 1.0 for all three).
  - Dense wins, and hybrid is *slightly below* dense. That does **not** match FiQA's 0.264 < 0.388 < 0.419.
- **Why 4 queries are unreliable:** every query is 25% of the average. One doc swapping places with its neighbour changes
  the average by ~0.03. With 24 easy, hand-written docs everything scores ~0.95. You need hundreds of real queries
  (FiQA dev: 500) before differences of 0.01–0.03 mean anything.
  Run `python exercises\ch06_pipeline\mini_pipeline.py --full` to see the real ordering appear.
