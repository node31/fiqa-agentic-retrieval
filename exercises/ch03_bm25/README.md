# ch03: Build BM25 yourself

Guide: `docs/from_zero/03_keyword_search_bm25.md`.

> 🎯 **Why:** BM25 is the baseline every search system is compared against, and the inverted index is the data
> structure you'll shard and update live in ch07. Matching the reference's scores to 4 decimals proves you understand
> every symbol in the formula.

## Steps
1. `PAPER.md`: tokenise and score the 3-doc example by hand.
2. Implement in `my_bm25.py`: `tokenize` → `build_index` → `idf` → `search`.
3. `python -m pytest exercises\ch03_bm25` (start with `-k tokenize`).
4. `python exercises\ch03_bm25\my_bm25.py` runs your BM25 on the 24-doc tiny FiQA.
5. Then the real thing: `python -m pytest exercises\ch03_bm25 --full` (see below).

## The exact recipe (so your numbers can match ours)
- tokenize: lowercase → `re.findall(r"[a-z0-9]+", text)` → drop the 33 Lucene stopwords (`STOPWORDS` below) →
  Snowball English stem (`snowballstemmer.stemmer("english").stemWord(w)`).
- idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5)); k1 = 1.2, b = 0.75; avgdl = mean token count per doc.
- A query term that appears twice counts **once** (dedupe the query terms).
- Return results sorted by score descending, ties broken by doc id ascending; only docs with score > 0.

## Two speeds: toy (seconds) and full (real FiQA)
| | command | what it proves | time |
|---|---|---|---|
| toy | `python -m pytest exercises\ch03_bm25` | the worksheet's 3 docs + the 24-doc tiny FiQA, scores equal to the reference to 4 decimals | seconds |
| full | `python -m pytest exercises\ch03_bm25 --full` | doc 336018 scores **10.68** for "What is short selling?"; all 500 dev queries get the reference top-10 and nDCG@10 **0.264** | ~3 min (building your pure-Python index over 57,638 docs is most of it) |
| demo | `python exercises\ch03_bm25\my_bm25.py [--full]` | index stats + rankings for a few dev queries | |
