# 01 · Lexical retrieval with BM25

**Code:** `src/retrieval/bm25.py` · **Toys:** 00, 01, 02, 03 · **Tests:** `tests/test_bm25.py`

## 1. Intuition

BM25 scores a passage by how well its **words** match the query's words, with two common-sense
corrections:
1. **Rare words matter more** (*IDF*). Matching "borrow" says more than matching "money".
2. **Repetition and length have diminishing returns.** Saying "loan" 10 times isn't 10× more
   relevant (*tf saturation*), and a match inside a 20-word answer is stronger evidence than the
   same match inside a 2,000-word essay (*length normalisation*).

Analogy: grading an essay for "is it about X?" by counting X. You'd cap the count (no reward for
spamming) and normalise for essay length.

## 2. The math

For query q and document d:

$$\text{score}(q,d)=\sum_{t\in q} \text{IDF}(t)\cdot\frac{tf(t,d)\,(k_1+1)}{tf(t,d)+k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}$$

$$\text{IDF}(t)=\ln\left(1+\frac{N-df(t)+0.5}{df(t)+0.5}\right)$$

| symbol | meaning | FiQA value |
|---|---|---|
| tf(t,d) | occurrences of term t in d | – |
| df(t) | docs containing t | "stock" 6,621 · "money" 15,009 · "borrow" 1,557 |
| N | number of docs | 57,638 |
| \|d\| | doc length in tokens (after stopwords) | median 65 |
| avgdl | average doc length | 95.2 (vocabulary 54,807 terms, 3.77M postings) |
| k1 | saturation (1.2) | – |
| b | length normalisation (0.75) | – |

**Worked example** (toy 02, 3 docs: `a="loan loan loan rate"`, `b="loan rate stock market investing ideas today"`, `c="stock market"`).
After tokenisation |a| = 4, |b| = 7, |c| = 2, so avgdl = 13/3 = 4.33. "loan" is in 2 of 3 docs:
IDF = ln(1 + (3 − 2 + 0.5)/(2 + 0.5)) = ln(1.6) = 0.470.
- doc a: tf = 3 → 0.470 × 3·2.2 / (3 + 1.2·(0.25 + 0.75·4/4.33)) = **0.751**
- doc b: tf = 1 → 0.470 × 2.2 / (1 + 1.2·(0.25 + 0.75·7/4.33)) = **0.376**

`python examples/02_bm25_by_hand.py` prints exactly these numbers, computed by hand and by the index.

## 3. Diagram: what the index looks like

```
            terms →  loan  rate  stock  market ...
docs  a  [ 3     1     .      .   ]   ← CSR row  = "which terms does doc a have?"  (used by delete)
  ↓   b  [ 1     1     1      1   ]
      c  [ .     .     1      1   ]
                ↑
         CSC column = postings list of "loan": [(a,3), (b,1)]   (used by search)
```
Plus a *delta segment* (Python dict) for docs added after the build, and an *alive* bit-mask for deletes.

## 4. How this repo implements it

- `tokenize()`: lowercase, `[a-z0-9]+`, Lucene's 33 stopwords, Snowball stemmer (cached).
  This is the same analysis chain as Pyserini's BEIR baseline, so numbers are comparable.
- `BM25Index.build()`: one pass builds a COO matrix, then converts it to CSC (search) and CSR (deletes).
- `search(terms, k, stats)`: for each query term, read its CSC column (`indptr[j]:indptr[j+1]`),
  compute the BM25 contribution **vectorised** with numpy for all docs in the postings list at once,
  and add it into a dense score array. Then `argpartition` picks the top-k in O(N).
- `stats` is the **sharding hook**: pass global N/avgdl/df from the router, and every shard scores
  identically to a single node. See [08](08_sharding_scatter_gather.md).
- Rejected: `rank_bm25` (no deletes, and it scores every document for every query term: O(N·|q|)) and
  Pyserini (JVM). See [ADR 0001](adr/0001-own-bm25.md).

## 5. Run the toys

```bash
python examples/00_tokenization.py      # what the tokenizer keeps and drops
python examples/01_tf_idf_by_hand.py    # df, idf by hand
python examples/02_bm25_by_hand.py      # k1 saturation table, b length table, hand == index
python examples/03_inverted_index_sparse.py  # postings, delta segment, tombstones
```

## 6. Observe it in the real system

`results/bench.json → configs.bm25`: **nDCG@10 0.264, Recall@10 0.314, MRR@10 0.322, p50 1.2 ms**
on dev. Published BM25 on FiQA *test* is 0.236 nDCG@10 (BEIR paper). We're in the same range, which is a
good sanity check that tokenisation and the formula are right. BM25 is ~16× faster than dense
(no neural encoder at query time) but ~12 nDCG points worse: finance questions are full of
paraphrase ("pay off early" vs "prepay").

## 7. Pitfalls and interview questions

- **Different analysis for query and docs**: an unstemmed query against a stemmed index silently loses matches.
- **IDF can go negative** in the original Robertson formula for terms in more than half the docs. Lucene adds 1 inside the log.
- **Stopwords remove negation** ("not"). See toy 00.
- *Q: Why is BM25 still a strong baseline in 2026?* It's zero-shot, exact for rare entities and codes, cheap and explainable.
- *Q: What does b = 0 mean?* No length normalisation.

## 8. Exercises

1. (Concept) Why does tf saturation make BM25 robust to keyword stuffing?
2. (Predict) If you set k1 = 0, what ranking do you get for the query "loan" in toy 02?
3. (Predict) A term appears in every document. What is its IDF, roughly?
4. (Do) Sweep b ∈ {0.3, 0.5, 0.75, 0.9} on dev (edit `config.BM25_B`, run `python -m src.eval --strategy bm25 --name b03`). Which wins?
5. (Do, harder) Implement `BM25Index.compact()` (see [DRILLS](DRILLS.md) drill 5).

Solutions: [solutions/01_bm25.md](solutions/01_bm25.md)

## 9. Further reading

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)
- Lucene `BM25Similarity` source (the IDF variant we use)
- Lin et al., *Pyserini* (2021): reproducible BM25 baselines for BEIR
