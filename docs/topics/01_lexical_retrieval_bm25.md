# 01 · Lexical retrieval with BM25

Code: `src/retrieval/bm25.py`. Tests: `tests/test_bm25.py`.

## How BM25 scores a passage

BM25 scores a passage by how well its words match the query's words, with two corrections. Rare words count more
(*IDF*): matching "borrow" says more than matching "money". And repetition and length have diminishing returns:
"loan" ten times is not ten times as relevant (*tf saturation*), and a match inside a 20-word answer is stronger
evidence than the same match inside a 2,000-word essay (*length normalisation*).

$$\text{score}(q,d)=\sum_{t\in q} \text{IDF}(t)\cdot\frac{tf(t,d)\,(k_1+1)}{tf(t,d)+k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}$$

$$\text{IDF}(t)=\ln\left(1+\frac{N-df(t)+0.5}{df(t)+0.5}\right)$$

The `1 +` inside the log is Lucene's variant. Without it, IDF goes negative for terms in more than half the docs.

| symbol | meaning | FiQA value (measured on the current tokenizer) |
|---|---|---|
| tf(t,d) | occurrences of term t in d | |
| df(t) | docs containing t | "stock" 6,621 · "money" 15,009 · "borrow" 1,557 |
| N | number of docs | 57,638 |
| \|d\| | doc length in tokens (after stopwords) | median 65 |
| avgdl | average doc length | 95.2 (vocabulary 54,807 terms, 3.77M postings) |
| k1 | saturation | 1.2 |
| b | length normalisation (0 = none) | 0.75 |

A small worked example with three docs: `a="loan loan loan rate"`, `b="loan rate stock market investing ideas today"`,
`c="stock market"`. After tokenisation |a| = 4, |b| = 7, |c| = 2, so avgdl = 13/3 = 4.33. "loan" is in 2 of 3 docs,
so IDF = ln(1 + (3 − 2 + 0.5)/(2 + 0.5)) = ln(1.6) = 0.470.

- doc a, tf = 3: 0.470 × 3·2.2 / (3 + 1.2·(0.25 + 0.75·4/4.33)) = 0.751
- doc b, tf = 1: 0.470 × 2.2 / (1 + 1.2·(0.25 + 0.75·7/4.33)) = 0.376

## What the index looks like

```
            terms →  loan  rate  stock  market ...
docs  a  [ 3     1     .      .   ]   ← CSR row  = "which terms does doc a have?"  (used by delete)
  ↓   b  [ 1     1     1      1   ]
      c  [ .     .     1      1   ]
                ↑
         CSC column = postings list of "loan": [(a,3), (b,1)]   (used by search)
```

Next to the matrices sit a *delta segment* (a Python dict) for docs added after the build, and an *alive* bit-mask
for deletes.

## How I implemented it

- `tokenize()`: lowercase, `[a-z0-9]+`, Lucene's 33 stopwords, Snowball stemmer (cached). This is the same analysis
  chain as Pyserini's BEIR baseline, so my numbers are comparable to published ones. Queries and docs go through the
  same chain, since an unstemmed query against a stemmed index silently loses matches.
- `BM25Index.build()`: one pass builds a COO matrix, then converts it to CSC (for search) and CSR (for deletes).
- `search(terms, k, stats)`: for each query term I read its CSC column (`indptr[j]:indptr[j+1]`), compute the BM25
  contribution for every doc in that postings list at once with numpy, and add it into a dense score array.
  `argpartition` then picks the top-k in O(N).
- `stats` is the sharding hook. When the router passes global N, avgdl and df, every shard scores exactly like a
  single node ([08](08_sharding_scatter_gather.md)).

I wrote my own index instead of using `rank_bm25` (no deletes, and it scores every document for every query term,
O(N·|q|)) or Pyserini (needs a JVM).

## Results

`results/bench.json → configs.bm25`, dev: nDCG@10 0.2642, Recall@10 0.3135, MRR@10 0.3222, p50 1.2 ms. The BEIR
paper reports 0.236 nDCG@10 for BM25 on FiQA test, so my number is in the expected range, a sanity check that the
tokenisation and the formula are right. BM25 is about 16× faster than dense (no neural encoder at query time) but
about 12 nDCG points worse, because finance questions are full of paraphrase ("pay off early" vs "prepay").

```bash
python -m src.eval --strategy bm25 --name bm25
```

Reference: Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009).
