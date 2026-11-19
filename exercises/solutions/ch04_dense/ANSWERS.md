# ch04 worksheet: answers

> Fill in `exercises/ch04_dense/PAPER.md` yourself **before** reading this.
> Run `python exercises\solutions\ch04_dense\my_dense.py` to see 1, 2, 4 and 6 computed.

**1.** |a| = √(9 + 16) = **5**; normalize(a) = (3/5, 4/5) = **(0.6, 0.8)**.

**2.**
- cos(a, b) = (12 + 12)/(5·5) = **0.96** (almost the same direction).
- cos(a, c) = (−9 − 16)/25 = **−1** (exactly opposite).
- cos(a, d) = (−12 + 12)/25 = **0** (perpendicular).

Geometrically:
- **1** = same direction (same meaning)
- **0** = 90°, unrelated
- **−1** = opposite

With real sentence embeddings almost all cosines fall between ~0 and ~0.9; truly negative ones are rare.

**3.** For unit vectors |a| = |b| = 1, so cos = a·b / (1·1) = a·b. Normalise once when indexing, and every search
is then a plain dot product, which is what faiss `IndexFlatIP` and `doc_vecs @ q` compute very fast.
That's why `encode()` uses `normalize_embeddings=True`.

**4.** **"Betting that a share price will fall is called shorting"** ranks first (cosine ≈ 0.56).

BM25 can't find it because the stems share **nothing**:
- query: `gambl equiti drop`
- doc: `bet share price fall call short`

Every BM25 score is 0. The embedding model learned from millions of sentences that gambling ≈ betting,
equities ≈ shares and dropping ≈ falling. This is the "vocabulary mismatch" that makes dense beat BM25 on FiQA (0.388 vs 0.264).

**5.** 256/1,800 ≈ **14%**. The model reads only the first 256 sub-word tokens and silently drops the rest.
That's **case 2 in `results/failures.md` (qid 4029)**: the answer sat in the part the encoder never read. Fixes:
- split long passages into chunks and index each one
- a model with a longer window
- rely on BM25 for that doc (BM25 has no length limit)

**6.** **High**: MiniLM gives ≈ **0.81**. An embedding averages meaning over all the tokens, and "not" is one small
token next to the same topic words, so the vectors stay close. Embeddings capture *topic* much better than *logic*
(negation, numbers, "before vs after"). That's a known weakness, and one reason cross-encoder rerankers exist.
