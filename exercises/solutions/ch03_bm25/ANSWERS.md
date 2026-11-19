# ch03 worksheet: answers

> Fill in `exercises/ch03_bm25/PAPER.md` yourself **before** reading this.
> Run `python exercises\solutions\ch03_bm25\my_bm25.py` to see these numbers printed.

**1. Tokens** (lowercase → words → drop "a"/"and" → Snowball stem):
- d1 = `short sell mean borrow share`
- d2 = `short stori about sell hous sell car` ("story" → "stori", "house" → "hous"; stems needn't be real words)
- d3 = `mortgag rate rose`

**2. Lengths.** |d1| = **5**, |d2| = **7**, |d3| = **3**; avgdl = 15/3 = **5**; N = 3.

**3. IDF.** df(short) = **2** (d1, d2), df(sell) = **2** (d1, d2).
idf = ln(1 + (3 − 2 + 0.5)/(2 + 0.5)) = ln(1.6) = **0.470** (the same for both terms).

**4. d1.**
- norm = 1 − 0.75 + 0.75·5/5 = **1.0** (average length: no penalty, no bonus).
- tf-part(short) = 1·2.2/(1 + 1.2·1.0) = **1.0**. tf-part(sell) is the same, **1.0**.

**5. Scores.**
- BM25(d1) = 0.470·1.0 + 0.470·1.0 = **0.940**.
- BM25(d2):
  - norm = 0.25 + 0.75·7/5 = **1.3** (longer than average, so it's penalised)
  - short (tf 1): 2.2/(1 + 1.56) = 0.859
  - sell (tf 2): 2·2.2/(2 + 1.56) = 1.236
  - BM25(d2) = 0.470·(0.859 + 1.236) = **0.985**

**d2 wins**: "selling" appears twice, and that beats the length penalty. BM25 doesn't know that "short story" isn't
"short selling": it counts matching stems, it doesn't read meaning. Note the saturation too: tf 2 gave 1.236, not 2 × 0.859.

**6. Knobs.**
- **k1 = 0**: tf-part = tf·1/tf = 1 for any tf > 0. BM25 becomes "sum of idf of the query terms present". Pure
  presence/absence: repeating a word does nothing, and length stops mattering (norm is multiplied by k1 = 0).
- **b = 0**: norm = 1 for every doc, so there's **no length normalisation**. Long docs aren't penalised and win more often
  just by containing more words. (b = 1 is full normalisation; 0.75 is the usual compromise.)
