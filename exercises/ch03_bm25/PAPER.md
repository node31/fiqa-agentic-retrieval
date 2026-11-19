# ch03 worksheet (pen and paper first)

Docs:
- d1 = "short selling means borrowing shares"
- d2 = "a short story about selling a house and selling a car"
- d3 = "mortgage rates rose"

Query: "short selling". Stopwords here: only "a" and "and" ("about" is NOT a Lucene stopword).

1. Tokens (stemmed): d1 = ____________   d2 = ____________   d3 = ____________
2. |d1| = ___  |d2| = ___  |d3| = ___   avgdl = ___   N = 3
3. df(short) = ___  df(sell) = ___   idf(short) = ln(1 + (3 − df + 0.5)/(df + 0.5)) = ___
4. For d1: norm = 1 − 0.75 + 0.75·|d1|/avgdl = ___ ; tf-part(short) = tf·2.2/(tf + 1.2·norm) = ___
5. BM25(d1) = ___   BM25(d2) = ___   Which wins, and why? ____________
   (The checker knows the answer: d2 ≈ 0.985, d1 ≈ 0.940. Compute yours first!)
6. What does k1 = 0 turn BM25 into? ____________  And b = 0? ____________

---
✅ Answers: [`solutions/ch03_bm25/ANSWERS.md`](../solutions/ch03_bm25/ANSWERS.md). Open it only *after* you've written yours.
