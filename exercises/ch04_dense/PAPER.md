# ch04 worksheet (pen and paper first)

1. a = (3, 4). |a| = ____. normalize(a) = (____, ____).
2. b = (4, 3). cosine(a, b) = ____.   c = (−3, −4): cosine(a, c) = ____.   d = (−4, 3): cosine(a, d) = ____.
   What do the values 1, 0 and −1 mean geometrically? ____________
3. Why can we use the dot product instead of cosine after normalising? ____________
4. Predict: query "gambling on equities dropping" vs these 4 docs. Which ranks first with MiniLM, and why can't BM25 find it?
   - "Short selling means borrowing shares and selling them"
   - "Paying off a mortgage early saves interest"
   - "Betting that a share price will fall is called shorting"
   - "A Roth IRA grows tax free"
   ________________________________________
5. MiniLM reads at most 256 sub-word tokens. A passage has 1,800 tokens. What fraction does the model see? ____
   Which failure case in `results/failures.md` is this? ____
6. Predict the cosine of "a good investment" vs "not a good investment": high or low? Why? ____________

---
✅ Answers: [`solutions/ch04_dense/ANSWERS.md`](../solutions/ch04_dense/ANSWERS.md). Open it only *after* you've written yours.
