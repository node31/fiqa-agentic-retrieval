# ch01 worksheet (pen and paper first)

1. Name the three files of a BEIR dataset and write one sample line of each from memory:
   - corpus: ______________________________________________
   - queries: _____________________________________________
   - qrels: _______________________________________________

2. In `tinyfiqa`, query `q1` is "how does short selling work". Open `corpus.jsonl` and, **before looking at the qrels**,
   write down which doc ids you think answer it: ____________
   Now check `qrels/dev.tsv`. Did you agree? Which did you miss or add? ____________

3. Why do we tune on dev and touch test only once? (2 sentences)
   ________________________________________________________________

4. FiQA has ~2.5 correct passages per query. If a system finds 1 of 2, what is Recall@10? ____
   What does that tell you about how "jumpy" per-query recall is? ____________

5. Predict for `tinyfiqa`: number of docs ____, dev queries ____, relevant pairs ____.
   (The checker will tell you if you were right.)

---
✅ Answers: [`solutions/ch01_data/ANSWERS.md`](../solutions/ch01_data/ANSWERS.md). Open it only *after* you've written yours.
