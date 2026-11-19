# Failure analysis (Part A4)

Chosen config: hybrid, linear min-max fusion α = 0.7 (dense weight), MiniLM, candidate depth 100, dev split.
**161 of 500 dev queries have Recall@10 = 0.** Candidates: `scripts/failure_cases.py` → `results/failure_candidates.json`.
Fix experiments: `scripts/failure_probe.py` → `results/failure_probe.json`. Scores below are fused scores in [0, 1]:
`0.7·dense_norm + 0.3·bm25_norm`, min-max normalised over each list's top-100.

How common each mode is among the 161 (a query can be in several):

| mode | count | meaning |
|---|---|---|
| near miss: a gold doc at fused rank 11–20 | 29 | Recall@20 would catch it; a reranker/agent has something to work with |
| neither retriever has any gold doc in its top-100 | 58 | first-stage recall failure: fusion and reranking cannot help |
| every gold doc is longer than the corpus p90 (270 words) | 18 | case 2 |
| a gold doc in BM25's top-10 but outside dense's top-100 | 10 | case 1 |

---

## Case 1: a lexical-only hit buried by the fusion weight (qid 4746)

**Query:** "Personal finance in EFU and NAFA"

| rank | doc | fused | snippet |
|---|---|---|---|
| 1 | 248368 | 0.700 | "look for an advisor who is a **NAPFA**-Registered Financial Advisor…" |
| 2 | 374518 | 0.577 | "online personal finance applications…" |
| 3 | 583912 | 0.519 | "Welcome to Money.SE… a personal finance question…" |
| 4 | 211767 | 0.468 | "invest in one of the funds… ask your Ulamas" |
| 5 | 119059 | 0.465 | "Being a successful CFP…" |
| **gold 19** | 216321 | **0.300** | "…the Pakistan stock market is going down… the funds invest in stock markets…" |

Gold ranks: **BM25 #1**, dense: not in top-1000.

**Diagnosis.** EFU and NAFA are Pakistani fund houses. They are rare tokens (max IDF 10.56, the corpus maximum), and
BM25 nails the one passage that contains them. MiniLM has never learned these acronyms: WordPiece splits them into
meaningless subwords (`e ##fu`, `na ##fa`), so the query embedding is just "personal finance" and lands on generic answers (and a lookalike
acronym, NAPFA). The fused score of the gold is **exactly 0.300**: BM25-normalised 1.0 × weight 0.3, plus 0 for the
missing dense component. With α = 0.7, *any* document that is a lexical-only hit is capped at 0.3, while a mediocre
dense match scores 0.4–0.5. Note that top-1 is 0.700 for the mirror reason: it is the dense #1 with no BM25 support.

**Fix (tested).** Lower α moves the gold to rank 2 (α = 0.5) or rank 1 (α = 0.3). A *query-adaptive* α (0.5 when the
query contains a term with IDF > 9, i.e. df ≤ 6; 25 of 500 dev queries) gives **nDCG 0.4206 vs 0.4190** and R@10
0.4956 vs 0.4914. With threshold 8 (52 queries) it is *worse* (0.4176). That is a +0.4% effect that flips sign with
the threshold, so it's noise-level on 500 queries, and we did not adopt it. A more robust fix: impute the missing
component with the list's minimum instead of 0, or always admit BM25's top-1 when its score margin over #2 is large
(rare exact matches), then validate on the test split once.

---

## Case 2: the answer is in the part of a long passage the encoder never reads (qid 4029)

**Query:** "Family suggests my first real estate. Advice?"

| rank | doc | fused | snippet |
|---|---|---|---|
| 1 | 531299 | 0.802 | "I would second the advice to not do this. Real estate ownership is complex…" |
| 2 | 117960 | 0.700 | "Be very careful to hold on tight to your money!… Don't let relatives convince you…" |
| 3 | 166531 | 0.546 | "…a family-run trust… probably not something you should get involved with…" |
| 4 | 429215 | 0.517 | "…current state of devaluation of real estate…" |
| 5 | 387717 | 0.487 | "Real estate investment is a proven creator of wealth…" |

Gold (3 docs): 208783 (173 words) fused **#13**, 0.368; 324386 (**422 words**), not in fused top-100 (dense #253, BM25 >1000);
63649 (**1,477 words**), not in fused top-100 (dense #448).

**Diagnosis.** The two missed gold passages are long. The encoder truncates at 256 WordPiece tokens: 324386 is 542 tokens
and 63649 is 1,871, so the encoder sees the first 47% and 14% respectively. Splitting each into 150-word windows and scoring
each window shows **the best-matching window is #4 of 5 and #10 of 15**, entirely past the truncation point.
63649 opens with job and location stability, and the real-estate advice comes later. BM25's length normalisation (b = 0.75)
also pushes a 1,477-word passage down. The top-5 are reasonable "don't mix family and real estate" answers, so the
query is also broad.

**Fix (partly tested).** Index passages as overlapping chunks (150 words, stride 100) and score a doc by its best chunk
(max-pooling). Cosine to the query rises **0.464 → 0.548** (324386) and **0.436 → 0.527** (63649), but dense rank-10 needs
0.583, so chunking alone narrows the gap without closing it. Costs: 1.8× vectors in the index (103,861 chunks for 57,638
passages; mean 133 words, but the tail is long), and other long docs gain too. That needs a full dev eval before adoption, which is listed
as future work. Cheaper partial fix: lower BM25 b for this corpus (part of the same experiment).

---

## Case 3: the query doesn't contain the question (qid 7490)

**Query:** "Clarify on some Stocks Terminology"

| rank | doc | fused | snippet |
|---|---|---|---|
| 1 | 330534 | 0.700 | "For all practical purposes the words [stock and share] mean the same thing…" |
| 2 | 208331 | 0.498 | "It is known as the range or the price spread of the stock…" |
| 3 | 512062 | 0.488 | "…'Google ordinary', 'Google preference'…" |
| 4 | 235391 | 0.484 | "stocks are a share of equity in the company, while bonds are…" |
| 5 | 286296 | 0.451 | "A stock represents your share of ownership…" |

Gold: 372153 fused **#12** (0.335; dense #12, BM25 none), 173836 not in fused top-100 (dense #753, BM25 none).
Both gold passages explain **trading volume** and ticker prints ("The volume is the number of shares traded that day…").

**Diagnosis.** BEIR's FiQA query is the *title* of a forum post; the actual question ("what do volume and the
numbers on the ticker mean?") was in the body, which the benchmark drops. The query shares only one stemmed term with
either gold passage, **"stock"** (df 6,621, so almost no IDF weight), and "volume" never appears in the query. Every retriever is
answering the question as written, and arguably the top-5 answer it well. It's an information-in-the-query failure,
and the qrels are incomplete with respect to the title. 58 of the 161 failures have *no* gold doc in either retriever's
top-100; many look like this.

**Fix.** Nothing at the retrieval layer can recover information that isn't in the query, and tuning for it would overfit
the benchmark's artefact. In a product: (1) send the full post body (title + body) as the query; (2) let the agent detect a
vague query (only high-df terms, low top-1 margin) and **ask a clarifying question** or search several interpretations
(Part C's multi-agent decomposition does the latter). In evaluation: report this class separately rather than "fixing" it.

---

### What we learned (across the three)

- Fusion weights act as *caps*: with α = 0.7 a lexical-only hit can never exceed 0.3. That's great on average and bad for rare entities.
- Truncation is silent: nothing errors, and the encoder simply never sees 86% of the longest gold passage.
- A third of the failures are first-stage recall misses where no later stage can help. That bounds what reranking (Part A
  ablation) and agents (Part C) can achieve, and it is why recall-oriented changes (chunking, query expansion) come first.
- Measurement detail that bit us while building this: min-max fusion depends on candidate depth. A depth-200 diagnostic
  list ranked some golds at 8–10 even though the served depth-100 list did not have them in the top 10, so diagnostics must use the serving depth.
