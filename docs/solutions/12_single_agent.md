# Solutions: single agent

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** The assignment caps the agent at 3 steps where it 'may rewrite the query and search again'. Our free initial search is not an LLM step, so we spend ≤ 3 LLM calls and ≤ 3 searches. Arguably within the rules; state it explicitly in DESIGN.md (we do).

**2.** See DESIGN.md §7 for the measured answer. Expectation: no gain or a small loss. Most plain FiQA queries are single-intent, and LLM reordering on 200-char snippets can hurt MRR.

**3.** Measure both; typically pure RRF keeps MRR closer to the baseline, while the LLM choice helps when the first search was off-topic.

**4.** Gate on the top-1 *dense cosine* or the score gap between rank 1 and 2 (fused min-max scores are relative). See drill 9.
