# ch08 worksheet: answers

> Fill in `exercises/ch08_agents/PAPER.md` yourself **before** reading this.

**1.** The LLM only ever **produces text**. Here that text is a JSON object *describing* a tool call:
`{"action": "search", "query": "…"}`. **Your code** does everything else:
- builds the prompt
- parses and validates the JSON
- decides whether the action is allowed
- runs the search
- puts the results into the next prompt
- enforces the caps
- assembles the final answer

"Native" tool calling (OpenAI/Azure function tools) is the same thing with the JSON format enforced by the API.

**2.** Return **`d2` first**, then fill the remaining slots from the searches (interleaved).
- `parse_action` finds the JSON inside the chatter and the fence.
- `d99` was never shown to the model, so it's **invented**. Drop it: an id the model never saw can't be a judgment,
  only a hallucination, and it could even be a valid-looking id of an unrelated doc.

**3.** Two reasons:
- **Robustness:** if the LLM fails, times out or returns garbage, you still have a sensible answer (plain retrieval).
  The agent can only *improve* on the baseline, never return nothing.
- **Cost:** the first prompt already contains evidence, so the model can finish in one call instead of
  spending a call just to ask for the obvious first search.

**4.** Step 1 = 600, step 2 = 600 + 500 = 1,100, **step 3 = 1,600** characters. The total sent over 3 calls is
600 + 1,100 + 1,600 = **3,300** characters (~825 tokens), not 3 × 600 = 1,800. Every call **re-sends the whole history**,
so the total grows like 1 + 2 + … + n ≈ n²/2. That's why step caps and compact prompts (ids + short snippets, not full
passages) matter. In the project the multi-agent design was cheaper than one long single-agent conversation:
940 vs 2,681 tokens per compound query.

**5.**
- Sort-by-score merge: **[a0 … a9]**. All of B's scores (0.2–0.3) are below all of A's (0.8–0.9), so B gets nothing → **Coverage@10 = 0**.
- Interleave: **[a0, b0, a1, b1, a2, b2, a3, b3, a4, b4]** → **Coverage@10 = 1** (assuming each list's top 5 holds a gold doc for its half).

Scores from different sub-queries aren't comparable (just like local IDF in ch07), so allocate *slots*, not scores.
The project's numbers on 100 real compound questions: Coverage@10 went from 0.10 (one search) to 0.26 (multi-agent).

**6.** Route by question type:
- **plain question → no agent.** Use the plain hybrid search: it's better (R 0.502 vs 0.494 single-agent / 0.476 multi-agent), ~20 ms instead of
  seconds, and free.
- **compound question → multi-agent** (planner splits, workers search each part, interleave). Recall@10 0.204 → 0.335,
  Coverage@10 0.10 → 0.26.
- **Who decides:** the planner itself. It kept 91/100 plain questions as one search and split 100/100 compound ones.
- **Guardrails:** a step cap, a token budget and a fallback to the plain search on any failure.

This is DESIGN.md §7. The `--full` ch08 check shows the same effect with your own loop: one search vs your agent on 100 real two-part questions.
