# ch08 worksheet (pen and paper first)

1. In tool calling, what does the LLM actually do, and what does YOUR code do? ____________
2. The LLM replies: `Sure! ```json {"action": "finish", "doc_ids": ["d2", "d99"]} ``` `. You only ever showed it d1–d5.
   What should your loop return first, and why? ____________
3. Why does the agent search once BEFORE the first LLM call? ____________
4. Step 1's prompt is 600 characters, and each search adds ~500. How long is step 3's prompt? ____
   Why does agent cost grow faster than linearly with the number of steps? ____________
5. Two sub-queries: A = [a0..a9] (scores 0.9 to 0.8), B = [b0..b9] (scores 0.3 to 0.2).
   - Sort-by-score merge top-10 = ____________ → Coverage@10 = ____
   - Interleave top-10 = ____________ → Coverage@10 = ____
6. On plain questions the agents were slightly WORSE than no agent; on two-part questions much better.
   Write the routing policy you'd ship: ____________

---
✅ Answers: [`solutions/ch08_agents/ANSWERS.md`](../solutions/ch08_agents/ANSWERS.md). Open it only *after* you've written yours.
