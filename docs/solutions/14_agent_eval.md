# Solutions: agent eval

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** Recall@10 counts gold docs; a system can score well by finding many docs for the easy half and none for the other. Coverage@10 asks whether *both needs* got served, which is what the user experiences.

**2.** 7/3 helps when one half is much harder (needs more slots to find gold at deeper ranks) and the planner knows which. It hurts when the guess is wrong. Equal split is the minimax-safe choice.

**3.** for b in range(1000): sample 100 query indices with replacement; compute ΔRecall = mean(r_A[i] - r_B[i]); the 2.5/97.5 percentiles give the CI. Paired resampling (same queries for both systems) gives much tighter intervals.

**4.** python data/build_compound.py --split test → data/compound_test.jsonl (template rewrites).
