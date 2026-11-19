# Answer key (spoilers!)

**Try each exercise yourself first.** Reading a solution before you've struggled with the problem teaches you much
less than it feels like it does. A good rule: open a solution only when your checker is green (to compare
approaches), or after ~30 minutes stuck on one function (then read *only* that function, close it, and write your own).

| chapter | worked solution | worksheet answers |
|---|---|---|
| ch01 data | [`ch01_data/loader.py`](ch01_data/loader.py) | [`ch01_data/ANSWERS.md`](ch01_data/ANSWERS.md) |
| ch02 metrics | [`ch02_metrics/my_metrics.py`](ch02_metrics/my_metrics.py) | [`ch02_metrics/ANSWERS.md`](ch02_metrics/ANSWERS.md) |
| ch03 BM25 | [`ch03_bm25/my_bm25.py`](ch03_bm25/my_bm25.py) | [`ch03_bm25/ANSWERS.md`](ch03_bm25/ANSWERS.md) |
| ch04 dense | [`ch04_dense/my_dense.py`](ch04_dense/my_dense.py) | [`ch04_dense/ANSWERS.md`](ch04_dense/ANSWERS.md) |
| ch05 fusion | [`ch05_fusion/my_fusion.py`](ch05_fusion/my_fusion.py) | [`ch05_fusion/ANSWERS.md`](ch05_fusion/ANSWERS.md) |
| ch06 pipeline | [`ch06_pipeline/mini_pipeline.py`](ch06_pipeline/mini_pipeline.py) | [`ch06_pipeline/ANSWERS.md`](ch06_pipeline/ANSWERS.md) (a filled-in NOTES.md) |
| ch07 shards | [`ch07_shards/mini_shards.py`](ch07_shards/mini_shards.py) | [`ch07_shards/ANSWERS.md`](ch07_shards/ANSWERS.md) |
| ch08 agents | [`ch08_agents/agent_loop.py`](ch08_agents/agent_loop.py) | [`ch08_agents/ANSWERS.md`](ch08_agents/ANSWERS.md) |
| ch09 rebuild | — the reference *is* the project (`src/`) | your numbers vs `results/` |

**The solutions are verified by the same checkers you use**, on both speeds:

```powershell
$env:EXERCISES_SOLUTIONS = "1"
python -m pytest exercises            # toy checks against the answer key
python -m pytest exercises --full     # full checks against the answer key (real FiQA)
Remove-Item Env:EXERCISES_SOLUTIONS   # back to grading YOUR code
```

Each solution file also runs on its own and prints the worksheet's numbers, e.g.
`python exercises\solutions\ch03_bm25\my_bm25.py` (add `--full` for the real corpus).

Things to look for when comparing with your version:
- the `# WHY:` comments: the reason behind each line, which is the part worth remembering
- the `# PITFALL:` comments: the bugs these functions usually have (most of which the checkers test for)
- **your solution doesn't need to look like this one.** If the checker is green, it's right. Different and correct is good.
