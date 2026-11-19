# ch01: Load the data yourself

Guide: `docs/from_zero/01_the_problem_and_the_data.md` (section "✍️ C. The toy, by you").

> 🎯 **Why:** every later exercise reads these three files. Writing the loader yourself makes the BEIR format
> (corpus / queries / qrels) second nature, and the checker catches the classic silent bugs (int ids, header line,
> wrong split).

## Steps
1. Fill in `PAPER.md` first.
2. Look at the raw files (you're reading the real format):
   ```powershell
   Get-Content tests\fixtures\tinyfiqa\corpus.jsonl  | Select -First 3
   Get-Content tests\fixtures\tinyfiqa\queries.jsonl
   Get-Content tests\fixtures\tinyfiqa\qrels\dev.tsv
   ```
3. Implement in `loader.py`, in this order (each one has its own checks):
   `load_jsonl` → `load_qrels` → `load_split` → `stats`.
4. Run `python -m pytest exercises\ch01_data` until all green. Then `python exercises\ch01_data\loader.py` prints your stats.
5. Then the real thing: `python -m pytest exercises\ch01_data --full` and `python exercises\ch01_data\loader.py --full`.

## Hints (read only if stuck)
- `json.loads(line)` per line; ids stay **strings** ("3" ≠ 3).
- The qrels TSV has a header line: `query-id	corpus-id	score`.
- `queries.jsonl` holds the queries of **all** splits; keep only those in the split's qrels.

## Two speeds: toy (seconds) and full (real FiQA)
| | command | what it proves | time |
|---|---|---|---|
| toy | `python -m pytest exercises\ch01_data` | your loader reads the 24-doc tiny FiQA exactly like the reference | seconds |
| full | `python -m pytest exercises\ch01_data --full` | 57,638 docs · dev 500/1,238 · test 648/1,706 · train 5,500/14,166, and every doc identical to the reference loader | seconds |
| demo | `python exercises\ch01_data\loader.py [--full]` | prints the stats | |
