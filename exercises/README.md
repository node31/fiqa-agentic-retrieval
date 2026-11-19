# Exercises: your workspace for the "✍️ C. The toy, by you" sections

This folder is **yours**. It's a separate top-level folder inside the project, apart from the reference code in
`src/`, so your attempts can be **evaluated automatically** (each chapter has a checker) without ever touching the
project's own code. Every chapter of the guide (`docs/from_zero/`) has a matching folder here.

```
fiqa-agentic-retrieval\
├── src\  docs\  examples\  tests\ ...   ← the finished project (reference, guide, toys, data)
└── exercises\                           ← YOU ARE HERE: your own code
    ├── README.md                ← this file
    ├── common.py                ← paths to the repo, the tiny dataset and full FiQA (you don't edit this)
    ├── conftest.py, pytest.ini  ← make the checkers work, incl. --full (you don't edit these)
    ├── fullkit.py               ← reference results for the --full checks (you don't edit this)
    ├── .cache\                  ← slow results saved by --full runs (gitignored; delete to recompute)
    ├── ch01_data\               ← each chapter folder has:
    │   ├── README.md            ←   1. the exercise, step by step
    │   ├── PAPER.md             ←   2. a pen-and-paper worksheet (fill it in BEFORE coding)
    │   ├── loader.py            ←   3. the starter file: function stubs with TODOs → YOU write the code here
    │   └── check_loader.py      ←   4. the automatic checker (don't edit; run it)
    ├── ch02_metrics\   ch03_bm25\   ch04_dense\   ch05_fusion\
    ├── ch06_pipeline\  ch07_shards\ ch08_agents\
    ├── ch09_rebuild\            ← pointer to the big "rebuild everything" project
    └── solutions\               ← ANSWER KEY: worked solutions + ANSWERS.md per chapter (spoilers!)
```

| folder | guide chapter | you write | what the checker compares against |
|---|---|---|---|
| `ch01_data` | 1. the data | `loader.py`: read the 3 BEIR files | the repo's `src/beir_data.py` |
| `ch02_metrics` | 2. measuring | `my_metrics.py`: Recall, MRR, nDCG, Coverage, percentile | the hand examples in the guide + `src/metrics.py` |
| `ch03_bm25` | 3. BM25 | `my_bm25.py`: tokenizer, inverted index, BM25 | `src/retrieval/bm25.py` (same scores to 4 decimals) |
| `ch04_dense` | 4. embeddings | `my_dense.py`: cosine by hand, encode, dense search | numpy + the real MiniLM model |
| `ch05_fusion` | 5. hybrid | `my_fusion.py`: min-max, linear fusion, RRF | toy 08's numbers + `src/retrieval/hybrid.py` |
| `ch06_pipeline` | 6. Part A | `mini_pipeline.py`: BM25 vs dense vs hybrid on 24 docs | the repo's BM25 rankings + metrics |
| `ch07_shards` | 7. Part B | `mini_shards.py`: 2 shards + router, versioned writes, a WAL | a single-node BM25; toy 16's write sequence; crash-twice recovery |
| `ch08_agents` | 8. Part C | `agent_loop.py`: JSON parsing, agent loop + guardrails, interleave | scripted fake-LLM scenarios |

## One-time setup

You reuse the project's virtual environment (it already has numpy, sentence-transformers, faiss, pytest...):

```powershell
cd C:\path\to\fiqa-agentic-retrieval
.\.venv\Scripts\Activate.ps1                       # prompt now shows (.venv)
$env:PYTHONIOENCODING = "utf-8"
python -m pytest exercises\ch01_data               # runs the chapter-1 checker
python -m pytest exercises                         # runs ALL chapters' checkers (your scoreboard)
```

At first every check **fails** with `NotImplementedError: TODO ...`. That's expected: nothing is written yet.
(ch06 shows them as `ERROR at setup`: same message, because its checks share one `run()` call.)

## Two speeds: toy (seconds) and full (the real 57,638-doc FiQA)

Every chapter has both. **Toy** checks run on the 24-doc tiny FiQA and finish in seconds: use them while you code.
**Full** checks run *your* code on the real corpus and compare it with the project's real numbers. They take minutes
(all 8 chapters: ~18 min the first time, measured; the longest is ch06's first run, ~11 min, because it encodes every passage with your `encode()`), and slow parts are cached
in `exercises\.cache\`, so a second run is quick. Run them in a **second terminal** and keep working:

```powershell
python -m pytest exercises                  # TOY checks only (the default): seconds
python -m pytest exercises --full           # FULL checks only: minutes (needs: python scripts\download_data.py)
python -m pytest exercises --all            # both
python -m pytest exercises\ch03_bm25 --full # one chapter, real data
python exercises\ch06_pipeline\mini_pipeline.py --full   # demos take --full too
```

| chapter | full check proves | ours |
|---|---|---|
| ch01 | your loader reads all 57,638 docs exactly like the reference | 500 dev / 1,238 pairs |
| ch02 | your metrics on a real 500-query run = the reference | BM25 nDCG 0.264 |
| ch03 | your BM25 on the real corpus | 336018 → 10.68; nDCG 0.264 |
| ch04 | your encode = the stored vectors; your top_k | dense nDCG 0.388 |
| ch05 | your fusion on real result lists | linear 0.419, RRF 0.373 |
| ch06 | **your whole Part A pipeline** | 0.264 / 0.388 / 0.419 |
| ch07 | 3 shards over 57,638 docs = one node | identical top-10s |
| ch08 | your agent loop on 100 real two-part questions | coverage clearly up |

## The answer key (`solutions\`)

Each chapter has a worked solution (same file name as your stub, with comments explaining every *why*) and an
`ANSWERS.md` with the worksheet answers. **Spoilers**: try first, compare after (see `solutions\README.md`).

```powershell
$env:EXERCISES_SOLUTIONS = "1"; python -m pytest exercises; Remove-Item Env:EXERCISES_SOLUTIONS   # grade the answer key
python exercises\solutions\ch05_fusion\my_fusion.py                                              # run a solution's demo
```
(The project's own test suite, `python -m pytest` with no path, does **not** include these. Your unfinished
exercises never break it.)

## The loop for every chapter

```
1. read the chapter in the guide            (docs/from_zero/0N_*.md)
2. fill in PAPER.md by hand                  (no computer!)
3. open the starter .py, replace each `raise NotImplementedError(...)` with your code
4. run the checker:  python -m pytest exercises\ch0N_xxx              (all of it)
                     python -m pytest exercises\ch0N_xxx -k recall    (only checks with "recall" in the name)
5. red → read the message → fix → re-run, until all green
6. try the file directly too:  python exercises\ch0N_xxx\<file>.py   (each has a small demo at the bottom)
7. commit:  git add exercises ; git commit -m "ex ch0N: <what you learned>"
```

**Rules that make this worth it:**
- Don't open `src/` until you've been stuck for ~30 minutes. When you do, read it, close it, and write your own version.
- Starter files only use the Python standard library unless the chapter says otherwise (ch03 uses `snowballstemmer`,
  ch04 uses numpy and sentence-transformers).
- When the toy checks are green, run the chapter with `--full`. Same code, real data, real numbers.

## Troubleshooting

| you see | meaning | fix |
|---|---|---|
| `NotImplementedError: TODO (...)` | you haven't written that function yet | write it |
| `ModuleNotFoundError: No module named 'numpy'` | not using the venv | run the Activate line above |
| `Project code not found` | the `exercises` folder was moved out of the repo | keep it at `fiqa-agentic-retrieval\exercises` |
| "N deselected" | the other speed's checks (full ones in a toy run, or toy ones with `--full`) | use `--all` for both |
| a full check is "skipped: full FiQA not found" | the real data isn't downloaded | `python scripts\download_data.py` |
| a full result looks stale after you fixed a bug | cached results | delete `exercises\.cache\` |
| everything passes but you didn't write anything | `EXERCISES_SOLUTIONS=1` is still set | `Remove-Item Env:EXERCISES_SOLUTIONS` |
