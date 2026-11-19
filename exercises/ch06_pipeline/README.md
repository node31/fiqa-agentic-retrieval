# ch06: Part A with your own pipeline: 24 documents, then all 57,638

Guide: `docs/from_zero/06_part_a_end_to_end.md`.

> 🎯 **Why:** Part A is graded as one system, not as parts. Wiring *your* loader, BM25, embeddings, fusion and metrics
> together (ch01–ch05) is the dress rehearsal for the full rebuild. With 24 docs you can read every ranking and
> explain each position, which is impossible at 57,638.

## Steps
1. Finish ch01–ch05 first (this chapter imports your files from them).
2. Implement `run()` in `mini_pipeline.py`: build the BM25 index and the doc vectors **once** (`doc_vectors()` is given:
   it encodes with *your* `encode` and caches the result), then for every query the top-10 for BM25, dense and hybrid
   (linear α = 0.7, fusing BM25 and dense lists of depth 100), then your `evaluate()` for each system.
3. `python -m pytest exercises\ch06_pipeline`.
4. `python exercises\ch06_pipeline\mini_pipeline.py` prints a comparison table and every ranking.
5. **Read** the rankings for 2 queries and write in `NOTES.md` *why* each gold doc is where it is
   (which words matched? did dense rescue it? did fusion help or hurt?). This is your first failure analysis.

## Two speeds: toy (seconds) and full (real FiQA)
| | command | what it proves | time |
|---|---|---|---|
| toy | `python -m pytest exercises\ch06_pipeline` | structure, BM25 equals the reference, metrics consistent, fusion uses both signals | ~10 s |
| full | `python -m pytest exercises\ch06_pipeline --full` | **your whole pipeline reproduces the project's Part A table**: BM25 0.264 · dense 0.388 · hybrid 0.419 | ~11 min the first time (your `encode` over 57,638 docs), ~3 min after (cached in `exercises\.cache\`) |
| demo | `python exercises\ch06_pipeline\mini_pipeline.py [--full]` | the comparison table + rankings | |

Start the full run in a second terminal and keep working on the toy one. If you change `encode()`, the cache recomputes itself.
