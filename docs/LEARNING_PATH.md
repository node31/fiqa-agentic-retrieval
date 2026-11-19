# Learning path

A suggested route through the repo. For every module use the same loop:

> **read** the LEARN doc → **run** its toy → **read** the real code (start at the module docstring) →
> **observe** it on FiQA → **do** the exercises → **check** `docs/solutions/` → tick the box.

Time estimates assume you are new to IR and agents but comfortable with Python.

## Stage 0: orientation (½ day)
- [ ] [00 RAG primer](00_rag_primer.md) + [GLOSSARY](GLOSSARY.md)
- [ ] Set up: `python -m venv .venv`, `pip install -r requirements.txt`, `python scripts/download_data.py`
- [ ] `pytest -q` (all green?) and `python -m src.beir_data`

## Stage 1: retrieval fundamentals (Part A, 2 days)
- [ ] [01 BM25](01_lexical_retrieval_bm25.md): toys 00, 01, 02, 03 · code `src/retrieval/bm25.py`
- [ ] [04 Metrics](04_ir_metrics.md): toy 07 · code `src/metrics.py` · tests `tests/test_metrics.py`
- [ ] [02 Dense](02_dense_retrieval.md): toys 04, 05 · code `src/retrieval/dense.py`, `src/index/build_index.py`
- [ ] [03 Vector indexes](03_vector_indexes.md): toy 06
- [ ] [05 Hybrid fusion](05_hybrid_fusion.md): toy 08 · `python scripts/ablations.py`
- [ ] [06 Reranking](06_reranking.md): toy 09
- [ ] [07 Benchmarking](07_benchmarking_latency_memory.md): toys 10, 11 · `src/eval.py`, `results/bench.json`
- [ ] Read `results/failures.md` and [walkthroughs/part_a_results.md](walkthroughs/part_a_results.md)

## Stage 2: distributed service (Part B, 2 days)
- [ ] [08 Sharding](08_sharding_scatter_gather.md): toys 12, 13, 14, 15 · `src/service/router.py`
- [ ] Run it: `python -m src.service.cluster` then `python src/search.py --query "what is short selling?"`
- [ ] [09 Live updates](09_live_updates_consistency.md): toys 16, 17, 18 · `src/service/store.py`, `wal.py`
- [ ] `pytest -q tests/test_store.py tests/test_integration_service.py`, then read every test
- [ ] [10 Migration](10_model_migration.md): toy 19 · `python scripts/migration_demo.py`
- [ ] Read [walkthroughs/part_b_results.md](walkthroughs/part_b_results.md)

## Stage 3: agents (Part C, 2 days)
- [ ] [11 Tool calling](11_llm_tool_calling.md): toy 20 · `src/agent/llm.py`
- [ ] [12 Single agent](12_single_agent.md): toy 21 · `src/agent/single_agent.py`
- [ ] [13 Multi-agent](13_multi_agent_orchestration.md): toys 22–25 · `src/agent/orchestrator.py`
- [ ] [14 Agent evaluation and cost](14_agent_evaluation_cost.md): toy 26 · `python scripts/agent_eval.py --backend fake`
- [ ] Hand-check 10 compound queries: [data/compound_review_sheet.md](../data/compound_review_sheet.md)

## Stage 4: system thinking + Project FAQ (1 day)
- [ ] [15 Scaling to 100M](15_scaling_to_100M.md)
- [ ] All [ADRs](adr/README.md): can you defend each choice in 2 sentences?
- [ ] [DESIGN.md](../DESIGN.md) end to end
- [ ] [DRILLS.md](DRILLS.md): do at least drills 1, 2 and 4 *without* looking at the solutions
- [ ] [project_faq.md](project_faq.md): answer each question out loud, then compare

## Map: concept → toy → code → test

| concept | toy | code | test |
|---|---|---|---|
| tokenisation, BM25 | 00–03 | `retrieval/bm25.py` | `test_bm25.py` |
| embeddings, flat index | 04–06 | `retrieval/dense.py` | `test_store.py` |
| metrics | 07 | `metrics.py` | `test_metrics.py` |
| fusion, merge | 08, 13 | `retrieval/hybrid.py` | `test_hybrid.py` |
| reranking | 09 | `retrieval/rerank.py` | – |
| latency, memory | 10, 11 | `eval.py`, `util.py` | – |
| global IDF, scatter-gather, timeouts | 12, 14 | `service/router.py` | `test_store.py`, `test_integration_service.py` |
| partitioning | 15 | `index/sharding.py` | doctest |
| versioned writes, WAL, freshness | 16–18 | `service/store.py`, `wal.py` | `test_store.py`, `test_integration_service.py` |
| migration | 19 | `store.start_build`, `router.cutover` | `test_store.py` |
| agent loop | 20, 21 | `agent/single_agent.py` | `test_agents.py` |
| orchestration, budgets, failure, slots | 22–25 | `agent/orchestrator.py` | `test_agents.py` |
| cost | 26 | `agent/llm.py` | – |
