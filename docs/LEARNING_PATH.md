# Reading order

If you want to understand the system, read in this order. Each step assumes the ones before it.

## 1. From zero
[from_zero/](from_zero/README.md) is me explaining the whole system to someone who knows Python and nothing about search. Read the chapters in order:

1. [The problem and the data](from_zero/01_the_problem_and_the_data.md)
2. [Measuring quality and speed](from_zero/02_measuring_quality_and_speed.md)
3. [Keyword search with BM25](from_zero/03_keyword_search_bm25.md)
4. [Meaning search with embeddings](from_zero/04_meaning_search_embeddings.md)
5. [Hybrid fusion and reranking](from_zero/05_hybrid_and_reranking.md)
6. [Part A end to end](from_zero/06_part_a_end_to_end.md)
7. [Part B: the distributed service](from_zero/07_part_b_distributed.md)
8. [Part C: agents](from_zero/08_part_c_agents.md)

Keep the [GLOSSARY](GLOSSARY.md) open while you read.

## 2. Topic docs
Each one goes deeper on one part of the system and says why I built it that way.

| Part | Docs | Code |
|---|---|---|
| Basics | [00 RAG primer](00_rag_primer.md) | `src/beir_data.py` |
| A: retrieval | [01 BM25](01_lexical_retrieval_bm25.md), [02 Dense](02_dense_retrieval.md), [03 Vector indexes](03_vector_indexes.md), [04 Metrics](04_ir_metrics.md), [05 Hybrid fusion](05_hybrid_fusion.md), [06 Reranking](06_reranking.md), [07 Benchmarking](07_benchmarking_latency_memory.md) | `src/retrieval/`, `src/metrics.py`, `src/eval.py` |
| B: service | [08 Sharding](08_sharding_scatter_gather.md), [09 Live updates](09_live_updates_consistency.md), [10 Migration](10_model_migration.md), [scenarios](scenarios.md) | `src/service/` |
| C: agents | [11 Tool calling](11_llm_tool_calling.md), [12 Single agent](12_single_agent.md), [13 Multi-agent](13_multi_agent_orchestration.md), [14 Agent evaluation and cost](14_agent_evaluation_cost.md) | `src/agent/` |
| Scale and evidence | [15 Scaling to 100M](15_scaling_to_100M.md), [16 Ablations](16_ablations.md), [eval](eval.md) | `scripts/ablations.py` |

The results write-ups are [part_a_results](walkthroughs/part_a_results.md), [part_b_results](walkthroughs/part_b_results.md) and `results/failures.md`.

## 3. Design and decisions
[DESIGN.md](../DESIGN.md), end to end. Its "Agent evaluation and routing" section compares the haiku dev run with the final Azure gpt-6-sol run and explains why the routing decision changed.

## 4. Questions
[project_faq.md](project_faq.md) and [concepts_faq.md](concepts_faq.md) hold the questions I expect about this system and my answers.

## Commands that run the real system
From the repo root, with the venv active:

```powershell
python -m venv .venv; pip install -r requirements.txt; python scripts/download_data.py   # setup, once
pytest -q                                                                                # all tests
python -m src.service.cluster                                                            # router :8000 + 3 shards
python src/search.py --query "what is short selling?"                                    # one search against it
pytest -q tests/test_store.py tests/test_integration_service.py tests/test_integration_async.py
python -m scripts.write_load --mode both --duration 30                                   # sync vs async under load
python scripts/migration_demo.py                                                         # zero-downtime model migration
python scripts/agent_eval.py --backend fake                                              # agent eval plumbing, no LLM cost
python -m scripts.docker_check        # Docker functional checks; tears the stack down unless --keep
python -m scripts.docker_trace        # shows the containers really talk; --failover stops a shard
```
