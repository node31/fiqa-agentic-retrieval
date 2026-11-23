# Entry points. Every target is a one-line Python command, so on Windows without `make`
# you can copy the command after the tab. PY defaults to the project venv when it exists.
PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)
ROUTER_URL ?= http://127.0.0.1:8000
NEW_MODEL ?= bge-small
OLD_MODEL ?= minilm
# The chosen configuration's model: MiniLM fine-tuned on FiQA train (make finetune).
FINAL_MODEL ?= minilm-ft

.PHONY: help data index package-index finetune package-finetune cluster up down eval bench ablations failures \
        migrate cutover rollback finalize load migration-demo compound agent-eval agent-dry-run \
        test toys test-final

help:
	@grep -E '^[a-z-]+:' Makefile | sed 's/:.*//' | tr '\n' ' '; echo

# ---- data + offline index ---------------------------------------------------
data:
	$(PY) scripts/download_data.py
index: data
	$(PY) -m src.index.build_index --models minilm bge-small
package-index:
	$(PY) -m src.index.build_index --package
finetune:           ## offline: fine-tune MiniLM on FiQA train (CPU, a few hours), then embed the corpus with it
	$(PY) -m scripts.finetune && $(PY) -m src.index.build_index --models minilm-ft
package-finetune:
	$(PY) -m src.index.build_index --package-local minilm-ft

# ---- service ----------------------------------------------------------------
cluster:            ## local processes, no docker
	$(PY) -m src.service.cluster --shards 3
up:
	docker compose up -d --build
down:
	docker compose down

# ---- Part A: evaluation ------------------------------------------------------
eval:               ## chosen operating point through the running router
	$(PY) -m src.eval --mode service --url $(ROUTER_URL) --strategy hybrid --fusion linear --alpha 0.7 --name svc-hybrid
bench:              ## full benchmark matrix -> results/bench.json (starts its own clusters)
	$(PY) scripts/bench.py
ablations:
	$(PY) scripts/ablations.py --model minilm && $(PY) scripts/ablations.py --model bge-small && \
	$(PY) scripts/ablations.py --model minilm --rerank
failures:
	$(PY) scripts/failure_cases.py

# ---- Part B3: zero-downtime migration (run `make load` in another terminal) --
migrate:
	$(PY) scripts/migration.py migrate --model $(NEW_MODEL) --wait --url $(ROUTER_URL)
cutover:
	$(PY) scripts/migration.py cutover --model $(NEW_MODEL) --url $(ROUTER_URL)
rollback:
	$(PY) scripts/migration.py rollback --url $(ROUTER_URL)
finalize:
	$(PY) scripts/migration.py finalize --model $(OLD_MODEL) --url $(ROUTER_URL)
load:
	$(PY) scripts/query_load.py --url $(ROUTER_URL) --duration 120
migration-demo:     ## all of the above, scripted, with load + writes
	$(PY) scripts/migration_demo.py

# ---- Part C: agents -------------------------------------------------------------
compound:
	$(PY) data/build_compound.py
agent-dry-run:      ## free: token/cost estimate before spending
	$(PY) scripts/agent_eval.py --backend azure --dry-run
agent-eval:         ## COSTS MONEY: needs .env with Azure key
	$(PY) scripts/agent_eval.py --backend azure --confirm-spend --max-usd 5
agent-eval-dev:     ## free dev run on the GitHub Copilot CLI backend
	$(PY) scripts/agent_eval.py --backend copilot --workers 8

# ---- tests + learning material ----------------------------------------------------
test:
	$(PY) -m pytest -q
toys:
	for f in examples/[0-9]*.py; do $(PY) $$f || exit 1; done
test-final:         ## the ONE test-split run of the chosen configuration
	$(PY) -m src.eval --mode local --split test --final-test-run --strategy hybrid --model $(FINAL_MODEL) --fusion linear --alpha 0.7 --name FINAL-test-hybrid-linear-$(FINAL_MODEL)
