# Replays

Open [index.html](index.html) in a browser. It lists every recorded replay, grouped by the five categories used in
the deck and the docs, and every page here opens straight from disk: the recorded data is inside it.

| page | what it shows | recorded by |
|---|---|---|
| [scenarios_sync_tiny.html](scenarios_sync_tiny.html), [scenarios_async_tiny.html](scenarios_async_tiny.html), [scenarios_sync_full.html](scenarios_sync_full.html), [scenarios_async_full.html](scenarios_async_full.html) | the eight scripted scenarios on one cluster type, with a nav over them; each embeds the recorded animated replay from `scenarios/SimpleScenarios/results/` unchanged, and two links open it on its own page (the original file, or the embedded copy in a new tab) | `scripts/run_simple_scenarios.py`, then `scripts/scenario_pages.py` |
| [security.html](security.html) | 14 cases, each protection off and on, grouped by category; a case opens with both sides hidden, and Show OFF, Show ON, Compare both and Hide both (keys 1, 2, 3, 0) reveal them | `scripts/security_replay.py` |
| [a2a.html](a2a.html) | two A2A agents and an LLM, one arrow per message; a dashed pink "from memory" tag marks the fact or words memory added to an LLM prompt or a search, and the detail pane highlights them in the JSON | `scripts/a2a_two_agents.py` |
| [memory.html](memory.html) | 16 steps of the agent's memory, with what the LLM read and answered | `scripts/memory_replay.py` |
| [ways_to_call.html](ways_to_call.html) | one question through Swagger `/docs`, `demo_call`, `search.py` and `ask.py`, with every HTTP call the router received (a logging proxy in between) | `scripts/ways_to_call_replay.py` |

`data/` holds the recorded JSON each page was rendered from; every generator has a `--render-only` mode that
rebuilds its page from that file without a cluster or an LLM. The colours and the nav come from
`scripts/replay_style.py`: Distributed blue, Agentic Security purple, Agentic Privacy teal, Agentic Cost Optimization
amber, Agentic Extensions pink. Red is kept for errors.
