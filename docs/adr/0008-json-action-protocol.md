# ADR 0008: Plain-JSON action protocol instead of native function calling

**Status:** accepted

## Context
Three LLM backends (fake, Copilot CLI, Azure) must run the same agent code.

## Decision
Prompts ask for exactly one JSON object; `parse_json` extracts it robustly; all arguments and ids are validated.

## Alternatives considered
Native `tools=[...]` on Azure only.

## Consequences
+ one code path, free dev iteration on the Copilot CLI, every byte visible for learning. − no server-side schema enforcement (mitigated by validation + fallbacks; measured as `queries_with_errors`).
