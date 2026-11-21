# ADR 0008: A plain-JSON action protocol instead of native function calling

Status: accepted

## Context
Three LLM backends (fake, Copilot CLI, Azure) must run the same agent code.

## Decision
My prompts ask for exactly one JSON object. `parse_json` extracts it from whatever the model returns, and I validate every argument and id.

## Alternatives considered
Native `tools=[...]` function calling, which only the Azure backend supports.

## Consequences
I have one code path, free development on the Copilot CLI, and every byte of the exchange is visible. I lose server-side schema enforcement; validation and fallbacks cover that, and I count the failures as `queries_with_errors`.
