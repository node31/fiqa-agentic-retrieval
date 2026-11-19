# Solutions: tool calling

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** JSON mode: the output is guaranteed valid JSON (any shape). Structured outputs: valid JSON matching *your* schema. Function calling: the model emits a structured call (name + args matching a schema) in a dedicated field, and you return tool results in typed messages.

**2.** {'a': '}'}. A naive regex like `\{.*?\}` stops at the first '}', which is inside the string. The brace scanner tracks string state and escapes.

**3.** Add `tools=[{'type': 'function', 'name': 'search', 'parameters': {...}}]` to responses.create, read `function_call` items from `response.output`, and map them to the same action dict the JSON path produces; fall back to parse_json(output_text).

**4.** Rarely on FiQA: passages are short and snippets already carry most content. It helps when the decision needs the full text (e.g. the key sentence is at the end). Cost: +1 call + ~500 tokens per use.
