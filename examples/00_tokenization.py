"""Toy 00: tokenisation. Why "Investing" can match "invest", and what we throw away.

Concept: lexical retrieval only matches what the tokenizer makes identical.
Real code: src/retrieval/bm25.py::tokenize       Doc: docs/01_lexical_retrieval_bm25.md
"""
from _common import banner, note, step, try_this

from src.retrieval.bm25 import STOPWORDS, _stem, tokenize

banner("Toy 00: tokenisation (lowercase -> split -> stopwords -> stem)")
text = "Is investing in Index-Funds better than SHORTING stocks? Not always!"
step(1, f"Raw text: {text!r}")
import re

raw = re.findall(r"[a-z0-9]+", text.lower())
step(2, f"Lowercase + split on non-alphanumerics -> {raw}")
note("'Index-Funds' became two tokens; punctuation vanished.")
kept = [t for t in raw if t not in STOPWORDS]
step(3, f"Drop stopwords ({len(STOPWORDS)} words) -> {kept}")
note("'not' was dropped: Lucene's stop list removes negation. A known BM25 weakness.")
step(4, "Stem each token (Snowball English):")
for t in kept:
    print(f"     {t:10s} -> {_stem(t)}")
step(5, f"Final tokens from tokenize(): {tokenize(text)}")
step(6, "Why it matters: query and document share NO raw words below, but match after stemming")
q, d = "investing in stocks", "He invests in a stock"
print(f"     query {tokenize(q)}  vs  doc {tokenize(d)}  -> overlap {set(tokenize(q)) & set(tokenize(d))}")
note("Synonyms ('equities' vs 'stocks') still never match. That gap is what dense retrieval fills (toy 04).")
try_this("Tokenise 'U.S. 401(k) contributions'. What happens to '401(k)'? Is that good for search?",
         "Remove stemming (replace _stem with identity) and re-run test_bm25.py: which test breaks?",
         "Find a word where stemming merges two unrelated meanings (hint: 'university'/'universe').")
