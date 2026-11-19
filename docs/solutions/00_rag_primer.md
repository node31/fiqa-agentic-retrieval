# Solutions: rag primer

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** Three reasons a correct passage scores as a miss: (a) qrels are sparse, so a good answer nobody judged counts as non-relevant; (b) duplicates/near-duplicates of the gold passage under another id; (c) the query is ambiguous and the gold covers one reading while we found the other.

**2.** BM25. '1099-B' and '1e' are rare exact tokens with high IDF; dense models tokenize them into sub-word pieces with little learned meaning, so semantic neighbours ('tax form', 'brokerage statement') dominate.

**3.** Run `python -m src.beir_data`: p90 = 270 words, so exactly ~10% of passages are longer than 270 words (that's the definition of p90). That same top-10% bucket is used for the stratified metric.

**4.** Typically BM25 returns passages that repeat 'short'/'sell'; dense returns paraphrases ('betting against a stock'); hybrid mixes both. Look for docs that only one strategy finds.
