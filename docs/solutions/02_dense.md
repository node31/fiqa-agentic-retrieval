# Solutions: dense

Try the exercise before reading. Several answers point to *your own* run output, because the numbers are hardware-dependent.

**1.** The bi-encoder embeds passages offline once; a query costs one forward pass + a matrix-vector product. A cross-encoder must run a forward pass for every (query, passage) pair at query time; nothing can be precomputed.

**2.** Poorly in general: form codes are rare tokens split into sub-words; the model maps the query near generic 'tax form' passages. BM25 (or hybrid) is better for exact identifiers. This is a core argument for hybrid.

**3.** Typical shape: 1 thread is slowest; 4 threads ~2–3× faster; beyond ~8 threads no gain or worse (a 1-query batch has too little work to split, and thread sync overhead dominates). That's why we pin 4 threads per process.

**4.** Build time drops roughly ~2× (attention cost grows with sequence length), and nDCG drops slightly: answers whose key sentence is beyond token 128 become invisible. Measure the delta and add it as an ablation row.
