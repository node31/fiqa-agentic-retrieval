"""Toy 03: the inverted index as a sparse matrix, plus live add/delete with tombstones.

Concept: postings lists, CSC columns, why deletes are "mark dead" instead of "erase".
Real code: src/retrieval/bm25.py::BM25Index (build/add/delete)   Doc: docs/topics/01_lexical_retrieval_bm25.md
"""
from _common import TOY_DOCS, banner, note, step, try_this

from src.retrieval.bm25 import BM25Index, tokenize

banner("Toy 03: inverted index = sparse (docs x terms) matrix")
ix = BM25Index().build(list(TOY_DOCS), list(TOY_DOCS.values()))
csc = ix._base_csc
step(1, f"Matrix shape {csc.shape} (docs x vocabulary), only {csc.nnz} non-zeros "
        f"({100 * csc.nnz / (csc.shape[0] * csc.shape[1]):.1f}% dense)")
note("Storing it dense would waste memory: FiQA is 57K x ~100K = 5.7 billion cells, but only ~4M are non-zero.")
t = "short"
j = ix.vocab[t]
lo, hi = csc.indptr[j], csc.indptr[j + 1]
rows, tfs = csc.indices[lo:hi].tolist(), csc.data[lo:hi].tolist()
step(2, f"Postings list for {t!r} = column {j}: rows {rows} with tf {tfs}")
print("     -> docs:", [ix.doc_ids[r] for r in rows])
note("Scoring a query reads only the columns of its terms: cost ~ sum of df, not N.")
step(3, "Live add: new docs go to a small 'delta' segment (a Python dict) instead of rebuilding the matrix")
ix.add("d11", "Short squeezes force short sellers to buy back shares")
print(f"     delta postings for 'short': {ix._delta['short']}  (row, tf)")
print(f"     search 'short squeez' -> {[(d, round(s, 3)) for d, s in ix.search(tokenize('short squeeze'), 3)]}")
step(4, "Delete: flip an 'alive' bit (a tombstone). Postings stay but are masked at query time")
ix.delete("d1")
print(f"     alive[d1] = {ix.alive[0]}, df('short') went from 4 to {ix.df['short']}")
print(f"     search 'short' -> {[d for d, _ in ix.search(['short'], 10)]}  (d1 gone)")
note("Erasing from a CSC matrix means shifting arrays: O(nnz). Tombstones are O(1). Real engines (Lucene) "
     "do the same, then merge segments in the background to reclaim space ('compaction').")
try_this("Implement compact(): rebuild the base matrix from live rows + delta.",
         "How many deletes before tombstones waste 50% of the postings? What would trigger compaction?",
         "Upsert d2 with new text. Check that BOTH the old row is dead and df is updated.")
