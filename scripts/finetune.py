"""Fine-tune all-MiniLM-L6-v2 on the FiQA TRAIN split -> artifacts/models/minilm-ft ("minilm-ft").

Easier to read, with a toy example: docs/topics/17_fine_tuning.md

WHAT (the offline path; the online path is the normal B3 migration: make migrate MODEL=minilm-ft)
    1. split    the 5,500 train questions into 5,000 for training and 500 "train-val" questions used
                only to pick the number of epochs. Dev stays untouched until the go/no-go, test until the end.
    2. mine     for every training question, run today's hybrid search (MiniLM, linear, alpha 0.7) and
                keep non-gold hits at ranks 10-30 as HARD NEGATIVES: passages that look relevant but aren't.
    3. triples  (question, gold passage, hard negative), one per gold passage.
    4. train    a plain PyTorch loop with the in-batch-negatives loss (below), CPU only.
    5. pick     the epoch with the best train-val nDCG@10, save it with a training.json record.
    6. embed    the corpus with the new weights: python -m src.index.build_index --models minilm-ft

    python -m scripts.finetune                       # the real run (~1 h on 32 CPU cores)
    python -m scripts.finetune --epochs 1 --limit 200 --out toy_runs/ft  # a quick plumbing check

THE LOSS (MultipleNegativesRankingLoss, written out by hand)
    A batch has B questions, B positives and B hard negatives. Encode them, then score every question
    against all 2B passages: S = 20 * q @ [pos; neg].T  (a B x 2B matrix of cosines; 20 = 1/temperature).
    Row i's right answer is column i (its own positive). Cross-entropy on each row pushes S[i, i] up and
    every other column down: the other questions' positives are free "in-batch" negatives, the mined
    ones are the hard negatives. With B = 64 every question sees 127 wrong answers per step.

    Toy (B = 2): q0 = "roth ira limits", q1 = "short selling".
        columns:   pos0  pos1  neg0  neg1
        S[0]   = [ 15,    2,   11,    1 ]  -> softmax puts most mass on pos0 (good), some on neg0 (the hard one)
        loss_0 = -log softmax(S[0])[0];  the gradient mostly pushes neg0 away, the easy pos1/neg1 barely matter.

PITFALLS
    * False in-batch negatives: two questions in a batch can share a gold passage (or one question appears
      twice, once per gold passage). Treating the other copy as "wrong" would punish a right answer, so those
      columns are masked out (in_batch_mask below).
    * Sparse labels: FiQA labels ~2.6 passages per question; many unlabelled top hits are real answers. Hard
      negatives therefore come from ranks 10-30, not 1-9.
    * Leakage: a passage that is gold for any dev or test question is never used as a negative, and dev/test
      questions are never trained on.
    * The model must be saved with the same max_seq_length (256) it is served with.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from src.config import DEFAULT_ALPHA, MODELS, ROOT, SEED

BASE = MODELS["minilm"].hf_name
OUT = ROOT / MODELS["minilm-ft"].hf_name


# ----------------------------------------------------------------------------- pure helpers (unit-tested)
def split_train(qids: list[str], n_val: int, seed: int = SEED) -> tuple[list[str], list[str]]:
    """Shuffle once with a fixed seed; the first n_val ids are train-val, the rest train."""
    ids = sorted(qids)
    random.Random(seed).shuffle(ids)
    return ids[n_val:], ids[:n_val]


def pick_negatives(ranked: list[str], gold: set[str], banned: set[str], lo: int = 10, hi: int = 30) -> list[str]:
    """Hard negatives = hits at 1-based ranks lo..hi that are neither this question's gold nor banned.

    >>> pick_negatives(["g", "a", "b", "c"], {"g"}, {"b"}, lo=1, hi=4)
    ['a', 'c']
    """
    return [d for d in ranked[lo - 1:hi] if d not in gold and d not in banned]


def build_triples(questions: dict[str, str], qrels: dict[str, dict[str, int]], negatives: dict[str, list[str]],
                  corpus_ids: list[str], banned: set[str], seed: int = SEED) -> list[tuple[str, str, str]]:
    """One (qid, positive doc id, negative doc id) per gold passage of every training question.

    The negative is a random mined hard negative; if mining found none, a random corpus passage
    that is neither gold nor banned (an "easy" negative is better than none).
    """
    rng = random.Random(seed)
    out = []
    for qid in questions:
        gold = {d for d, g in qrels[qid].items() if g > 0}
        for pos in sorted(gold):
            negs = negatives.get(qid) or []
            if negs:
                neg = rng.choice(negs)
            else:
                while (neg := rng.choice(corpus_ids)) in gold or neg in banned:
                    pass
            out.append((qid, pos, neg))
    rng.shuffle(out)
    return out


def in_batch_mask(batch: list[tuple[str, str, str]], qrels: dict[str, dict[str, int]]) -> np.ndarray:
    """B x 2B boolean matrix: True where column j is a RIGHT answer for row i other than its own positive.

    Columns are [pos_0..pos_{B-1}, neg_0..neg_{B-1}]. Those cells get -inf before the softmax, so a
    shared gold passage is never pushed away.
    """
    docs = [p for _, p, _ in batch] + [n for _, _, n in batch]
    m = np.zeros((len(batch), len(docs)), dtype=bool)
    for i, (qid, _, _) in enumerate(batch):
        gold = qrels[qid]
        for j, d in enumerate(docs):
            if j != i and gold.get(d, 0) > 0:
                m[i, j] = True
    return m


def lr_at(step: int, total: int, peak: float, warmup_frac: float = 0.1) -> float:
    """Linear warmup to `peak` over the first 10% of steps, then linear decay to 0."""
    warm = max(1, int(total * warmup_frac))
    return peak * step / warm if step < warm else peak * max(0.0, (total - step) / max(1, total - warm))


# ----------------------------------------------------------------------------- the run
def mine_negatives(questions: dict[str, str], qrels, banned: set[str], lo: int, hi: int) -> dict[str, list[str]]:
    """Run today's chosen hybrid (MiniLM, linear, alpha 0.7) in-process and keep ranks lo..hi."""
    import asyncio

    from src.service.router import single_node_router

    router = single_node_router(models=("minilm",), active_model="minilm")
    loop = asyncio.new_event_loop()
    out, t0 = {}, time.time()
    for n, (qid, q) in enumerate(questions.items(), 1):
        res = loop.run_until_complete(router.search(q, k=hi, strategy="hybrid", fusion="linear", alpha=DEFAULT_ALPHA))
        gold = {d for d, g in qrels[qid].items() if g > 0}
        out[qid] = pick_negatives([r["doc_id"] for r in res["results"]], gold, banned, lo, hi)
        if n % 1000 == 0:
            print(f"[finetune] mined {n}/{len(questions)} ({time.time() - t0:.0f}s)")
    return out


def val_ndcg(model, val_q: dict[str, str], qrels, corpus: dict[str, str], sub_ids: list[str]) -> float:
    """Dense nDCG@10 of `model` for the train-val questions over a fixed sub-corpus (cheap epoch picking)."""
    from src.metrics import evaluate_run

    docs = model.encode([corpus[d] for d in sub_ids], batch_size=128, normalize_embeddings=True,
                        convert_to_numpy=True)
    qv = model.encode(list(val_q.values()), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
    top = np.argsort(-(qv @ docs.T), axis=1)[:, :10]
    run = {qid: [sub_ids[j] for j in row] for qid, row in zip(val_q, top)}
    return round(evaluate_run(run, {q: qrels[q] for q in val_q})["ndcg@10"], 4)


def train(args) -> dict:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    from src.beir_data import load_corpus, load_qrels, load_split

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    corpus = load_corpus()
    corpus_ids = sorted(corpus)
    train_q_all, qrels = load_split("train")
    # Never use a passage that answers a dev or test question as a negative (no leakage, no false negatives).
    banned = {d for s in ("dev", "test") for rel in load_qrels(s).values() for d, g in rel.items() if g > 0}

    tr_ids, val_ids = split_train(list(train_q_all), args.val, args.seed)
    if args.limit:
        tr_ids, val_ids = tr_ids[:args.limit], val_ids[:max(20, args.limit // 10)]
    tr_q = {q: train_q_all[q] for q in tr_ids}
    val_q = {q: train_q_all[q] for q in val_ids}
    print(f"[finetune] {len(tr_q)} training questions, {len(val_q)} train-val questions, "
          f"{len(banned)} dev/test gold passages banned as negatives")

    t0 = time.time()
    negatives = mine_negatives(tr_q, qrels, banned, args.neg_lo, args.neg_hi)
    mine_s = time.time() - t0
    triples = build_triples(tr_q, qrels, negatives, corpus_ids, banned, args.seed)
    n_hard = sum(1 for q in tr_q if negatives.get(q))
    print(f"[finetune] {len(triples)} triples; {n_hard}/{len(tr_q)} questions have mined negatives ({mine_s:.0f}s)")

    # Fixed train-val sub-corpus: every train-val gold passage + 20,000 random passages (same for every epoch).
    rng = np.random.default_rng(args.seed)
    val_gold = {d for q in val_q for d, g in qrels[q].items() if g > 0}
    rest = [d for d in corpus_ids if d not in val_gold]
    sub_ids = sorted(val_gold) + list(rng.choice(rest, size=min(args.val_corpus, len(rest)), replace=False))

    model = SentenceTransformer(BASE, device="cpu")
    model.max_seq_length = MODELS["minilm-ft"].max_seq_length
    history = [{"epoch": 0, "val_ndcg@10": val_ndcg(model, val_q, qrels, corpus, sub_ids)}]
    print(f"[finetune] epoch 0 (base model): train-val nDCG@10 {history[0]['val_ndcg@10']}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(triples) / args.batch)
    total, step = steps_per_epoch * args.epochs, 0
    best, best_dir = history[0]["val_ndcg@10"], None
    out = Path(args.out)

    def embed(texts: list[str]):
        # preprocess() (tokenize() in older sentence-transformers) pads and truncates to max_seq_length.
        prep = getattr(model, "preprocess", None) or model.tokenize
        return F.normalize(model(prep(texts))["sentence_embedding"], dim=-1)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(triples)
        te, losses = time.time(), []
        for b in range(0, len(triples), args.batch):
            batch = triples[b:b + args.batch]
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, args.lr)
            q = embed([tr_q[qid] for qid, _, _ in batch])
            p = embed([corpus[d] for _, d, _ in batch] + [corpus[d] for _, _, d in batch])
            scores = args.scale * q @ p.T                                   # B x 2B cosines, scaled
            mask = torch.from_numpy(in_batch_mask(batch, qrels))
            scores = scores.masked_fill(mask, float("-inf"))                # don't punish other right answers
            loss = F.cross_entropy(scores, torch.arange(len(batch)))        # row i's answer is column i
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
            done = b // args.batch + 1
            if done % 20 == 0:
                print(f"[finetune] epoch {epoch} step {done}/{steps_per_epoch} "
                      f"loss {np.mean(losses[-20:]):.3f} ({time.time() - te:.0f}s)", flush=True)
        model.eval()
        score = val_ndcg(model, val_q, qrels, corpus, sub_ids)
        history.append({"epoch": epoch, "mean_loss": round(float(np.mean(losses)), 4),
                        "val_ndcg@10": score, "seconds": round(time.time() - te, 1)})
        print(f"[finetune] epoch {epoch}: loss {np.mean(losses):.3f}, train-val nDCG@10 {score} "
              f"({time.time() - te:.0f}s)")
        if score > best:
            best, best_dir = score, epoch
            model.save(str(out))

    if best_dir is None:
        print("[finetune] no epoch beat the base model on train-val; saving the last epoch anyway for inspection")
        model.save(str(out))
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
    except OSError:
        sha = ""
    record = {
        "base_model": BASE, "out": out.resolve().relative_to(ROOT).as_posix() if out.resolve().is_relative_to(ROOT) else str(out),
        "chosen_epoch": best_dir, "history": history,
        "data": {"train_questions": len(tr_q), "train_val_questions": len(val_q), "triples": len(triples),
                 "questions_with_mined_negatives": n_hard, "banned_dev_test_gold": len(banned),
                 "val_subcorpus": len(sub_ids)},
        "hyperparameters": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "warmup_frac": 0.1,
                            "scale": args.scale, "weight_decay": 0.01, "max_seq_length": model.max_seq_length,
                            "negative_ranks": [args.neg_lo, args.neg_hi], "mining": f"hybrid linear alpha {DEFAULT_ALPHA}, minilm",
                            "seed": args.seed, "threads": args.threads},
        "seconds": {"mining": round(mine_s, 1), "total": round(time.time() - t0, 1)},
        "git": sha,
    }
    (out / "training.json").write_text(json.dumps(record, indent=1))
    print(f"[finetune] saved epoch {best_dir} -> {out} (train-val nDCG@10 {best}); record in training.json")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--scale", type=float, default=20.0, help="1/temperature on the cosines (sentence-transformers default)")
    ap.add_argument("--val", type=int, default=500, help="train questions held out to pick the epoch")
    ap.add_argument("--val-corpus", type=int, default=20000, help="random passages in the train-val sub-corpus")
    ap.add_argument("--neg-lo", type=int, default=10)
    ap.add_argument("--neg-hi", type=int, default=30)
    ap.add_argument("--limit", type=int, help="train on the first N questions only (plumbing check)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT))
    train(ap.parse_args())


if __name__ == "__main__":
    main()
