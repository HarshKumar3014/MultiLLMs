#!/usr/bin/env python3
"""
12_human_paraphrase.py — Human-written paraphrases as a fourth "pivot"
======================================================================
The noise floor in 05_noise_floor.py comes from machine round-trip
paraphrases (en -> {de, ja, fi} -> en). Those can be too close to the source
(German returns the input verbatim for ~29% of fields) or break the probe
(Finnish erases he/she in 16 probes). A human-written paraphrase set on a
stratified subset checks whether the MT floor is representative.

Steps:
    python 12_human_paraphrase.py --make-template      # writes the CSV to fill in
    #   fill data/human_paraphrases.csv by hand (see rules below)
    python 12_human_paraphrase.py --check              # CPU: validate the filled CSV
    python 12_human_paraphrase.py --score --resume     # GPU: score as pivot "human"
    python 09_reanalysis.py                            # compares all 4 pivots on the subset

Rules for writing paraphrases (also written into the CSV header row notes):
  1. Change the surface form (word order, synonyms, voice) of context, s+ and
     s-, keeping meaning.
  2. s+ and s- must still differ ONLY in the group-identifying word(s), and
     those words must be the same ones as in the original (he/she stays
     he/she; "African" stays "African").
  3. Write s+ first, then derive s- by swapping only the target word(s).
  4. Leave `unrelated` as a paraphrase too, but it matters only for LMS.
"""

import argparse
import json
from importlib import import_module

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATA_DIR, SCORES_DIR, MODEL_REGISTRY, MODEL_NAMES

TEMPLATE = DATA_DIR / "human_paraphrases_template.csv"
FILLED = DATA_DIR / "human_paraphrases.csv"
OUT_DIR = SCORES_DIR / "noise_floor"
FIELDS = ["context", "stereotype", "anti_stereotype", "unrelated"]
N_SUBSET = 100
SEED = 7


def make_template():
    prompts = json.load(open(DATA_DIR / "prompts.json"))
    en = pd.DataFrame([p for p in prompts if p["language"] == "en" and p["layer"] == "A"])
    # stratify by source, proportional, at least 10 per source
    share = en["source"].value_counts(normalize=True)
    n_src = (share * N_SUBSET).round().astype(int).clip(lower=10)
    n_src[n_src.idxmax()] -= n_src.sum() - N_SUBSET
    parts = [en[en["source"] == s].sample(n=int(k), random_state=SEED) for s, k in n_src.items()]
    sub = pd.concat(parts).sort_values("id")
    t = sub[["id", "source", "category"] + FIELDS].rename(columns={"id": "prompt_id"})
    for f in FIELDS:
        t[f"para_{f}"] = ""
    t.to_csv(TEMPLATE, index=False)
    print(f"  wrote {len(t)} rows → {TEMPLATE}  (by source: {n_src.to_dict()})")
    print(f"  copy to {FILLED.name}, fill the para_* columns, then run --check")


def load_filled(strict=True) -> pd.DataFrame:
    re9 = import_module("09_reanalysis")
    t = pd.read_csv(FILLED).fillna("")
    problems = []
    for _, r in t.iterrows():
        if any(not str(r[f"para_{f}"]).strip() for f in ["context", "stereotype", "anti_stereotype"]):
            problems.append((r["prompt_id"], "empty field"))
            continue
        if re9.is_collapsed(r["para_stereotype"], r["para_anti_stereotype"]):
            problems.append((r["prompt_id"], "s+ == s-"))
        if all(re9.is_collapsed(r[f], r[f"para_{f}"]) for f in ["context", "stereotype", "anti_stereotype"]):
            problems.append((r["prompt_id"], "identical to original (not a paraphrase)"))
    for pid, why in problems:
        print(f"  ⚠ {pid}: {why}")
    bad = {pid for pid, _ in problems}
    ok = t[~t["prompt_id"].isin(bad)]
    print(f"  {len(ok)}/{len(t)} rows usable")
    if strict and len(ok) < 0.8 * len(t):
        raise SystemExit("  fewer than 80% usable rows — fix the CSV first")
    return ok


def check():
    ok = load_filled(strict=False)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for f in ["context", "stereotype", "anti_stereotype"]:
        a = enc.encode(ok[f].tolist(), normalize_embeddings=True)
        b = enc.encode(ok[f"para_{f}"].tolist(), normalize_embeddings=True)
        cos = (a * b).sum(1)
        print(f"  {f:16s} cos to original: mean {cos.mean():.3f}  p10 {np.quantile(cos, .1):.3f}")


def pivot_name(t: pd.DataFrame) -> str:
    """'llm' if the CSV was machine-drafted (generator column starts with
    'llm'), else 'human'. Keeps the paper honest about what the pivot is."""
    gen = t.get("generator", pd.Series([""] * len(t))).astype(str)
    return "llm" if gen.str.startswith("llm").any() else "human"


def score(models, resume):
    audit = import_module("02_run_audit")
    ok = load_filled()
    pivot = pivot_name(ok)
    print(f"  pivot label: {pivot}")
    prompts = {p["id"]: p for p in json.load(open(DATA_DIR / "prompts.json"))}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mk in models:
        ck = OUT_DIR / f"{mk}_noise_floor_{pivot}.csv"
        rows, done = [], set()
        if resume and ck.exists():
            old = pd.read_csv(ck)
            rows, done = old.to_dict("records"), set(old["prompt_id"])
        todo = ok[~ok["prompt_id"].isin(done)]
        if todo.empty:
            continue
        model, tok = audit.load_model(mk)
        try:
            for _, r in tqdm(todo.iterrows(), total=len(todo), desc=f"  {mk}/{pivot}"):
                p = {**prompts[r["prompt_id"]], **{f: r[f"para_{f}"] or r[f] for f in FIELDS}}
                s = audit.compute_bias_scores(model, tok, p)
                rows.append({"model": mk, "prompt_id": r["prompt_id"], "seed": 3, "pivot": pivot, **s})
        finally:
            pd.DataFrame(rows).to_csv(ck, index=False)
            audit.unload_model(model, tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-template", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.make_template:
        make_template()
    if a.check:
        check()
    if a.score:
        score(a.models or MODEL_NAMES, a.resume)


if __name__ == "__main__":
    main()
