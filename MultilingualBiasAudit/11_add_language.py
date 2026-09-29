#!/usr/bin/env python3
"""
11_add_language.py — Add a language to the audit after the fact
================================================================
Translates the 387 English Layer-A probes into a new language (default:
Amharic, a second low-resource language), validates them more strictly than
01_build_prompts.py did, and scores them with all ten models using the exact
scoring code of 02_run_audit.py.

Validation differs from 01 in two ways, both prompted by the collapsed-pair
audit in 09_reanalysis.py:
  * every field (context, s+, s-) is back-translated and embedded, not only
    the context;
  * a probe whose translated s+ and s- are identical (the minimal-pair
    contrast was erased) is flagged `collapsed` and excluded downstream.

Steps (each resumable):
    python 11_add_language.py --lang am --build        # CPU: translate + validate
    python 11_add_language.py --lang am --score --resume   # GPU: score 10 models
    python 09_reanalysis.py                            # picks up results/extra_lang/

Outputs:
    data/prompts_<lang>.json
    data/validation_<lang>.json
    results/extra_lang/<model>_<lang>.csv   (same columns as all_results.csv)
"""

import argparse
import json
from importlib import import_module

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATA_DIR, SCORES_DIR, MODEL_REGISTRY, MODEL_NAMES, EXTRA_LANGUAGES, ANALYSIS

OUT_DIR = SCORES_DIR / "extra_lang"
FIELDS = ["context", "stereotype", "anti_stereotype", "unrelated"]


def translate(text: str, target: str, source: str = "en", retries: int = 6) -> str:
    """Google Translate with throttling and exponential backoff. Unlike
    01_build_prompts.translate_text, never falls back to the input text:
    a silent English fallback would look like a valid (uncollapsed) probe."""
    import time
    from deep_translator import GoogleTranslator
    for k in range(retries):
        try:
            time.sleep(0.25)
            out = GoogleTranslator(source=source, target=target).translate(text)
            if out and out.strip():
                return out
        except Exception as e:
            wait = 2 ** k * 5
            print(f"  ⚠ translate retry {k + 1}/{retries} in {wait}s: {type(e).__name__}")
            time.sleep(wait)
    raise RuntimeError(f"translation failed after {retries} tries: {text[:60]!r} → {target}")


def build(lang: str):
    re9 = import_module("09_reanalysis")
    prompts = json.load(open(DATA_DIR / "prompts.json"))
    en = [p for p in prompts if p["language"] == "en" and p["layer"] == "A"]
    path = DATA_DIR / f"prompts_{lang}.json"
    done = {p["base_id"]: p for p in json.load(open(path))} if path.exists() else {}

    out = []
    for p in tqdm(en, desc=f"  translate → {lang}"):
        if p["base_id"] in done:
            out.append(done[p["base_id"]])
            continue
        t = {k: v for k, v in p.items() if not k.endswith("_en")}
        t.update({"id": f"{p['base_id']}_{lang}", "base_id": p["base_id"], "language": lang, "layer": "A"})
        for f in FIELDS:
            t[f"{f}_en"] = p[f]
            t[f] = translate(p[f], lang)
        out.append(t)
        if len(out) % 25 == 0:
            json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)

    # ── validation: back-translate context, s+, s- ──
    vpath = DATA_DIR / f"validation_{lang}.json"
    val = json.load(open(vpath)) if vpath.exists() else {}
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for t in tqdm(out, desc="  back-translate"):
        if t["id"] in val:
            continue
        rec = {"collapsed": re9.is_collapsed(t["stereotype"], t["anti_stereotype"]),
               "untranslated_fields": [f for f in FIELDS if t[f].strip() == t[f"{f}_en"].strip()]}
        for f in ["context", "stereotype", "anti_stereotype"]:
            bt = translate(t[f], "en", lang)
            e = enc.encode([t[f"{f}_en"], bt], normalize_embeddings=True)
            rec[f"{f}_bt"] = bt
            rec[f"{f}_sim"] = round(float(e[0] @ e[1]), 4)
        # the contrast survives only if back-translated s+ and s- still differ
        rec["contrast_lost_in_bt"] = re9.is_collapsed(rec["stereotype_bt"], rec["anti_stereotype_bt"])
        rec["passed"] = (not rec["collapsed"]) and all(
            rec[f"{f}_sim"] >= ANALYSIS["backtranslation_threshold"] for f in ["context", "stereotype", "anti_stereotype"])
        val[t["id"]] = rec
        if len(val) % 25 == 0:
            json.dump(val, open(vpath, "w"), ensure_ascii=False, indent=2)
    json.dump(val, open(vpath, "w"), ensure_ascii=False, indent=2)

    v = pd.DataFrame.from_dict(val, orient="index")
    print(f"\n  {lang}: {len(v)} probes | collapsed {v['collapsed'].sum()} | "
          f"contrast lost in back-translation {v['contrast_lost_in_bt'].sum()} | "
          f"passed all-field check {v['passed'].mean():.1%}")
    print(v[[c for c in v.columns if c.endswith('_sim')]].describe().round(3).to_string())


def score(lang: str, models, resume: bool):
    audit = import_module("02_run_audit")
    info_l = EXTRA_LANGUAGES[lang]
    probes = json.load(open(DATA_DIR / f"prompts_{lang}.json"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mk in models:
        ck = OUT_DIR / f"{mk}_{lang}.csv"
        rows, done = [], set()
        if resume and ck.exists():
            old = pd.read_csv(ck)
            rows, done = old.to_dict("records"), set(old["prompt_id"])
        todo = [p for p in probes if p["id"] not in done]
        if not todo:
            continue
        model, tok = audit.load_model(mk)
        info = MODEL_REGISTRY[mk]
        try:
            for i, p in enumerate(tqdm(todo, desc=f"  {mk}/{lang}")):
                s = audit.compute_bias_scores(model, tok, p)
                rows.append({"model": mk, "model_group": info["group"], "model_group_label": info["group_label"],
                             "model_org": info["org"], "prompt_id": p["id"], "base_prompt_id": p["base_id"],
                             "language": lang, "language_name": info_l["name"],
                             "resource_level": info_l["resource"], "category": p["category"], "layer": "A",
                             "context": p["context"], **s})
                if i % 50 == 0:
                    pd.DataFrame(rows).to_csv(ck, index=False)
        finally:
            pd.DataFrame(rows).to_csv(ck, index=False)
            audit.unload_model(model, tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="am", choices=list(EXTRA_LANGUAGES))
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.build:
        build(args.lang)
    if args.score:
        score(args.lang, args.models or MODEL_NAMES, args.resume)
    if not (args.build or args.score):
        ap.print_help()


if __name__ == "__main__":
    main()
