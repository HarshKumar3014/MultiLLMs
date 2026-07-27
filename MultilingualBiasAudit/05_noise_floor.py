#!/usr/bin/env python3
"""
05_noise_floor.py — Within-Language Perturbation Noise Floor (GPU required)
=============================================================================
Addresses review item #1 ("kill shot"): CLFI range is 0.973–0.992, DFG range
is 0.004–0.013, and the paper never establishes what DFG value is
distinguishable from noise. This script computes that zero point.

Method (as specified in the review):
  1. Take the English probe set (Layer A, handcrafted + benchmark templates).
  2. Build 3 independent English *paraphrase* variants via round-trip
     pivot-translation (en -> pivot_lang -> en), using 3 different pivot
     languages as 3 "seeds" — this is the cheapest reproducible way to get
     3 independent realizations of "same meaning, different surface form"
     without a second stochastic model call.
  3. Score both the original English prompts and each paraphrase variant
     with the SAME log-prob scoring function used in 02_run_audit.py
     (imported directly — do not reimplement, must be identical methodology).
  4. DFG_noise(model, seed) = |SS(model, en_paraphrase_seed) - SS(model, en_original)|
     This is the within-language, zero-translation, zero-culture noise floor.
     Any cross-lingual DFG in results/all_results.csv that is not
     distinguishable from this floor is not evidence of cross-lingual bias.

Requires: GPU (loads all 10 models sequentially, same as 02_run_audit.py).
Not runnable in a CPU-only / no-GPU environment.

Usage:
    python 05_noise_floor.py                          # full run, all models, 3 seeds
    python 05_noise_floor.py --models llama3.1-8b      # single model (fast check)
    python 05_noise_floor.py --resume                  # resume from checkpoint
    python 05_noise_floor.py --n-seeds 3                # override seed count
    python 05_noise_floor.py --build-paraphrases-only   # just build+cache paraphrases, no GPU
"""
from __future__ import annotations  # postpone type-hint evaluation so `pd.DataFrame` hints don't force pandas import

import argparse
import gc
import json
from pathlib import Path

from tqdm import tqdm

from config import DATA_DIR, RESULTS_DIR, MODEL_REGISTRY, MODEL_NAMES, ANALYSIS


def _get_audit_functions():
    """
    Lazily import 02_run_audit (pulls in torch/transformers). Deferred so
    --build-paraphrases-only works on any machine without ML deps installed —
    only the scoring path (GPU-bound) needs this module.
    Reuse the EXACT scoring methodology from the main audit — do not reimplement.
    """
    from importlib import import_module
    audit = import_module("02_run_audit")
    return audit.load_model, audit.unload_model, audit.compute_bias_scores

# 3 pivot languages for 3 independent paraphrase "seeds" — chosen for
# typological distance from English so round-trip translation actually
# perturbs surface form rather than round-tripping to a near-identical string.
PIVOT_LANGS = ["de", "ja", "fi"]

PARAPHRASE_CACHE = DATA_DIR / "noise_floor_paraphrases.json"
CHECKPOINT_DIR = RESULTS_DIR / "noise_floor"


# ──────────────────────────────────────────────────────────────────────────────
# Paraphrase construction (CPU-only, no GPU needed — can run standalone)
# ──────────────────────────────────────────────────────────────────────────────

def round_trip_paraphrase(text: str, pivot: str) -> str:
    """en -> pivot -> en round-trip translation as a cheap paraphrase generator."""
    from deep_translator import GoogleTranslator
    try:
        pivoted = GoogleTranslator(source="en", target=pivot).translate(text)
        back = GoogleTranslator(source=pivot, target="en").translate(pivoted)
        return back if back else text
    except Exception as e:
        print(f"  ⚠ Paraphrase failed ({pivot}) for '{text[:40]}...': {e}")
        return text


def build_paraphrase_sets(prompts: list[dict], n_seeds: int = 3) -> dict:
    """
    Build n_seeds independent English paraphrase variants of every English
    Layer-A prompt. Cached to disk since translation API calls are slow.
    Returns {seed_idx: {prompt_id: {context, stereotype, anti_stereotype, unrelated}}}.
    """
    en_prompts = [p for p in prompts if p["language"] == "en" and p["layer"] == "A"]
    print(f"\n▸ Building {n_seeds} paraphrase seed-sets for {len(en_prompts)} English prompts...")

    cache = {}
    if PARAPHRASE_CACHE.exists():
        with open(PARAPHRASE_CACHE) as f:
            cache = json.load(f)

    fields = ["context", "stereotype", "anti_stereotype", "unrelated"]

    for seed_idx in range(n_seeds):
        pivot = PIVOT_LANGS[seed_idx % len(PIVOT_LANGS)]
        seed_key = str(seed_idx)
        cache.setdefault(seed_key, {})
        print(f"\n  Seed {seed_idx} (pivot={pivot}): {len(en_prompts)} prompts")

        for p in tqdm(en_prompts, desc=f"  seed {seed_idx}"):
            if p["id"] in cache[seed_key]:
                continue
            variant = {"pivot": pivot}
            for field in fields:
                variant[field] = round_trip_paraphrase(p[field], pivot)
            cache[seed_key][p["id"]] = variant

        with open(PARAPHRASE_CACHE, "w") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ Paraphrase cache → {PARAPHRASE_CACHE}")
    return cache


# ──────────────────────────────────────────────────────────────────────────────
# Scoring (GPU required)
# ──────────────────────────────────────────────────────────────────────────────

def run_noise_floor_for_model(model_key: str, en_prompts: list[dict], paraphrase_cache: dict,
                               n_seeds: int, resume: bool = False) -> pd.DataFrame:
    import pandas as pd
    load_model, unload_model, compute_bias_scores = _get_audit_functions()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{model_key}_noise_floor.csv"

    completed_keys = set()
    results = []
    if resume and checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        completed_keys = set(zip(existing["prompt_id"], existing["seed"]))
        results = existing.to_dict("records")
        print(f"  ↻ Resuming: {len(completed_keys)} (prompt, seed) pairs already done")

    model, tokenizer = load_model(model_key)
    info = MODEL_REGISTRY[model_key]

    try:
        for p in tqdm(en_prompts, desc=f"  {model_key} original"):
            key = (p["id"], -1)  # seed -1 = original English
            if key in completed_keys:
                continue
            scores = compute_bias_scores(model, tokenizer, p)
            results.append({"model": model_key, "prompt_id": p["id"], "seed": -1,
                             "pivot": "none", **scores})

        for seed_idx in range(n_seeds):
            seed_key = str(seed_idx)
            for p in tqdm(en_prompts, desc=f"  {model_key} seed {seed_idx}"):
                key = (p["id"], seed_idx)
                if key in completed_keys:
                    continue
                variant = paraphrase_cache[seed_key].get(p["id"])
                if variant is None:
                    continue
                para_prompt = {**p, "context": variant["context"], "stereotype": variant["stereotype"],
                                "anti_stereotype": variant["anti_stereotype"], "unrelated": variant["unrelated"]}
                scores = compute_bias_scores(model, tokenizer, para_prompt)
                results.append({"model": model_key, "prompt_id": p["id"], "seed": seed_idx,
                                 "pivot": variant["pivot"], **scores})

            if len(results) % 100 < n_seeds + 1:
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    finally:
        df = pd.DataFrame(results)
        df.to_csv(checkpoint_path, index=False)
        print(f"  ✓ Checkpoint saved → {checkpoint_path}")
        unload_model(model, tokenizer)

    return df


def compute_dfg_noise(all_noise_df: pd.DataFrame) -> pd.DataFrame:
    """DFG_noise(model, seed) = |SS(paraphrase) - SS(original)|, per prompt then averaged."""
    orig = all_noise_df[all_noise_df["seed"] == -1][["model", "prompt_id", "stereotype_score"]] \
        .rename(columns={"stereotype_score": "ss_original"})
    para = all_noise_df[all_noise_df["seed"] != -1]

    merged = para.merge(orig, on=["model", "prompt_id"])
    merged["dfg_noise"] = (merged["stereotype_score"] - merged["ss_original"]).abs()

    summary = (
        merged.groupby(["model", "seed"])["dfg_noise"]
        .mean()
        .reset_index()
    )
    overall = (
        merged.groupby("model")["dfg_noise"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_dfg_noise", "std": "std_dfg_noise"})
        .reset_index()
    )
    return summary, overall


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute within-language noise floor for DFG")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--build-paraphrases-only", action="store_true",
                         help="Only build the paraphrase cache (no GPU needed) — run this first "
                              "on any machine, then copy data/noise_floor_paraphrases.json to the GPU box")
    args = parser.parse_args()

    prompts_path = DATA_DIR / "prompts.json"
    assert prompts_path.exists(), f"Prompts not found at {prompts_path}. Run 01_build_prompts.py first."
    with open(prompts_path) as f:
        prompts = json.load(f)

    en_prompts = [p for p in prompts if p["language"] == "en" and p["layer"] == "A"]

    paraphrase_cache = build_paraphrase_sets(prompts, n_seeds=args.n_seeds)

    if args.build_paraphrases_only:
        print("\n  Paraphrase cache built. Copy data/noise_floor_paraphrases.json to the GPU "
              "machine, then run: python 05_noise_floor.py --resume")
        return

    import pandas as pd  # deferred: only needed past this point (GPU scoring path)

    model_keys = args.models if args.models else MODEL_NAMES
    invalid = [m for m in model_keys if m not in MODEL_REGISTRY]
    if invalid:
        print(f"  ✗ Unknown models: {invalid}")
        return

    print("=" * 70)
    print("  NOISE FLOOR EXPERIMENT — within-language perturbation baseline")
    print("=" * 70)
    print(f"  English prompts: {len(en_prompts)} | Seeds (pivot langs): {args.n_seeds} {PIVOT_LANGS[:args.n_seeds]}")
    print(f"  Models: {model_keys}")

    all_dfs = []
    for i, model_key in enumerate(model_keys):
        print(f"\n{'━'*60}\n  MODEL {i+1}/{len(model_keys)}: {model_key}\n{'━'*60}")
        df = run_noise_floor_for_model(model_key, en_prompts, paraphrase_cache,
                                        n_seeds=args.n_seeds, resume=args.resume)
        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)
    merged_path = RESULTS_DIR / "noise_floor_all_results.csv"
    merged.to_csv(merged_path, index=False)
    print(f"\n  ✓ Merged noise-floor results → {merged_path}")

    per_seed, overall = compute_dfg_noise(merged)
    per_seed.to_csv(RESULTS_DIR / "noise_floor_by_seed.csv", index=False)
    overall.to_csv(RESULTS_DIR / "noise_floor_summary.csv", index=False)

    print(f"\n{'═'*70}")
    print("  NOISE FLOOR SUMMARY (DFG_noise = |SS(paraphrase) - SS(original English)|)")
    print(f"{'═'*70}")
    print(overall.to_string(index=False))

    grand_mean = float(overall["mean_dfg_noise"].mean())
    print(f"\n  Grand mean noise floor across all models: {grand_mean:.4f}")
    print(f"  Compare against Table 2's DFG range in the paper.")
    print(f"  Any model×language DFG below this is NOT distinguishable from within-language noise.")

    with open(RESULTS_DIR / "noise_floor_verdict.json", "w") as f:
        json.dump({
            "grand_mean_noise_floor": round(grand_mean, 6),
            "per_model": overall.set_index("model")["mean_dfg_noise"].round(6).to_dict(),
            "interpretation": (
                "Any reported cross-lingual DFG value at or below the noise floor for that "
                "model is not distinguishable from within-language paraphrase perturbation "
                "and cannot be attributed to cross-lingual bias."
            ),
        }, f, indent=2)
    print(f"\n  → {RESULTS_DIR / 'noise_floor_verdict.json'}")


if __name__ == "__main__":
    main()
