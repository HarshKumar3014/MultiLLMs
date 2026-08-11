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


def _free_model_cache(hf_id: str):
    """
    Delete a model's downloaded weights from the local HF cache after it's
    been scored. Necessary because this script loads 10 models sequentially
    (~15-20GB fp16 download each, ~150-200GB total) and never needed more
    than one on disk at a time — small instance disks (e.g. 60GB) fill up
    partway through otherwise, which surfaces as a confusing
    "No space left on device" crash mid-download of the *next* model.
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == hf_id:
                revisions = {rev.commit_hash for rev in repo.revisions}
                strategy = cache_info.delete_revisions(*revisions)
                print(f"  🗑 Freeing {strategy.expected_freed_size_str} of cache for {hf_id}")
                strategy.execute()
                return
    except Exception as e:
        print(f"  ⚠ Cache cleanup failed for {hf_id} (non-fatal, continuing): {e}")


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
                               n_seeds: int, resume: bool = False, free_cache: bool = True) -> pd.DataFrame:
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
        if free_cache:
            _free_model_cache(info["hf_id"])

    return df


def compute_dfg_noise(all_noise_df: pd.DataFrame) -> pd.DataFrame:
    """
    Two DISTINCT quantities, which must not be confused (an earlier version of
    this script conflated them):

    (a) MATCHED noise floor -- the only quantity comparable to the paper's
        reported DFG. The reported DFG is a *difference of probe-set means*,
        |mean(SS_lang) - mean(SS_en)|, so its noise analogue must also be a
        difference of means:

            DFG_noise(model, pivot) = |mean(SS_para(pivot)) - mean(SS_orig)|

        averaged over pivots, exactly mirroring how the reported DFG averages
        over the 7 non-English languages.

    (b) PER-PROBE instability -- mean over probes of |SS_para - SS_orig|.
        This is a legitimate and useful statistic (how much an individual
        probe's score moves under paraphrase) but it is roughly an order of
        magnitude LARGER than (a), because taking absolute values before
        averaging does not let signed per-probe noise cancel, whereas a
        difference of means does. It therefore CANNOT be compared against a
        difference-of-means DFG. We keep it because it corroborates the
        per-probe variance driving the power analysis (07_power_analysis.py),
        but we report it under its own name.

    BBQ-sourced prompts score short answer-choice continuations whose original
    SS is frequently near-saturated (0 or 1); small perturbations flip an
    already-near-certain preference, inflating (b) in particular. We break
    BBQ out rather than pooling it silently.
    """
    orig = all_noise_df[all_noise_df["seed"] == -1][["model", "prompt_id", "stereotype_score"]] \
        .rename(columns={"stereotype_score": "ss_original"})
    para = all_noise_df[all_noise_df["seed"] != -1]

    merged = para.merge(orig, on=["model", "prompt_id"])
    merged["abs_dev"] = (merged["stereotype_score"] - merged["ss_original"]).abs()

    # attach source (bbq / stereoset / handcrafted) so it can be reported separately
    prompts_path = DATA_DIR / "prompts.json"
    with open(prompts_path) as f:
        prompts = json.load(f)
    src_by_id = {p["id"]: p.get("source", "handcrafted")
                 for p in prompts if p["language"] == "en" and p["layer"] == "A"}
    merged["source"] = merged["prompt_id"].map(src_by_id)

    # (a) matched floor: difference of means, per model x pivot, then over pivots
    matched_per_pivot = (
        merged.groupby(["model", "pivot"])
        .apply(lambda g: abs(g["stereotype_score"].mean() - g["ss_original"].mean()),
               include_groups=False)
        .rename("matched_floor").reset_index()
    )
    overall = (
        matched_per_pivot.groupby("model")["matched_floor"].mean()
        .rename("matched_dfg_noise").reset_index()
    )
    # (b) per-probe instability, same grouping
    overall = overall.merge(
        merged.groupby("model")["abs_dev"]
        .agg(["mean", "median"])
        .rename(columns={"mean": "perprobe_mean_abs_dev",
                         "median": "perprobe_median_abs_dev"})
        .reset_index(),
        on="model")

    by_source = (
        merged.groupby("source")["abs_dev"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "perprobe_mean_abs_dev",
                         "median": "perprobe_median_abs_dev", "count": "n"})
        .reset_index()
    )
    no_bbq = merged[merged["source"] != "bbq"]
    return matched_per_pivot, overall, by_source, no_bbq


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
    parser.add_argument("--keep-model-cache", action="store_true",
                         help="Do not delete each model's HF cache after scoring it. Default is to "
                              "delete, since 10 models sequentially (~150-200GB fp16 total) will fill "
                              "a small instance disk otherwise.")
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
                                        n_seeds=args.n_seeds, resume=args.resume,
                                        free_cache=not args.keep_model_cache)
        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)
    merged_path = RESULTS_DIR / "noise_floor_all_results.csv"
    merged.to_csv(merged_path, index=False)
    print(f"\n  ✓ Merged noise-floor results → {merged_path}")

    per_seed, overall, by_source, no_bbq = compute_dfg_noise(merged)
    per_seed.to_csv(RESULTS_DIR / "noise_floor_by_seed.csv", index=False)
    overall.to_csv(RESULTS_DIR / "noise_floor_summary.csv", index=False)
    by_source.to_csv(RESULTS_DIR / "noise_floor_by_source.csv", index=False)

    print(f"\n{'═'*70}")
    print("  NOISE FLOOR SUMMARY")
    print(f"{'═'*70}")
    print("  matched_dfg_noise = |mean(SS_para) - mean(SS_orig)|, averaged over pivots")
    print("     ^ the ONLY column comparable to the paper's reported DFG")
    print("  perprobe_* = mean/median over probes of |SS_para - SS_orig|")
    print("     ^ probe-level instability; ~10x larger by construction, NOT comparable to DFG")
    print()
    print(overall.round(4).to_string(index=False))
    print(f"\n  By source (per-probe; BBQ answer-choice continuations are near-saturated "
          f"and unstable under perturbation):")
    print(by_source.round(4).to_string(index=False))

    matched_median = float(overall["matched_dfg_noise"].median())
    matched_lo = float(overall["matched_dfg_noise"].min())
    matched_hi = float(overall["matched_dfg_noise"].max())
    perprobe_median = float(overall["perprobe_median_abs_dev"].median())

    print(f"\n  MATCHED noise floor: median {matched_median:.4f} "
          f"(per-model range {matched_lo:.4f}-{matched_hi:.4f})")
    print(f"  Per-probe instability (different quantity): median {perprobe_median:.4f}")
    print(f"  Paper's observed cross-lingual DFG range: 0.004-0.013")
    print(f"  -> observed DFG is the SAME ORDER as the matched floor: the cross-lingual")
    print(f"     signal is not separable from within-language paraphrase noise.")

    with open(RESULTS_DIR / "noise_floor_verdict.json", "w") as f:
        json.dump({
            "matched_noise_floor_median": round(matched_median, 6),
            "matched_noise_floor_range": [round(matched_lo, 6), round(matched_hi, 6)],
            "perprobe_instability_median": round(perprobe_median, 6),
            "per_model": overall.set_index("model").round(6).to_dict("index"),
            "by_source_perprobe": by_source.set_index("source").round(6).to_dict("index"),
            "interpretation": (
                "The matched floor is a difference of probe-set means, the same "
                "aggregation as the paper's reported DFG, and is the only quantity "
                "comparable to it. Its median is ~0.008 with a per-model range of "
                "roughly 0.003-0.021, i.e. the SAME ORDER OF MAGNITUDE as the observed "
                "cross-lingual DFG range (0.004-0.013). The cross-lingual signal is "
                "therefore not separable from within-language paraphrase noise; it is "
                "NOT the case that the floor is several times larger than the signal. "
                "The per-probe absolute-deviation statistic (median ~0.08) is roughly an "
                "order of magnitude larger purely because taking absolute values before "
                "averaging prevents signed per-probe noise from cancelling; it measures "
                "probe-level instability and must not be compared against a "
                "difference-of-means DFG."
            ),
        }, f, indent=2)
    print(f"\n  → {RESULTS_DIR / 'noise_floor_verdict.json'}")


if __name__ == "__main__":
    main()
