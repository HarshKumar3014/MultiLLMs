#!/usr/bin/env python3
"""
04_robustness.py — Early robustness checks (superseded by 09–22)
==============================================================
Checks:
  #3 Translation-quality confound: per-language back-translation pass-rate,
     regression rerun restricted to high-similarity subset only.
  #2 CLFI redundancy: proves current CLFI is a monotone transform of mean DFG,
     adds a dispersion-based CLFI that doesn't reward uniformly-biased models.
  #5 Non-independence: cluster-robust SEs (cluster on base_prompt_id) and
     mixed-effects model with crossed random intercepts (model, probe).

Usage:
    python 04_robustness.py                  # run everything
    python 04_robustness.py --translation-only
    python 04_robustness.py --clfi-only
    python 04_robustness.py --regression-only
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR, DATA_DIR, LANGUAGES, ANALYSIS, MODEL_REGISTRY

RESOURCE_ORDER = ["high", "mid", "low"]


# ══════════════════════════════════════════════════════════════════════════════
# #3 — Translation-quality confound
# ══════════════════════════════════════════════════════════════════════════════

def translation_pass_rate_table() -> pd.DataFrame:
    """
    Per-language back-translation pass rate (from validate_translations()
    output in data/validation_results.json). Tests the hypothesis
    that low-resource-language SS/LMS effects are a translation-quality
    artifact rather than a model-bias finding.
    """
    print("\n▸ Computing per-language translation pass-rate...")

    val_path = DATA_DIR / "validation_results.json"
    prompts_path = DATA_DIR / "prompts.json"
    assert val_path.exists(), f"Missing {val_path} — run 01_build_prompts.py first."
    assert prompts_path.exists(), f"Missing {prompts_path}."

    with open(val_path) as f:
        val = json.load(f)
    with open(prompts_path) as f:
        prompts = json.load(f)

    lang_by_id = {p["id"]: p["language"] for p in prompts}

    rows = []
    for pid, v in val.items():
        lang = lang_by_id.get(pid)
        if lang is None:
            continue
        rows.append({
            "prompt_id": pid,
            "language": lang,
            "similarity": v["similarity"],
            "passed": v["passed"],
        })
    vdf = pd.DataFrame(rows)

    table = (
        vdf.groupby("language")
        .agg(
            n=("prompt_id", "count"),
            pass_rate=("passed", "mean"),
            mean_similarity=("similarity", "mean"),
            min_similarity=("similarity", "min"),
        )
        .reset_index()
    )
    table["resource_level"] = table["language"].map(lambda l: LANGUAGES.get(l, {}).get("resource", "?"))
    table["language_name"] = table["language"].map(lambda l: LANGUAGES.get(l, {}).get("name", l))
    table = table.sort_values(
        by="resource_level", key=lambda s: s.map({r: i for i, r in enumerate(RESOURCE_ORDER)})
    )

    print(table.to_string(index=False))

    # Key check: does failure rate correlate with resource level?
    resource_map = {"high": 0, "mid": 1, "low": 2}
    table["resource_numeric"] = table["resource_level"].map(resource_map)
    corr, p = stats.spearmanr(table["resource_numeric"], 1 - table["pass_rate"])
    print(f"\n  Spearman(resource_level, fail_rate) = {corr:.3f} (p={p:.4f})")
    if corr > 0.3 and p < ANALYSIS["significance"]:
        print("  ⚠ CONFIRMED: translation failures concentrate in lower-resource languages.")
        print("    §4.3/§4.4 resource-level findings are confounded until re-run on clean subset.")
    else:
        print("  ✓ No significant concentration of failures by resource level — confound less likely.")

    out_path = RESULTS_DIR / "translation_pass_rate_by_language.csv"
    table.to_csv(out_path, index=False)
    print(f"  → {out_path}")

    return vdf  # per-prompt table needed downstream for the clean-subset regression


def regression_clean_subset(df: pd.DataFrame, vdf: pd.DataFrame, threshold: float = None) -> dict:
    """
    Rerun the resource-level regression restricted to prompts whose
    back-translation similarity is >= threshold (default: ANALYSIS threshold, 0.80,
    but review specifically asks for a 0.90 cut — report both).
    English/Layer-B rows have no back-translation entry (validation only runs
    on Layer A non-English prompts) — they are always kept.
    """
    import statsmodels.formula.api as smf

    print("\n▸ Re-running regression on high-similarity subset only...")

    passed_ids = set(vdf[vdf["similarity"] >= (threshold or ANALYSIS["backtranslation_threshold"])]["prompt_id"])
    validated_lang_ids = set(vdf["prompt_id"])  # all Layer-A non-en prompts that went through validation

    # keep: rows not subject to validation (en, Layer B) + rows that passed at this threshold
    clean = df[(~df["prompt_id"].isin(validated_lang_ids)) | (df["prompt_id"].isin(passed_ids))].copy()

    print(f"  Full dataset: {len(df)} rows | Clean subset (sim >= {threshold or ANALYSIS['backtranslation_threshold']}): {len(clean)} rows "
          f"({100*len(clean)/len(df):.1f}%)")

    resource_map = {"high": 0, "mid": 1, "low": 2}
    results = {}
    for label, data in [("full", df), ("clean_subset", clean)]:
        d = data.copy()
        d["resource_numeric"] = d["resource_level"].map(resource_map)
        formula = "stereotype_score ~ resource_numeric + C(model_group_label) + C(layer) + C(category)"
        model = smf.ols(formula, data=d).fit()
        coef = model.params.get("resource_numeric")
        p = model.pvalues.get("resource_numeric")
        results[label] = {"n": len(d), "coef": round(float(coef), 6), "pvalue": round(float(p), 6),
                           "significant": bool(p < ANALYSIS["significance"])}
        print(f"    {label:>13}: n={len(d):>6} resource_numeric coef={coef:.4f} p={p:.4f} "
              f"{'SIGNIFICANT' if p < ANALYSIS['significance'] else 'not significant'}")

    survived = results["full"]["significant"] and results["clean_subset"]["significant"] and \
        (np.sign(results["full"]["coef"]) == np.sign(results["clean_subset"]["coef"]))
    print(f"\n  {'✓ Resource-level effect SURVIVES clean-subset check.' if survived else '⚠ Resource-level effect DOES NOT survive — likely translation-quality artifact.'}")
    results["survived"] = survived

    out_path = RESULTS_DIR / "regression_clean_subset.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → {out_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# #2 — CLFI redundancy with DFG
# ══════════════════════════════════════════════════════════════════════════════

def compute_dfg(df: pd.DataFrame) -> pd.DataFrame:
    en_ss = df[df["language"] == "en"].groupby("model")["stereotype_score"].mean().rename("ss_english")
    lang_ss = (
        df.groupby(["model", "language"])["stereotype_score"].mean().reset_index()
        .rename(columns={"stereotype_score": "ss_lang"})
    )
    lang_ss = lang_ss.merge(en_ss, on="model")
    lang_ss["dfg"] = (lang_ss["ss_lang"] - lang_ss["ss_english"]).abs()
    return lang_ss


def clfi_redundancy_check(df: pd.DataFrame) -> dict:
    """
    Prove (or disprove) that CLFI = 1 - mean(DFG)/0.5 carries no information
    beyond mean DFG, i.e. that model rankings by CLFI and by mean DFG are
    identical (Spearman rho = -1.0 exactly, since it's a monotone transform).

    Then compute a dispersion-based alternative:
      CLFI_dispersion(model) = 1 - std_over_languages(SS) / max_possible_std
    which does NOT reward a model that is uniformly biased toward the
    stereotype in every language (old CLFI gives such a model ~1.0; dispersion
    version correctly flags it as non-zero spread only if spread is non-zero —
    the point is that this must be explicitly a *different signal*, not
    that uniform bias should score low. We report both so the paper can state
    "equity" (old CLFI = deployment-consistency) vs "calibration spread" (new)
    as two distinct, non-redundant axes.)
    """
    print("\n▸ Checking CLFI redundancy vs mean DFG...")

    dfg_df = compute_dfg(df)
    non_en = dfg_df[dfg_df["language"] != "en"]

    old_clfi = (
        non_en.groupby("model")["dfg"].mean().rename("mean_dfg").reset_index()
    )
    old_clfi["clfi_old"] = (1 - old_clfi["mean_dfg"] / 0.5).clip(0, 1)

    # Dispersion-based CLFI: std of raw SS across languages (not relative to English)
    ss_by_lang = df.groupby(["model", "language"])["stereotype_score"].mean().reset_index()
    disp = ss_by_lang.groupby("model")["stereotype_score"].std().rename("ss_std_across_langs").reset_index()
    # Max possible std across k languages of values bounded in [0,1] with values at 0/1 extremes
    n_langs = df["language"].nunique()
    max_std = 0.5  # theoretical max std for a [0,1]-bounded variable (half split at 0 and 1)
    disp["clfi_dispersion"] = (1 - disp["ss_std_across_langs"] / max_std).clip(0, 1)

    merged = old_clfi.merge(disp, on="model")

    rho, p = stats.spearmanr(merged["clfi_old"], merged["mean_dfg"])
    print(f"  Spearman(CLFI_old, mean_DFG) = {rho:.6f} (p={p:.2e})  "
          f"[expected: exactly -1.0, monotone transform]")

    rho2, p2 = stats.spearmanr(merged["clfi_old"], merged["clfi_dispersion"])
    print(f"  Spearman(CLFI_old, CLFI_dispersion) = {rho2:.4f} (p={p2:.4f})")
    if abs(rho2) < 0.9:
        print("  ✓ Dispersion-based CLFI is NOT redundant with old CLFI — carries independent signal.")
    else:
        print("  ⚠ Dispersion-based CLFI is still highly correlated with old CLFI in this data.")

    merged["model_group_label"] = merged["model"].map(lambda m: MODEL_REGISTRY[m]["group_label"])
    merged = merged.sort_values("clfi_old", ascending=False)
    print("\n" + merged[["model", "model_group_label", "mean_dfg", "clfi_old", "ss_std_across_langs", "clfi_dispersion"]]
          .to_string(index=False))

    out_path = RESULTS_DIR / "clfi_redundancy_check.csv"
    merged.to_csv(out_path, index=False)
    print(f"  → {out_path}")

    summary = {
        "rho_clfi_old_vs_mean_dfg": round(float(rho), 6),
        "is_monotone_transform": bool(abs(rho) > 0.999),
        "rho_clfi_old_vs_clfi_dispersion": round(float(rho2), 4),
        "dispersion_is_independent_metric": bool(abs(rho2) < 0.9),
    }
    with open(RESULTS_DIR / "clfi_redundancy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# #5 — Non-independence: cluster-robust SEs + mixed-effects
# ══════════════════════════════════════════════════════════════════════════════

def cluster_robust_regression(df: pd.DataFrame) -> dict:
    """
    Same OLS specification as 03_analyze.py, but with cluster-robust standard
    errors clustered on base_prompt_id (each base prompt is repeated across
    languages and models — the flat OLS treats those as independent draws,
    which is anti-conservative). This is the cheap, always-converges fix.
    """
    import statsmodels.formula.api as smf

    print("\n▸ Cluster-robust regression (cluster = base_prompt_id)...")

    d = df.copy()
    resource_map = {"high": 0, "mid": 1, "low": 2}
    d["resource_numeric"] = d["resource_level"].map(resource_map)
    formula = "stereotype_score ~ resource_numeric + C(model_group_label) + C(layer) + C(category)"

    ols = smf.ols(formula, data=d).fit()
    clustered = smf.ols(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["base_prompt_id"]}
    )

    print(f"\n  {'term':<35}{'coef':>10}{'p (flat OLS)':>16}{'p (clustered)':>16}")
    rows = {}
    for name in ols.params.index:
        p_flat = float(ols.pvalues[name])
        p_clust = float(clustered.pvalues[name])
        rows[name] = {
            "coef": round(float(ols.params[name]), 6),
            "p_flat_ols": round(p_flat, 6),
            "p_clustered": round(p_clust, 6),
            "significant_flat": p_flat < ANALYSIS["significance"],
            "significant_clustered": p_clust < ANALYSIS["significance"],
        }
        flag = "" if rows[name]["significant_flat"] == rows[name]["significant_clustered"] else "  <-- FLIPS"
        print(f"  {name:<35}{ols.params[name]:>10.4f}{p_flat:>16.4f}{p_clust:>16.4f}{flag}")

    n_flip = sum(1 for r in rows.values() if r["significant_flat"] and not r["significant_clustered"])
    print(f"\n  {n_flip} term(s) significant under flat OLS but NOT under clustering "
          f"(i.e. were anti-conservative false positives).")

    out = {"n_obs": len(d), "n_clusters": int(d["base_prompt_id"].nunique()), "terms": rows,
           "n_terms_flipped_to_nonsignificant": n_flip}
    with open(RESULTS_DIR / "regression_cluster_robust.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → {RESULTS_DIR / 'regression_cluster_robust.json'}")
    return out


def mixed_effects_regression(df: pd.DataFrame, sample_frac: float = None, label: str = "full") -> dict:
    """
    Full mixed-effects model with random intercepts for base_prompt_id
    (repeated-measures across languages/models) and a variance component
    for model (repeated-measures across prompts). This is the model the
    paper's own §5 names as the fix.

    statsmodels MixedLM only supports one primary `groups` grouping factor
    natively; we use base_prompt_id as the primary group (probes are the
    tighter cluster — same probe, same wording, translated) and add `model`
    as a variance component (crossed random effect) via vc_formula.

    This can be slow on 30k+ rows with ~400 groups; if it fails to converge
    in a reasonable number of iterations we report that explicitly rather
    than silently falling back.
    """
    import statsmodels.formula.api as smf

    print("\n▸ Mixed-effects regression (random intercepts: base_prompt_id, model)...")

    d = df.copy()
    resource_map = {"high": 0, "mid": 1, "low": 2}
    d["resource_numeric"] = d["resource_level"].map(resource_map)

    if sample_frac:
        d = d.sample(frac=sample_frac, random_state=42)
        print(f"  (subsampled to {len(d)} rows for feasibility, frac={sample_frac})")

    formula = "stereotype_score ~ resource_numeric + C(model_group_label) + C(layer) + C(category)"
    vc = {"model": "0 + C(model)"}

    try:
        mixed = smf.mixedlm(formula, data=d, groups=d["base_prompt_id"], vc_formula=vc)
        fit = mixed.fit(reml=True, method="lbfgs", maxiter=200)
        print(fit.summary())

        resource_coef = fit.params.get("resource_numeric")
        resource_p = fit.pvalues.get("resource_numeric")
        result = {
            "converged": bool(fit.converged),
            "n_obs": len(d),
            "n_groups_base_prompt_id": int(d["base_prompt_id"].nunique()),
            "resource_numeric_coef": round(float(resource_coef), 6) if resource_coef is not None else None,
            "resource_numeric_pvalue": round(float(resource_p), 6) if resource_p is not None else None,
            "significant": bool(resource_p < ANALYSIS["significance"]) if resource_p is not None else None,
        }
        print(f"\n  ★ Mixed-effects resource_numeric: coef={result['resource_numeric_coef']} "
              f"p={result['resource_numeric_pvalue']} converged={result['converged']}")

    except Exception as e:
        print(f"  ✗ Mixed-effects model failed: {e}")
        result = {"converged": False, "error": str(e)}

    out_path = RESULTS_DIR / f"regression_mixed_effects_{label}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {out_path}")
    return result


def combined_robustness_verdict(df: pd.DataFrame, vdf: pd.DataFrame) -> dict:
    """
    The decisive test: apply BOTH fixes at once (mixed-effects model +
    clean-subset restriction to sim >= 0.90). If resource_numeric survives
    only in specifications missing one of the two fixes, the original
    "significant" flat-OLS result was a compound artifact of non-independence
    AND translation-quality confound, not a real cross-lingual bias effect.
    """
    print("\n▸ Combined verdict: mixed-effects model on clean (sim>=0.90) subset...")

    passed_ids = set(vdf[vdf["similarity"] >= 0.90]["prompt_id"])
    validated_ids = set(vdf["prompt_id"])
    clean = df[(~df["prompt_id"].isin(validated_ids)) | (df["prompt_id"].isin(passed_ids))].copy()

    result = mixed_effects_regression(clean, label="clean_subset")

    verdict = {
        "flat_ols_full_data_significant": True,   # paper's original result, p=0.039
        "mixed_effects_plus_clean_subset_significant": result.get("significant"),
        "conclusion": (
            "Resource-level effect does NOT survive when both non-independence "
            "and translation-quality confound are corrected simultaneously — "
            "original finding was a compound artifact."
            if not result.get("significant")
            else "Resource-level effect SURVIVES both corrections applied together — finding holds."
        ),
    }
    print(f"\n  VERDICT: {verdict['conclusion']}")

    with open(RESULTS_DIR / "robustness_final_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    return verdict


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Early robustness checks")
    parser.add_argument("--translation-only", action="store_true")
    parser.add_argument("--clfi-only", action="store_true")
    parser.add_argument("--regression-only", action="store_true")
    parser.add_argument("--mixed-effects-sample-frac", type=float, default=None,
                         help="Subsample fraction for mixed-effects fit if full data is too slow")
    args = parser.parse_args()

    results_path = RESULTS_DIR / "all_results.csv"
    assert results_path.exists(), f"Results not found at {results_path}. Run 02_run_audit.py first."
    df = pd.read_csv(results_path)

    run_all = not (args.translation_only or args.clfi_only or args.regression_only)

    print("=" * 70)
    print("  ROBUSTNESS CHECKS")
    print("=" * 70)
    print(f"  Loaded {len(df)} rows, {df['model'].nunique()} models, {df['language'].nunique()} languages")

    vdf = None
    if run_all or args.translation_only:
        vdf = translation_pass_rate_table()
        regression_clean_subset(df, vdf, threshold=0.90)

    if run_all or args.clfi_only:
        clfi_redundancy_check(df)

    if run_all or args.regression_only:
        if vdf is None:
            vdf = translation_pass_rate_table()
        cluster_robust_regression(df)
        mixed_effects_regression(df, sample_frac=args.mixed_effects_sample_frac, label="full")
        combined_robustness_verdict(df, vdf)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
