#!/usr/bin/env python3
"""
07_power_analysis.py — Power analysis + per-pivot noise-floor breakdown
=======================================================================
Two analyses that turn the paper's negative result into a concrete design
recommendation, and test whether the noise floor is a real property of the
measurement rather than an artifact of one machine-translation route.

(1) POWER ANALYSIS
    Given the per-probe variance actually observed in the audit, how many
    probes per language are needed to detect a true DFG of size delta at
    80% power? DFG is a paired contrast (the same base prompt, translated),
    so the relevant quantity is the SD of the per-probe difference
    d = SS(lang) - SS(en), and

        n = (z_{alpha/2} + z_beta)^2 * sigma_d^2 / delta^2

    We report n across a range of delta including the paper's own observed
    DFG range (0.004-0.013) and the measured noise floor (~0.046).

(2) PER-PIVOT NOISE FLOOR
    The noise floor is built from round-trip pivot translation
    en -> {de, ja, fi} -> en. Pooling those hides whether the floor is a
    stable property or an artifact of a particular MT route. Breaking it
    out by pivot tests this directly: German is typologically close to
    English, Japanese and Finnish are distant. If close and distant pivots
    give similar floors, the floor is a property of the SS measurement
    under paraphrase, not of any one translation direction.

Usage:
    python 07_power_analysis.py
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR, DATA_DIR

ALPHA = 0.05
POWER = 0.80


# ══════════════════════════════════════════════════════════════════════════════
# (1) Power analysis
# ══════════════════════════════════════════════════════════════════════════════

def paired_difference_sd(df: pd.DataFrame) -> dict:
    """
    SD of the per-probe paired difference d = SS(lang) - SS(en), pooled
    across models and non-English languages. This is the noise term that
    governs how precisely a mean DFG can be estimated from n probes.
    """
    en = (df[df["language"] == "en"]
          [["model", "base_prompt_id", "stereotype_score"]]
          .rename(columns={"stereotype_score": "ss_en"}))
    non_en = df[df["language"] != "en"][
        ["model", "base_prompt_id", "language", "stereotype_score"]]

    merged = non_en.merge(en, on=["model", "base_prompt_id"], how="inner")
    merged["d"] = merged["stereotype_score"] - merged["ss_en"]

    per_lang = (merged.groupby("language")["d"].std()
                .rename("sigma_d").reset_index())
    return {
        "sigma_d_pooled": float(merged["d"].std()),
        "sigma_d_per_language": per_lang,
        "n_paired_observations": int(len(merged)),
    }


def n_for_power(delta: float, sigma_d: float, alpha: float = ALPHA,
                power: float = POWER) -> float:
    """Probes per language needed to detect a paired mean difference `delta`."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return (z_alpha + z_beta) ** 2 * sigma_d ** 2 / delta ** 2


def run_power_analysis(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("  (1) POWER ANALYSIS")
    print("=" * 70)

    sd_info = paired_difference_sd(df)
    sigma_d = sd_info["sigma_d_pooled"]
    print(f"  Per-probe paired-difference SD (sigma_d): {sigma_d:.4f}")
    print(f"  (from {sd_info['n_paired_observations']:,} paired observations)")
    print(f"\n  Per-language sigma_d:")
    print(sd_info["sigma_d_per_language"].to_string(index=False))

    # Effect sizes spanning the paper's observed DFG range and the noise floor
    deltas = [0.004, 0.005, 0.010, 0.013, 0.020, 0.046, 0.05, 0.08, 0.10]
    rows = []
    for d in deltas:
        n = n_for_power(d, sigma_d)
        rows.append({"delta_DFG": d, "n_probes_per_language": int(np.ceil(n))})
    table = pd.DataFrame(rows)

    print(f"\n  Probes per language for {int(POWER*100)}% power "
          f"(alpha={ALPHA}, two-sided, paired):")
    print(table.to_string(index=False))

    # What the current design can actually detect
    n_current = int(df[df["language"] != "en"]
                    .groupby(["model", "language"])["base_prompt_id"]
                    .nunique().median())
    z_alpha = stats.norm.ppf(1 - ALPHA / 2)
    z_beta = stats.norm.ppf(POWER)
    mde_current = (z_alpha + z_beta) * sigma_d / np.sqrt(n_current)
    print(f"\n  Current design: {n_current} probes per language per model")
    print(f"  Minimum detectable DFG at {int(POWER*100)}% power: {mde_current:.4f}")
    print(f"  Observed DFG range in this audit: 0.004-0.013")
    print(f"  -> the current design is {mde_current/0.013:.1f}-{mde_current/0.004:.1f}x "
          f"underpowered for the effects it reports.")

    table.to_csv(RESULTS_DIR / "power_analysis.csv", index=False)

    return {
        "sigma_d_pooled": round(sigma_d, 6),
        "n_paired_observations": sd_info["n_paired_observations"],
        "n_probes_current": n_current,
        "min_detectable_dfg_current": round(float(mde_current), 6),
        "alpha": ALPHA,
        "power": POWER,
        "n_required": {str(r["delta_DFG"]): int(r["n_probes_per_language"])
                       for _, r in table.iterrows()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# (2) Per-pivot noise floor
# ══════════════════════════════════════════════════════════════════════════════

def run_per_pivot(noise_df: pd.DataFrame, exclude_bbq: bool = True) -> dict:
    print("\n" + "=" * 70)
    print("  (2) NOISE FLOOR BY PIVOT LANGUAGE")
    print("=" * 70)

    orig = (noise_df[noise_df["seed"] == -1]
            [["model", "prompt_id", "stereotype_score"]]
            .rename(columns={"stereotype_score": "ss_original"}))
    para = noise_df[noise_df["seed"] != -1]
    merged = para.merge(orig, on=["model", "prompt_id"])
    merged["dfg_noise"] = (merged["stereotype_score"] - merged["ss_original"]).abs()

    if exclude_bbq:
        with open(DATA_DIR / "prompts.json") as f:
            prompts = json.load(f)
        src = {p["id"]: p.get("source", "handcrafted")
               for p in prompts if p["language"] == "en" and p["layer"] == "A"}
        merged["source"] = merged["prompt_id"].map(src)
        merged = merged[merged["source"] != "bbq"]
        print("  (BBQ-sourced probes excluded, matching the paper's "
              "most-conservative reading)")

    by_pivot = (merged.groupby("pivot")["dfg_noise"]
                .agg(["mean", "median", "std", "count"])
                .reset_index()
                .rename(columns={"mean": "mean_noise", "median": "median_noise"}))

    typology = {"de": "Germanic (close to English)",
                "ja": "Japonic (distant)",
                "fi": "Uralic (distant)"}
    by_pivot["typological_distance"] = by_pivot["pivot"].map(typology)

    print()
    print(by_pivot.to_string(index=False))

    medians = by_pivot.set_index("pivot")["median_noise"].to_dict()
    spread = max(medians.values()) - min(medians.values())
    rel_spread = spread / np.mean(list(medians.values()))
    print(f"\n  Median noise floor spread across pivots: {spread:.4f} "
          f"({rel_spread*100:.1f}% of the mean floor)")

    # Kruskal-Wallis: are the three pivot distributions different?
    groups = [g["dfg_noise"].values for _, g in merged.groupby("pivot")]
    h, p = stats.kruskal(*groups)
    print(f"  Kruskal-Wallis across pivots: H={h:.2f}, p={p:.4f}")

    if rel_spread < 0.25:
        print("  -> Close (de) and distant (ja, fi) pivots give a similar floor:")
        print("     the floor is a property of the SS measurement under paraphrase,")
        print("     not an artifact of one translation route.")
    else:
        print("  -> Pivots differ substantially; the floor is route-dependent and")
        print("     should be reported per-pivot rather than pooled.")

    by_pivot.to_csv(RESULTS_DIR / "noise_floor_by_pivot.csv", index=False)

    return {
        "by_pivot": by_pivot.set_index("pivot")[
            ["mean_noise", "median_noise", "count"]].round(6).to_dict("index"),
        "median_spread_absolute": round(float(spread), 6),
        "median_spread_relative": round(float(rel_spread), 4),
        "kruskal_H": round(float(h), 4),
        "kruskal_p": float(p),
        "floor_is_route_independent": bool(rel_spread < 0.25),
    }


def main():
    results_path = RESULTS_DIR / "all_results.csv"
    noise_path = RESULTS_DIR / "noise_floor_all_results.csv"
    assert results_path.exists(), f"Missing {results_path}"
    assert noise_path.exists(), f"Missing {noise_path}"

    df = pd.read_csv(results_path)
    noise_df = pd.read_csv(noise_path)

    power = run_power_analysis(df)
    pivot = run_per_pivot(noise_df)

    out = {"power_analysis": power, "noise_floor_by_pivot": pivot}
    with open(RESULTS_DIR / "power_and_pivot_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {RESULTS_DIR / 'power_and_pivot_analysis.json'}")


if __name__ == "__main__":
    main()
