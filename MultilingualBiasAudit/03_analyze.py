#!/usr/bin/env python3
"""
03_analyze.py — Metrics, Regression, Visualizations, Paper Artifacts
====================================================================
Computes all analysis from audit results:
  • Per-model, per-language bias metrics (SS, LMS, ICAT)
  • Deployment Fairness Gap (DFG) per language
  • Cross-Lingual Fairness Index (CLFI) per model
  • Mixed-effects regression
  • Publication-quality figures and LaTeX tables

Usage:
    python 03_analyze.py                     # full analysis
    python 03_analyze.py --figures-only      # regenerate figures
    python 03_analyze.py --tables-only       # regenerate LaTeX tables
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from config import (
    RESULTS_DIR, FIGURES_DIR, TABLES_DIR, DATA_DIR,
    MODEL_REGISTRY, LANGUAGES, ANALYSIS,
)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Color palette for model groups
GROUP_COLORS = {
    "A_english_centric": "#4C72B0",
    "B_multilingual_native": "#55A868",
    "C_regional_centric": "#C44E52",
}
GROUP_LABELS = {
    "A_english_centric": "English-Centric",
    "B_multilingual_native": "Multilingual-Native",
    "C_regional_centric": "Regional-Centric",
}
RESOURCE_ORDER = ["high", "mid", "low"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_dfg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Deployment Fairness Gap (DFG) for each model × language.
    DFG(model, L) = |SS(model, L) - SS(model, English)|
    """
    # Get English SS per model
    en_ss = (
        df[df["language"] == "en"]
        .groupby("model")["stereotype_score"]
        .mean()
        .rename("ss_english")
    )

    # Get per-language SS per model
    lang_ss = (
        df.groupby(["model", "language"])["stereotype_score"]
        .mean()
        .reset_index()
        .rename(columns={"stereotype_score": "ss_lang"})
    )

    # Merge and compute DFG
    lang_ss = lang_ss.merge(en_ss, on="model")
    lang_ss["dfg"] = (lang_ss["ss_lang"] - lang_ss["ss_english"]).abs()

    return lang_ss


def compute_clfi(dfg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Cross-Lingual Fairness Index (CLFI) per model.
    CLFI = 1 - mean(DFG) / max_possible_DFG
    Where max_possible_DFG = 0.5 (theoretical max shift from 0.5 to 0 or 1)
    CLFI ∈ [0, 1], where 1 = perfect cross-lingual equity.
    """
    # Exclude English (DFG with itself is always 0)
    non_en = dfg_df[dfg_df["language"] != "en"]

    clfi = (
        non_en.groupby("model")["dfg"]
        .agg(["mean", "std", "max"])
        .rename(columns={"mean": "mean_dfg", "std": "std_dfg", "max": "max_dfg"})
    )
    clfi["clfi"] = 1 - clfi["mean_dfg"] / 0.5
    clfi["clfi"] = clfi["clfi"].clip(0, 1)

    # Add model metadata
    clfi = clfi.reset_index()
    clfi["model_group"] = clfi["model"].map(lambda m: MODEL_REGISTRY[m]["group"])
    clfi["model_group_label"] = clfi["model"].map(lambda m: MODEL_REGISTRY[m]["group_label"])
    clfi["model_org"] = clfi["model"].map(lambda m: MODEL_REGISTRY[m]["org"])

    return clfi.sort_values("clfi", ascending=False)


def bootstrap_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval for the mean."""
    boot_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100])


def compute_full_metrics(df: pd.DataFrame) -> dict:
    """Compute all metrics and return a summary dict."""
    print("\n▸ Computing bias metrics...")

    # Per-model summary
    model_summary = (
        df.groupby(["model", "model_group_label"])
        .agg(
            mean_ss=("stereotype_score", "mean"),
            std_ss=("stereotype_score", "std"),
            mean_lms=("lm_score", "mean"),
            mean_icat=("icat_score", "mean"),
            n_prompts=("stereotype_score", "count"),
        )
        .reset_index()
    )

    # Per-model × language
    model_lang = (
        df.groupby(["model", "model_group_label", "language"])
        .agg(
            mean_ss=("stereotype_score", "mean"),
            std_ss=("stereotype_score", "std"),
            mean_lms=("lm_score", "mean"),
            mean_icat=("icat_score", "mean"),
            n_prompts=("stereotype_score", "count"),
        )
        .reset_index()
    )

    # DFG and CLFI
    dfg_df = compute_dfg(df)
    clfi_df = compute_clfi(dfg_df)

    # Layer comparison
    layer_comp = (
        df.groupby(["model", "layer"])["stereotype_score"]
        .mean()
        .unstack(fill_value=np.nan)
        .reset_index()
    )

    # Bootstrap CIs for CLFI per group
    print("  Computing bootstrap confidence intervals...")
    group_cis = {}
    for group in df["model_group_label"].unique():
        group_models = clfi_df[clfi_df["model_group_label"] == group]["clfi"].values
        if len(group_models) > 1:
            ci = bootstrap_ci(group_models, n_boot=ANALYSIS["bootstrap_n"])
            group_cis[group] = {"mean": float(np.mean(group_models)),
                                "ci_low": float(ci[0]), "ci_high": float(ci[1])}
        else:
            group_cis[group] = {"mean": float(group_models[0]) if len(group_models) else 0,
                                "ci_low": None, "ci_high": None}

    return {
        "model_summary": model_summary,
        "model_lang": model_lang,
        "dfg": dfg_df,
        "clfi": clfi_df,
        "layer_comp": layer_comp,
        "group_cis": group_cis,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. REGRESSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_regression(df: pd.DataFrame) -> dict:
    """
    Mixed-effects-style regression (OLS with categorical predictors):
    SS ~ Language_Resource + Model_Group + Prompt_Layer + Category
    """
    print("\n▸ Running regression analysis...")

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  ⚠ statsmodels not installed — skipping regression")
        return {}

    # Prepare data
    reg_df = df.copy()
    reg_df["resource_level"] = pd.Categorical(
        reg_df["resource_level"], categories=RESOURCE_ORDER, ordered=True
    )

    # Encode resource level numerically for regression
    resource_map = {"high": 0, "mid": 1, "low": 2}
    reg_df["resource_numeric"] = reg_df["resource_level"].map(resource_map)

    # OLS regression
    formula = "stereotype_score ~ resource_numeric + C(model_group_label) + C(layer) + C(category)"
    try:
        model = smf.ols(formula, data=reg_df).fit()
        print(f"\n{model.summary2().tables[1].to_string()}")

        results = {
            "r_squared": round(model.rsquared, 4),
            "adj_r_squared": round(model.rsquared_adj, 4),
            "f_pvalue": float(model.f_pvalue),
            "coefficients": {
                name: {"coef": round(float(model.params[name]), 6),
                       "pvalue": round(float(model.pvalues[name]), 6),
                       "significant": float(model.pvalues[name]) < ANALYSIS["significance"]}
                for name in model.params.index
            },
        }

        # Key finding: is resource level significant?
        resource_coef = model.params.get("resource_numeric", None)
        resource_p = model.pvalues.get("resource_numeric", None)
        if resource_coef is not None:
            print(f"\n  ★ Key Finding: Language resource level coefficient = {resource_coef:.4f} "
                  f"(p = {resource_p:.4f})")
            if resource_p < ANALYSIS["significance"]:
                print(f"    → SIGNIFICANT: Lower-resource languages show {'more' if resource_coef > 0 else 'less'} bias")
            else:
                print(f"    → Not significant at α={ANALYSIS['significance']}")

        return results

    except Exception as e:
        print(f"  ✗ Regression failed: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(model_lang: pd.DataFrame):
    """
    Figure 1: Models × Languages Stereotype Score Heatmap
    The "money figure" — shows the full experimental matrix at a glance.
    """
    print("  📊 Generating heatmap...")

    pivot = model_lang.pivot(index="model", columns="language", values="mean_ss")

    # Order columns by resource level
    lang_order = sorted(LANGUAGES.keys(),
                        key=lambda x: (RESOURCE_ORDER.index(LANGUAGES[x]["resource"]), x))
    pivot = pivot.reindex(columns=[c for c in lang_order if c in pivot.columns])

    # Order rows by model group
    group_order = sorted(MODEL_REGISTRY.keys(),
                         key=lambda m: (MODEL_REGISTRY[m]["group"], m))
    pivot = pivot.reindex(index=[m for m in group_order if m in pivot.index])

    # Rename columns to language names
    pivot.columns = [LANGUAGES.get(c, {}).get("name", c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        # PuOr instead of RdYlGn: red-green diverging palettes are not
        # colorblind-safe (fail for deuteranopia/protanopia); PuOr is a
        # ColorBrewer-validated CVD-safe diverging pair. Cells are also
        # directly annotated with values (secondary encoding), so identity
        # never depends on color alone. vmin/vmax zoomed to just beyond the
        # actual data range (~0.469-0.517) rather than a token +/-0.15
        # window, so the real cell-to-cell variation actually shows up in
        # color instead of being washed out to near-uniform pale yellow.
        pivot, annot=True, fmt=".3f", cmap="PuOr_r",
        center=0.5, vmin=0.46, vmax=0.54,
        linewidths=1.2, linecolor="white",
        annot_kws={"fontsize": 9.5, "fontweight": "medium"},
        cbar_kws={"label": "Stereotype Score (SS)", "shrink": 0.8},
        ax=ax,
    )

    # Add group separators
    group_counts = {}
    for m in pivot.index:
        g = MODEL_REGISTRY.get(m, {}).get("group", "?")
        group_counts[g] = group_counts.get(g, 0) + 1

    y = 0
    for g, count in group_counts.items():
        if y > 0:
            ax.axhline(y=y, color="white", linewidth=4)
            ax.axhline(y=y, color="#333333", linewidth=1.2)
        y += count

    ax.set_title("Stereotype Score by Model × Language", fontsize=16, fontweight="bold",
                 pad=32, loc="left")
    ax.text(0, 1.045, "0.5 = unbiased · warmer (orange) = pro-stereotype · cooler (purple) = anti-stereotype "
                       "· color scale zoomed to the data range",
            transform=ax.transAxes, fontsize=10, color="#6B6B6B", ha="left")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()

    path = FIGURES_DIR / "fig1_heatmap.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"    → {path}")


def _dfg_per_language(df: pd.DataFrame, model: str) -> np.ndarray:
    sub = df[df["model"] == model]
    en_ss = sub[sub["language"] == "en"]["stereotype_score"].mean()
    lang_ss = sub.groupby("language")["stereotype_score"].mean()
    lang_ss = lang_ss[lang_ss.index != "en"]
    return (lang_ss - en_ss).abs().values


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    return np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)


def plot_clfi_radar(clfi_df: pd.DataFrame, df: pd.DataFrame):
    """
    Figure 3: CLFI forest plot (point + 95% bootstrap CI per model).

    Formerly a polar/radar bar chart. Replaced: a radar's wedge area scales
    with r^2, so it visually exaggerated the ~2-percentage-point CLFI spread
    (0.973-0.992) into what looked like a dramatic gap between models,
    actively misleading given this paper's own finding that the underlying
    DFG differences are not distinguishable from noise. This is now a
    standard forest-plot-style figure (point estimate + CI per row), the
    convention for "is this estimate distinguishable from a reference"
    comparisons, matching Figure 1's visual language. CLFI = 1 - 2*mean(DFG)
    is a monotone transform of DFG, so its CI is derived directly from the
    same per-language bootstrap used for the DFG forest plot.
    """
    print("  📊 Generating CLFI forest plot...")

    models_all = clfi_df["model"].tolist()
    per_lang = {m: _dfg_per_language(df, m) for m in models_all}
    dfg_ci = {m: _bootstrap_ci(v) for m, v in per_lang.items()}
    # CLFI = 1 - 2*DFG is decreasing in DFG, so CI bounds flip under the transform.
    clfi_ci = {m: (1 - 2 * hi, 1 - 2 * lo) for m, (lo, hi) in dfg_ci.items()}

    clfi_val = dict(zip(clfi_df["model"], clfi_df["clfi"]))
    group_of = dict(zip(clfi_df["model"], clfi_df["model_group"]))
    models = sorted(models_all, key=lambda m: clfi_val[m])
    colors = [GROUP_COLORS.get(group_of[m], "#888888") for m in models]

    PANEL_BG = "#FAFAF8"
    ZEBRA_COLOR = "#EFEEEA"
    TEXT_MUTED = "#6B6B6B"

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PANEL_BG)
    y = np.arange(len(models))
    n = len(models)

    for yi in range(n):
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color=ZEBRA_COLOR, zorder=0, linewidth=0)

    all_los = [clfi_ci[m][0] for m in models]
    x_min = min(all_los) - 0.004

    for yi, (m, c) in enumerate(zip(models, colors)):
        lo, hi = clfi_ci[m]
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=2.2, alpha=0.9, zorder=3,
                 solid_capstyle="round")
        ax.plot([lo, lo], [yi - 0.14, yi + 0.14], color=c, linewidth=2.2, zorder=3)
        ax.plot([hi, hi], [yi - 0.14, yi + 0.14], color=c, linewidth=2.2, zorder=3)
        ax.plot(clfi_val[m], yi, "o", color=c, markersize=10.5, markeredgecolor="white",
                 markeredgewidth=1.4, zorder=4)
        ax.text(hi + 0.0015, yi, f"{clfi_val[m]:.3f}", va="center", ha="left",
                 fontsize=9.5, color=TEXT_MUTED, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_ylim(-0.5, n - 0.1)
    ax.set_xlim(x_min, 1.012)
    ax.set_xlabel("Cross-Lingual Fairness Index (CLFI), mean $\\pm$ 95% bootstrap CI",
                  fontsize=11, color=TEXT_MUTED)
    ax.set_title("CLFI confidence intervals overlap across the entire ranking",
                 fontsize=15, fontweight="bold", pad=28, loc="left")
    ax.text(0, 1.03, "Point = observed CLFI · bar = 95% CI, derived from the same per-language bootstrap as panel (a)",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_MUTED, ha="left")

    ax.grid(axis="x", alpha=0.5, zorder=0, color="white", linewidth=1.2)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="both", length=0)

    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=GROUP_COLORS[g],
                           markeredgecolor="white", markersize=9, label=GROUP_LABELS[g])
               for g in GROUP_COLORS]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95, edgecolor="#DDDDDD")

    plt.tight_layout()
    path = FIGURES_DIR / "fig2_clfi_radar.pdf"
    fig.savefig(path, facecolor="white")
    fig.savefig(path.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    print(f"    → {path}")


def plot_dfg_by_resource(dfg_df: pd.DataFrame):
    """
    Figure 3: DFG by Language Resource Level, grouped by model group.
    """
    print("  📊 Generating DFG by resource level...")

    # Add metadata
    plot_df = dfg_df[dfg_df["language"] != "en"].copy()
    plot_df["resource_level"] = plot_df["language"].map(
        lambda x: LANGUAGES.get(x, {}).get("resource", "unknown")
    )
    plot_df["model_group"] = plot_df["model"].map(
        lambda m: MODEL_REGISTRY.get(m, {}).get("group", "?")
    )
    plot_df["model_group_label"] = plot_df["model"].map(
        lambda m: MODEL_REGISTRY.get(m, {}).get("group_label", "?")
    )
    plot_df["resource_level"] = pd.Categorical(
        plot_df["resource_level"], categories=RESOURCE_ORDER, ordered=True
    )

    # Aggregate: mean DFG per resource level x model group. Standard error
    # of the mean (std / sqrt(n)), not raw std: raw std across the
    # underlying per-language DFG draws is often as large as the mean
    # itself here (coefficient of variation ~70-100%), which made a
    # bar+errorbar rendering dominated by whisker length rather than
    # legible bar comparison. SEM is the standard, defensible uncertainty
    # band for "how precisely is this group mean estimated" and is an
    # order of magnitude tighter, consistent with how uncertainty is
    # shown everywhere else in this paper (Figures 1, 3).
    agg = (
        plot_df.groupby(["model_group_label", "resource_level"], observed=True)["dfg"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])

    PANEL_BG = "#FAFAF8"
    TEXT_MUTED = "#6B6B6B"

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PANEL_BG)

    group_labels = list(GROUP_LABELS.values())
    # One row per (resource level, group) combination, forest-plot style,
    # grouped visually by resource level with zebra banding per level.
    rows = [(r, g) for r in RESOURCE_ORDER for g in group_labels]
    y = np.arange(len(rows))

    for ri in range(len(RESOURCE_ORDER)):
        if ri % 2 == 0:
            ax.axhspan(ri * len(group_labels) - 0.5, ri * len(group_labels) + len(group_labels) - 0.5,
                       color="#EFEEEA", zorder=0, linewidth=0)

    for yi, (resource, group_label) in enumerate(rows):
        cell = agg[(agg["resource_level"] == resource) & (agg["model_group_label"] == group_label)]
        if cell.empty:
            continue
        mean, sem = cell["mean"].values[0], cell["sem"].values[0]
        group_key = [k for k, v in GROUP_LABELS.items() if v == group_label][0]
        color = GROUP_COLORS[group_key]
        ax.plot([mean - sem, mean + sem], [yi, yi], color=color, linewidth=2, zorder=2,
                 solid_capstyle="round")
        ax.plot(mean, yi, "o", color=color, markersize=8, markeredgecolor="white",
                 markeredgewidth=1.1, zorder=3)
        ax.text(mean + sem + 0.0006, yi, f"{mean:.3f}", va="center", ha="left",
                 fontsize=8, color=TEXT_MUTED, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([g for _, g in rows], fontsize=8.5)
    # Resource-level group labels on the right margin
    for ri, r in enumerate(RESOURCE_ORDER):
        y0 = ri * len(group_labels) - 0.5
        y1 = y0 + len(group_labels)
        ax.annotate(r.capitalize(), xy=(1.02, (y0 + y1) / 2), xycoords=("axes fraction", "data"),
                    fontsize=10, fontweight="bold", color=TEXT_MUTED, va="center", ha="left",
                    rotation=270)

    ax.invert_yaxis()
    ax.set_xlabel("Deployment Fairness Gap (DFG), mean $\\pm$ SEM", color=TEXT_MUTED)
    ax.set_title("DFG by language resource level (descriptive)", fontsize=14,
                 fontweight="bold", pad=26, loc="left")
    ax.text(0, 1.04, "Not noise-floor tested; the apparent low-resource pattern does not "
                      "survive the translation-quality confound check (§4.3)",
            transform=ax.transAxes, fontsize=9, color=TEXT_MUTED, ha="left")
    ax.set_xlim(0, agg["mean"].max() + agg["sem"].max() + 0.004)
    ax.grid(axis="x", alpha=0.5, color="white", linewidth=1.2, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="both", length=0)

    plt.tight_layout()
    path = FIGURES_DIR / "fig3_dfg_resource.pdf"
    fig.savefig(path, facecolor="white")
    fig.savefig(path.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    print(f"    → {path}")


def plot_layer_comparison(df: pd.DataFrame):
    """
    Figure 4: Layer A (Translated) vs. Layer B (Culturally-Native) comparison.
    """
    print("  📊 Generating layer comparison...")

    layer_data = (
        df.groupby(["model", "model_group_label", "layer"])["stereotype_score"]
        .mean()
        .unstack(fill_value=np.nan)
        .reset_index()
    )

    if "A" not in layer_data.columns or "B" not in layer_data.columns:
        print("    ⚠ Both layers required — skipping")
        return

    PANEL_BG = "#FAFAF8"
    TEXT_MUTED = "#6B6B6B"

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PANEL_BG)

    # Real data clusters tightly around (0.49, 0.49), and three points
    # (solar-10.7b, llama3.1-8b, yi-1.5-9b) sit within ~0.002-0.004 of each
    # other in both dimensions -- close enough that full-name text labels
    # collide regardless of repulsion strength. Numbered markers (alphabetic
    # order, matching Table 4's row order) plus a compact in-plot index
    # sidesteps the problem entirely: numbers are small enough to place
    # cleanly even when points nearly coincide.
    models_sorted = sorted(layer_data["model"].tolist())
    index_of = {m: i + 1 for i, m in enumerate(models_sorted)}

    for _, row in layer_data.iterrows():
        model = row["model"]
        group = MODEL_REGISTRY.get(model, {}).get("group", "?")
        color = GROUP_COLORS.get(group, "#888888")
        ax.scatter(row["A"], row["B"], c=color, s=210, alpha=0.9,
                   edgecolors="white", linewidth=1.5, zorder=3)
        ax.text(row["A"], row["B"], str(index_of[model]), fontsize=8.5, zorder=4,
                 ha="center", va="center", color="white", fontweight="bold")

    index_text = "\n".join(f"{i}. {m}" for m, i in sorted(index_of.items(), key=lambda kv: kv[1]))
    ax.text(0.02, 0.98, index_text, transform=ax.transAxes, fontsize=8, color=TEXT_MUTED,
             va="top", ha="left", linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#DDDDDD", alpha=0.92))

    # Reference line (equal bias in both layers). Zoom to the actual
    # combined data range (union of A and B, padded) rather than a fixed
    # 0.35-0.65 window: the real data spans roughly [0.42, 0.50], so the
    # old fixed window left 90% of the plot empty and crushed all ten
    # points into a small central cluster, which is what made labels
    # unreadable regardless of repulsion. Equal aspect is kept so the
    # diagonal reference stays meaningful.
    all_vals = pd.concat([layer_data["A"], layer_data["B"]])
    pad = (all_vals.max() - all_vals.min()) * 0.25
    lims = [all_vals.min() - pad, all_vals.max() + pad]
    ax.plot(lims, lims, "--", color="#AAAAAA", alpha=0.7, zorder=1)
    ax.axhline(0.5, color="#AAAAAA", alpha=0.3, linewidth=0.8)
    ax.axvline(0.5, color="#AAAAAA", alpha=0.3, linewidth=0.8)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Layer A — Translated Stereotypes (SS)", color=TEXT_MUTED)
    ax.set_ylabel("Layer B — Culturally-Native Stereotypes (SS)", color=TEXT_MUTED)
    ax.set_title("Translated vs. Culturally-Native Bias Scores", fontsize=15,
                 fontweight="bold", pad=28, loc="left")
    ax.text(0, 1.035, "Deviation from the diagonal = cultural calibration gap",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_MUTED, ha="left")
    ax.set_aspect("equal")
    ax.grid(alpha=0.5, color="white", linewidth=1.2)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color("#CCCCCC")
    ax.tick_params(axis="both", length=0)

    # Legend
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=GROUP_COLORS[g],
                           markeredgecolor="white", markersize=9, label=GROUP_LABELS[g])
               for g in GROUP_LABELS]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95, edgecolor="#DDDDDD")

    path = FIGURES_DIR / "fig4_layer_comparison.pdf"
    fig.savefig(path, facecolor="white")
    fig.savefig(path.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    print(f"    → {path}")


def plot_ss_distribution(df: pd.DataFrame):
    """
    Figure 5: Violin plot of SS distributions per model group.
    """
    print("  📊 Generating SS distribution violins...")

    fig, ax = plt.subplots(figsize=(12, 6))

    order = list(GROUP_LABELS.values())
    palette = [GROUP_COLORS[k] for k in GROUP_LABELS.keys()]

    sns.violinplot(
        data=df, x="model_group_label", y="stereotype_score",
        order=order, palette=palette, inner="box", alpha=0.8, ax=ax,
    )
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Unbiased baseline")
    ax.set_xlabel("Model Provenance Group")
    ax.set_ylabel("Stereotype Score (SS)")
    ax.set_title("Distribution of Stereotype Scores by Model Provenance")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    path = FIGURES_DIR / "fig5_ss_distribution.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"    → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. LATEX TABLE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_tables(metrics: dict):
    """Generate LaTeX tables for the paper."""
    print("\n▸ Generating LaTeX tables...")

    # Table 1: CLFI rankings
    # Row-tinted by provenance group (very light versions of the same three
    # hues used in every figure in the paper -- GROUP_COLORS at ~12% tint --
    # so the table reads consistently with the rest of the figure set and
    # provenance groupings are scannable at a glance, matching Table 4's
    # level of visual polish. No value is bolded: the paper's own finding
    # is that these rankings are not distinguishable from noise, so nothing
    # here should visually imply a "winner".
    GROUP_TINTS = {
        "A_english_centric": "EAEEF6",
        "B_multilingual_native": "EBF5ED",
        "C_regional_centric": "F8EAEA",
    }
    GROUP_CODE = {
        "A_english_centric": "A",
        "B_multilingual_native": "B",
        "C_regional_centric": "C",
    }
    clfi = metrics["clfi"]
    # Single-column table (matches the rest of the single-column body flow,
    # not table*): "Multilingual-Native" was the width culprit, so the
    # Provenance column now uses the A/B/C codes already established in
    # Table 3 ("Multilingual-Native (B)" etc.) instead of the full label --
    # row tint still carries the group identity redundantly.
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cross-Lingual Fairness Index (CLFI) Rankings (nominal; see \S4.1 for the noise-floor test). "
        r"Higher CLFI indicates more equitable bias treatment across languages. "
        r"Provenance: A=English-Centric, B=Multilingual-Native, C=Regional-Centric "
        r"(row tint matches Figure 1).}",
        r"\label{tab:clfi}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Prov.} & \textbf{CLFI} $\uparrow$ & \textbf{DFG} $\downarrow$ \\",
        r"\midrule",
    ]
    for _, row in clfi.iterrows():
        tint = GROUP_TINTS.get(row["model_group"], "FFFFFF")
        code = GROUP_CODE.get(row["model_group"], "?")
        lines.append(
            f"  \\rowcolor[HTML]{{{tint}}} {row['model']} & {code} & "
            f"{row['clfi']:.3f} & {row['mean_dfg']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = TABLES_DIR / "table1_clfi.tex"
    path.write_text("\n".join(lines))
    print(f"  → {path}")

    # Table 2: SS breakdown by model × language
    model_lang = metrics["model_lang"]
    pivot = model_lang.pivot(index="model", columns="language", values="mean_ss")

    lang_order = sorted(LANGUAGES.keys(),
                       key=lambda x: (RESOURCE_ORDER.index(LANGUAGES[x]["resource"]), x))
    pivot = pivot.reindex(columns=[c for c in lang_order if c in pivot.columns])

    lang_headers = " & ".join([f"\\textbf{{{LANGUAGES[c]['name'][:3]}}}" for c in pivot.columns])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Stereotype Score (SS) by Model and Language. The value "
        r"closest to ideal neutrality (0.500) in each language is highlighted. "
        r"Values $<0.480$ indicate strong anti-stereotype bias (highlighted green).}",
        r"\label{tab:ss_matrix}",
        r"\begin{tabular}{l" + "c" * len(pivot.columns) + "}",
        r"\toprule",
        f"\\textbf{{Model}} & {lang_headers} \\\\",
        r"\midrule",
    ]
    # Per column (language): bold the value closest to 0.5 neutrality;
    # separately, cellcolor green any value < 0.48 (strong anti-stereotype).
    # These are independent checks, not mutually exclusive or ordered --
    # matches the scheme actually used in the released table.
    closest_per_lang = {
        lang: (pivot[lang] - 0.5).abs().idxmin() for lang in pivot.columns
    }
    for model in pivot.index:
        vals = []
        for lang in pivot.columns:
            v = pivot.loc[model, lang]
            if pd.isna(v):
                vals.append("--")
                continue
            text = f"{v:.3f}"
            if model == closest_per_lang[lang]:
                text = f"\\textbf{{{text}}}"
            if v < 0.48:
                text = f"\\cellcolor{{green!15}}{text}"
            vals.append(text)
        lines.append(f"  {model} & {' & '.join(vals)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    path = TABLES_DIR / "table2_ss_matrix.tex"
    path.write_text("\n".join(lines))
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze multilingual bias audit results")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()

    # Load results
    results_path = RESULTS_DIR / "all_results.csv"
    assert results_path.exists(), (
        f"Results not found at {results_path}. Run 02_run_audit.py first."
    )
    df = pd.read_csv(results_path)

    print("=" * 60)
    print("  MULTILINGUAL BIAS AUDIT — ANALYSIS")
    print("=" * 60)
    print(f"  Results loaded: {len(df)} rows")
    print(f"  Models: {df['model'].nunique()} | Languages: {df['language'].nunique()}")

    # ── Compute Metrics ───────────────────────────────────────────────────────
    metrics = compute_full_metrics(df)

    # Print key results
    clfi = metrics["clfi"]
    print(f"\n{'═'*60}")
    print("  CROSS-LINGUAL FAIRNESS INDEX (CLFI) RANKINGS")
    print(f"{'═'*60}")
    for _, row in clfi.iterrows():
        bar = "█" * int(row["clfi"] * 30)
        print(f"  {row['model']:>18} ({row['model_group_label']:>20}) "
              f"CLFI={row['clfi']:.3f} {bar}")

    # Group averages
    print(f"\n  Group Averages:")
    for group, ci_data in metrics["group_cis"].items():
        ci_str = ""
        if ci_data["ci_low"] is not None:
            ci_str = f" [{ci_data['ci_low']:.3f}, {ci_data['ci_high']:.3f}]"
        print(f"    {group:>20}: CLFI = {ci_data['mean']:.3f}{ci_str}")

    # ── Regression ────────────────────────────────────────────────────────────
    if not args.figures_only and not args.tables_only:
        reg_results = run_regression(df)
        if reg_results:
            reg_path = RESULTS_DIR / "regression_results.json"
            with open(reg_path, "w") as f:
                json.dump(reg_results, f, indent=2)
            print(f"\n  Regression saved → {reg_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.tables_only:
        print(f"\n{'═'*60}")
        print("  GENERATING FIGURES")
        print(f"{'═'*60}")

        plot_heatmap(metrics["model_lang"])
        plot_clfi_radar(metrics["clfi"], df)
        plot_dfg_by_resource(metrics["dfg"])
        plot_layer_comparison(df)
        plot_ss_distribution(df)

    # ── Tables ────────────────────────────────────────────────────────────────
    if not args.figures_only:
        generate_tables(metrics)

    # ── Save metrics summary ──────────────────────────────────────────────────
    summary = {
        "n_models": int(df["model"].nunique()),
        "n_languages": int(df["language"].nunique()),
        "n_prompts": int(len(df)),
        "clfi_rankings": metrics["clfi"][["model", "model_group_label", "clfi", "mean_dfg"]].to_dict("records"),
        "group_averages": metrics["group_cis"],
    }
    summary_path = RESULTS_DIR / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*60}")
    print("  ANALYSIS COMPLETE")
    print(f"{'═'*60}")
    print(f"  Figures → {FIGURES_DIR}")
    print(f"  Tables  → {TABLES_DIR}")
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
