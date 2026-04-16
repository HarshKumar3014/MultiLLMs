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
        pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
        center=0.5, vmin=0.35, vmax=0.65,
        linewidths=0.5, linecolor="white",
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
            ax.axhline(y=y, color="black", linewidth=2)
        y += count

    ax.set_title("Stereotype Score by Model × Language\n"
                 "(0.5 = unbiased, >0.5 = pro-stereotype, <0.5 = anti-stereotype)",
                 fontsize=13, pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()

    path = FIGURES_DIR / "fig1_heatmap.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"    → {path}")


def plot_clfi_radar(clfi_df: pd.DataFrame):
    """
    Figure 2: CLFI Radar Chart — compact comparison of cross-lingual fairness.
    """
    print("  📊 Generating CLFI radar chart...")

    models = clfi_df["model"].tolist()
    values = clfi_df["clfi"].tolist()
    groups = clfi_df["model_group"].tolist()

    N = len(models)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot each point colored by group
    for i, (model, val, group) in enumerate(zip(models, values, groups)):
        color = GROUP_COLORS.get(group, "#888888")
        ax.bar(angles[i], val, width=0.4, alpha=0.7, color=color,
               edgecolor=color, linewidth=1.5, label=GROUP_LABELS.get(group, group))

    # Clean up labels and legend
    ax.set_xticks(angles)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("Cross-Lingual Fairness Index (CLFI)\n"
                 "(1.0 = perfect equity across languages)", pad=20)

    # Deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right",
              bbox_to_anchor=(1.3, 1.1), fontsize=10)

    path = FIGURES_DIR / "fig2_clfi_radar.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
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

    # Aggregate: mean DFG per resource level × model group
    agg = (
        plot_df.groupby(["model_group_label", "resource_level"], observed=True)["dfg"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    group_labels = list(GROUP_LABELS.values())
    x = np.arange(len(RESOURCE_ORDER))
    width = 0.25

    for i, group_label in enumerate(group_labels):
        group_data = agg[agg["model_group_label"] == group_label]
        means = [group_data[group_data["resource_level"] == r]["mean"].values
                 for r in RESOURCE_ORDER]
        stds = [group_data[group_data["resource_level"] == r]["std"].values
                for r in RESOURCE_ORDER]
        means = [m[0] if len(m) > 0 else 0 for m in means]
        stds = [s[0] if len(s) > 0 else 0 for s in stds]

        group_key = [k for k, v in GROUP_LABELS.items() if v == group_label][0]
        color = GROUP_COLORS[group_key]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
               color=color, alpha=0.85, label=group_label, edgecolor="white")

    ax.set_xlabel("Language Resource Level")
    ax.set_ylabel("Deployment Fairness Gap (DFG)")
    ax.set_title("Deployment Fairness Gap by Language Resource Level\n"
                 "(higher = less fair treatment vs. English)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([r.capitalize() for r in RESOURCE_ORDER])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = FIGURES_DIR / "fig3_dfg_resource.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
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

    fig, ax = plt.subplots(figsize=(8, 8))

    for _, row in layer_data.iterrows():
        model = row["model"]
        group = MODEL_REGISTRY.get(model, {}).get("group", "?")
        color = GROUP_COLORS.get(group, "#888888")
        ax.scatter(row["A"], row["B"], c=color, s=120, alpha=0.8,
                   edgecolors="white", linewidth=1.5, zorder=3)
        ax.annotate(model, (row["A"], row["B"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)

    # Reference line (equal bias in both layers)
    lims = [0.35, 0.65]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, zorder=1)
    ax.axhline(0.5, color="gray", alpha=0.2, linewidth=0.8)
    ax.axvline(0.5, color="gray", alpha=0.2, linewidth=0.8)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Layer A — Translated Stereotypes (SS)")
    ax.set_ylabel("Layer B — Culturally-Native Stereotypes (SS)")
    ax.set_title("Translated vs. Culturally-Native Bias Scores\n"
                 "(deviation from diagonal = cultural calibration gap)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    # Legend
    for group_key, label in GROUP_LABELS.items():
        ax.scatter([], [], c=GROUP_COLORS[group_key], s=80, label=label)
    ax.legend(loc="lower right", fontsize=9)

    path = FIGURES_DIR / "fig4_layer_comparison.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
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
    clfi = metrics["clfi"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-Lingual Fairness Index (CLFI) Rankings. "
        r"Higher CLFI indicates more equitable bias treatment across languages.}",
        r"\label{tab:clfi}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Provenance} & \textbf{CLFI} $\uparrow$ & \textbf{Mean DFG} $\downarrow$ \\",
        r"\midrule",
    ]
    for _, row in clfi.iterrows():
        lines.append(
            f"  {row['model']} & {row['model_group_label']} & "
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
        r"\caption{Stereotype Score (SS) by Model and Language. "
        r"Values {>}0.5 indicate pro-stereotype bias (highlighted).}",
        r"\label{tab:ss_matrix}",
        r"\begin{tabular}{l" + "c" * len(pivot.columns) + "}",
        r"\toprule",
        f"\\textbf{{Model}} & {lang_headers} \\\\",
        r"\midrule",
    ]
    for model in pivot.index:
        vals = []
        for lang in pivot.columns:
            v = pivot.loc[model, lang]
            if pd.isna(v):
                vals.append("--")
            elif v > 0.52:
                vals.append(f"\\cellcolor{{red!15}}{v:.3f}")
            elif v < 0.48:
                vals.append(f"\\cellcolor{{green!15}}{v:.3f}")
            else:
                vals.append(f"{v:.3f}")
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
        plot_clfi_radar(metrics["clfi"])
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
