#!/usr/bin/env python3
"""
08_robustness_figure.py — Drop-BLOOMZ robustness figure (Appendix)

Generates fig7_robustness: group-level mean CLFI computed with all models
vs. with BLOOMZ-7B excluded, showing that the apparent Group B gap is
carried by a single model.

This figure previously shipped in the repository with no generating script,
i.e. it could not be regenerated from the released data. This script closes
that gap.

Descriptive only: per the paper's noise-floor result, none of the
group-level differences plotted here is distinguishable from within-language
measurement noise, and the figure should not be read as establishing a
curation effect.

Usage:
    python 08_robustness_figure.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import RESULTS_DIR, FIGURES_DIR, MODEL_REGISTRY

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})

GROUP_COLORS = {
    "A_english_centric": "#3B5FA0",
    "B_multilingual_native": "#3F9142",
    "C_regional_centric": "#C0392B",
}
GROUP_LABELS = {
    "A_english_centric": "English-Centric",
    "B_multilingual_native": "Multilingual-Native",
    "C_regional_centric": "Regional-Centric",
}
PANEL_BG = "#FAFAF8"
TEXT_MUTED = "#6B6B6B"
EXCLUDED_MODEL = "bloomz-7b"


def clfi_per_model(df: pd.DataFrame) -> pd.Series:
    en = df[df.language == "en"].groupby("model")["stereotype_score"].mean().rename("ss_en")
    lang = (df.groupby(["model", "language"])["stereotype_score"].mean()
            .reset_index().merge(en, on="model"))
    lang["dfg"] = (lang["stereotype_score"] - lang["ss_en"]).abs()
    mean_dfg = lang[lang.language != "en"].groupby("model")["dfg"].mean()
    return (1 - 2 * mean_dfg).clip(0, 1)


def main():
    results_path = RESULTS_DIR / "all_results.csv"
    assert results_path.exists(), f"Missing {results_path}"
    df = pd.read_csv(results_path)

    clfi = clfi_per_model(df)
    group_of = {m: MODEL_REGISTRY[m]["group"] for m in clfi.index}
    groups = list(GROUP_LABELS.keys())

    def group_means(exclude=None):
        out = []
        for g in groups:
            members = [m for m in clfi.index
                       if group_of[m] == g and m != exclude]
            out.append(clfi[members].mean())
        return out

    all_models = group_means()
    without = group_means(exclude=EXCLUDED_MODEL)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PANEL_BG)

    x = np.arange(len(groups))
    width = 0.34
    for i, g in enumerate(groups):
        ax.bar(x[i] - width / 2, all_models[i], width, color=GROUP_COLORS[g],
               alpha=0.9, edgecolor="white", linewidth=0.8, zorder=2)
        ax.bar(x[i] + width / 2, without[i], width, color=GROUP_COLORS[g],
               alpha=0.45, edgecolor="white", linewidth=0.8, hatch="///", zorder=2)
        ax.text(x[i] - width / 2, all_models[i] + 0.0004, f"{all_models[i]:.3f}",
                ha="center", va="bottom", fontsize=8, color=TEXT_MUTED, zorder=3)
        ax.text(x[i] + width / 2, without[i] + 0.0004, f"{without[i]:.3f}",
                ha="center", va="bottom", fontsize=8, color=TEXT_MUTED, zorder=3)

    lo = min(min(all_models), min(without))
    ax.set_ylim(lo - 0.004, max(max(all_models), max(without)) + 0.003)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in groups], fontsize=10)
    ax.set_ylabel("Mean CLFI", color=TEXT_MUTED)
    ax.set_title(f"Excluding {EXCLUDED_MODEL} brings group means to near-parity",
                 fontsize=13.5, fontweight="bold", pad=26, loc="left")
    ax.text(0, 1.04, "Descriptive only: no group difference here clears the noise floor",
            transform=ax.transAxes, fontsize=9, color=TEXT_MUTED, ha="left")
    ax.grid(axis="y", alpha=0.5, color="white", linewidth=1.2, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="both", length=0)

    ax.legend(handles=[
        mpatches.Patch(facecolor="#888888", alpha=0.9, label="All models"),
        mpatches.Patch(facecolor="#888888", alpha=0.45, hatch="///",
                       label=f"Excluding {EXCLUDED_MODEL}"),
    ], loc="lower right", fontsize=9, framealpha=0.95, edgecolor="#DDDDDD")

    plt.tight_layout()
    path = FIGURES_DIR / "fig7_robustness.pdf"
    fig.savefig(path, facecolor="white")
    fig.savefig(path.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    print(f"✓ Saved → {path}")
    for g, a, w in zip(groups, all_models, without):
        print(f"  {GROUP_LABELS[g]:>20}: all={a:.4f}  excl={w:.4f}")


if __name__ == "__main__":
    main()
