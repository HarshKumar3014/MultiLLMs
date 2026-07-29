#!/usr/bin/env python3
"""
06_noise_floor_figure.py — The paper's central figure
========================================================
Forest plot: the standard chart for "is this estimate distinguishable from
a reference/null" (meta-analysis / clinical-trial convention -- point
estimate + 95% CI per row, plotted against a reference zone/line for the
null). Here: each model's mean DFG with a bootstrap 95% CI (resampled over
the 7 non-English languages), against a shaded zone for the measured
within-language noise-floor range. Styled as an editorial/report-style
figure (soft panel, zebra rows, direct value labels) rather than a bare
matplotlib default.

Usage:
    python 06_noise_floor_figure.py
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
    "font.size": 11.5,
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
NOISE_ZONE_COLOR = "#D99A2B"   # warm amber "caution zone" -- status color, not a categorical hue
PANEL_BG = "#FAFAF8"
ZEBRA_COLOR = "#EFEEEA"
TEXT_MUTED = "#6B6B6B"


def dfg_per_language(df: pd.DataFrame, model: str) -> np.ndarray:
    """Per-language DFG values for one model (7 non-English languages)."""
    sub = df[df["model"] == model]
    en_ss = sub[sub["language"] == "en"]["stereotype_score"].mean()
    lang_ss = sub.groupby("language")["stereotype_score"].mean()
    lang_ss = lang_ss[lang_ss.index != "en"]
    return (lang_ss - en_ss).abs().values


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple:
    """Bootstrap 95% CI for the mean, resampling the 7 language-level DFG values."""
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    return np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)


def main():
    results_path = RESULTS_DIR / "all_results.csv"
    noise_path = RESULTS_DIR / "noise_floor_summary.csv"
    assert results_path.exists(), f"Missing {results_path}"
    assert noise_path.exists(), f"Missing {noise_path} -- run 05_noise_floor.py first"

    df = pd.read_csv(results_path)
    noise = pd.read_csv(noise_path).set_index("model")

    models_all = df["model"].unique().tolist()
    per_lang = {m: dfg_per_language(df, m) for m in models_all}
    mean_dfg = {m: v.mean() for m, v in per_lang.items()}
    cis = {m: bootstrap_ci(v) for m, v in per_lang.items()}

    models = sorted(models_all, key=lambda m: mean_dfg[m])
    groups = [MODEL_REGISTRY[m]["group"] for m in models]
    colors = [GROUP_COLORS[g] for g in groups]

    noise_lo = noise["median_dfg_noise"].min()
    noise_hi = noise["median_dfg_noise"].max()
    noise_grand_median = float(noise["median_dfg_noise"].median())

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(PANEL_BG)
    y = np.arange(len(models))
    n = len(models)

    # Zebra striping for row legibility
    for yi in range(n):
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color=ZEBRA_COLOR, zorder=0, linewidth=0)

    x_max = noise_hi * 1.35

    # Reference zone: measured within-language noise-floor range, a warm
    # "caution" tone (status color, not a categorical hue -- distinguished
    # by position and legend text, not by hue-discrimination against the
    # group colors).
    ax.axvspan(noise_lo, noise_hi, facecolor=NOISE_ZONE_COLOR, alpha=0.18, zorder=1)
    ax.axvline(noise_grand_median, color=NOISE_ZONE_COLOR, linestyle=(0, (5, 3)),
               linewidth=1.6, zorder=2, alpha=0.9)

    # Forest-plot rows: point estimate + 95% CI
    for yi, m in enumerate(models):
        lo, hi = cis[m]
        ax.plot([lo, hi], [yi, yi], color=colors[yi], linewidth=2.2, alpha=0.9, zorder=3,
                 solid_capstyle="round")
        ax.plot([lo, lo], [yi - 0.14, yi + 0.14], color=colors[yi], linewidth=2.2, zorder=3)
        ax.plot([hi, hi], [yi - 0.14, yi + 0.14], color=colors[yi], linewidth=2.2, zorder=3)
        ax.plot(mean_dfg[m], yi, "o", color=colors[yi], markersize=10.5,
                 markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        ax.text(hi + x_max * 0.018, yi, f"{mean_dfg[m]:.3f}", va="center", ha="left",
                 fontsize=9.5, color=TEXT_MUTED, zorder=4)

    ax.text(noise_grand_median, n - 0.15, "noise floor", va="bottom", ha="center",
             fontsize=9, color=NOISE_ZONE_COLOR, fontweight="bold", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_ylim(-0.5, n - 0.1)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Deployment Fairness Gap (DFG), mean $\\pm$ 95% bootstrap CI",
                  fontsize=11, color=TEXT_MUTED)
    ax.set_title("No model's DFG clears the within-language noise floor",
                 fontsize=15, fontweight="bold", pad=28, loc="left")
    ax.text(0, 1.03, "Point = observed mean DFG · bar = 95% CI · shaded zone = measured noise-floor range",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_MUTED, ha="left")

    ax.grid(axis="x", alpha=0.5, zorder=0, color="white", linewidth=1.2)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="both", length=0)

    group_handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=GROUP_COLORS[g],
                                 markeredgecolor="white", markersize=9, label=GROUP_LABELS[g])
                      for g in GROUP_COLORS]
    zone_handle = mpatches.Patch(facecolor=NOISE_ZONE_COLOR, alpha=0.18, label="Noise-floor range")
    ax.legend(handles=group_handles + [zone_handle], loc="lower right",
              fontsize=9, framealpha=0.95, edgecolor="#DDDDDD", ncol=1)

    plt.tight_layout()
    path = FIGURES_DIR / "fig0_noise_floor_headline.pdf"
    fig.savefig(path, facecolor="white")
    fig.savefig(path.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    print(f"✓ Saved → {path}")


if __name__ == "__main__":
    main()
