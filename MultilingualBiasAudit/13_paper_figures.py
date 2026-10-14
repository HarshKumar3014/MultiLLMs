#!/usr/bin/env python3
"""
13_paper_figures.py — Every number and figure in the paper (CPU)
======================================================================
Reads v2 scores through 09_reanalysis.load_inputs (benchmark roles fixed,
collapsed pairs removed) and writes

    results/reanalysis/paper_numbers.json     every number quoted in the text
    paper/figures/fig_*.pdf                   the figures
    paper/tables/tab_*.tex                    the tables

Run 09_reanalysis.py first (it writes the per-cell tests this reuses).
"""

import json
from importlib import import_module

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from config import RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODEL_REGISTRY

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

re9 = import_module("09_reanalysis")
OUT = RESULTS_DIR / "reanalysis"
RNG = np.random.default_rng(7)

# ── style: one primary hue, gray for context, orange only for contrast ──
BLUE, BLUE_D, ORANGE, RED = "#2a78d6", "#1c5cab", "#eb6834", "#e34948"
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#f0efec"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL_W, FULL_W = 3.15, 6.5   # ACL column / page width (in)

LANG_NAME = {"en": "English", "es": "Spanish", "fr": "French", "zh-CN": "Chinese", "ar": "Arabic",
             "ko": "Korean", "hi": "Hindi", "sw": "Swahili"}
SHORT = re9.SHORT
LANG_HDR = re9.LANG_HDR
PIVOT_NAME = {"de": "via German", "fi": "via Finnish", "ja": "via Japanese", "llm": "LLM rewrite"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


# ══════════════════════════════════════════════════════════════════════════════
def main():
    prompts, para, df, noise = re9.load_inputs()
    collapsed, collapsed_para, _ = re9.audit_collapses(prompts, para)
    paired = re9.build_paired(df, collapsed)
    npaired = re9.build_noise_paired(noise, collapsed_para)
    cells = pd.read_csv(OUT / "cell_tests.csv")
    floors = pd.read_csv(OUT / "floor_tests.csv")
    boot = pd.read_csv(OUT / "bootstrap_dfg_minus_floor.csv")
    s09 = json.load(open(OUT / "summary.json"))
    assert s09["scores_version"] == "v2", "run 09 on v2 scores first"
    langs = [l for l in ["es", "fr", "zh-CN", "ar", "ko", "hi", "sw"] if l in set(paired["language"])]
    N = {}

    # ── data description ──
    a = df[(df["layer"] == "A") & ~df["prompt_id"].isin(collapsed)]
    en = a[a["language"] == "en"]
    N["n_models"] = int(df["model"].nunique())
    N["n_probes_en"] = int(en["base_prompt_id"].nunique())
    N["n_by_source"] = en.drop_duplicates("base_prompt_id")["prompt_id"].str.extract(r"^(ss|bbq|gp|gt|rt|re|ac)")[0] \
        .map({"ss": "stereoset", "bbq": "bbq"}).fillna("handcrafted").value_counts().to_dict()
    N["n_pairs"] = int(len(paired))
    N["collapsed"] = s09["collapses"]
    N["label_fix"] = s09["label_fix"]
    N["cells_n_median"] = float(cells["n"].median())

    # ── 1. the gap, per language ──
    by_lang = []
    for l in langs:
        c = cells[cells["language"] == l]
        m = c["mean_d"].to_numpy()
        t = stats.t.ppf(0.975, len(m) - 1) * m.std(ddof=1) / np.sqrt(len(m))
        by_lang.append({"language": l, "mean_d": m.mean(), "lo": m.mean() - t, "hi": m.mean() + t,
                        "n_sig": int((c["q_bh"] < 0.05).sum()), "n_neg": int((m < 0).sum()),
                        "n_sig_bin": int((c["q_bh_bin"] < 0.05).sum())})
    by_lang = pd.DataFrame(by_lang)
    N["gap_by_language"] = by_lang.round(4).set_index("language").to_dict("index")
    N["cells"] = {k: s09["cells"][k] for k in ["n_cells", "n_sig_bh05", "n_sig_bh05_binary", "analytic_null_mean",
                                                 "sigma_d_pooled", "dfg_range"]}
    N["cells_negative"] = int((cells["mean_d"] < 0).sum())
    N["en_mean_ss"] = float(en["stereotype_score"].mean())
    N["en_binary_ss"] = float((en["logprob_stereotype"] > en["logprob_anti_stereotype"]).mean())
    N["floor"] = {k: s09["floor"][k] for k in ["median_floor", "floor_range", "median_floor_by_pivot", "n_sig_bh05",
                                                 "analytic_null_floor_mean"]}
    N["bootstrap"] = {"n_ci_above_0": s09["bootstrap_n_ci_excludes_0"],
                      "n_ci_above_0_binary": s09["bootstrap_binary_n_ci_excludes_0"],
                      "dfg_range": [float(boot["mean_dfg"].min()), float(boot["mean_dfg"].max())]}

    # ── 2. transfer: does a probe's English preference carry over? ──
    paired["x"] = paired["ss_en"] - .5
    paired["y"] = paired["stereotype_score"] - .5
    npaired["x"] = npaired["ss_orig"] - .5
    npaired["y"] = npaired["stereotype_score"] - .5
    N["slope_cross"] = slope(paired["x"], paired["y"])
    N["slope_para"] = slope(npaired["x"], npaired["y"])
    N["slope_by_lang"] = {l: slope(g["x"], g["y"]) for l, g in paired.groupby("language")}
    N["slope_by_pivot"] = {p: slope(g["x"], g["y"]) for p, g in npaired.groupby("pivot")}
    sl_model = paired.groupby(["model", "language"]).apply(lambda g: slope(g["x"], g["y"]), include_groups=False)
    N["slope_model_lang_range"] = [float(sl_model.min()), float(sl_model.max())]
    N["corr_cross"] = float(np.corrcoef(paired["ss_en"], paired["stereotype_score"])[0, 1])
    N["corr_para"] = float(np.corrcoef(npaired["ss_orig"], npaired["stereotype_score"])[0, 1])
    paired["agree"] = (paired["stereotype_score"] > .5) == (paired["ss_en"] > .5)
    npaired["agree"] = (npaired["stereotype_score"] > .5) == (npaired["ss_orig"] > .5)
    N["agree_by_lang"] = paired.groupby("language")["agree"].mean().to_dict()
    N["agree_by_pivot"] = npaired.groupby("pivot")["agree"].mean().to_dict()
    fx = smf.ols("y ~ x", paired).fit()
    N["shrink_intercept"] = float(fx.params["Intercept"])
    N["mean_x"] = float(paired["x"].mean())
    N["shrink_predicted_d"] = float((fx.params["x"] - 1) * paired["x"].mean() + fx.params["Intercept"])
    N["observed_mean_d"] = float(paired["d"].mean())
    ext = a.assign(e=(a["stereotype_score"] - .5).abs()).groupby("language")["e"].mean()
    N["strength_by_lang"] = ext.to_dict()

    # ── 3. what explains it: translation fidelity, competence ──
    v = json.load(open(re9.DATA_DIR / "validation_results.json"))
    paired["sim"] = paired["prompt_id"].map({k: x["similarity"] for k, x in v.items()})
    lms_en = a[a["language"] == "en"][["model", "base_prompt_id", "lm_score"]].rename(columns={"lm_score": "lms_en"})
    pp = paired.merge(lms_en, on=["model", "base_prompt_id"])
    pp["dlms"] = pp["lm_score"] - pp["lms_en"]
    # fluency: BBQ's third option is the "unknown" answer, not an unrelated sentence, so exclude BBQ
    N["lms_by_lang"] = a[~a["base_prompt_id"].str.startswith("bbq")].groupby("language")["lm_score"].mean().to_dict()
    subsets = {"all": pp, "faithful": pp[pp["sim"] >= .9],
               "faithful_and_fluent": pp[(pp["sim"] >= .9) & (pp["dlms"].abs() < .1)]}
    N["subsets"] = {}
    for k, q in subsets.items():
        cm = q.groupby(["model", "language"])["d"].mean()
        N["subsets"][k] = {"n": int(len(q)), "mean_d": float(q["d"].mean()),
                           "cells_negative": int((cm < 0).sum()), "n_cells": int(len(cm)),
                           "slope": slope(q["x"], q["y"]),
                           "by_lang": q.groupby("language")["d"].mean().to_dict()}
    f = smf.ols("d ~ dlms + sim + C(language)", pp).fit(
        cov_type="cluster", cov_kwds={"groups": pp["base_prompt_id"].astype("category").cat.codes})
    N["d_on_competence"] = {"coef_dlms": float(f.params["dlms"]), "p_dlms": float(f.pvalues["dlms"]),
                            "coef_sim": float(f.params["sim"]), "p_sim": float(f.pvalues["sim"])}

    # ── 4. noise sources, side by side ──
    vd = s09["variance_decomposition"]
    N["per_probe"] = {"cross_median_abs": vd["cross_lingual"]["median_abs"],
                      "para_median_abs": vd["paraphrase"]["median_abs"],
                      "ratio": vd["ratio_median_abs_cross_over_para"],
                      "excess_share": vd["excess_share_of_cross_variance"],
                      "models_cross_gt_para": s09["variance_models_cross_gt_para_bh05"]}
    N["extra_pivot"] = s09.get("extra_pivot")
    N["fidelity"] = s09.get("paraphrase_fidelity")

    # ── 5. can the test see a real shift? (a) spike-in into known-null pairs ──
    null_cells = list(npaired.groupby(["model", "pivot"]).groups.items())
    lo_null = (npaired["logprob_stereotype"] - npaired["logprob_anti_stereotype"])
    rows = []
    for delta in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        for rep in range(40):
            k = rng_choice = null_cells[RNG.integers(len(null_cells))]
            ps = {}
            for key, idx in null_cells:
                d = npaired.loc[idx, "d"].to_numpy()
                if key == k[0]:
                    ss = 1 / (1 + np.exp(-(lo_null.loc[idx] + delta)))
                    d = (ss - npaired.loc[idx, "ss_orig"]).to_numpy()
                    shift = float(d.mean() - npaired.loc[idx, "d"].mean())
                ps[key] = re9.sign_flip_p(d, RNG, 2000)
            keys = list(ps)
            q = dict(zip(keys, re9.bh(np.array([ps[x] for x in keys]))))
            rows.append({"delta": delta, "shift": shift, "hit": q[k[0]] < .05,
                         "false_pos": sum(q[x] < .05 for x in keys if x != k[0]) / (len(keys) - 1)})
    spike = pd.DataFrame(rows).groupby("delta").agg(shift=("shift", "mean"), power=("hit", "mean"),
                                                     false_pos=("false_pos", "mean")).reset_index()
    if (OUT / "spike_in_null_200.csv").exists():   # 200 runs per size (15_robustness_checks.py)
        spike = pd.read_csv(OUT / "spike_in_null_200.csv").rename(columns={"fa_rate": "false_pos"})
    spike.to_csv(OUT / "spike_in_null.csv", index=False)
    N["spike_in"] = spike.round(4).to_dict("records")

    # (b) priming prefix
    pc = pd.read_csv(re9.SCORES_DIR / "positive_control" / "positive_control_tests.csv")
    w = pc[pc["comparison"] == "within_language"]
    N["priming"] = {}
    for (lang, cond), g in w.groupby(["language", "condition"]):
        N["priming"][f"{lang}_{cond}"] = {"mean": float(g["mean_d"].mean()), "n_pos": int((g["mean_d"] > 0).sum()),
                                          "n_detected": int(g["detected"].sum()),
                                          "sign_p": float(stats.binomtest(int((g["mean_d"] > 0).sum()), len(g)).pvalue)}

    # ── 6. power with v2 variance ──
    sd = paired["d"].std()
    z = stats.norm.ppf(.975) + stats.norm.ppf(.8)
    N["power"] = {"sigma_d": float(sd), "mde_at_n": float(z * sd / np.sqrt(N["cells_n_median"])),
                  "n_for": {str(dl): int(np.ceil((z * sd / dl) ** 2)) for dl in [0.005, 0.01, 0.02, 0.03, 0.05]}}

    # ── 7. models, groups, rankings ──
    N["clfi"] = s09["clfi"]
    N["korean_reference_rho"] = s09["korean_reference"]["spearman_rho"]
    N["provenance"] = {k: v for k, v in s09["regression"]["provenance_model_level"].items() if k != "per_model_mean_ss"}
    N["regression_terms"] = s09["regression"]["terms"]
    N["layer_b"] = s09["layer_b"]
    N["worst_language"] = s09["worst_language"]
    # bootstrap CI over probes for each model's CLFI rank stability
    N["rank_spearman_binary_vs_soft"] = float(stats.spearmanr(
        cells.groupby("model")["dfg"].mean(), cells.groupby("model")["dfg_bin"].mean()).statistic)

    json.dump(N, open(OUT / "paper_numbers.json", "w"), indent=1, default=float)

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════
    order = by_lang.sort_values("mean_d", ascending=False)["language"].tolist()   # smallest gap first

    # Fig 1 — how much lower than English, per language (plain bars)
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    fl = float(np.median(floors.groupby("model")["floor"].mean()))
    bl = by_lang.set_index("language")
    xs_ = np.arange(len(order))
    drop = -bl.loc[order, "mean_d"].to_numpy()
    lo_, hi_ = -bl.loc[order, "hi"].to_numpy(), -bl.loc[order, "lo"].to_numpy()
    ax.bar(xs_, drop, width=0.62, color=BLUE, lw=0, zorder=2)
    ax.errorbar(xs_, drop, yerr=[drop - lo_, hi_ - drop], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2,
                zorder=3)
    hfl = ax.axhline(fl, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=4, label="noise floor (rewording English)")
    ax.legend(handles=[hfl], loc="upper left", frameon=False, fontsize=6.6, borderaxespad=0.1, handlelength=2.2)
    for i, l in enumerate(order):
        ax.text(i, hi_[i] + 0.0012, f"{int(bl.loc[l, 'n_sig'])}/10", ha="center", va="bottom", fontsize=6.8,
                color=INK)
    ax.text(-0.42, 0.0478, "n/10 above bars: models whose drop is reliable", ha="left", va="top",
            fontsize=6.6, color=INK2)
    ax.set_xticks(xs_, [LANG_NAME[l] for l in order], fontsize=7.2, rotation=25, ha="right",
                  rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 0.052)
    ax.set_ylabel("Drop in stereotype score\nbelow English")
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_gap_by_language")

    # Fig 2 — same probe reworded vs translated (the central picture)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.78, 2.45), sharey=True)
    panels = [(npaired, "ss_orig", "Same item, reworded in English", N["slope_para"]),
              (paired, "ss_en", "Same item, translated", N["slope_cross"])]
    for ax, (d, xcol, title, sl) in zip(axes, panels):
        hb = ax.hexbin(d[xcol], d["stereotype_score"], gridsize=34, extent=(0, 1, 0, 1), bins="log",
                       cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#86b6ef", BLUE_D]),
                       mincnt=1, linewidths=0)
        ax.plot([0, 1], [0, 1], ls=(0, (3, 2)), color=MUTED, lw=0.9)
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, .5 + sl * (xs - .5), color=ORANGE, lw=1.8)
        ax.text(0.04, 0.94, f"slope {sl:.2f}", transform=ax.transAxes, fontsize=8, color=INK, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.85))
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("Stereotype score, original English")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    axes[0].set_ylabel("Stereotype score after change")
    axes[1].text(0.97, 0.06, "dashed: perfect carry-over", transform=axes[1].transAxes, ha="right",
                 fontsize=6.8, color=INK2)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig_transfer_scatter")

    # Fig 3 — (a) same choice after the change? (b) how strong is the preference?
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.1), gridspec_kw={"width_ratios": [1.25, 1]})
    piv_order = ["de", "fi", "ja"]
    labels_a = [PIVOT_NAME[p].split()[-1] for p in piv_order] + [LANG_NAME[l] for l in order]
    vals_a = [N["agree_by_pivot"][p] for p in piv_order] + [N["agree_by_lang"][l] for l in order]
    cols_a = [MUTED] * 3 + [BLUE] * len(order)
    xa = np.r_[np.arange(3), np.arange(len(order)) + 3.6]
    a1.bar(xa, np.array(vals_a) * 100, width=0.68, color=cols_a, lw=0, zorder=2)
    a1.axhline(50, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=3)
    a1.text(xa[-1] + 0.5, 50.4, "chance", ha="left", va="bottom", fontsize=6.6, color=INK2)
    a1.set_xlim(-0.6, xa[-1] + 1.55)
    a1.text(1, 83.5, "English reworded via", ha="center", va="top", fontsize=6.8, color=INK2)
    a1.text(xa[3:].mean(), 83.5, "Translated into", ha="center", va="top", fontsize=6.8, color=BLUE_D)
    for x_, v_ in zip(xa, vals_a):
        a1.text(x_, v_ * 100 + 0.6, f"{v_*100:.0f}", ha="center", va="bottom", fontsize=6.6, color=INK)
    a1.set_xticks(xa, labels_a, fontsize=6.6, rotation=35, ha="right", rotation_mode="anchor")
    a1.set_ylim(40, 84); a1.set_yticks([40, 50, 60, 70, 80]); a1.tick_params(axis="x", length=0)
    a1.set_ylabel("Same choice as\nin English (%)")
    a1.set_title("(a) Does the preference carry over?", loc="left", fontsize=8.5)
    strength = N["strength_by_lang"]
    lab_b = ["English"] + [LANG_NAME[l] for l in order]
    val_b = [strength["en"]] + [strength[l] for l in order]
    cols_b = [ORANGE] + [BLUE] * len(order)
    xb = np.arange(len(lab_b))
    a2.bar(xb, val_b, width=0.68, color=cols_b, lw=0, zorder=2)
    a2.axhline(strength["en"], color=ORANGE, lw=0.9, ls=(0, (4, 2)), zorder=3)
    a2.text(xb[-1] + 0.5, strength["en"] + 0.002, "English", ha="left", va="bottom", fontsize=6.6, color=ORANGE)
    a2.set_xlim(-0.6, xb[-1] + 1.5)
    for x_, v_ in zip(xb, val_b):
        a2.text(x_, v_ + 0.003, f"{v_:.2f}"[1:], ha="center", va="bottom", fontsize=6.6, color=INK,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none") if abs(v_ - strength["en"]) < .012 else None)
    a2.set_xticks(xb, lab_b, fontsize=6.6, rotation=35, ha="right", rotation_mode="anchor")
    a2.set_ylim(0, 0.19); a2.tick_params(axis="x", length=0)
    a2.set_ylabel("Strength of preference\n|score $-$ 0.5|")
    a2.set_title("(b) Is the preference weaker?", loc="left", fontsize=8.5)
    for ax in (a1, a2):
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig_carryover")

    # Fig 4 — can the test see a real shift? (spike-in power curve)
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    gap_lo = by_lang["mean_d"].abs().min(); gap_hi = by_lang["mean_d"].abs().max()
    ax.axvspan(gap_lo, gap_hi, color=BAND, lw=0, zorder=0)
    ax.text((gap_lo + gap_hi) / 2, 1.0, "average gap\nper language", ha="center", va="top", fontsize=6.8,
            color=INK2, linespacing=0.95)
    ax.plot(spike["shift"], spike["power"], "-o", color=BLUE, lw=2, ms=4.5, mec="white", mew=0.7, zorder=3)
    ax.plot(spike["shift"], spike["false_pos"], "-", color=MUTED, lw=1.2, zorder=2)
    ax.text(spike["shift"].iloc[-1], spike["false_pos"].iloc[-1] + 0.04, "false alarms", ha="right", fontsize=6.8,
            color=INK2)
    ax.text(spike["shift"].iloc[4] + 0.004, spike["power"].iloc[4] - 0.06, "shift detected", fontsize=6.8,
            color=BLUE_D, ha="left", va="top")
    ax.set_xlabel("Injected shift in mean stereotype score")
    ax.set_ylabel("Share of runs")
    ax.set_ylim(-0.03, 1.05); ax.set_xlim(0, spike["shift"].max() * 1.03)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_detection")

    # Appendix A — per-model gap vs own floor
    b = boot.sort_values("diff").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    yy = np.arange(len(b))
    ax.axvline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.hlines(yy, b["diff_lo"], b["diff_hi"], color=BLUE, lw=1.6)
    ax.plot(b["diff"], yy, "o", color=BLUE, ms=5, mec="white", mew=0.7)
    ax.set_yticks(yy, [SHORT[m] for m in b["model"]]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Average gap minus model's own noise floor")
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_model_gap_vs_floor")

    # Appendix B — score heatmap (diverging around 0.5)
    ss = pd.read_csv(OUT / "ss_matrix_clean.csv").pivot(index="model", columns="language", values="soft")
    ss = ss[["en"] + order].rename(columns=LANG_NAME).rename(index=SHORT)
    ss = ss.loc[ss["English"].sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(FULL_W * 0.86, 2.5))
    span = float(np.abs(ss.values - .5).max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [BLUE_D, "#86b6ef", "#f0efec", "#f19a95", RED])
    im = ax.imshow(ss.values, cmap=cmap, vmin=.5 - span, vmax=.5 + span, aspect="auto")
    for i in range(ss.shape[0]):
        for j in range(ss.shape[1]):
            ax.text(j, i, f"{ss.values[i, j]:.3f}"[1:], ha="center", va="center", fontsize=7.2, color=INK)
    ax.set_xticks(range(ss.shape[1]), ss.columns)
    ax.xaxis.tick_top()
    ax.set_yticks(range(ss.shape[0]), ss.index)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
    cb.set_label("stereotype score (0.5 = no preference)", fontsize=6.8)
    save(fig, "fig_score_heatmap")

    # ══════════════════════════════════════════════════════════════════════
    # TABLES
    # ══════════════════════════════════════════════════════════════════════
    grp = {m: MODEL_REGISTRY[m]["group_label"] for m in N["clfi"]}
    sig = cells.groupby("model")["q_bh"].apply(lambda q: int((q < .05).sum()))

    # ── Is the item-level signal in other languages systematic or noise? ──
    import itertools
    models_ = sorted(a["model"].unique())

    def xmodel(frame, key, val="stereotype_score"):
        w = frame.pivot_table(index=key, columns="model", values=val)
        return float(np.mean([w[m1].corr(w[m2]) for m1, m2 in itertools.combinations(models_, 2)]))

    agree_rows = {"en": {"scores": xmodel(a[a["language"] == "en"], "base_prompt_id")}}
    for l in order:
        g = paired[paired["language"] == l]
        b_ = np.polyfit(g["x"], g["y"], 1)
        agree_rows[l] = {"scores": xmodel(a[a["language"] == l], "base_prompt_id"),
                         "change": xmodel(g, "base_prompt_id", "d"), "sd_change": float(g["d"].std()),
                         "center": float(0.5 + b_[1])}
    for pv in ["de", "fi", "ja"]:
        g = npaired[npaired["pivot"] == pv]
        b_ = np.polyfit(g["x"], g["y"], 1)
        agree_rows[f"rw_{pv}"] = {"scores": xmodel(noise[noise["pivot"] == pv], "prompt_id"),
                                  "change": xmodel(g, "prompt_id", "d"), "sd_change": float(g["d"].std()),
                                  "center": float(0.5 + b_[1])}
    N["agreement"] = agree_rows
    N["sd_change_translated"] = float(paired["d"].std())
    N["sd_change_reworded"] = float(npaired["d"].std())
    N["null_n100"] = float(npaired["d"].std() * np.sqrt(2 / (np.pi * 100)))
    N["null_n360"] = float(npaired["d"].std() * np.sqrt(2 / (np.pi * N["cells_n_median"])))
    rows = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{lcccc}", r"\toprule",
            r" & \multicolumn{2}{c}{\textbf{Models agree}} & & \\", r"\cmidrule(lr){2-3}",
            r"\textbf{Version} & score & change & \textbf{Spread} & \textbf{Center} \\", r"\midrule",
            f"English & {agree_rows['en']['scores']:.2f} & -- & -- & -- \\\\", r"\midrule"]
    for l in order:
        r_ = agree_rows[l]
        rows.append(f"{LANG_NAME[l]} & {r_['scores']:.2f} & {r_['change']:.2f} & {r_['sd_change']:.2f} & "
                    f"{r_['center']:.3f} \\\\")
    rows.append(r"\midrule")
    for pv in ["de", "fi", "ja"]:
        r_ = agree_rows[f"rw_{pv}"]
        rows.append(f"Reworded via {PIVOT_NAME[pv].split()[-1]} & {r_['scores']:.2f} & {r_['change']:.2f} & "
                    f"{r_['sd_change']:.2f} & {r_['center']:.3f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Are item-level scores outside English systematic or noise? \textbf{Models agree}: average "
             r"correlation, over the 45 pairs of models, of their scores on the same items (\emph{score}) and of the "
             r"change from English (\emph{change}). \textbf{Spread}: standard deviation of the item-level change. "
             r"\textbf{Center}: fitted score of an item on which the model has no preference in English.}",
             r"\label{tab:agreement}", r"\end{table}"]
    (TABLES_DIR / "tab_agreement.tex").write_text("\n".join(rows) + "\n")

    # ── English lean vs gap without sharing items (split halves) ──
    en_items = a[a["language"] == "en"]
    items_ = np.array(sorted(en_items["base_prompt_id"].unique()))
    rs = []
    for _ in range(500):
        half = set(RNG.choice(items_, size=len(items_) // 2, replace=False))
        e_ = en_items[en_items["base_prompt_id"].isin(half)].groupby("model")["stereotype_score"].mean()
        q_ = paired[~paired["base_prompt_id"].isin(half)]
        g_ = q_.groupby(["model", "language"])["d"].mean().abs().groupby(level="model").mean()
        rs.append(np.corrcoef(e_.reindex(g_.index), g_)[0, 1])
    N["lean_gap_r_splithalf"] = {"median": float(np.median(rs)), "p5": float(np.percentile(rs, 5)),
                                 "p95": float(np.percentile(rs, 95))}
    tgt_mean = a[a["language"] != "en"].groupby(["model", "language"])["stereotype_score"].mean()
    N["sd_models_en"] = float(en_items.groupby("model")["stereotype_score"].mean().std())
    N["sd_models_target"] = float(tgt_mean.groupby(level="model").mean().std())
    N["range_models_target"] = [float(tgt_mean.groupby(level="model").mean().min()),
                                float(tgt_mean.groupby(level="model").mean().max())]

    # LLM-rewrite items actually used (after dropping unrecoverable BBQ items)
    N["llm_items_used"] = int(len(set(npaired["prompt_id"]) & set(
        pd.read_csv(re9.DATA_DIR / "human_paraphrases.csv")["prompt_id"])))

    # ── Appendix table: gap for every model and language ──
    rows = [r"\begin{table*}[t]", r"\centering", r"\small",
            r"\begin{tabular}{l" + "c" * len(order) + "c}", r"\toprule",
            r"\textbf{Model} & " + " & ".join(rf"\textbf{{{LANG_NAME[l]}}}" for l in order) +
            r" & \textbf{Noise only} \\", r"\midrule"]
    for m in sorted(cells["model"].unique(), key=lambda m: SHORT[m]):
        c = cells[cells["model"] == m].set_index("language")
        vals = []
        for l in order:
            v_ = f"{-c.loc[l, 'mean_d']:.3f}"
            vals.append(rf"\textbf{{{v_}}}" if c.loc[l, "q_bh"] < 0.05 else v_)
        rows.append(f"{SHORT[m]} & " + " & ".join(vals) + f" & {c['analytic_null'].mean():.3f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{How far each language's average stereotype score falls below English, for every model "
             r"(negative values: above English). Bold: the drop survives the permutation test with "
             r"false-discovery correction. \textbf{Noise only}: the distance from zero expected if the "
             r"item-level changes were pure noise, averaged over languages.}",
             r"\label{tab:gap_matrix}", r"\end{table*}"]
    (TABLES_DIR / "tab_gap_matrix.tex").write_text("\n".join(rows) + "\n")

    # ── Table: per language (the core numbers) ──
    v_sim = pd.Series({k: x["similarity"] for k, x in v.items()})
    lang_of = v_sim.index.str.rsplit("_", n=1).str[-1]
    used = set(a["prompt_id"])
    sim_by_lang = v_sim[v_sim.index.isin(used)].groupby(lang_of[v_sim.index.isin(used)]).mean()
    score_by_lang = a.groupby("language")["stereotype_score"].mean()
    bin_by_lang = (a["logprob_stereotype"] > a["logprob_anti_stereotype"]).groupby(a["language"]).mean()
    rows = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{lccccccccc}", r"\toprule",
            r" & \multicolumn{2}{c}{\textbf{Score}} & \multicolumn{2}{c}{\textbf{Drop vs.\ English}} & "
            r"\multicolumn{2}{c}{\textbf{Reliable drops}} & \multicolumn{2}{c}{\textbf{Carry-over}} & \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            r"\textbf{Language} & soft & binary & mean & 95\% CI & soft & binary & slope & same choice & "
            r"\textbf{Fluency} \\", r"\midrule",
            f"English & {score_by_lang['en']:.3f} & {bin_by_lang['en']:.2f} & -- & -- & -- & -- & -- & -- & "
            f"{N['lms_by_lang']['en']:.2f} \\\\", r"\midrule"]
    for l in order:
        r = bl.loc[l]
        rows.append(f"{LANG_NAME[l]} & {score_by_lang[l]:.3f} & {bin_by_lang[l]:.2f} & {-r['mean_d']:.3f} & "
                    f"[{-r['hi']:.3f}, {-r['lo']:.3f}] & {int(r['n_sig'])}/10 & {int(r['n_sig_bin'])}/10 & "
                    f"{N['slope_by_lang'][l]:.2f} & {100*N['agree_by_lang'][l]:.0f}\\% & "
                    f"{N['lms_by_lang'][l]:.2f} \\\\")
    rows += [r"\midrule",
             f"English reworded & -- & -- & {N['floor']['median_floor']:.3f} & -- & 0/10 & -- & "
             f"{N['slope_para']:.2f} & {100*np.mean(list(N['agree_by_pivot'].values())):.0f}\\% & -- \\\\",
             r"\bottomrule", r"\end{tabular}",
             r"\caption{Results per language, averaged over the ten models. \textbf{Score}: mean stereotype score "
             r"(soft) and share of items on which the stereotyped continuation wins (binary). \textbf{Drop}: how "
             r"far the mean score falls below English (95\% interval across models). \textbf{Reliable drops}: "
             r"models whose drop survives a permutation test with false-discovery correction, using soft or binary "
             r"scores. \textbf{Carry-over}: slope of the translated score on the English score for the same item "
             r"(1 = fully kept), and how often the model picks the same continuation in both versions. "
             r"\textbf{Fluency}: how well the model tells the two candidate continuations from the unrelated one "
             r"(excluding BBQ, whose third option is an ``unknown'' answer). "
             r"Last row: the same measures when English items are only reworded (median over models and the three "
             r"machine rewordings).}",
             r"\label{tab:languages}", r"\end{table*}"]
    (TABLES_DIR / "tab_languages.tex").write_text("\n".join(rows) + "\n")
    N["sim_by_lang"] = sim_by_lang.to_dict()
    N["score_by_lang"] = score_by_lang.to_dict()
    N["bin_by_lang"] = bin_by_lang.to_dict()

    # ── Table: per model (enriched, full width) ──
    en_m = a[a["language"] == "en"]
    en_soft = en_m.groupby("model")["stereotype_score"].mean()
    en_bin = (en_m["logprob_stereotype"] > en_m["logprob_anti_stereotype"]).groupby(en_m["model"]).mean()
    sig_bin = cells.groupby("model")["q_bh_bin"].apply(lambda q: int((q < .05).sum()))
    gshort = {"English-Centric": "English", "Multilingual-Native": "Multiling.", "Regional-Centric": "Regional"}
    rows = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{llcccccccc}", r"\toprule",
            r" & & \multicolumn{2}{c}{\textbf{English score}} & & & & \multicolumn{2}{c}{\textbf{Reliable drops}} "
            r"& \\", r"\cmidrule(lr){3-4}\cmidrule(lr){8-9}",
            r"\textbf{Model} & \textbf{Focus} & soft & binary & \textbf{Gap} & \textbf{Floor} & "
            r"\textbf{Gap $-$ floor [95\% CI]} & soft & binary & \textbf{CLCI} \\", r"\midrule"]
    for _, r in boot.sort_values("mean_dfg").iterrows():
        m = r["model"]
        star = r"$^{\ast}$" if r["diff_lo"] >= 0.0005 else ""
        rows.append(f"{SHORT[m]} & {gshort[grp[m]]} & {en_soft[m]:.3f} & {en_bin[m]:.2f} & {r['mean_dfg']:.3f} & "
                    f"{r['floor']:.3f} & ${r['diff']:+.3f}$ [${r['diff_lo']:+.3f}$, ${r['diff_hi']:+.3f}$]{star} & "
                    f"{sig[m]}/7 & {sig_bin[m]}/7 & {N['clfi'][m]:.3f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Results per model, ordered by gap. \textbf{Gap}: mean distance between a language's average "
             r"score and English's, over the seven languages (the cross-lingual score shift, CSS). \textbf{Floor}: the same distance between "
             r"English and reworded English. \textbf{Gap $-$ floor}: with a 95\% bootstrap interval over items; "
             r"$^{\ast}$ marks intervals above zero. \textbf{Reliable drops}: languages (of seven) whose drop "
             r"survives the permutation test with false-discovery correction. \textbf{CLCI} $= 1 - $ mean gap. "
             r"These numbers describe consistency with English, not fairness, and should not be used to rank "
             r"models for deployment.}",
             r"\label{tab:models}", r"\end{table*}"]
    (TABLES_DIR / "tab_models.tex").write_text("\n".join(rows) + "\n")
    N["en_soft_by_model"] = en_soft.to_dict(); N["en_bin_by_model"] = en_bin.to_dict()

    # ── Table: noise floor by rewording method ──
    fid = pd.read_csv(OUT / "paraphrase_fidelity.csv")
    fid = fid[fid["prompt_id"].isin(set(npaired["prompt_id"]))]
    llm_files = sorted((re9.SCORES_DIR / "noise_floor").glob("*_noise_floor_llm.csv"))
    import label_fix
    llm = label_fix.fix_scores(pd.concat([pd.read_csv(f) for f in llm_files], ignore_index=True))
    sub = set(llm["prompt_id"])
    nz_sub = pd.concat([noise[noise["prompt_id"].isin(sub)], llm], ignore_index=True)
    np_sub = re9.build_noise_paired(nz_sub, collapsed_para)
    np_sub["x"] = np_sub["ss_orig"] - .5; np_sub["y"] = np_sub["stereotype_score"] - .5
    np_sub["agree"] = (np_sub["stereotype_score"] > .5) == (np_sub["ss_orig"] > .5)
    hp = json.load(open(OUT / "summary.json")).get("extra_pivot", {})
    hcsv = pd.read_csv(re9.DATA_DIR / "human_paraphrases.csv")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cos_llm = []
    for f in ["context", "stereotype", "anti_stereotype"]:
        e1 = enc.encode(hcsv[f].tolist(), normalize_embeddings=True); e2 = enc.encode(hcsv[f"para_{f}"].tolist(), normalize_embeddings=True)
        cos_llm += list((e1 * e2).sum(1))
    cp_count = pd.Series([pv for _, pv in collapsed_para]).value_counts()
    floor_med = floors.groupby("pivot")["floor"].median()
    rows = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.2pt}",
            r"\begin{tabular}{lccccc}", r"\toprule",
            r"\textbf{Rewording} & \textbf{Sim.} & \textbf{Same} & \textbf{Floor} & \textbf{Slope} & "
            r"\textbf{Agree} \\", r"\midrule",
            r"\multicolumn{6}{l}{\emph{All items}} \\"]
    for pv in ["de", "fi", "ja"]:
        ff = fid[fid["pivot"] == pv]
        rows.append(f"via {PIVOT_NAME[pv].split()[-1]} & {ff['cos'].mean():.2f} & {100*ff['identical'].mean():.0f}\\% & "
                    f"{floor_med[pv]:.3f} & {N['slope_by_pivot'][pv]:.2f} & {100*N['agree_by_pivot'][pv]:.0f}\\% \\\\")
    rows.append(r"\multicolumn{6}{l}{\emph{92-item subset (LLM rewrites)}} \\")
    for pv in ["de", "fi", "ja", "llm"]:
        g = np_sub[np_sub["pivot"] == pv]
        fl_ = g.groupby("model")["d"].mean().abs().median()
        name = "LLM rewrite" if pv == "llm" else f"via {PIVOT_NAME[pv].split()[-1]}"
        sim_ = f"{np.mean(cos_llm):.2f}" if pv == "llm" else f"{fid[(fid['pivot']==pv) & fid['prompt_id'].isin(sub)]['cos'].mean():.2f}"
        same_ = "0\\%" if pv == "llm" else f"{100*fid[(fid['pivot']==pv) & fid['prompt_id'].isin(sub)]['identical'].mean():.0f}\\%"
        rows.append(f"{name} & {sim_} & {same_} & {fl_:.3f} & {slope(g['x'], g['y']):.2f} & "
                    f"{100*g['agree'].mean():.0f}\\% \\\\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{How the noise floor depends on the rewording. \textbf{Sim.}: embedding similarity to the "
             r"original; \textbf{Same}: share of fields returned unchanged; \textbf{Floor}: median over models of the "
             r"gap between English and its rewording; \textbf{Slope}, \textbf{Agree}: carry-over of the English "
             r"preference, as in \Cref{tab:languages}.}",
             r"\label{tab:floor}", r"\end{table}"]
    (TABLES_DIR / "tab_floor.tex").write_text("\n".join(rows) + "\n")

    # (Table "controls" is written by 16_controls_and_sources.py: nested filter + paired bootstrap)
    rb = json.load(open(OUT / "robustness_checks.json"))

    # ── Table: summed instead of per-token log-probability ──
    sm = rb["summed"]
    rows = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lcccccc}", r"\toprule",
            r" & \multicolumn{2}{c}{\textbf{Drop}} & \multicolumn{2}{c}{\textbf{Reliable}} & "
            r"\multicolumn{2}{c}{\textbf{Slope}} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"\textbf{Language} & avg. & sum & avg. & sum & avg. & sum \\", r"\midrule"]
    for l in order:
        rows.append(f"{LANG_NAME[l]} & {-bl.loc[l, 'mean_d']:.3f} & {sm['drop_by_lang'][l]:.3f} & "
                    f"{int(bl.loc[l, 'n_sig'])} & {sm['sig_by_lang'][l]} & {N['slope_by_lang'][l]:.2f} & "
                    f"{sm['slope_by_lang'][l]:.2f} \\\\")
    rows += [r"\midrule", f"All & {-N['observed_mean_d']:.3f} & -- & {N['cells']['n_sig_bh05']} & {sm['n_sig_bh']} & "
             f"{N['slope_cross']:.2f} & {sm['slope']:.2f} \\\\", r"\bottomrule", r"\end{tabular}",
             r"\caption{Main results with the per-token average log-probability (avg.) and with the summed "
             r"log-probability (sum), which does not dilute a one-word contrast in a long continuation. "
             r"\textbf{Reliable}: models (of 10; last row: pairs of 70) with a drop that survives the corrected "
             r"permutation test. Summed scores are more extreme, so drops are larger; the drop and the overall "
             r"carry-over hold, while carry-over by language shifts (Swahili falls, Arabic and Hindi rise).}",
             r"\label{tab:summed}", r"\end{table}"]
    (TABLES_DIR / "tab_summed.tex").write_text("\n".join(rows) + "\n")
    N["robustness"] = rb

    # ── Table: what each pipeline error did (from 14_pipeline_ablation.py) ──
    ab = pd.read_csv(OUT / "pipeline_ablation.csv").set_index(["scorer", "labels"])
    rows = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.6pt}",
            r"\begin{tabular}{lcccccc}", r"\toprule",
            r" & \multicolumn{2}{c}{\textbf{English}} & \textbf{Below} & & \multicolumn{2}{c}{\textbf{Slope}} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){6-7}",
            r"\textbf{Errors present} & soft & bin. & \textbf{En.} & \textbf{Rel.} & rew. & trans. \\", r"\midrule"]
    cfg = [("v1", "raw", "All three errors"), ("v2", "raw", "Only label errors"),
           ("v1", "fixed", "Only scorer error"), ("v2", "fixed", "None (ours)")]
    for sc, lb, name in cfg:
        r = ab.loc[(sc, lb)]
        rows.append(f"{name} & {r['en_ss']:.3f} & {r['en_bin']:.2f} & {int(r['neg'])}/70 & {int(r['sig'])}/70 & "
                    f"{r['slope_rw']:.2f} & {r['slope_tr']:.2f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{The same analysis under each combination of errors. \textbf{English}: mean soft and binary "
             r"score on English items. \textbf{Below En.}: model--language pairs scoring below English; "
             r"\textbf{Rel.}: of these, reliable after correction. \textbf{Slope}: carry-over for rewordings and "
             r"translations. With the label errors, every gap looks like noise.}",
             r"\label{tab:pipeline}", r"\end{table}"]
    (TABLES_DIR / "tab_pipeline.tex").write_text("\n".join(rows) + "\n")
    N["ablation"] = ab.reset_index().to_dict("records")

    # ── Table: one item across languages ──
    ex_models = ["qwen2.5-7b", "aya-23-8b", "mistral-7b"]
    ex = a[(a["base_prompt_id"] == "ac_003") & a["model"].isin(ex_models)].pivot_table(
        index="model", columns="language", values="stereotype_score")
    cols_ex = ["en", "es", "fr", "zh-CN", "ar", "ko", "hi", "sw"]
    rows = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.4pt}",
            r"\begin{tabular}{l" + "c" * len(cols_ex) + "}", r"\toprule",
            r"\textbf{Model} & " + " & ".join(rf"\textbf{{{LANG_HDR[c]}}}" for c in cols_ex) + r" \\", r"\midrule"]
    fmt = lambda x: (rf"\textbf{{{x:.2f}}}" if x > .5 else f"{x:.2f}")[0:] if not np.isnan(x) else "--"
    for m in ex_models:
        rows.append(f"{SHORT[m]} & " + " & ".join(fmt(ex.loc[m, c]) for c in cols_ex) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{One item in eight languages: \emph{``The elderly woman tried to set up her new "
             r"smartphone.''} followed by \emph{``She was confused by the interface and needed constant help.''} "
             r"(stereotyped) or \emph{``She configured it independently using the online guide.''} Bold: the "
             r"model prefers the stereotyped continuation. Every model leans strongly toward it in English; after "
             r"translation the preference holds in some languages, reverses in others, and differs between "
             r"models.}",
             r"\label{tab:example}", r"\end{table}"]
    (TABLES_DIR / "tab_example.tex").write_text("\n".join(rows) + "\n")
    json.dump(N, open(OUT / "paper_numbers.json", "w"), indent=1, default=float)  # incl. figure-time numbers

    print(json.dumps({k: N[k] for k in ["slope_cross", "slope_para", "slope_by_lang", "slope_by_pivot",
                                         "agree_by_lang", "agree_by_pivot", "shrink_predicted_d", "observed_mean_d",
                                         "subsets", "spike_in", "priming", "power", "corr_cross", "corr_para",
                                         "rank_spearman_binary_vs_soft"]}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
