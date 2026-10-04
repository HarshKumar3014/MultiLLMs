#!/usr/bin/env python3
"""
13_paper_figures.py — Every number and figure in the NAACL paper (CPU)
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
    N["lms_by_lang"] = a.groupby("language")["lm_score"].mean().to_dict()
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

    # Fig 1 — the gap per language: one row per language, one dot per model
    #          (filled = drop survives the permutation test with FDR, hollow = not)
    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    fl = float(np.median(floors.groupby("model")["floor"].mean()))
    ax.axvspan(-fl, fl, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
    bl = by_lang.set_index("language")
    for i, l in enumerate(order):
        r = bl.loc[l]
        ax.barh(i, r["mean_d"], height=0.62, color="#cde2fb", lw=0, zorder=1)
        ax.hlines(i, r["lo"], r["hi"], color=BLUE_D, lw=1.6, zorder=3)
        ax.vlines(r["mean_d"], i - 0.31, i + 0.31, color=BLUE_D, lw=1.6, zorder=3)
        c = cells[cells["language"] == l]
        yj = i + (RNG.random(len(c)) - .5) * 0.42
        sig_ = c["q_bh"].to_numpy() < .05
        ax.scatter(c["mean_d"][sig_], yj[sig_], s=13, color=BLUE, lw=0.5, ec="white", zorder=4)
        ax.scatter(c["mean_d"][~sig_], yj[~sig_], s=13, facecolor="white", ec=MUTED, lw=0.8, zorder=4)
        ax.text(0.0135, i, f"{int(r['n_sig'])}/10", va="center", ha="left", fontsize=7, color=INK)
    ax.text(0.0135, -0.85, "reliable", va="center", ha="left", fontsize=6.8, color=INK2)
    ax.text(0, -0.85, "noise floor", ha="center", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(order)), [LANG_NAME[l] for l in order])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(order) - 0.45, -1.25)
    ax.set_xlim(-0.062, 0.013)
    ax.set_xticks([-0.06, -0.04, -0.02, 0])
    ax.set_xlabel("Change in stereotype score vs. English")
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    h1 = ax.scatter([], [], s=13, color=BLUE, ec="white", lw=0.5, label="model, reliable drop")
    h2 = ax.scatter([], [], s=13, facecolor="white", ec=MUTED, lw=0.8, label="model, not reliable")
    h3 = matplotlib.patches.Patch(color="#cde2fb", label="average of 10 models")
    ax.legend(handles=[h3, h1, h2], loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, frameon=False,
              fontsize=6.5, handletextpad=0.25, columnspacing=0.8, borderaxespad=0)
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

    # Fig 3 — carry-over vs. gap: weaker carry-over goes with a bigger drop
    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    cm = cells.set_index(["model", "language"])["mean_d"]
    pm = npaired.groupby(["model", "pivot"])["d"].mean()
    psl = npaired.groupby(["model", "pivot"]).apply(lambda g: slope(g["x"], g["y"]), include_groups=False)
    ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
    ax.scatter(sl_model.values, cm.reindex(sl_model.index).values, s=9, color=MUTED, alpha=.55, lw=0, zorder=2)
    ax.scatter(psl.values, pm.reindex(psl.index).values, s=11, facecolor="white", ec=MUTED, lw=0.7, zorder=2)
    N["corr_carryover_gap_cells"] = float(np.corrcoef(sl_model.values, cm.reindex(sl_model.index).values)[0, 1])
    pmean = npaired.groupby("pivot")["d"].mean()
    for p, val in N["slope_by_pivot"].items():
        ax.scatter(val, pmean[p], s=34, marker="D", color=INK2, ec="white", lw=0.6, zorder=4)
    px = np.mean(list(N["slope_by_pivot"].values()))
    ax.text(px, 0.0105, "English reworded", ha="center", va="bottom", fontsize=6.8, color=INK2)
    offs = {"es": (6, 3), "fr": (7, 4), "zh-CN": (7, -7), "ar": (6, 0), "ko": (-6, 3), "sw": (6, -3),
            "hi": (-6, -4)}
    for l in order:
        x_, y_ = N["slope_by_lang"][l], bl.loc[l, "mean_d"]
        ax.scatter(x_, y_, s=40, color=BLUE, ec="white", lw=0.7, zorder=5)
        dx, dy = offs.get(l, (5, 0))
        ax.annotate(LANG_NAME[l], (x_, y_), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    ha="left" if dx > 0 else "right", va="center", color=INK, zorder=6)
    ax.text(0.98, 0.04, f"r = {N['corr_carryover_gap_cells']:.2f} across\n70 model–language pairs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6, color=INK2, linespacing=0.95)
    hs = [ax.scatter([], [], s=40, color=BLUE, ec="white", lw=0.7, label="language (all models)"),
          ax.scatter([], [], s=9, color=MUTED, alpha=.55, lw=0, label="one model, one language"),
          ax.scatter([], [], s=11, facecolor="white", ec=MUTED, lw=0.7, label="one model, English reworded")]
    ax.legend(handles=hs, loc="upper left", frameon=False, fontsize=6.3, handletextpad=0.2, borderaxespad=0.2,
              labelspacing=0.3)
    ax.set_xlabel("Carry-over of English preference (slope)")
    ax.set_ylabel("Change in stereotype score\nvs. English")
    ax.set_xlim(-0.1, 0.95); ax.set_ylim(-0.066, 0.024)
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_carryover_vs_gap")

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
    slm = sl_model.groupby(level="model").mean()
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"\textbf{Model} & \textbf{Gap} & \textbf{Floor} & \textbf{Sig.} & \textbf{Carry-over} \\",
             r"\midrule"]
    for _, r in boot.sort_values("mean_dfg").iterrows():
        m = r["model"]
        star = r"$^{\ast}$" if r["diff_lo"] > 0 else ""
        lines.append(f"{SHORT[m]} & {r['mean_dfg']:.3f}{star} & {r['floor']:.3f} & {sig[m]}/7 & {slm[m]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Per model. \textbf{Gap}: average distance between a language's mean stereotype score and "
              r"English's (DFG, averaged over the seven languages). \textbf{Floor}: the same distance between English "
              r"and reworded English. $^{\ast}$: gap exceeds the model's own floor (95\% bootstrap interval of the "
              r"difference above zero). \textbf{Sig.}: languages whose shift survives a permutation test with "
              r"false-discovery correction. \textbf{Carry-over}: slope of translated on English scores (1 = full).}",
              r"\label{tab:models}", r"\end{table}"]
    (TABLES_DIR / "tab_models.tex").write_text("\n".join(lines) + "\n")
    json.dump(N, open(OUT / "paper_numbers.json", "w"), indent=1, default=float)  # incl. figure-time numbers

    print(json.dumps({k: N[k] for k in ["slope_cross", "slope_para", "slope_by_lang", "slope_by_pivot",
                                         "agree_by_lang", "agree_by_pivot", "shrink_predicted_d", "observed_mean_d",
                                         "subsets", "spike_in", "priming", "power", "corr_cross", "corr_para",
                                         "rank_spearman_binary_vs_soft"]}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
