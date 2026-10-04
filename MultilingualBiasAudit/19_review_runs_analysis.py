#!/usr/bin/env python3
"""
19_review_runs_analysis.py — Analyse the GPU runs of 18_review_gpu_runs.py (CPU)
================================================================================
  (1) Within-language carry-over: translated Hindi/Spanish item -> the same
      item reworded in Hindi/Spanish. Compared with English -> reworded English
      (within-language reference) and English -> Hindi/Spanish (cross-language).
  (2) StereoSet scored without its blank context: drop below English, reliable
      cells, and English -> target carry-over, next to the with-context values.
Writes results/reanalysis/review_runs.json and paper/tables/tab_review_runs.tex.
"""

import json
from importlib import import_module

import numpy as np
import pandas as pd

from config import RESULTS_DIR, SCORES_DIR, TABLES_DIR

re9 = import_module("09_reanalysis")
import label_fix

OUT = RESULTS_DIR / "reanalysis"
RNG = np.random.default_rng(41)
ORDER = ["es", "fr", "zh-CN", "ar", "ko", "sw", "hi"]
NAME = {"en": "English", "es": "Spanish", "fr": "French", "zh-CN": "Chinese", "ar": "Arabic", "ko": "Korean",
        "sw": "Swahili", "hi": "Hindi"}


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def main():
    files = sorted((SCORES_DIR / "review_runs").glob("*.csv"))
    assert files, "no results/v2/review_runs/*.csv — run 18_review_gpu_runs.py --score first"
    rr = label_fix.fix_scores(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    prompts, para, df, noise = re9.load_inputs()
    col, cp, _ = re9.audit_collapses(prompts, para)
    a = df[(df["layer"] == "A") & ~df["prompt_id"].isin(col)]
    N = {"models": sorted(rr["model"].unique())}

    # ── (1) within-language rewording ──
    rw = rr[rr["condition"] == "rw_target"][["model", "prompt_id", "language", "stereotype_score"]] \
        .rename(columns={"stereotype_score": "ss_rw"})
    m = rw.merge(a[["model", "prompt_id", "base_prompt_id", "stereotype_score"]], on=["model", "prompt_id"])
    m["x"] = m["stereotype_score"] - .5; m["y"] = m["ss_rw"] - .5; m["d"] = m["ss_rw"] - m["stereotype_score"]
    n = re9.build_noise_paired(noise, cp)
    n["x"] = n["ss_orig"] - .5; n["y"] = n["stereotype_score"] - .5
    p = re9.build_paired(df, col)
    p["x"] = p["ss_en"] - .5; p["y"] = p["stereotype_score"] - .5
    within = {}
    for l in ["hi", "es"]:
        q = m[m["language"] == l]
        if q.empty:
            continue
        common = set(q["base_prompt_id"])
        pe = p[(p["language"] == l) & p["base_prompt_id"].isin(common)]
        ne = n[n["base_prompt_id"].isin(common)]
        within[l] = {"pairs": int(len(q)), "items": int(q["prompt_id"].nunique()),
                     "slope_within": slope(q["x"], q["y"]),
                     "agree_within": float(((q["ss_rw"] > .5) == (q["stereotype_score"] > .5)).mean()),
                     "corr_within": float(np.corrcoef(q["stereotype_score"], q["ss_rw"])[0, 1]),
                     "floor_within": float(q.groupby("model")["d"].mean().abs().median()),
                     "slope_cross_same_items": slope(pe["x"], pe["y"]),
                     "slope_en_rewording_same_items": slope(ne["x"], ne["y"])}
    N["within_language"] = within

    # ── (2) StereoSet without context ──
    nc = rr[rr["condition"] == "nocontext"].copy()
    nc = nc[~nc["prompt_id"].isin(col)]
    en = nc[nc["language"] == "en"][["model", "base_prompt_id", "stereotype_score"]].rename(
        columns={"stereotype_score": "ss_en"})
    pn = nc[nc["language"] != "en"].merge(en, on=["model", "base_prompt_id"])
    pn["d"] = pn["stereotype_score"] - pn["ss_en"]
    pn["x"] = pn["ss_en"] - .5; pn["y"] = pn["stereotype_score"] - .5
    pn["d_bin"] = 0.0  # cell_tests needs it; binary not used here
    cells = re9.cell_tests(pn, RNG)
    ps = p[p["base_prompt_id"].str.startswith("ss_")]
    cells_ctx = re9.cell_tests(ps, RNG)
    N["nocontext"] = {
        "en_mean": float(en["ss_en"].mean()),
        "drop_by_lang": (-cells.groupby("language")["mean_d"].mean()).reindex(ORDER).round(4).to_dict(),
        "reliable_by_lang": cells[cells["q_bh"] < .05].groupby("language").size().reindex(ORDER).fillna(0).astype(int).to_dict(),
        "reliable": int((cells["q_bh"] < .05).sum()), "negative": int((cells["mean_d"] < 0).sum()),
        "slope": slope(pn["x"], pn["y"]),
        "slope_by_lang": {l: slope(g["x"], g["y"]) for l, g in pn.groupby("language")},
        "with_context": {"drop_by_lang": (-cells_ctx.groupby("language")["mean_d"].mean()).reindex(ORDER).round(4).to_dict(),
                         "reliable": int((cells_ctx["q_bh"] < .05).sum()),
                         "slope": slope(ps["x"], ps["y"])},
    }
    json.dump(N, open(OUT / "review_runs.json", "w"), indent=1, default=float)

    # ── table ──
    t = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"\textbf{Change} & \textbf{Slope} & \textbf{Same choice} & \textbf{Items} \\", r"\midrule"]
    for l, w in within.items():
        t.append(f"{NAME[l]} $\\rightarrow$ reworded {NAME[l]} & {w['slope_within']:.2f} & "
                 f"{100*w['agree_within']:.0f}\\% & {w['items']} \\\\")
        t.append(f"English $\\rightarrow$ {NAME[l]} & {w['slope_cross_same_items']:.2f} & -- & {w['items']} \\\\")
    if within:
        w0 = next(iter(within.values()))
        t.append(f"English $\\rightarrow$ reworded English & {w0['slope_en_rewording_same_items']:.2f} & -- & -- \\\\")
    t += [r"\midrule", r"\multicolumn{4}{l}{\emph{StereoSet scored without the blank context}} \\",
          f"Reliable drops & \\multicolumn{{3}}{{l}}{{{N['nocontext']['reliable']}/70 "
          f"(with context: {N['nocontext']['with_context']['reliable']}/70)}} \\\\",
          f"Carry-over slope & \\multicolumn{{3}}{{l}}{{{N['nocontext']['slope']:.2f} "
          f"(with context: {N['nocontext']['with_context']['slope']:.2f})}} \\\\",
          r"\bottomrule", r"\end{tabular}",
          r"\caption{Top: carry-over of item-level preferences within a language (each translated item reworded "
          r"in the same language by an LLM that is not among the audited models) and across languages, on the "
          r"same items. Bottom: StereoSet items scored as sentences on their own, without the template that "
          r"contains the blank.}",
          r"\label{tab:review_runs}", r"\end{table}"]
    (TABLES_DIR / "tab_review_runs.tex").write_text("\n".join(t) + "\n")
    print(json.dumps(N, indent=1, default=float)[:5000])


if __name__ == "__main__":
    main()
