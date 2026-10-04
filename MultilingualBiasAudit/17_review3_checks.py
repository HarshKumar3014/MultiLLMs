#!/usr/bin/env python3
"""
17_review3_checks.py — Third-round reviewer checks (CPU)
========================================================
  (1) Results by source: reliable drops (permutation + BH within source),
      mixed-model drop per source, and the main findings on BBQ + written
      items only (no StereoSet).
  (2) Mixed model with rewording as a condition: stack the item-level change
      for translations (7 languages) and for English rewordings (3 pivots),
      fit  d ~ 0 + C(condition)  with crossed random effects for item and
      model, and test each language against the mean rewording condition.
  (3) Sample for a contrast-preservation judgment (30 items per language).
Writes results/reanalysis/review3_checks.json, data/contrast_judgment_sample.csv
and paper tables.
"""

import itertools
import json
from importlib import import_module

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import RESULTS_DIR, TABLES_DIR, DATA_DIR

re9 = import_module("09_reanalysis")
OUT = RESULTS_DIR / "reanalysis"
RNG = np.random.default_rng(31)
ORDER = ["es", "fr", "zh-CN", "ar", "ko", "sw", "hi"]
NAME = {"es": "Spanish", "fr": "French", "zh-CN": "Chinese", "ar": "Arabic", "ko": "Korean", "sw": "Swahili",
        "hi": "Hindi", "de": "German", "fi": "Finnish", "ja": "Japanese"}


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def source(bid):
    return "StereoSet" if bid.startswith("ss_") else "BBQ" if bid.startswith("bbq_") else "Written"


def mixed(frame, formula):
    f = frame.assign(g=1)
    return smf.mixedlm(formula, f, groups="g",
                       vc_formula={"item": "0 + C(base_prompt_id)", "model": "0 + C(model)"}).fit(method="lbfgs")


def main():
    prompts, para, df, noise = re9.load_inputs()
    col, cp, _ = re9.audit_collapses(prompts, para)
    p = re9.build_paired(df, col)
    p["x"] = p["ss_en"] - .5; p["y"] = p["stereotype_score"] - .5
    p["src"] = p["base_prompt_id"].map(source)
    n = re9.build_noise_paired(noise, cp)
    n["x"] = n["ss_orig"] - .5; n["y"] = n["stereotype_score"] - .5
    n["src"] = n["base_prompt_id"].map(source)
    N = {}

    # ── (1) by source ──
    bys = {}
    for name, q, r in [("StereoSet", p[p["src"] == "StereoSet"], n[n["src"] == "StereoSet"]),
                       ("BBQ", p[p["src"] == "BBQ"], n[n["src"] == "BBQ"]),
                       ("Written", p[p["src"] == "Written"], n[n["src"] == "Written"]),
                       ("BBQ+Written", p[p["src"] != "StereoSet"], n[n["src"] != "StereoSet"])]:
        cells = re9.cell_tests(q, RNG)
        mm = mixed(q[["d", "base_prompt_id", "model"]], "d ~ 1")
        ci = mm.conf_int().loc["Intercept"]
        mm_l = mixed(q[["d", "language", "base_prompt_id", "model"]], "d ~ 0 + C(language)")
        cil = mm_l.conf_int()
        bys[name] = {
            "items": int(q["base_prompt_id"].nunique()),
            "cells_negative": int((cells["mean_d"] < 0).sum()), "reliable": int((cells["q_bh"] < .05).sum()),
            "reliable_binary": int((cells["q_bh_bin"] < .05).sum()),
            "mixed_drop": float(-mm.params["Intercept"]), "mixed_lo": float(-ci[1]), "mixed_hi": float(-ci[0]),
            "mixed_p": float(mm.pvalues["Intercept"]),
            "mixed_by_lang": {l: {"est": float(-mm_l.params[f"C(language)[{l}]"]),
                                  "lo": float(-cil.loc[f"C(language)[{l}]", 1]),
                                  "hi": float(-cil.loc[f"C(language)[{l}]", 0])} for l in ORDER},
            "slope_tr": slope(q["x"], q["y"]), "slope_rw": slope(r["x"], r["y"]),
            "agree_tr": float(((q["stereotype_score"] > .5) == (q["ss_en"] > .5)).mean()),
            "agree_rw": float(((r["stereotype_score"] > .5) == (r["ss_orig"] > .5)).mean()),
        }
    N["by_source"] = bys

    # ── (2) mixed model: translation vs rewording ──
    st = pd.concat([
        p[["d", "language", "base_prompt_id", "model"]].rename(columns={"language": "cond"}),
        n[["d", "pivot", "base_prompt_id", "model"]].rename(columns={"pivot": "cond"}).assign(cond=lambda t: "rw_" + t["cond"]),
    ], ignore_index=True)
    fit = mixed(st, "d ~ 0 + C(cond)")
    params, cov = fit.params, fit.cov_params()
    rw = [f"C(cond)[rw_{v}]" for v in ["de", "fi", "ja"]]
    res = {}
    from scipy import stats
    for l in ORDER:
        c = pd.Series(0.0, index=params.index)
        c[f"C(cond)[{l}]"] = 1.0
        for k in rw:
            c[k] -= 1 / 3
        est = float(c @ params)
        se = float(np.sqrt(c @ cov.loc[params.index, params.index] @ c))
        res[l] = {"diff": -est, "lo": -est - 1.96 * se, "hi": -est + 1.96 * se,
                  "p": float(2 * stats.norm.sf(abs(est / se))),
                  "level": float(-params[f"C(cond)[{l}]"])}
    from statsmodels.stats.multitest import multipletests
    q_ = multipletests([res[l]["p"] for l in ORDER], method="fdr_bh")[1]
    for l, qq in zip(ORDER, q_):
        res[l]["q"] = float(qq)
    N["vs_rewording"] = res
    N["rewording_levels"] = {v: float(-params[f"C(cond)[rw_{v}]"]) for v in ["de", "fi", "ja"]}
    N["vs_rewording_vc"] = {k: float(v) for k, v in zip(fit.model.exog_vc.names, fit.vcomp)}
    N["vs_rewording_resid"] = float(fit.scale)

    # ── (3) contrast-judgment sample ──
    by_id = {q["id"]: q for q in prompts}
    en = {q["base_id"]: q for q in prompts if q["language"] == "en" and q["layer"] == "A"}
    rows = []
    for l in ORDER:
        ids = sorted(set(p.loc[p["language"] == l, "prompt_id"]))
        for pid in RNG.choice(ids, size=30, replace=False):
            t = by_id[pid]; e = en[t["base_id"]]
            rows.append({"language": l, "prompt_id": pid, "source": source(t["base_id"]),
                         "en_stereotype": e["stereotype"], "en_anti": e["anti_stereotype"],
                         "tr_stereotype": t["stereotype"], "tr_anti": t["anti_stereotype"],
                         "judgment": "", "note": ""})
    sample_path = DATA_DIR / "contrast_judgment_sample.csv"
    if not sample_path.exists():   # never overwrite a judged sample
        pd.DataFrame(rows).to_csv(sample_path, index=False)
    j = pd.read_csv(sample_path)
    if j["judgment"].notna().all() and (j["judgment"] != "").all():
        N["contrast_judgment"] = {"by_lang": j.groupby(["language", "judgment"]).size().unstack(fill_value=0)
                                  .to_dict("index"), "total": j["judgment"].value_counts().to_dict()}

    json.dump(N, open(OUT / "review3_checks.json", "w"), indent=1, default=float)

    # ── tables ──
    t = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcccccccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{\textbf{Reliable drops}} & \textbf{Mixed-model drop} & "
         r"\multicolumn{2}{c}{\textbf{Carry-over}} & \multicolumn{2}{c}{\textbf{Same choice}} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"\textbf{Items} & $n$ & soft & binary & [95\% CI] & transl. & rew. & transl. & rew. \\", r"\midrule"]
    for k in ["StereoSet", "BBQ", "Written", "BBQ+Written"]:
        r = bys[k]
        if k == "BBQ+Written":
            t.append(r"\midrule")
        t.append(f"{k.replace('+', ' + ')} & {r['items']} & {r['reliable']}/70 & {r['reliable_binary']}/70 & "
                 f"{r['mixed_drop']:.3f} [${r['mixed_lo']:.3f}$, ${r['mixed_hi']:.3f}$] & {r['slope_tr']:.2f} & "
                 f"{r['slope_rw']:.2f} & {100*r['agree_tr']:.0f}\\% & {100*r['agree_rw']:.0f}\\% \\\\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Results by source of the items. \textbf{Reliable drops}: model--language pairs (of 70) whose "
          r"drop survives the permutation test with false-discovery correction within that source. "
          r"\textbf{Mixed-model drop}: average drop below English with crossed random effects for item and model. "
          r"\textbf{Carry-over} and \textbf{Same choice}: for translations and for English rewordings. The last "
          r"row excludes StereoSet entirely.}",
          r"\label{tab:sources}", r"\end{table*}"]
    (TABLES_DIR / "tab_sources.tex").write_text("\n".join(t) + "\n")

    def f3(x):
        v = f"{abs(x):.3f}"[1:]
        return f"$-${v}" if x < -0.0005 else v
    t = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"\textbf{Language} & \textbf{Drop} & \textbf{Beyond rewording} & $q$ \\", r"\midrule"]
    for l in ORDER:
        r = res[l]
        qs = "$<$.001" if r["q"] < .001 else f"{r['q']:.3f}"[1:]
        t.append(f"{NAME[l]} & {f3(r['level'])} & {f3(r['diff'])} [{f3(r['lo'])}, {f3(r['hi'])}] & {qs} \\\\")
    rl = N["rewording_levels"]
    t += [r"\midrule", f"Reworded & \\multicolumn{{3}}{{l}}{{De {f3(rl['de'])}, Fi {f3(rl['fi'])}, "
          f"Ja {f3(rl['ja'])}}} \\\\", r"\bottomrule", r"\end{tabular}",
          r"\caption{Mixed model of the item-level change from English, $d \sim 0 + \text{condition} + "
          r"(1 \mid \text{item}) + (1 \mid \text{model})$, where the condition is one of the seven translations or "
          r"the three English rewordings. \textbf{Drop}: estimated drop below English. \textbf{Beyond "
          r"rewording}: the language's drop minus the average drop under rewording, with 95\% Wald interval; "
          r"$q$: false-discovery corrected.}",
          r"\label{tab:mixed}", r"\end{table}"]
    (TABLES_DIR / "tab_mixed.tex").write_text("\n".join(t) + "\n")
    print(json.dumps({k: N[k] for k in ["by_source", "vs_rewording", "rewording_levels"]}, indent=1, default=float)[:7000])


if __name__ == "__main__":
    main()
