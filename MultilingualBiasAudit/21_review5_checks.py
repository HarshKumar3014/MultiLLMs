#!/usr/bin/env python3
"""
21_review5_checks.py — Fifth-round reviewer analyses (CPU)
==========================================================
  (1) Association between English and changed scores: correlation, SD ratio
      and slope with bootstrap intervals over items; pooled, per language,
      per source, per model, and for each rewording.
  (2) Chance baseline for "same choice": p_en p_l + (1-p_en)(1-p_l).
  (3) Absolute gap vs absolute rewording floor, per language, items as unit.
  (4) BBQ: how often the "unknown" answer (correct in ambiguous contexts)
      outscores both group answers, per language.
  (5) What happens to items that lean stereotyped in English: kept, moved to
      neutral, or reversed, after translation vs after rewording; by source
      and social dimension.
  (6) Drop and carry-over by social dimension.
  (7) Translation error types in the judged sample; drop on preserved vs not.
  (8) Item flow through every stage.
Writes results/reanalysis/review5_checks.json and paper tables.
"""

import json
from importlib import import_module

import numpy as np
import pandas as pd

from config import RESULTS_DIR, SCORES_DIR, TABLES_DIR, DATA_DIR

re9 = import_module("09_reanalysis")
import label_fix

OUT = RESULTS_DIR / "reanalysis"
RNG = np.random.default_rng(61)
ORDER = ["es", "fr", "zh-CN", "ar", "ko", "sw", "hi"]
NAME = {"en": "English", "es": "Spanish", "fr": "French", "zh-CN": "Chinese", "ar": "Arabic", "ko": "Korean",
        "sw": "Swahili", "hi": "Hindi", "de": "German", "fi": "Finnish", "ja": "Japanese"}
LO, HI = 0.45, 0.55   # "leans stereotyped" / "leans counter-stereotyped" thresholds


def source(bid):
    return "StereoSet" if bid.startswith("ss_") else "BBQ" if bid.startswith("bbq_") else "Written"


def dimension(cat):
    c = cat.lower()
    if "gender" in c:
        return "Gender"
    if "race" in c:
        return "Race/ethnicity"
    if "relig" in c:
        return "Religion"
    if "nation" in c:
        return "Nationality"
    if "age" in c:
        return "Age"
    if "profession" in c:
        return "Profession"
    return "Other"


def assoc(x, y):
    r = np.corrcoef(x, y)[0, 1]
    sdr = np.std(y, ddof=1) / np.std(x, ddof=1)
    return r, sdr, r * sdr


def boot_assoc(frame, xcol, ycol, n_boot=1000):
    """Bootstrap over items (base_prompt_id); all pairs of a drawn item come along."""
    items = frame["base_prompt_id"].to_numpy()
    uniq, inv = np.unique(items, return_inverse=True)
    x = frame[xcol].to_numpy(); y = frame[ycol].to_numpy()
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    est = assoc(x, y)
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in RNG.integers(0, len(groups), len(groups))])
        bs.append(assoc(x[idx], y[idx]))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {"r": est[0], "r_lo": lo[0], "r_hi": hi[0], "sd_ratio": est[1], "slope": est[2],
            "slope_lo": lo[2], "slope_hi": hi[2], "n_items": int(len(uniq)), "n_pairs": int(len(x))}


def main():
    prompts, para, df, noise = re9.load_inputs()
    col, cp, _ = re9.audit_collapses(prompts, para)
    cat = {p["base_id"]: p["category"] for p in prompts if p["language"] == "en"}
    p = re9.build_paired(df, col)
    n = re9.build_noise_paired(noise, cp)
    for f in (p, n):
        f["src"] = f["base_prompt_id"].map(source)
        f["dim"] = f["base_prompt_id"].map(cat).map(dimension)
    p["x"], p["y"] = p["ss_en"], p["stereotype_score"]
    n["x"], n["y"] = n["ss_orig"], n["stereotype_score"]
    N = {}

    # ── (1) association with bootstrap intervals ──
    A = {"translated_all": boot_assoc(p, "x", "y"), "reworded_all": boot_assoc(n, "x", "y")}
    for l in ORDER:
        A[f"lang_{l}"] = boot_assoc(p[p["language"] == l], "x", "y", 500)
    for pv in ["de", "fi", "ja"]:
        A[f"pivot_{pv}"] = boot_assoc(n[n["pivot"] == pv], "x", "y", 500)
    for s in ["StereoSet", "BBQ", "Written"]:
        A[f"src_tr_{s}"] = boot_assoc(p[p["src"] == s], "x", "y", 500)
        A[f"src_rw_{s}"] = boot_assoc(n[n["src"] == s], "x", "y", 500)
    for m in sorted(p["model"].unique()):
        A[f"model_tr_{m}"] = boot_assoc(p[p["model"] == m], "x", "y", 300)
        A[f"model_rw_{m}"] = boot_assoc(n[n["model"] == m], "x", "y", 300)
    N["association"] = A

    # ── (2) same-choice chance baseline ──
    a = df[(df["layer"] == "A") & ~df["prompt_id"].isin(col)]
    sc = {}
    for l in ORDER:
        q = p[p["language"] == l]
        pe = (q["ss_en"] > .5).mean(); pl = (q["stereotype_score"] > .5).mean()
        sc[l] = {"observed": float(((q["ss_en"] > .5) == (q["stereotype_score"] > .5)).mean()),
                 "chance": float(pe * pl + (1 - pe) * (1 - pl))}
    for pv in ["de", "fi", "ja"]:
        q = n[n["pivot"] == pv]
        pe = (q["ss_orig"] > .5).mean(); pl = (q["stereotype_score"] > .5).mean()
        sc[pv] = {"observed": float(((q["ss_orig"] > .5) == (q["stereotype_score"] > .5)).mean()),
                  "chance": float(pe * pl + (1 - pe) * (1 - pl))}
    N["same_choice"] = sc

    # ── (3) absolute gap vs absolute floor, items as unit ──
    # per item: mean over models of d (translation) and of d (rewording, mean over pivots);
    # statistic: |mean_items tr| - mean_pivots |mean_items rw_pivot|, bootstrap over items
    rw_piv = n.groupby(["base_prompt_id", "pivot"])["d"].mean().unstack()
    absres = {}
    for l in ORDER:
        tr = p[p["language"] == l].groupby("base_prompt_id")["d"].mean()
        j = rw_piv.join(tr.rename("tr"), how="inner").dropna()
        def stat(J):
            return abs(J["tr"].mean()) - np.mean([abs(J[pv].mean()) for pv in ["de", "fi", "ja"]])
        obs = stat(j)
        bs = [stat(j.iloc[RNG.integers(0, len(j), len(j))]) for _ in range(2000)]
        absres[l] = {"abs_gap": float(abs(j["tr"].mean())),
                     "abs_floor": float(np.mean([abs(j[pv].mean()) for pv in ["de", "fi", "ja"]])),
                     "diff": float(obs), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                     "items": int(len(j))}
    N["abs_vs_floor"] = absres

    # ── (4) BBQ unknown-answer selection ──
    bb = a[a["base_prompt_id"].str.startswith("bbq_")]
    unk = bb["logprob_unrelated"] > bb[["logprob_stereotype", "logprob_anti_stereotype"]].max(axis=1)
    N["bbq_unknown_chosen"] = unk.groupby(bb["language"]).mean().round(3).to_dict()

    # ── (5) fate of items that lean stereotyped in English ──
    def fate(frame, xcol, ycol):
        f = frame[frame[xcol] > HI]
        y = f[ycol]
        return {"n": int(len(f)), "kept": float((y > HI).mean()), "neutral": float(((y >= LO) & (y <= HI)).mean()),
                "reversed": float((y < LO).mean())}
    fates = {"translated": fate(p, "ss_en", "stereotype_score"), "reworded": fate(n, "ss_orig", "stereotype_score")}
    for l in ORDER:
        fates[f"lang_{l}"] = fate(p[p["language"] == l], "ss_en", "stereotype_score")
    for s in ["StereoSet", "BBQ", "Written"]:
        fates[f"src_{s}"] = fate(p[p["src"] == s], "ss_en", "stereotype_score")
        fates[f"src_rw_{s}"] = fate(n[n["src"] == s], "ss_orig", "stereotype_score")
    # and the reverse direction: counter-leaning in English
    f2 = p[p["ss_en"] < LO]
    fates["counter_translated"] = {"n": int(len(f2)), "to_stereo": float((f2["stereotype_score"] > HI).mean())}
    N["fate"] = fates

    # ── (6) by social dimension ──
    dims = {}
    for d_, g in p.groupby("dim"):
        it = -g.groupby("base_prompt_id")["d"].mean().to_numpy()
        bs = it[RNG.integers(0, len(it), (3000, len(it)))].mean(1)
        r = n[n["dim"] == d_]
        dims[d_] = {"items": int(len(it)), "drop": float(it.mean()), "lo": float(np.percentile(bs, 2.5)),
                    "hi": float(np.percentile(bs, 97.5)), "r_tr": float(np.corrcoef(g["x"], g["y"])[0, 1]),
                    "r_rw": float(np.corrcoef(r["x"], r["y"])[0, 1]) if len(r) > 10 else None,
                    "reversed": fate(g, "ss_en", "stereotype_score")["reversed"]}
    N["by_dimension"] = dims

    # ── (7) translation error types in the judged sample ──
    j = pd.read_csv(DATA_DIR / "contrast_judgment_sample.csv")
    def etype(note):
        nt = str(note).lower()
        if "name" in nt:
            return "Name translated as a word"
        if "context" in nt or "->" in nt and ("mover" in nt or "guy" in nt):
            return "Context or noun mistranslated"
        if "person" in nt or "neutral" in nt or "eomma" in nt:
            return "Gender or contrast neutralized"
        return "Contrast word shifted in meaning"
    bad = j[j["judgment"] != "preserved"].copy()
    bad["type"] = bad["note"].map(etype)
    N["error_types"] = bad.groupby(["type"]).size().to_dict()
    N["error_types_by_lang"] = bad.groupby(["language", "type"]).size().unstack(fill_value=0).to_dict("index")
    jp = p.merge(j[["prompt_id", "judgment"]], on="prompt_id")
    N["drop_by_judgment"] = {k: {"pairs": int(len(g)), "drop": float(-g["d"].mean()),
                                 "r": float(np.corrcoef(g["x"], g["y"])[0, 1])} for k, g in
                             jp.assign(k=np.where(jp["judgment"] == "preserved", "preserved", "not preserved"))
                             .groupby("k")}

    # ── (8) item flow ──
    rw = json.load(open(DATA_DIR / "target_rewordings.json"))
    N["flow"] = {"stereoset": 200, "bbq_sampled": 150, "bbq_kept": 125, "written": 37,
                 "english_items": int(a[a["language"] == "en"]["base_prompt_id"].nunique()),
                 "translated": 362 * 7, "collapsed_translations": len([c for c in col if not c.endswith("_en")]),
                 "pairs_per_model": int(len(p) / p["model"].nunique()),
                 "mt_rewordings": 362 * 3, "collapsed_rewordings": len(cp),
                 "llm_rewordings_sampled": 100, "llm_rewordings_used": 92,
                 "target_rewordings_attempted": len(rw),
                 "target_rewordings_passed": sum("context" in v for v in rw.values())}
    json.dump(N, open(OUT / "review5_checks.json", "w"), indent=1, default=float)

    # ── tables ──
    # tab_assoc is written by 22_review6_checks.py (adds preserved-only and within-language rows)

    t = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"\textbf{Dimension} & $n$ & \textbf{Drop} [95\% CI] & $r$ tr. & $r$ rew. & \textbf{Rev.} \\",
         r"\midrule"]
    for d_, r in sorted(dims.items(), key=lambda kv: -kv[1]["items"]):
        rw_ = f"{r['r_rw']:.2f}" if r["r_rw"] is not None else "--"
        f3 = lambda x: (f"$-${abs(x):.3f}"[0:] if x < 0 else f"{x:.3f}").replace("0.", ".")
        t.append(f"{d_.replace('Race/ethnicity','Race/eth.')} & {r['items']} & {f3(r['drop'])} [{f3(r['lo'])}, {f3(r['hi'])}] & {r['r_tr']:.2f} & "
                 f"{rw_} & {100*r['reversed']:.0f}\\% \\\\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Results by social dimension (pooled over sources). \textbf{Drop}: mean drop below English, "
          r"bootstrap over items. $r$: correlation of item scores with English after translation and after "
          r"rewording. \textbf{Rev.}: share of model--item pairs leaning stereotyped in English (score $> 0.55$) "
          r"that lean counter-stereotyped after translation ($< 0.45$).}",
          r"\label{tab:dims}", r"\end{table}"]
    (TABLES_DIR / "tab_dims.tex").write_text("\n".join(t) + "\n")

    fl = N["flow"]
    t = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{p{5.6cm}r}", r"\toprule", r"\textbf{Stage} & \textbf{Count} \\", r"\midrule",
         f"StereoSet intrasentence items sampled & {fl['stereoset']} \\\\",
         f"BBQ ambiguous items sampled / kept (roles recovered) & {fl['bbq_sampled']} / {fl['bbq_kept']} \\\\",
         f"Items written by the authors & {fl['written']} \\\\",
         f"English items & {fl['english_items']} \\\\",
         f"Translated items (7 languages) & {fl['translated']:,} \\\\",
         f"\\quad removed: contrast collapsed & {fl['collapsed_translations']} \\\\",
         f"Translated--English pairs per model & {fl['pairs_per_model']:,} \\\\",
         f"Machine rewordings of English (3 pivots) & {fl['mt_rewordings']:,} \\\\",
         f"\\quad removed: contrast collapsed & {fl['collapsed_rewordings']} \\\\",
         f"LLM rewordings of English sampled / usable & {fl['llm_rewordings_sampled']} / {fl['llm_rewordings_used']} \\\\",
         f"Hindi/Spanish rewordings attempted / passed & {fl['target_rewordings_attempted']} / {fl['target_rewordings_passed']} \\\\",
         r"\quad usable after removing collapsed translations & 129 / 101 \\",
         r"Translations judged for contrast preservation & 2,497 \\",
         r"\bottomrule", r"\end{tabular}",
         r"\caption{Items at each stage. Every translated or reworded item is compared only with its own English "
         r"original, scored by the same model.}",
         r"\label{tab:flow}", r"\end{table}"]
    (TABLES_DIR / "tab_flow.tex").write_text("\n".join(t) + "\n")
    print(json.dumps({k: N[k] for k in ["same_choice", "abs_vs_floor", "bbq_unknown_chosen", "fate", "by_dimension",
                                        "error_types", "drop_by_judgment", "flow"]}, indent=1, default=float)[:7000])
    print({k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}
           for k, v in A.items() if not k.startswith("model")})


if __name__ == "__main__":
    main()
