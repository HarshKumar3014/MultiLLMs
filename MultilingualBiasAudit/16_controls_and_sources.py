#!/usr/bin/env python3
"""
16_controls_and_sources.py — Controls and results by source (CPU)
=================================================================
  (1) Mixed model: item-level change d ~ language, with crossed random effects
      for item and model (models are not independent evidence).
  (2) Faithful-translation filter, nested in the full set, per language and
      equal-weighted, with a paired bootstrap interval for the reduction.
  (3) Drop and carry-over by source (StereoSet / BBQ / written), translated vs
      reworded, and by whether StereoSet's "BLANK" survived translation.
  (4) Surface controls: model agreement on d before/after removing length and
      word-frequency differences (wordfreq; no data for Swahili, Korean).
Writes results/reanalysis/controls_and_sources.json and paper tables.
"""

import itertools
import json
from importlib import import_module

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import RESULTS_DIR, TABLES_DIR

re9 = import_module("09_reanalysis")
OUT = RESULTS_DIR / "reanalysis"
RNG = np.random.default_rng(23)
ORDER = ["es", "fr", "zh-CN", "ar", "ko", "sw", "hi"]
NAME = {"en": "English", "es": "Spanish", "fr": "French", "zh-CN": "Chinese", "ar": "Arabic", "ko": "Korean",
        "hi": "Hindi", "sw": "Swahili"}
WF_LANG = {"en": "en", "es": "es", "fr": "fr", "zh-CN": "zh", "ar": "ar", "hi": "hi"}


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def source(bid):
    return "StereoSet" if bid.startswith("ss_") else "BBQ" if bid.startswith("bbq_") else "Written"


def lang_drop(q):
    """Per-language drop as the mean over models of each model's mean change (as in Table 1)."""
    return -q.groupby(["language", "model"])["d"].mean().groupby(level="language").mean()


def main():
    prompts, para, df, noise = re9.load_inputs()
    col, cp, _ = re9.audit_collapses(prompts, para)
    p = re9.build_paired(df, col)
    p["x"] = p["ss_en"] - .5; p["y"] = p["stereotype_score"] - .5
    p["src"] = p["base_prompt_id"].map(source)
    models = sorted(p["model"].unique())
    N = {}

    # ── (1) crossed mixed model ──
    m = p[["d", "language", "model", "base_prompt_id"]].copy()
    m["g"] = 1
    md = smf.mixedlm("d ~ 0 + C(language)", m, groups="g",
                     vc_formula={"item": "0 + C(base_prompt_id)", "model": "0 + C(model)"})
    fit = md.fit(method="lbfgs")
    ci = fit.conf_int()
    N["mixed"] = {l: {"est": float(-fit.params[f"C(language)[{l}]"]),
                      "lo": float(-ci.loc[f"C(language)[{l}]", 1]), "hi": float(-ci.loc[f"C(language)[{l}]", 0]),
                      "p": float(fit.pvalues[f"C(language)[{l}]"])} for l in ORDER}
    N["mixed_vc"] = {k: float(v) for k, v in zip(fit.model.exog_vc.names, fit.vcomp)}
    N["mixed_resid_var"] = float(fit.scale)
    m2 = smf.mixedlm("d ~ 1", m, groups="g",
                     vc_formula={"item": "0 + C(base_prompt_id)", "model": "0 + C(model)"}).fit(method="lbfgs")
    N["mixed_overall"] = {"est": float(-m2.params["Intercept"]), "lo": float(-m2.conf_int().loc["Intercept", 1]),
                          "hi": float(-m2.conf_int().loc["Intercept", 0]), "p": float(m2.pvalues["Intercept"])}

    # ── (2) faithful filter, nested, paired bootstrap over items ──
    v = json.load(open(re9.DATA_DIR / "validation_results.json"))
    p["sim"] = p["prompt_id"].map({k: x["similarity"] for k, x in v.items()})
    p["faithful"] = p["sim"] >= .9
    full = lang_drop(p).reindex(ORDER); faith = lang_drop(p[p["faithful"]]).reindex(ORDER)
    items = np.array(sorted(p["base_prompt_id"].unique()))
    # pre-aggregate: per item x language x model mean d (all / faithful) for fast resampling
    agg_all = p.groupby(["base_prompt_id", "language", "model"])["d"].mean().rename("d").reset_index()
    agg_f = p[p["faithful"]].groupby(["base_prompt_id", "language", "model"])["d"].mean().rename("d").reset_index()
    diffs = []
    for _ in range(1000):
        draw = pd.Series(RNG.choice(items, size=len(items), replace=True)).value_counts()
        w_all = agg_all["base_prompt_id"].map(draw).fillna(0)
        w_f = agg_f["base_prompt_id"].map(draw).fillna(0)

        def ew(a, w):
            t = a.assign(w=w, wd=a["d"] * w).groupby(["language", "model"])[["wd", "w"]].sum()
            cell = t["wd"] / t["w"].replace(0, np.nan)
            return float(-cell.groupby(level="language").mean().mean())
        diffs.append(ew(agg_all, w_all) - ew(agg_f, w_f))
    N["faithful"] = {"drop_all_by_lang": full.round(4).to_dict(), "drop_faithful_by_lang": faith.round(4).to_dict(),
                     "ew_all": float(full.mean()), "ew_faithful": float(faith.mean()),
                     "reduction": float(full.mean() - faith.mean()),
                     "reduction_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                     "share": float((full.mean() - faith.mean()) / full.mean()),
                     "pairs_all": int(len(p)), "pairs_faithful": int(p["faithful"].sum()),
                     "slope_all": slope(p["x"], p["y"]),
                     "slope_faithful": slope(p.loc[p["faithful"], "x"], p.loc[p["faithful"], "y"])}

    # ── (3) by source ──
    n = re9.build_noise_paired(noise, cp)
    n["x"] = n["ss_orig"] - .5; n["y"] = n["stereotype_score"] - .5
    n["src"] = n["prompt_id"].str.replace("_en$", "", regex=True).map(source)
    ctx = {q["id"]: q["context"] for q in prompts}
    p["agree"] = (p["stereotype_score"] > .5) == (p["ss_en"] > .5)
    n["agree"] = (n["stereotype_score"] > .5) == (n["ss_orig"] > .5)
    rows = {}
    for s_ in ["StereoSet", "BBQ", "Written"]:
        q = p[p["src"] == s_]; r = n[n["src"] == s_]
        rows[s_] = {"items": int(q["base_prompt_id"].nunique()), "drop": float(lang_drop(q).mean()),
                    "slope_tr": slope(q["x"], q["y"]), "slope_rw": slope(r["x"], r["y"]),
                    "agree_tr": float(q["agree"].mean()), "agree_rw": float(r["agree"].mean()),
                    "drop_by_lang": lang_drop(q).reindex(ORDER).round(4).to_dict()}
    s = p[p["src"] == "StereoSet"].copy()
    s["blank_kept"] = s["prompt_id"].map(lambda i: "BLANK" in ctx.get(i, ""))
    N["blank"] = {str(k): {"pairs": int(len(g)), "slope": slope(g["x"], g["y"]), "drop": float(-g["d"].mean())}
                  for k, g in s.groupby("blank_kept")}
    N["blank_kept_share_by_lang"] = s.groupby("language")["blank_kept"].mean().round(3).to_dict()
    N["by_source"] = rows

    # ── (4) surface: length + word frequency ──
    import wordfreq
    tc = pd.read_csv(OUT / "token_counts.csv")
    texts = {q["id"]: q for q in prompts}

    def zipf(text, lang):
        wl = WF_LANG.get(lang)
        if wl is None:
            return np.nan
        toks = wordfreq.tokenize(text, wl)
        return float(np.mean([wordfreq.zipf_frequency(t, wl) for t in toks])) if toks else np.nan

    freq = {}
    for pid, q in texts.items():
        if q["layer"] != "A":
            continue
        freq[pid] = zipf(q["stereotype"], q["language"]) - zipf(q["anti_stereotype"], q["language"])
    p2 = p.merge(tc, on=["model", "prompt_id"])
    p2["len_diff"] = p2["n_stereotype"] - p2["n_anti_stereotype"]
    en_tc = tc[tc["prompt_id"].str.endswith("_en")].assign(
        base_prompt_id=lambda t: t["prompt_id"].str.replace("_en$", "", regex=True),
        len_diff_en=lambda t: t["n_stereotype"] - t["n_anti_stereotype"],
        len_mean_en=lambda t: (t["n_stereotype"] + t["n_anti_stereotype"]) / 2)
    p2 = p2.merge(en_tc[["model", "base_prompt_id", "len_diff_en", "len_mean_en"]], on=["model", "base_prompt_id"])
    p2["len_ratio"] = np.log(((p2["n_stereotype"] + p2["n_anti_stereotype"]) / 2) / p2["len_mean_en"])
    p2["freq_diff"] = p2["prompt_id"].map(freq)
    p2["freq_diff_en"] = (p2["base_prompt_id"] + "_en").map(freq)

    def xmodel(frame, val):
        w = frame.pivot_table(index="base_prompt_id", columns="model", values=val)
        return float(np.mean([w[a].corr(w[b]) for a, b in itertools.combinations(models, 2)]))
    surf = {}
    for l in ORDER:
        q = p2[p2["language"] == l].copy()
        q["r_len"] = smf.ols("d ~ len_diff + len_diff_en + len_ratio", q).fit().resid
        row = {"agree_raw": xmodel(q, "d"), "agree_len": xmodel(q, "r_len"),
               "r2_len": float(smf.ols("d ~ len_diff + len_diff_en + len_ratio", q).fit().rsquared)}
        if q["freq_diff"].notna().mean() > .9:
            qf = q.dropna(subset=["freq_diff", "freq_diff_en"]).copy()
            ff = smf.ols("d ~ len_diff + len_diff_en + len_ratio + freq_diff + freq_diff_en", qf).fit()
            qf["r_all"] = ff.resid
            row.update({"agree_len_freq": xmodel(qf, "r_all"), "r2_len_freq": float(ff.rsquared)})
        surf[l] = row
    N["surface"] = surf
    eq = p2[(p2["len_diff"] == 0) & (p2["len_diff_en"] == 0)]
    N["length_matched"] = {"pairs": int(len(eq)), "drop": float(-eq["d"].mean()), "slope": slope(eq["x"], eq["y"]),
                           "drop_all": float(-p2["d"].mean()), "slope_all": slope(p2["x"], p2["y"])}

    # ── German subset floor ──
    h = set(pd.read_csv(re9.DATA_DIR / "human_paraphrases.csv")["prompt_id"])
    g = n[(n["pivot"] == "de") & n["prompt_id"].isin(h)]
    N["german_subset"] = {"items": int(g["prompt_id"].nunique()), "mean_change": float(g["d"].mean()),
                          "se_items": float(g.groupby("prompt_id")["d"].mean().std() / np.sqrt(g["prompt_id"].nunique()))}

    json.dump(N, open(OUT / "controls_and_sources.json", "w"), indent=1, default=float)

    # ── tables ──
    t = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{\textbf{Carry-over}} & \\", r"\cmidrule(lr){4-5}",
         r"\textbf{Source} & \textbf{Items} & \textbf{Drop} & transl. & rew. & \textbf{Same choice} \\", r"\midrule"]
    for s_ in ["StereoSet", "BBQ", "Written"]:
        r = rows[s_]
        t.append(f"{s_} & {r['items']} & {r['drop']:.3f} & {r['slope_tr']:.2f} & {r['slope_rw']:.2f} & "
                 f"{100*r['agree_tr']:.0f}\\% / {100*r['agree_rw']:.0f}\\% \\\\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Results by source of the items. \textbf{Drop}: mean over languages of the drop below English. "
          r"\textbf{Carry-over}: slope for translations and for English rewordings. \textbf{Same choice}: "
          r"translated / reworded. Carry-over is far below the rewording level for every source.}",
          r"\label{tab:sources}", r"\end{table}"]
    (TABLES_DIR / "tab_sources.tex").write_text("\n".join(t) + "\n")

    t = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{3}{c}{\textbf{Models agree on change}} & \\", r"\cmidrule(lr){2-4}",
         r"\textbf{Language} & raw & $-$length & $-$length, freq. & $R^2$ \\", r"\midrule"]
    for l in ORDER:
        r = surf[l]
        f_ = f"{r['agree_len_freq']:.2f}" if "agree_len_freq" in r else "--"
        r2 = r.get("r2_len_freq", r["r2_len"])
        t.append(f"{NAME[l]} & {r['agree_raw']:.2f} & {r['agree_len']:.2f} & {f_} & {r2:.3f} \\\\")
    t += [r"\midrule", f"\\multicolumn{{5}}{{l}}{{Equal length ($n$={N['length_matched']['pairs']:,}): drop {N['length_matched']['drop']:.3f}, "
          f"slope {N['length_matched']['slope']:.2f}}} \\\\",
          f"\\multicolumn{{5}}{{l}}{{All pairs: drop {N['length_matched']['drop_all']:.3f}, "
          f"slope {N['length_matched']['slope_all']:.2f}}} \\\\",
          r"\bottomrule", r"\end{tabular}",
          r"\caption{Do surface properties drive the shared item-level changes? Average correlation between "
          r"models of the change from English, before and after regressing out differences in token count "
          r"between the two continuations and in overall length (\emph{length}), and additionally in word "
          r"frequency (\emph{freq.}; no frequency data for Korean and Swahili). $R^2$: variance in the change "
          r"explained by these properties. Last two rows: items whose two continuations have the same number of "
          r"tokens in both languages, and all pairs.}",
          r"\label{tab:surface}", r"\end{table}"]
    (TABLES_DIR / "tab_surface.tex").write_text("\n".join(t) + "\n")

    t = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.4pt}",
         r"\begin{tabular}{l" + "c" * len(ORDER) + "c}", r"\toprule",
         r"\textbf{Items} & " + " & ".join(rf"\textbf{{{NAME[l][:2]}}}" for l in ORDER) + r" & \textbf{Avg.} \\",
         r"\midrule",
         "All & " + " & ".join(f"{full[l]:.3f}"[1:] for l in ORDER) + f" & {full.mean():.3f}".replace(" 0.", " .") + r" \\",
         "Faithful & " + " & ".join(f"{faith[l]:.3f}"[1:] for l in ORDER) + f" & {faith.mean():.3f}".replace(" 0.", " .") + r" \\",
         r"\bottomrule", r"\end{tabular}",
         rf"\caption{{Drop below English with all translations and with faithful ones only (back-translation "
         rf"similarity $\ge 0.9$, a subset of the first row). \textbf{{Avg.}}: equal weight per language. The "
         rf"reduction is {N['faithful']['reduction']:.3f} (95\% paired bootstrap interval over items "
         rf"[{N['faithful']['reduction_ci'][0]:.3f}, {N['faithful']['reduction_ci'][1]:.3f}]).}}",
         r"\label{tab:controls}", r"\end{table}"]
    (TABLES_DIR / "tab_controls.tex").write_text("\n".join(t) + "\n")
    print(json.dumps({k: N[k] for k in ["mixed", "mixed_overall", "mixed_vc", "faithful", "blank", "by_source",
                                        "surface", "length_matched", "german_subset"]}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
