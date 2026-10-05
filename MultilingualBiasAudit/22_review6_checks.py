#!/usr/bin/env python3
"""
22_review6_checks.py — Sixth-round reviewer checks (CPU)
========================================================
  (1) Contrast preservation for every translated item (data/contrast_judgment_full.csv,
      judged by an LLM with the same criteria as the 210-pair sample, applied
      strictly: "partly" whenever the pair differs outside the contrast slot or
      the contrast word shifts in meaning). Counts by language and source.
  (2) Item-level test against English rewording (as in 20_review4_checks) on all
      items, on fully preserved items only, and on items whose contrast is not lost.
  (3) English-to-translation association (r, SD ratio, slope) on the same subsets.
  (4) Drop by judgment, per language, with bootstrap intervals over items.
  (5) Power of the item-level test for true extra drops of 0.02 and 0.03.
  (6) Within-language reliability (Hindi, Spanish): correlation between a
      translated item's score and its target-language rewording, bootstrap over items.
  (7) StereoSet drop without the gender items.
Writes results/reanalysis/review6_checks.json and paper tables.
"""

import json
from importlib import import_module

import numpy as np
import pandas as pd

from config import RESULTS_DIR, SCORES_DIR, TABLES_DIR, DATA_DIR

re9 = import_module("09_reanalysis")
r4 = import_module("20_review4_checks")
r5 = import_module("21_review5_checks")
import label_fix

OUT = RESULTS_DIR / "reanalysis"
RNG = np.random.default_rng(67)
ORDER = r4.ORDER
NAME = r4.NAME


def item_diffs(p, n, l):
    rw_item = n.groupby("base_prompt_id")["d"].mean()
    tr_item = p[p["language"] == l].groupby("base_prompt_id")["d"].mean()
    j = pd.concat([tr_item.rename("tr"), rw_item.rename("rw")], axis=1).dropna()
    return (-(j["tr"] - j["rw"])).to_numpy()


def power(diff, delta, reps=200, n_perm=2000, alpha=0.05):
    """Share of runs in which the sign-flip test rejects when the true extra drop is
    `delta`: centre the observed item differences, add delta, resample items."""
    c = diff - diff.mean()
    hits = 0
    for _ in range(reps):
        x = c[RNG.integers(0, len(c), len(c))] + delta
        s = RNG.choice([-1.0, 1.0], size=(n_perm, len(x)))
        pv = (1 + np.sum(np.abs(s @ x) / len(x) >= abs(x.mean()) - 1e-15)) / (n_perm + 1)
        hits += pv < alpha
    return hits / reps


def boot_drop(frame):
    it = -frame.groupby("base_prompt_id")["d"].mean().to_numpy()
    if len(it) < 5:
        return {"items": int(len(it)), "drop": float(it.mean()) if len(it) else None, "lo": None, "hi": None}
    bs = it[RNG.integers(0, len(it), (3000, len(it)))].mean(1)
    return {"items": int(len(it)), "drop": float(it.mean()),
            "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def main():
    prompts, para, df, noise = re9.load_inputs()
    col, cp, _ = re9.audit_collapses(prompts, para)
    p = re9.build_paired(df, col)
    n = re9.build_noise_paired(noise, cp)
    p["x"], p["y"] = p["ss_en"], p["stereotype_score"]
    n["x"], n["y"] = n["ss_orig"], n["stereotype_score"]
    n["src"] = n["base_prompt_id"].map(r5.source)
    N = {}

    # ── (1) judgments ──
    j = pd.read_csv(DATA_DIR / "contrast_judgment_full.csv")
    p = p.merge(j[["prompt_id", "judgment"]], on="prompt_id", how="left")
    N["coverage"] = {"pairs": int(len(p)), "judged_pairs": int(p["judgment"].notna().sum()),
                     "items_judged": int(len(j))}
    N["judgments"] = {"by_lang": j.groupby(["language", "judgment"]).size().unstack(fill_value=0).to_dict("index"),
                      "by_source": j.groupby(["source", "judgment"]).size().unstack(fill_value=0).to_dict("index"),
                      "total": j["judgment"].value_counts().to_dict()}

    subsets = {"all": p, "preserved": p[p["judgment"] == "preserved"],
               "not_lost": p[p["judgment"].isin(["preserved", "partly"])]}

    # ── (2) item-level test ──
    N["item_level"] = {k: r4.item_level_tests(v, n) for k, v in subsets.items()}

    # pooled over languages: per English item, mean change over the languages and
    # models where it is kept, minus its rewording change; bootstrap over items
    rw_item = n.groupby("base_prompt_id")["d"].mean()
    def pooled(frame):
        tr = frame.groupby("base_prompt_id")["d"].mean()
        x = (-(tr - rw_item.reindex(tr.index))).dropna().to_numpy()
        bs = x[RNG.integers(0, len(x), (5000, len(x)))].mean(1)
        return {"items": int(len(x)), "beyond": float(x.mean()),
                "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
    src = p["base_prompt_id"].map(r5.source)
    N["pooled_item_level"] = {k: {"all_sources": pooled(v), **{s: pooled(v[src.loc[v.index] == s])
                                                               for s in ["StereoSet", "BBQ", "Written"]}}
                              for k, v in subsets.items()}

    # ── (3) association ──
    A = {}
    for k, v in subsets.items():
        A[k] = {"pooled": r5.boot_assoc(v, "x", "y", 500)}
        for l in ORDER:
            A[k][l] = r5.boot_assoc(v[v["language"] == l], "x", "y", 300)
    A["reworded"] = r5.boot_assoc(n, "x", "y", 500)
    N["association"] = A

    # ── (4) drop by judgment ──
    N["drop_by_judgment"] = {jd: {"all": boot_drop(g), **{l: boot_drop(g[g["language"] == l]) for l in ORDER}}
                             for jd, g in p.groupby("judgment")}

    # ── (5) power of the item-level test ──
    N["power"] = {l: {str(dl): power(item_diffs(p, n, l), dl) for dl in (0.02, 0.03)} for l in ORDER}

    # ── (6) within-language reliability ──
    rr = label_fix.fix_scores(pd.concat([pd.read_csv(f) for f in sorted((SCORES_DIR / "review_runs").glob("*.csv"))]))
    rw = rr[rr["condition"] == "rw_target"][["model", "prompt_id", "language", "stereotype_score"]] \
        .rename(columns={"stereotype_score": "ss_rw"})
    a = df[(df["layer"] == "A") & ~df["prompt_id"].isin(col)]
    m = rw.merge(a[["model", "prompt_id", "base_prompt_id", "stereotype_score"]], on=["model", "prompt_id"])
    N["within"] = {l: r5.boot_assoc(m[m["language"] == l], "stereotype_score", "ss_rw", 1000) for l in ["hi", "es"]}

    # ── (7) StereoSet without gender ──
    cat = {q["base_id"]: q["category"] for q in prompts if q["language"] == "en"}
    ss = p[p["base_prompt_id"].str.startswith("ss_")]
    ng = ss[~ss["base_prompt_id"].map(cat).str.lower().str.contains("gender")]
    N["stereoset_no_gender"] = boot_drop(ng)
    N["stereoset_all"] = boot_drop(ss)
    json.dump(N, open(OUT / "review6_checks.json", "w"), indent=1, default=float)

    # ── tables ──
    def f3(x):
        v = f"{abs(x):.3f}"[1:]
        return f"$-${v}" if x < -0.0005 else v
    qs = lambda q: "$<$.001" if q < .001 else f"{q:.3f}"[1:]
    jc = N["judgments"]["by_lang"]
    t = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{lccccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{\textbf{All items}} & \multicolumn{3}{c}{\textbf{Fully preserved only}} & "
         r"\multicolumn{2}{c}{\textbf{BBQ + written}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
         r"\textbf{Language} & beyond rew. & $q$ & kept & beyond rew. & $q$ & beyond rew. & $q$ \\", r"\midrule"]
    bw = json.load(open(OUT / "review4_checks.json"))["item_level_bbq_written"]
    for l in ORDER:
        r1, r2, r3 = N["item_level"]["all"][l], N["item_level"]["preserved"][l], bw[l]
        kept = jc[l].get("preserved", 0); tot = sum(jc[l].values())
        t.append(f"{NAME[l]} & {f3(r1['beyond'])} [{f3(r1['lo'])}, {f3(r1['hi'])}] & {qs(r1['q'])} & "
                 f"{kept}/{tot} & {f3(r2['beyond'])} [{f3(r2['lo'])}, {f3(r2['hi'])}] & {qs(r2['q'])} & "
                 f"{f3(r3['beyond'])} [{f3(r3['lo'])}, {f3(r3['hi'])}] & {qs(r3['q'])} \\\\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Item-level test of each language against English rewording. The change is first averaged "
          r"over the ten models within each item, so that items, not model--item pairs, are the unit. "
          r"\textbf{beyond rew.}: how much more the language drops below English than rewording does, with a 95\% "
          r"bootstrap interval over items; $q$: paired sign-flip permutation test over items, false-discovery "
          r"corrected. \textbf{Fully preserved only}: translations whose contrast was judged fully preserved "
          r"(\textbf{kept}: fully preserved out of all judged translations). \textbf{BBQ + written}: all items "
          r"from these two sources, without StereoSet.}",
          r"\label{tab:mixed}", r"\end{table*}"]
    (TABLES_DIR / "tab_mixed.tex").write_text("\n".join(t) + "\n")

    def ci(d, k):
        return f"{d[k]:.2f} [{d[k + '_lo']:.2f}, {d[k + '_hi']:.2f}]"
    t = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{\textbf{All translated items}} & \multicolumn{2}{c}{\textbf{Fully preserved only}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
         r"\textbf{Items} & $r$ [95\% CI] & SD ratio & slope [95\% CI] & $r$ [95\% CI] & SD ratio \\", r"\midrule"]
    ra, rp = A["all"], A["preserved"]
    t.append(f"All languages & {ci(ra['pooled'], 'r')} & {ra['pooled']['sd_ratio']:.2f} & {ci(ra['pooled'], 'slope')} & "
             f"{ci(rp['pooled'], 'r')} & {rp['pooled']['sd_ratio']:.2f} \\\\")
    t.append(r"\midrule")
    for l in ORDER:
        t.append(f"{NAME[l]} & {ci(ra[l], 'r')} & {ra[l]['sd_ratio']:.2f} & {ci(ra[l], 'slope')} & "
                 f"{ci(rp[l], 'r')} & {rp[l]['sd_ratio']:.2f} \\\\")
    t.append(r"\midrule")
    rwd = A["reworded"]
    t.append(f"English rewording & {ci(rwd, 'r')} & {rwd['sd_ratio']:.2f} & {ci(rwd, 'slope')} & -- & -- \\\\")
    for l in ["hi", "es"]:
        w = N["within"][l]
        t.append(f"{NAME[l]} rewording & {ci(w, 'r')} & {w['sd_ratio']:.2f} & {ci(w, 'slope')} & -- & -- \\\\")
    t += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Association between an item's score before and after a change: correlation $r$, ratio of "
          r"standard deviations, and least-squares slope ($= r \times$ SD ratio), each with a 95\% bootstrap "
          r"interval over items. Translated rows compare English with the translation; \textbf{English rewording} "
          r"compares English with its rewordings; the Hindi and Spanish rewording rows compare a translation with "
          r"its rewording in the same language. Low slopes after translation come from low correlation; the SD "
          r"ratio does not fall consistently below 1.}",
          r"\label{tab:assoc}", r"\end{table*}"]
    (TABLES_DIR / "tab_assoc.tex").write_text("\n".join(t) + "\n")

    show = {k: N[k] for k in ["coverage", "judgments", "drop_by_judgment", "power", "stereoset_no_gender",
                              "stereoset_all"]}
    print(json.dumps(show, indent=1, default=float)[:6000])
    for k in subsets:
        print(k, {l: (round(v["beyond"], 4), round(v["lo"], 4), round(v["hi"], 4), round(v["q"], 4), v["items"])
                  for l, v in N["item_level"][k].items()})
        print(k, "assoc", {l: (round(v["r"], 3), round(v["sd_ratio"], 2)) for l, v in A[k].items()})
    print("rew", {kk: round(vv, 3) for kk, vv in A["reworded"].items()})
    print("within", {l: {kk: round(vv, 3) for kk, vv in v.items()} for l, v in N["within"].items()})


if __name__ == "__main__":
    main()
