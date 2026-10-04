#!/usr/bin/env python3
"""
15_robustness_checks.py — Reviewer-requested robustness checks (CPU)
====================================================================
  (1) Surface features: how much of the item-level change d is explained by
      token counts / length differences of the two continuations, and does
      cross-model agreement survive after removing them?
  (2) Summed instead of per-token-averaged log-probability (mean x number of
      continuation tokens, counted with each model's own tokenizer).
  (3) Translation-quality filter broken down per language, and re-averaged
      with equal language weights.
  (4) Fluency computed without BBQ (whose "unrelated" option is the unknown
      answer, not an unrelated sentence).
  (5) Injected-shift check with 200 runs per size.
Writes results/reanalysis/robustness_checks.json.
"""

import itertools
import json
from importlib import import_module

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import RESULTS_DIR, MODEL_REGISTRY

re9 = import_module("09_reanalysis")
audit = import_module("02_run_audit")
OUT = RESULTS_DIR / "reanalysis"
RNG = np.random.default_rng(11)
LANGS = ["es", "fr", "zh-CN", "ar", "ko", "hi", "sw"]


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def xmodel(frame, key, val, models):
    w = frame.pivot_table(index=key, columns="model", values=val)
    return float(np.mean([w[a].corr(w[b]) for a, b in itertools.combinations(models, 2)]))


def token_counts(prompts):
    """n tokens of s+ and s- for every prompt and model tokenizer (same span as the scorer)."""
    from transformers import AutoTokenizer
    rows = []
    for mk, info in MODEL_REGISTRY.items():
        tok = AutoTokenizer.from_pretrained(info["hf_id"], trust_remote_code=True)
        for p in prompts:
            r = {"model": mk, "prompt_id": p["id"]}
            for f in ["stereotype", "anti_stereotype"]:
                ids, st = audit.continuation_span(tok, p["context"], p[f])
                r[f"n_{f}"] = len(ids) - st
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    prompts, para, df, noise = re9.load_inputs()
    col, cp, _ = re9.audit_collapses(prompts, para)
    models = sorted(df["model"].unique())
    N = {}

    # ── token counts ──
    cache = OUT / "token_counts.csv"
    if cache.exists():
        tc = pd.read_csv(cache)
    else:
        tc = token_counts([p for p in prompts if p["layer"] == "A"])
        tc.to_csv(cache, index=False)
    d2 = df.merge(tc, on=["model", "prompt_id"], how="inner")
    d2["len_diff"] = d2["n_stereotype"] - d2["n_anti_stereotype"]
    d2["len_mean"] = (d2["n_stereotype"] + d2["n_anti_stereotype"]) / 2

    # ── (2) summed log-probability ──
    s = d2.copy()
    ls = s["logprob_stereotype"] * s["n_stereotype"]
    la = s["logprob_anti_stereotype"] * s["n_anti_stereotype"]
    s["stereotype_score"] = 1 / (1 + np.exp(-(ls - la)))
    ps = re9.build_paired(s, col)
    cs = re9.cell_tests(ps, RNG)
    ps["x"] = ps["ss_en"] - .5; ps["y"] = ps["stereotype_score"] - .5
    ps["agree"] = (ps["stereotype_score"] > .5) == (ps["ss_en"] > .5)
    en_s = s[(s["language"] == "en") & (s["layer"] == "A") & ~s["prompt_id"].isin(col)]
    N["summed"] = {
        "en_mean": float(en_s["stereotype_score"].mean()),
        "cells_negative": int((cs["mean_d"] < 0).sum()),
        "n_sig_bh": int((cs["q_bh"] < .05).sum()),
        "drop_by_lang": (-cs.groupby("language")["mean_d"].mean()).reindex(LANGS).round(4).to_dict(),
        "sig_by_lang": cs[cs["q_bh"] < .05].groupby("language").size().reindex(LANGS).fillna(0).astype(int).to_dict(),
        "slope": slope(ps["x"], ps["y"]),
        "slope_by_lang": {l: slope(g["x"], g["y"]) for l, g in ps.groupby("language")},
        "agree_by_lang": ps.groupby("language")["agree"].mean().round(3).to_dict(),
        "strength_by_lang": s[(s["layer"] == "A") & ~s["prompt_id"].isin(col)]
            .assign(e=lambda t: (t["stereotype_score"] - .5).abs()).groupby("language")["e"].mean().round(3).to_dict(),
    }
    # reworded with summed scores needs token counts of paraphrases: skip (English only, same tokenizer)

    # ── (1) surface features ──
    p = re9.build_paired(d2, col)
    en_tc = d2[d2["language"] == "en"][["model", "base_prompt_id", "len_diff", "len_mean"]] \
        .rename(columns={"len_diff": "len_diff_en", "len_mean": "len_mean_en"})
    p = p.merge(en_tc, on=["model", "base_prompt_id"])
    p["len_ratio"] = np.log(p["len_mean"] / p["len_mean_en"])
    p["d_len_diff"] = p["len_diff"] - p["len_diff_en"]
    p["abs_len_diff"] = p["len_diff"].abs()
    f = smf.ols("d ~ len_diff + d_len_diff + len_ratio + abs_len_diff + C(language)*C(model)", p).fit()
    f0 = smf.ols("d ~ C(language)*C(model)", p).fit()
    N["surface"] = {"r2_language_model": float(f0.rsquared), "r2_with_surface": float(f.rsquared),
                    "r2_added_by_surface": float(f.rsquared - f0.rsquared),
                    "coefs": {k: [float(f.params[k]), float(f.pvalues[k])]
                              for k in ["len_diff", "d_len_diff", "len_ratio", "abs_len_diff"]}}
    p["d_resid"] = smf.ols("d ~ len_diff + d_len_diff + len_ratio + abs_len_diff", p).fit().resid
    N["agreement_change_raw"] = {l: xmodel(p[p["language"] == l], "base_prompt_id", "d", models) for l in LANGS}
    N["agreement_change_resid"] = {l: xmodel(p[p["language"] == l], "base_prompt_id", "d_resid", models)
                                   for l in LANGS}
    # are length-matched pairs (same token count for s+ and s-) different?
    eq = p[(p["len_diff"] == 0) & (p["len_diff_en"] == 0)]
    eq = eq.assign(x=eq["ss_en"] - .5, y=eq["stereotype_score"] - .5)
    N["length_matched"] = {"n": int(len(eq)), "mean_drop": float(-eq["d"].mean()),
                           "slope": slope(eq["x"], eq["y"])}

    # ── (3) translation-quality filter per language, equal weights ──
    v = json.load(open(re9.DATA_DIR / "validation_results.json"))
    p["sim"] = p["prompt_id"].map({k: x["similarity"] for k, x in v.items()})
    p["x"] = p["ss_en"] - .5; p["y"] = p["stereotype_score"] - .5
    lms_en = df[(df["language"] == "en") & (df["layer"] == "A")][["model", "base_prompt_id", "lm_score"]] \
        .rename(columns={"lm_score": "lms_en"})
    p = p.merge(lms_en, on=["model", "base_prompt_id"])
    p["dlms"] = p["lm_score"] - p["lms_en"]
    p["is_bbq"] = p["base_prompt_id"].str.startswith("bbq")
    subsets = {"all": p, "faithful": p[p["sim"] >= .9],
               "faithful_fluent_nobbq": p[(p["sim"] >= .9) & (p["dlms"].abs() < .1) & ~p["is_bbq"]]}
    N["filters"] = {}
    for k, q in subsets.items():
        by = q.groupby("language").agg(n=("d", "size"), drop=("d", lambda x: -x.mean()))
        by["slope"] = [slope(g["x"], g["y"]) for _, g in q.groupby("language")]
        N["filters"][k] = {"n": int(len(q)), "share_by_lang": (by["n"] / by["n"].sum()).round(3).to_dict(),
                           "drop_by_lang": by["drop"].round(4).to_dict(), "slope_by_lang": by["slope"].round(3).to_dict(),
                           "drop_equal_weight": float(by["drop"].mean()), "slope_pooled": slope(q["x"], q["y"]),
                           "drop_pooled": float(-q["d"].mean())}
    # fluency without BBQ
    a = df[(df["layer"] == "A") & ~df["prompt_id"].isin(col)]
    nb = a[~a["base_prompt_id"].str.startswith("bbq")]
    N["fluency_nobbq_by_lang"] = nb.groupby("language")["lm_score"].mean().round(3).to_dict()
    N["fluency_bbq_by_lang"] = a[a["base_prompt_id"].str.startswith("bbq")].groupby("language")["lm_score"].mean().round(3).to_dict()

    # ── (5) injected shift, 200 runs per size ──
    npaired = re9.build_noise_paired(noise, cp)
    null_cells = list(npaired.groupby(["model", "pivot"]).groups.items())
    lo = npaired["logprob_stereotype"] - npaired["logprob_anti_stereotype"]
    base_p = {k: re9.sign_flip_p(npaired.loc[idx, "d"].to_numpy(), RNG, 2000) for k, idx in null_cells}
    rows = []
    for delta in [0.0, 0.1, 0.15, 0.2, 0.3, 0.5]:
        for rep in range(200):
            key, idx = null_cells[RNG.integers(len(null_cells))]
            ss = 1 / (1 + np.exp(-(lo.loc[idx] + delta)))
            dnew = (ss - npaired.loc[idx, "ss_orig"]).to_numpy()
            ps_ = dict(base_p); ps_[key] = re9.sign_flip_p(dnew, RNG, 2000)
            keys = list(ps_)
            q = dict(zip(keys, re9.bh(np.array([ps_[k] for k in keys]))))
            rows.append({"delta": delta, "shift": float(dnew.mean() - npaired.loc[idx, "d"].mean()),
                         "hit": q[key] < .05, "fa": sum(q[k] < .05 for k in keys if k != key)})
    sp = pd.DataFrame(rows)
    summ = sp.groupby("delta").agg(shift=("shift", "mean"), power=("hit", "mean"), fa_rate=("fa", "mean"),
                                   runs=("hit", "size")).reset_index()
    summ["fa_rate"] = summ["fa_rate"] / (len(null_cells) - 1)
    summ.to_csv(OUT / "spike_in_null_200.csv", index=False)
    N["spike_200"] = summ.round(4).to_dict("records")
    N["spike_200_total_false_alarm_tests"] = int(len(sp) * (len(null_cells) - 1))
    N["spike_200_false_alarms"] = int(sp["fa"].sum())

    # ── LLM floor expectation ──
    from config import SCORES_DIR
    import label_fix
    llm = label_fix.fix_scores(pd.concat([pd.read_csv(f) for f in sorted((SCORES_DIR / "noise_floor").glob("*_llm.csv"))]))
    nl = re9.build_noise_paired(pd.concat([noise[noise["prompt_id"].isin(set(llm["prompt_id"]))], llm]), cp)
    g = nl[nl["pivot"] == "llm"]
    N["llm_floor"] = {"sd": float(g["d"].std()), "n_items": int(g["prompt_id"].nunique()),
                      "expected": float(g["d"].std() * np.sqrt(2 / (np.pi * g["prompt_id"].nunique()))),
                      "observed_median": float(g.groupby("model")["d"].mean().abs().median())}

    json.dump(N, open(OUT / "robustness_checks.json", "w"), indent=1, default=float)
    print(json.dumps(N, indent=1, default=float)[:7000])


if __name__ == "__main__":
    main()
