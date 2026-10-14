#!/usr/bin/env python3
"""
20_item_level_tests.py — Item-level tests (CPU)
==============================================
  (1) Item-level inference. All models read the same translation, so the item
      is the unit: average the change over models within each item, then test
      each language against rewording with a paired sign-flip permutation test
      and a bootstrap over items (both conditions resampled jointly).
  (2) The same item-level test on BBQ + written items only, per language.
  (3) How much the accepted Hindi/Spanish rewordings changed (embedding
      similarity, share of fields unchanged), and carry-over on items where
      context and both continuations changed; without the two Aya models;
      English comparisons on exactly the same items.
Writes results/reanalysis/item_level_tests.json.
"""

import json
from importlib import import_module

import numpy as np
import pandas as pd

from config import RESULTS_DIR, SCORES_DIR, TABLES_DIR, DATA_DIR

re9 = import_module("09_reanalysis")
import label_fix

OUT = RESULTS_DIR / "reanalysis"
RNG = np.random.default_rng(53)
ORDER = ["es", "fr", "zh-CN", "ar", "ko", "sw", "hi"]
NAME = {"es": "Spanish", "fr": "French", "zh-CN": "Chinese", "ar": "Arabic", "ko": "Korean", "sw": "Swahili",
        "hi": "Hindi"}
FIELDS = ["context", "stereotype", "anti_stereotype"]


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def item_level_tests(p, n, n_perm=20000, n_boot=5000):
    """Per language: item-level mean change (over models) minus the item-level
    mean change under English rewording (over models and pivots), on items
    present in both. Sign-flip test over items + bootstrap CI over items."""
    rw_item = n.groupby("base_prompt_id")["d"].mean()
    res = {}
    for l in ORDER:
        tr_item = p[p["language"] == l].groupby("base_prompt_id")["d"].mean()
        j = pd.concat([tr_item.rename("tr"), rw_item.rename("rw")], axis=1).dropna()
        diff = (-(j["tr"] - j["rw"])).to_numpy()          # positive = drops more than rewording
        drop = (-j["tr"]).to_numpy()
        obs = diff.mean()
        signs = RNG.choice([-1.0, 1.0], size=(n_perm, len(diff)))
        p_perm = (1 + np.sum(np.abs(signs @ diff) / len(diff) >= abs(obs) - 1e-15)) / (n_perm + 1)
        boot = RNG.integers(0, len(diff), size=(n_boot, len(diff)))
        bd = diff[boot].mean(1); bdrop = drop[boot].mean(1)
        res[l] = {"items": int(len(diff)), "drop": float(drop.mean()),
                  "drop_lo": float(np.percentile(bdrop, 2.5)), "drop_hi": float(np.percentile(bdrop, 97.5)),
                  "beyond": float(obs), "lo": float(np.percentile(bd, 2.5)), "hi": float(np.percentile(bd, 97.5)),
                  "p": float(p_perm)}
    q = re9.bh(np.array([res[l]["p"] for l in ORDER]))
    for l, qq in zip(ORDER, q):
        res[l]["q"] = float(qq)
    return res


def main():
    prompts, para, df, noise = re9.load_inputs()
    col, cp, _ = re9.audit_collapses(prompts, para)
    p = re9.build_paired(df, col)
    n = re9.build_noise_paired(noise, cp)
    N = {}

    # ── (1) all items ──
    N["item_level"] = item_level_tests(p, n)
    # ── (2) BBQ + written only ──
    keep = ~p["base_prompt_id"].str.startswith("ss_")
    keep_n = ~n["base_prompt_id"].str.startswith("ss_")
    N["item_level_bbq_written"] = item_level_tests(p[keep], n[keep_n])

    # ── (3) target-language rewording: how much changed ──
    rr = label_fix.fix_scores(pd.concat([pd.read_csv(f) for f in sorted((SCORES_DIR / "within_language_runs").glob("*.csv"))]))
    rw = rr[rr["condition"] == "rw_target"][["model", "prompt_id", "language", "stereotype_score"]] \
        .rename(columns={"stereotype_score": "ss_rw"})
    a = df[(df["layer"] == "A") & ~df["prompt_id"].isin(col)]
    m = rw.merge(a[["model", "prompt_id", "base_prompt_id", "stereotype_score"]], on=["model", "prompt_id"])
    used = set(m["prompt_id"])
    rewd = json.load(open(DATA_DIR / "target_rewordings.json"))
    by_id = {q["id"]: q for q in json.load(open(DATA_DIR / "prompts.json"))}
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    rows = []
    for pid in sorted(used):
        o, r = by_id[pid], rewd[pid]
        e = enc.encode([o[f] for f in FIELDS] + [r[f] for f in FIELDS], normalize_embeddings=True)
        sims = (e[:3] * e[3:]).sum(1)
        same = [re9.is_collapsed(o[f], r[f]) for f in FIELDS]
        rows.append({"prompt_id": pid, "language": o["language"], "sim_mean": float(sims.mean()),
                     "n_unchanged": int(sum(same)), "all_changed": not any(same)})
    ch = pd.DataFrame(rows)
    N["rewording_change"] = {
        l: {"items": int(len(g)), "sim_mean": float(g["sim_mean"].mean()),
            "share_fields_unchanged": float(g["n_unchanged"].sum() / (3 * len(g))),
            "share_items_all_changed": float(g["all_changed"].mean())}
        for l, g in ch.groupby("language")}
    m = m.merge(ch[["prompt_id", "all_changed"]], on="prompt_id")
    m["x"] = m["stereotype_score"] - .5; m["y"] = m["ss_rw"] - .5
    aya = {"aya-23-8b", "aya-expanse-8b"}
    p["x"] = p["ss_en"] - .5; p["y"] = p["stereotype_score"] - .5
    n["x"] = n["ss_orig"] - .5; n["y"] = n["stereotype_score"] - .5
    llm = label_fix.fix_scores(pd.concat([pd.read_csv(f) for f in sorted((SCORES_DIR / "noise_floor").glob("*_llm.csv"))]))
    nl = re9.build_noise_paired(pd.concat([noise[noise["prompt_id"].isin(set(llm["prompt_id"]))], llm]), cp)
    nl = nl[nl["pivot"] == "llm"].copy(); nl["x"] = nl["ss_orig"] - .5; nl["y"] = nl["stereotype_score"] - .5
    within = {}
    for l in ["hi", "es"]:
        q = m[m["language"] == l]
        items = set(q["base_prompt_id"])
        qa = q[q["all_changed"]]
        qn = q[~q["model"].isin(aya)]
        pe = p[(p["language"] == l) & p["base_prompt_id"].isin(items)]
        ne = n[n["base_prompt_id"].isin(items)]
        le = nl[nl["base_prompt_id"].isin(items)]
        within[l] = {"items": int(q["prompt_id"].nunique()), "slope": slope(q["x"], q["y"]),
                     "items_all_changed": int(qa["prompt_id"].nunique()),
                     "slope_all_changed": slope(qa["x"], qa["y"]) if len(qa) > 20 else None,
                     "slope_no_aya": slope(qn["x"], qn["y"]),
                     "slope_cross_same": slope(pe["x"], pe["y"]),
                     "slope_cross_same_no_aya": slope(pe.loc[~pe["model"].isin(aya), "x"],
                                                      pe.loc[~pe["model"].isin(aya), "y"]),
                     "slope_en_mt_same": slope(ne["x"], ne["y"]),
                     "items_en_llm_overlap": int(le["base_prompt_id"].nunique()),
                     "slope_en_llm_overlap": slope(le["x"], le["y"]) if le["base_prompt_id"].nunique() >= 15 else None}
    N["within"] = within
    json.dump(N, open(OUT / "item_level_tests.json", "w"), indent=1, default=float)

    # tab_mixed is written by 22_translation_quality.py (adds the fully-preserved columns)
    print(json.dumps(N, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
