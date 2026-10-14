#!/usr/bin/env python3
"""
09_reanalysis.py — Paired-test reanalysis of the audit (CPU only)
==================================================================
Recomputes every headline number from the existing score files
(results/all_results.csv, results/noise_floor_all_results.csv) under a
corrected protocol. No model is re-run.

  (0) Collapsed minimal pairs. Translation (and round-trip paraphrase) can
      erase the only token distinguishing s+ from s- (Swahili "a-" and
      Spanish pro-drop erase he/she; Hindi "usne" is gender-neutral). Such a
      probe has SS = 0.5 by construction and carries no information. We flag
      every probe whose s+ and s- are identical after NFKC, case-folding, and
      stripping punctuation/whitespace, and drop the (probe, language) pair.
  (1) Layer B probes are written natively in one language and have no
      English counterpart, so they cannot enter a paired contrast. DFG and
      every test below use Layer A only.
  (2) Per model x language: sign-flip permutation test on paired
      d_i = SS_lang,i - SS_en,i, BH-FDR across all cells.
  (3) Analytic null: under H0 (E[d]=0) the expected |mean d| is
      sigma_d * sqrt(2 / (pi n)). Reported next to the empirical floor.
  (4) Bootstrap (resampling base prompts jointly across languages and
      pivots) of DFG minus the model's own paraphrase floor.
  (5) Binary-preference SS (StereoSet's definition: share of probes where
      l(s+) > l(s-)) alongside the soft SS used in the paper.
  (6) Probe-level variance decomposition: cross-lingual vs paraphrase vs
      rerun (4-bit inference nondeterminism) per-probe deviation.
  (7) Paraphrase fidelity: embedding similarity of each round-trip
      paraphrase to its source, and the floor's spread across pivots.
  (8) Regression refit on Layer A with cluster-robust SEs; provenance
      re-tested at the model level (n = 10) to avoid pseudo-replication.
  (9) Korean-reference ranking check.

Outputs go to results/reanalysis/ and paper/tables/, figures to
paper/figures/.

Usage:
    python 09_reanalysis.py                 # everything
    python 09_reanalysis.py --skip-embeddings  # no sentence-transformers
"""

import argparse
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_DIR, RESULTS_DIR, SCORES_DIR, FIGURES_DIR, TABLES_DIR, MODEL_REGISTRY, EXTRA_LANGUAGES

OUT_DIR = RESULTS_DIR / "reanalysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 20000
N_BOOT = 5000
SEED = 20260928
LANG_ORDER = ["fr", "es", "zh-CN", "ko", "ar", "hi", "sw"]
PIVOTS = ["de", "ja", "fi"]


# ══════════════════════════════════════════════════════════════════════════════
# Inputs, with benchmark stereotype roles corrected (label_fix.py)
# ══════════════════════════════════════════════════════════════════════════════

def scores_dir():
    """v2 scores (fixed scorer) when the re-run is complete, else the v1
    scores with a loud warning: v1 skipped the first continuation token."""
    if (SCORES_DIR / "all_results.csv").exists() and (SCORES_DIR / "noise_floor_all_results.csv").exists():
        return SCORES_DIR
    print("  ⚠⚠ USING v1 SCORES (first continuation token unscored for 9/10 models). "
          "Numbers are NOT valid for the paper until 02 + 05 are re-run into results/v2/.")
    return RESULTS_DIR


def load_inputs(raw: bool = False):
    """prompts, paraphrases, audit scores, noise-floor scores. Unless raw,
    StereoSet/BBQ s+/s-/unrelated roles are corrected and unrecoverable BBQ
    probes dropped, in the texts and the scores alike."""
    import label_fix
    prompts = json.load(open(DATA_DIR / "prompts.json"))
    paraphrases = json.load(open(DATA_DIR / "noise_floor_paraphrases.json"))
    src = scores_dir()
    df = pd.read_csv(src / "all_results.csv")
    noise = pd.read_csv(src / "noise_floor_all_results.csv")
    if raw:
        return prompts, paraphrases, df, noise
    prompts = [q for q in map(label_fix.fix_prompt, prompts) if q is not None]
    fixed_para = {}
    for seed, variants in paraphrases.items():
        fixed_para[seed] = {}
        for pid, v in variants.items():
            q = label_fix.fix_prompt({**v, "id": pid})
            if q is not None:
                fixed_para[seed][pid] = {k: q[k] for k in v}
    return prompts, fixed_para, label_fix.fix_scores(df), label_fix.fix_scores(noise)


# ══════════════════════════════════════════════════════════════════════════════
# (0) Collapsed-pair audit
# ══════════════════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    """NFKC + casefold, then drop punctuation (P*), separators (Z*), and
    whitespace. Letters and combining marks are kept, so Devanagari/Hangul
    vowel signs still distinguish e.g. ladki vs ladka."""
    s = unicodedata.normalize("NFKC", s).casefold()
    return "".join(ch for ch in s
                   if not unicodedata.category(ch)[0] in ("P", "Z") and not ch.isspace())


def is_collapsed(a: str, b: str) -> bool:
    return _norm(a) == _norm(b)


def audit_collapses(prompts: list[dict], paraphrases: dict) -> tuple[set, set, pd.DataFrame]:
    """Returns (collapsed prompt ids, collapsed (prompt_id, pivot) pairs, report)."""
    rows = []
    collapsed_ids = set()
    for p in prompts:
        if is_collapsed(p["stereotype"], p["anti_stereotype"]):
            collapsed_ids.add(p["id"])
            rows.append({"set": "audit", "language": p["language"], "prompt_id": p["id"],
                         "layer": p["layer"], "source": p.get("source", "cultural_native"),
                         "stereotype": p["stereotype"], "anti_stereotype": p["anti_stereotype"],
                         "stereotype_en": p.get("stereotype_en"),
                         "anti_stereotype_en": p.get("anti_stereotype_en")})
    collapsed_para = set()
    for seed_key, variants in paraphrases.items():
        for pid, v in variants.items():
            if is_collapsed(v["stereotype"], v["anti_stereotype"]):
                collapsed_para.add((pid, v["pivot"]))
                rows.append({"set": f"paraphrase_{v['pivot']}", "language": "en",
                             "prompt_id": pid, "layer": "A", "source": None,
                             "stereotype": v["stereotype"], "anti_stereotype": v["anti_stereotype"],
                             "stereotype_en": None, "anti_stereotype_en": None})
    report = pd.DataFrame(rows)
    return collapsed_ids, collapsed_para, report


# ══════════════════════════════════════════════════════════════════════════════
# Data assembly
# ══════════════════════════════════════════════════════════════════════════════

def binary_pref(lp_s, lp_a):
    """StereoSet SS: 1 if s+ preferred, 0 if s- preferred, 0.5 on exact tie."""
    return np.where(lp_s > lp_a, 1.0, np.where(lp_s < lp_a, 0.0, 0.5))


def build_paired(df: pd.DataFrame, collapsed_ids: set) -> pd.DataFrame:
    """Layer A, one row per (model, base prompt, non-English language) with
    English partner; pairs where the translated probe collapsed are dropped."""
    a = df[df["layer"] == "A"].copy()
    a["ss_bin"] = binary_pref(a["logprob_stereotype"].values, a["logprob_anti_stereotype"].values)
    en = a[a["language"] == "en"][["model", "base_prompt_id", "stereotype_score", "ss_bin"]] \
        .rename(columns={"stereotype_score": "ss_en", "ss_bin": "bin_en"})
    xl = a[a["language"] != "en"]
    xl = xl[~xl["prompt_id"].isin(collapsed_ids)]
    m = xl.merge(en, on=["model", "base_prompt_id"], how="inner")
    m["d"] = m["stereotype_score"] - m["ss_en"]
    m["d_bin"] = m["ss_bin"] - m["bin_en"]
    return m


def build_noise_paired(noise: pd.DataFrame, collapsed_para: set) -> pd.DataFrame:
    orig = noise[noise["seed"] == -1][["model", "prompt_id", "stereotype_score",
                                       "logprob_stereotype", "logprob_anti_stereotype"]] \
        .rename(columns={"stereotype_score": "ss_orig", "logprob_stereotype": "lps_o",
                         "logprob_anti_stereotype": "lpa_o"})
    para = noise[noise["seed"] != -1].copy()
    keep = [(pid, pv) not in collapsed_para for pid, pv in zip(para["prompt_id"], para["pivot"])]
    para = para[keep]
    m = para.merge(orig, on=["model", "prompt_id"])
    m["d"] = m["stereotype_score"] - m["ss_orig"]
    m["d_bin"] = binary_pref(m["logprob_stereotype"].values, m["logprob_anti_stereotype"].values) \
        - binary_pref(m["lps_o"].values, m["lpa_o"].values)
    m["base_prompt_id"] = m["prompt_id"].str.replace(r"_en$", "", regex=True)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# (2)-(4) Tests
# ══════════════════════════════════════════════════════════════════════════════

def sign_flip_p(d: np.ndarray, rng: np.random.Generator, n_perm: int = N_PERM) -> float:
    """Two-sided sign-flip permutation p for H0: d symmetric about 0."""
    d = d[~np.isnan(d)]
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = np.abs(signs @ d) / d.size
    return (1 + np.sum(null >= obs - 1e-15)) / (n_perm + 1)


def analytic_null(d: np.ndarray) -> float:
    """E|mean d| under H0 with d ~ N(0, sigma^2): sigma * sqrt(2/(pi n))."""
    d = d[~np.isnan(d)]
    return d.std(ddof=1) * np.sqrt(2 / (np.pi * d.size))


def bh(p: np.ndarray) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests
    return multipletests(p, method="fdr_bh")[1]


def cell_tests(paired: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for (model, lang), g in paired.groupby(["model", "language"]):
        d = g["d"].to_numpy()
        db = g["d_bin"].to_numpy()
        rows.append({
            "model": model, "language": lang, "n": len(d),
            "mean_d": d.mean(), "dfg": abs(d.mean()),
            "sigma_d": d.std(ddof=1), "analytic_null": analytic_null(d),
            "p_perm": sign_flip_p(d, rng),
            "mean_d_bin": db.mean(), "dfg_bin": abs(db.mean()),
            "p_perm_bin": sign_flip_p(db, rng),
        })
    out = pd.DataFrame(rows)
    out["q_bh"] = bh(out["p_perm"].to_numpy())
    out["q_bh_bin"] = bh(out["p_perm_bin"].to_numpy())
    return out


def floor_tests(npaired: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for (model, pivot), g in npaired.groupby(["model", "pivot"]):
        d = g["d"].to_numpy()
        db = g["d_bin"].to_numpy()
        rows.append({"model": model, "pivot": pivot, "n": len(d),
                     "mean_d": d.mean(), "floor": abs(d.mean()),
                     "sigma_d": d.std(ddof=1), "analytic_null": analytic_null(d),
                     "p_perm": sign_flip_p(d, rng),
                     "floor_bin": abs(db.mean())})
    out = pd.DataFrame(rows)
    out["q_bh"] = bh(out["p_perm"].to_numpy())
    return out


def bootstrap_dfg_minus_floor(paired: pd.DataFrame, npaired: pd.DataFrame, rng,
                              col: str = "d", n_boot: int = N_BOOT) -> pd.DataFrame:
    """Per model: resample English base prompts with replacement (one draw
    shared by all 7 languages and 3 pivots, preserving the dependence the
    shared English score induces), and recompute
        mean_lang |mean d_lang|  -  mean_pivot |mean d_pivot|."""
    rows = []
    for model in sorted(paired["model"].unique()):
        pm = paired[paired["model"] == model].pivot_table(
            index="base_prompt_id", columns="language", values=col)
        nm = npaired[npaired["model"] == model].pivot_table(
            index="base_prompt_id", columns="pivot", values=col)
        M = pm.join(nm, how="outer").reindex(columns=LANG_ORDER + PIVOTS)
        k = len(LANG_ORDER)
        X = M.to_numpy()
        W = (~np.isnan(X)).astype(float)
        X0 = np.nan_to_num(X)
        n = X.shape[0]
        idx = rng.integers(0, n, size=(n_boot, n))
        # counts[b, i] = how many times probe i drawn in bootstrap b
        counts = np.zeros((n_boot, n))
        np.add.at(counts, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1)
        means = (counts @ X0) / np.maximum(counts @ W, 1)
        dfg_b = np.abs(means[:, :k]).mean(1)
        floor_b = np.abs(means[:, k:]).mean(1)
        diff_b = dfg_b - floor_b
        full = np.nansum(X0, 0) / W.sum(0)
        dfg, floor = np.abs(full[:k]).mean(), np.abs(full[k:]).mean()
        rows.append({
            "model": model, "mean_dfg": dfg, "floor": floor, "diff": dfg - floor,
            "dfg_lo": np.percentile(dfg_b, 2.5), "dfg_hi": np.percentile(dfg_b, 97.5),
            "floor_lo": np.percentile(floor_b, 2.5), "floor_hi": np.percentile(floor_b, 97.5),
            "diff_lo": np.percentile(diff_b, 2.5), "diff_hi": np.percentile(diff_b, 97.5),
            "p_diff_le_0": float(np.mean(diff_b <= 0)),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# (6) Probe-level variance decomposition
# ══════════════════════════════════════════════════════════════════════════════

def variance_decomposition(paired: pd.DataFrame, npaired: pd.DataFrame,
                           df: pd.DataFrame, noise: pd.DataFrame) -> dict:
    # rerun noise: English original re-scored in the noise-floor job vs main audit
    en_main = df[(df["language"] == "en") & (df["layer"] == "A")][["model", "prompt_id", "stereotype_score"]]
    en_rerun = noise[noise["seed"] == -1][["model", "prompt_id", "stereotype_score"]]
    rr = en_main.merge(en_rerun, on=["model", "prompt_id"], suffixes=("_main", "_rerun"))
    rr["d"] = rr["stereotype_score_rerun"] - rr["stereotype_score_main"]

    def summ(d):
        d = np.asarray(d)
        return {"n": int(d.size), "median_abs": float(np.median(np.abs(d))),
                "mean_abs": float(np.mean(np.abs(d))), "sd": float(np.std(d, ddof=1)),
                "mean_signed": float(np.mean(d))}

    out = {"cross_lingual": summ(paired["d"]), "paraphrase": summ(npaired["d"]),
           "rerun_4bit": summ(rr["d"])}
    out["ratio_median_abs_cross_over_para"] = out["cross_lingual"]["median_abs"] / out["paraphrase"]["median_abs"]
    out["ratio_sd_cross_over_para"] = out["cross_lingual"]["sd"] / out["paraphrase"]["sd"]
    # variance attributable to language beyond paraphrase-level perturbation
    out["excess_variance_cross_minus_para"] = out["cross_lingual"]["sd"] ** 2 - out["paraphrase"]["sd"] ** 2
    out["excess_share_of_cross_variance"] = out["excess_variance_cross_minus_para"] / out["cross_lingual"]["sd"] ** 2

    # per model: probe-matched Wilcoxon of mean_lang |d| vs mean_pivot |d|
    per_model = []
    for model in sorted(paired["model"].unique()):
        c = paired[paired["model"] == model].groupby("base_prompt_id")["d"].apply(lambda x: np.abs(x).mean())
        p = npaired[npaired["model"] == model].groupby("base_prompt_id")["d"].apply(lambda x: np.abs(x).mean())
        j = pd.concat([c.rename("cross"), p.rename("para")], axis=1).dropna()
        w = stats.wilcoxon(j["cross"], j["para"])
        sd_c = paired.loc[paired["model"] == model, "d"].std()
        sd_p = npaired.loc[npaired["model"] == model, "d"].std()
        per_model.append({"model": model, "n_probes": len(j),
                          "median_abs_cross": float(np.median(np.abs(paired.loc[paired["model"] == model, "d"]))),
                          "median_abs_para": float(np.median(np.abs(npaired.loc[npaired["model"] == model, "d"]))),
                          "sd_cross": sd_c, "sd_para": sd_p, "sd_ratio": sd_c / sd_p,
                          "wilcoxon_p": float(w.pvalue)})
    pm = pd.DataFrame(per_model)
    pm["wilcoxon_q"] = bh(pm["wilcoxon_p"].to_numpy())
    # per language (pooled over models)
    per_lang = paired.groupby("language")["d"].agg(
        median_abs=lambda x: np.median(np.abs(x)), sd="std").reindex(LANG_ORDER)
    per_pivot = npaired.groupby("pivot")["d"].agg(
        median_abs=lambda x: np.median(np.abs(x)), sd="std").reindex(PIVOTS)
    return out, pm, per_lang, per_pivot


# ══════════════════════════════════════════════════════════════════════════════
# (7) Paraphrase fidelity
# ══════════════════════════════════════════════════════════════════════════════

def paraphrase_fidelity(prompts, paraphrases, collapsed_para) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    en = {p["id"]: p for p in prompts if p["language"] == "en" and p["layer"] == "A"}
    rows = []
    for seed_key, variants in paraphrases.items():
        for pid, v in variants.items():
            for f in ["context", "stereotype", "anti_stereotype"]:
                rows.append({"prompt_id": pid, "pivot": v["pivot"], "field": f,
                             "orig": en[pid][f], "para": v[f],
                             "identical": en[pid][f].strip() == v[f].strip()})
    t = pd.DataFrame(rows)
    e1 = model.encode(t["orig"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    e2 = model.encode(t["para"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    t["cos"] = (e1 * e2).sum(1)
    # a pair "flips" if the paraphrased s+ now matches the original s- better than the original s+
    return t


# ══════════════════════════════════════════════════════════════════════════════
# (8) Regression
# ══════════════════════════════════════════════════════════════════════════════

def regressions(df: pd.DataFrame, collapsed_ids: set) -> dict:
    import statsmodels.formula.api as smf
    a = df[(df["layer"] == "A") & (~df["prompt_id"].isin(collapsed_ids))].copy()
    res_map = {"high": 0, "mid": 1, "low": 2}
    a["resource_numeric"] = a["resource_level"].map(res_map)
    out = {"n_obs": int(len(a)), "n_clusters_prompt": int(a["base_prompt_id"].nunique())}
    f = "stereotype_score ~ C(resource_level, Treatment('high')) + C(model_group_label, Treatment('English-Centric')) + C(category)"
    fit = smf.ols(f, data=a).fit()
    fit_cl = smf.ols(f, data=a).fit(cov_type="cluster", cov_kwds={"groups": a["base_prompt_id"].astype("category").cat.codes})
    # two-way: base prompt and model -> use model clusters as the conservative variant (10 clusters)
    fit_m = smf.ols(f, data=a).fit(cov_type="cluster", cov_kwds={"groups": a["model"].astype("category").cat.codes})
    terms = {}
    for t in fit.params.index:
        if t.startswith("C(category)"):
            continue
        terms[t] = {"coef": fit.params[t], "p_ols": fit.pvalues[t],
                    "p_cluster_prompt": fit_cl.pvalues[t], "p_cluster_model": fit_m.pvalues[t]}
    out["terms"] = terms
    out["r2"] = fit.rsquared
    out["f_pvalue"] = float(fit.f_pvalue)
    out["category_coefs"] = {t: {"coef": fit.params[t], "p_cluster_prompt": fit_cl.pvalues[t]}
                             for t in fit.params.index if t.startswith("C(category)")}

    # provenance at the unit where it varies: the model (n = 10)
    per_model = a.groupby(["model", "model_group_label"])["stereotype_score"].mean().reset_index()
    groups = [g["stereotype_score"].to_numpy() for _, g in per_model.groupby("model_group_label")]
    h, p = stats.kruskal(*groups)
    out["provenance_model_level"] = {
        "per_model_mean_ss": per_model.set_index("model")["stereotype_score"].to_dict(),
        "kruskal_H": float(h), "kruskal_p": float(p),
        "group_means": per_model.groupby("model_group_label")["stereotype_score"].mean().to_dict(),
    }
    # permutation of group labels over the 10 models (exact: 10!/(4!3!3!) = 4200 partitions)
    from itertools import combinations
    vals = per_model.set_index("model")["stereotype_score"]
    labels = per_model.set_index("model")["model_group_label"]
    obs_b_minus_a = vals[labels == "Multilingual-Native"].mean() - vals[labels == "English-Centric"].mean()
    models = list(vals.index)
    null = []
    for b in combinations(models, 3):
        rest = [m for m in models if m not in b]
        for a_ in combinations(rest, 4):
            null.append(vals[list(b)].mean() - vals[list(a_)].mean())
    null = np.array(null)
    out["provenance_model_level"]["B_minus_A_obs"] = float(obs_b_minus_a)
    out["provenance_model_level"]["B_minus_A_exact_perm_p"] = float(np.mean(np.abs(null) >= abs(obs_b_minus_a) - 1e-12))
    return out



# ══════════════════════════════════════════════════════════════════════════════
# Post-hoc additions: extra languages (11_add_language.py), human pivot (12_…)
# ══════════════════════════════════════════════════════════════════════════════

def load_extra_languages(df: pd.DataFrame, collapsed_ids: set) -> tuple[pd.DataFrame, list]:
    """Append results/extra_lang/*.csv for languages scored by all models;
    add their collapsed probes to collapsed_ids (mutated)."""
    added = []
    for lang in EXTRA_LANGUAGES:
        files = sorted((SCORES_DIR / "extra_lang").glob(f"*_{lang}.csv"))
        if not files:
            continue
        import label_fix
        x = label_fix.fix_scores(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
        if x["model"].nunique() < df["model"].nunique():
            print(f"  ⚠ {lang}: only {x['model'].nunique()} models scored — skipping")
            continue
        import label_fix
        for p in json.load(open(DATA_DIR / f"prompts_{lang}.json")):
            p = label_fix.fix_prompt(p)
            if p is not None and is_collapsed(p["stereotype"], p["anti_stereotype"]):
                collapsed_ids.add(p["id"])
        df = pd.concat([df, x[[c for c in df.columns if c in x.columns]]], ignore_index=True)
        added.append(lang)
    return df, added


def human_pivot_comparison(noise: pd.DataFrame, collapsed_para: set, rng) -> pd.DataFrame | None:
    files = sorted(f for pv in ("human", "llm") for f in (SCORES_DIR / "noise_floor").glob(f"*_noise_floor_{pv}.csv"))
    if not files:
        return None
    import label_fix
    h = label_fix.fix_scores(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    subset = set(h["prompt_id"])
    nz = pd.concat([noise[noise["prompt_id"].isin(subset)], h], ignore_index=True)
    np_ = build_noise_paired(nz, collapsed_para)
    rows = []
    for (model, pivot), g in np_.groupby(["model", "pivot"]):
        d = g["d"].to_numpy()
        rows.append({"model": model, "pivot": pivot, "n": len(d), "floor": abs(d.mean()),
                     "median_abs_d": float(np.median(np.abs(d))), "sd_d": d.std(ddof=1),
                     "analytic_null": analytic_null(d), "p_perm": sign_flip_p(d, rng)})
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# Tables / figure
# ══════════════════════════════════════════════════════════════════════════════

SHORT = {"aya-23-8b": "Aya-23-8B", "aya-expanse-8b": "Aya-Expanse-8B", "bloomz-7b": "BLOOMZ-7B",
         "gemma2-9b": "Gemma-2-9B", "llama3.1-8b": "Llama-3.1-8B", "mistral-7b": "Mistral-7B",
         "olmo2-7b": "OLMo-2-7B", "qwen2.5-7b": "Qwen-2.5-7B", "solar-10.7b": "SOLAR-10.7B",
         "yi-1.5-9b": "Yi-1.5-9B"}
LANG_HDR = {"en": "En", "fr": "Fr", "es": "Es", "zh-CN": "Zh", "ko": "Ko", "ar": "Ar", "hi": "Hi", "sw": "Sw",
            "am": "Am"}


def write_tables(ss_mat, cells, boot, clfi):
    # Table: DFG + CLFI + floor (main text)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Mean DFG (Layer~A, collapsed pairs removed) against each model's own "
             r"paraphrase floor, with 95\% bootstrap CIs on the difference (base prompts resampled "
             r"jointly across languages and pivots). CLFI $=1-\overline{\text{DFG}}$. "
             rf"Sig.: cells (of {len(LANG_ORDER)}) with BH-FDR $q<0.05$ under a sign-flip permutation test.}}",
             r"\label{tab:clfi}", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"\textbf{Model} & \textbf{CLFI} & $\overline{\textbf{DFG}}$ & \textbf{Floor} & \textbf{DFG$-$Floor} [95\% CI] & \textbf{Sig.} \\",
             r"\midrule"]
    sig = cells.groupby("model")["q_bh"].apply(lambda q: int((q < 0.05).sum()))
    for _, r in boot.sort_values("mean_dfg").iterrows():
        m = r["model"]
        lines.append(f"{SHORT[m]} & {clfi[m]:.3f} & {r['mean_dfg']:.3f} & {r['floor']:.3f} & "
                     f"${r['diff']:+.3f}$ [${r['diff_lo']:+.3f}$, ${r['diff_hi']:+.3f}$] & {sig[m]}/{len(LANG_ORDER)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "table1_clfi.tex").write_text("\n".join(lines) + "\n")

    # SS matrix
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{Mean soft SS by model and language (Layer~A only, collapsed minimal pairs removed). "
             r"Bracketed: binary-preference SS (StereoSet definition).}",
             r"\label{tab:ss_matrix}", r"\begin{tabular}{l" + "c" * (len(LANG_ORDER) + 1) + "}", r"\toprule",
             r"\textbf{Model} & " + " & ".join(rf"\textbf{{{LANG_HDR[l]}}}" for l in ["en"] + LANG_ORDER) + r" \\",
             r"\midrule"]
    for m in sorted(ss_mat.index.get_level_values(0).unique()):
        vals = [f"{ss_mat.loc[(m, l), 'soft']:.3f} [{ss_mat.loc[(m, l), 'bin']:.2f}]" for l in ["en"] + LANG_ORDER]
        lines.append(f"{SHORT[m]} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES_DIR / "table2_ss_matrix.tex").write_text("\n".join(lines) + "\n")

    # DFG matrix with significance marks
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{DFG $=|\overline{d}|$ by model and language (Layer~A, paired, collapsed pairs removed). "
             rf"$^{{\dagger}}$: sign-flip permutation $q<0.05$ after BH-FDR over all {len(cells)} cells. "
             r"Last column: analytic null $\mathbb{E}|\overline{d}|$ under $H_0$, averaged over languages.}",
             r"\label{tab:dfg_full}", r"\begin{tabular}{l" + "c" * (len(LANG_ORDER) + 1) + "}", r"\toprule",
             r"\textbf{Model} & " + " & ".join(rf"\textbf{{{LANG_HDR[l]}}}" for l in LANG_ORDER) + r" & \textbf{Null} \\",
             r"\midrule"]
    for m in sorted(cells["model"].unique()):
        c = cells[cells["model"] == m].set_index("language")
        vals = []
        for l in LANG_ORDER:
            v = f".{round(c.loc[l, 'dfg'] * 1000):03d}"
            vals.append(v + (r"$^{\dagger}$" if c.loc[l, "q_bh"] < 0.05 else ""))
        lines.append(f"{SHORT[m]} & " + " & ".join(vals) + f" & .{round(c['analytic_null'].mean() * 1000):03d} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES_DIR / "table6_dfg_matrix.tex").write_text("\n".join(lines) + "\n")


def plot_fig1a(boot: pd.DataFrame, cells: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = boot.sort_values("diff").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    y = np.arange(len(b))
    ax.axvline(0, color="#888888", lw=1, ls="--", zorder=1)
    ax.hlines(y, b["diff_lo"], b["diff_hi"], color="#3B6EA8", lw=2.2, zorder=2)
    ax.plot(b["diff"], y, "o", color="#3B6EA8", ms=7, mec="white", zorder=3)
    nsig = cells.groupby("model")["q_bh"].apply(lambda q: int((q < 0.05).sum()))
    for i, r in b.iterrows():
        ax.text(max(b["diff_hi"]) + 0.0015, i, f"{nsig[r['model']]}/{len(LANG_ORDER)}", va="center", fontsize=8, color="#555555")
    ax.set_yticks(y, [SHORT[m] for m in b["model"]])
    ax.set_xlabel(r"mean DFG $-$ own paraphrase floor (95% bootstrap CI)")
    ax.text(max(b["diff_hi"]) + 0.0015, len(b) - 0.35, "sig. cells\n(BH q<.05)", va="bottom",
            fontsize=7.5, color="#555555")
    ax.set_ylim(-0.6, len(b) + 0.4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig0_noise_floor_headline.{ext}", dpi=300)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    prompts, paraphrases, df, noise = load_inputs()
    summary = {"scores_version": "v2" if scores_dir() == SCORES_DIR else "v1 (INVALID: first-token bug)",
               "label_fix": json.load(open(DATA_DIR / "label_fix.json"))["report"]}
    summary["label_fix"]["bbq_dropped"] = len(summary["label_fix"]["bbq_dropped"])

    # ── counts (for the text) ──
    en = [p for p in prompts if p["language"] == "en"]
    summary["counts"] = {
        "prompt_instances": len(prompts),
        "unique_base_ids": len({p["base_id"] for p in prompts}),
        "english_layerA": len(en),
        "english_by_source": pd.Series([p.get("source") for p in en]).value_counts().to_dict(),
        "layerB_probes": sum(p["layer"] == "B" for p in prompts),
        "translated_layerA": sum(p["language"] != "en" and p["layer"] == "A" for p in prompts),
        "rows_scored": len(df),
    }

    # ── (0) collapses ──
    collapsed_ids, collapsed_para, report = audit_collapses(prompts, paraphrases)
    report.to_csv(OUT_DIR / "collapsed_pairs.csv", index=False)
    by_lang = report[report["set"] == "audit"].groupby("language").size()
    by_piv = report[report["set"] != "audit"].groupby("set").size()
    summary["collapses"] = {"audit_by_language": by_lang.to_dict(), "paraphrase_by_pivot": by_piv.to_dict(),
                            "audit_total": int(by_lang.sum()), "paraphrase_total": int(by_piv.sum())}
    print("Collapsed pairs by language:", by_lang.to_dict())
    print("Collapsed paraphrase pairs:", by_piv.to_dict())

    # ── effect of collapses on SS: before/after ──
    a = df[df["layer"] == "A"]
    before = a.groupby(["model", "language"])["stereotype_score"].mean()
    after = a[~a["prompt_id"].isin(collapsed_ids)].groupby(["model", "language"])["stereotype_score"].mean()
    summary["collapse_max_shift_in_cell_mean"] = float((before - after).abs().max())

    # ── post-hoc languages ──
    df, extra = load_extra_languages(df, collapsed_ids)
    LANG_ORDER.extend(l for l in extra if l not in LANG_ORDER)
    summary["extra_languages"] = extra
    a = df[df["layer"] == "A"]

    # ── paired sets ──
    paired = build_paired(df, collapsed_ids)
    npaired = build_noise_paired(noise, collapsed_para)

    # ── SS matrix (soft + binary) ──
    clean_a = a[~a["prompt_id"].isin(collapsed_ids)].copy()
    clean_a["ss_bin"] = binary_pref(clean_a["logprob_stereotype"].values, clean_a["logprob_anti_stereotype"].values)
    ss_mat = clean_a.groupby(["model", "language"]).agg(soft=("stereotype_score", "mean"),
                                                        bin=("ss_bin", "mean"), n=("stereotype_score", "size"))
    ss_mat.to_csv(OUT_DIR / "ss_matrix_clean.csv")

    # ── (2)(3) per-cell tests ──
    cells = cell_tests(paired, rng)
    cells.to_csv(OUT_DIR / "cell_tests.csv", index=False)
    floors = floor_tests(npaired, rng)
    floors.to_csv(OUT_DIR / "floor_tests.csv", index=False)
    summary["cells"] = {
        "n_cells": len(cells),
        "n_sig_p05_uncorrected": int((cells["p_perm"] < 0.05).sum()),
        "n_sig_bh05": int((cells["q_bh"] < 0.05).sum()),
        "n_sig_bh05_binary": int((cells["q_bh_bin"] < 0.05).sum()),
        "sig_cells": cells.loc[cells["q_bh"] < 0.05, ["model", "language", "mean_d", "q_bh"]].to_dict("records"),
        "dfg_range": [float(cells.groupby("model")["dfg"].mean().min()), float(cells.groupby("model")["dfg"].mean().max())],
        "analytic_null_mean": float(cells["analytic_null"].mean()),
        "sigma_d_pooled": float(paired["d"].std()),
        "n_per_cell_median": float(cells["n"].median()),
        "analytic_null_pooled": float(paired["d"].std() * np.sqrt(2 / (np.pi * cells["n"].median()))),
    }
    summary["floor"] = {
        "n_pivot_cells": len(floors),
        "n_sig_bh05": int((floors["q_bh"] < 0.05).sum()),
        "per_model_floor": floors.groupby("model")["floor"].mean().to_dict(),
        "per_model_floor_sd_across_pivots": floors.groupby("model")["floor"].std().to_dict(),
        "median_floor": float(floors.groupby("model")["floor"].mean().median()),
        "floor_range": [float(floors.groupby("model")["floor"].mean().min()),
                        float(floors.groupby("model")["floor"].mean().max())],
        "median_floor_by_pivot": floors.groupby("pivot")["floor"].median().to_dict(),
        "analytic_null_floor_mean": float(floors["analytic_null"].mean()),
    }

    # ── (4) bootstrap ──
    boot = bootstrap_dfg_minus_floor(paired, npaired, rng)
    boot_bin = bootstrap_dfg_minus_floor(paired, npaired, rng, col="d_bin")
    boot.to_csv(OUT_DIR / "bootstrap_dfg_minus_floor.csv", index=False)
    boot_bin.to_csv(OUT_DIR / "bootstrap_dfg_minus_floor_binary.csv", index=False)
    summary["bootstrap"] = boot.set_index("model").round(5).to_dict("index")
    summary["bootstrap_n_ci_excludes_0"] = int((boot["diff_lo"] > 0).sum())
    summary["bootstrap_binary_n_ci_excludes_0"] = int((boot_bin["diff_lo"] > 0).sum())

    # ── CLFI (max DFG = 1) ──
    clfi = (1 - cells.groupby("model")["dfg"].mean()).to_dict()
    summary["clfi"] = clfi
    # group CLFI mean and SD across models
    grp = pd.Series({m: MODEL_REGISTRY[m]["group_label"] for m in clfi})
    cl = pd.Series(clfi)
    summary["group_clfi"] = {g: {"mean": float(cl[grp == g].mean()), "sd_across_models": float(cl[grp == g].std())}
                             for g in grp.unique()}
    summary["group_ss_sd_per_probe"] = clean_a.groupby("model_group_label")["stereotype_score"].std().to_dict()
    kw = stats.kruskal(*[cells[cells["model"].map(grp) == g]["dfg"] for g in grp.unique()])
    summary["group_kruskal_on_cell_dfg"] = {"H": float(kw.statistic), "p": float(kw.pvalue)}

    # ── (6) variance decomposition ──
    vd, vd_model, vd_lang, vd_piv = variance_decomposition(paired, npaired, df, noise)
    vd_model.to_csv(OUT_DIR / "variance_decomposition_by_model.csv", index=False)
    summary["variance_decomposition"] = vd
    summary["variance_by_language"] = vd_lang.round(4).to_dict("index")
    summary["variance_by_pivot"] = vd_piv.round(4).to_dict("index")
    summary["variance_models_cross_gt_para_bh05"] = int((vd_model["wilcoxon_q"] < 0.05).sum())

    # ── (8) regressions ──
    summary["regression"] = regressions(df, collapsed_ids)

    # ── (9) Korean reference ──
    ko_ref = {}
    for m in sorted(df["model"].unique()):
        s = clean_a[clean_a["model"] == m].groupby("language")["stereotype_score"].mean()
        ko_ref[m] = 1 - (s.drop("ko") - s["ko"]).abs().mean()
    en_rank = pd.Series(clfi).rank(ascending=False)
    ko_rank = pd.Series(ko_ref).rank(ascending=False)
    rho = stats.spearmanr(en_rank, ko_rank.reindex(en_rank.index))
    summary["korean_reference"] = {"clfi_ko": ko_ref, "rank_en": en_rank.to_dict(), "rank_ko": ko_rank.to_dict(),
                                   "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
                                   "identical_ordering": bool((en_rank == ko_rank.reindex(en_rank.index)).all())}

    # ── worst-case language per model (model-card table) ──
    summary["worst_language"] = cells.loc[cells.groupby("model")["dfg"].idxmax(), ["model", "language", "dfg"]] \
        .set_index("model").to_dict("index")

    # ── Layer B descriptive only ──
    lb = df[df["layer"] == "B"]
    summary["layer_b"] = {"n_rows": int(len(lb)), "n_probes": int(lb["prompt_id"].nunique()),
                          "mean_ss": float(lb["stereotype_score"].mean()),
                          "mean_ss_layerA_same_languages": float(clean_a[clean_a["language"] != "en"]["stereotype_score"].mean())}

    # ── (7) paraphrase fidelity ──
    if not args.skip_embeddings:
        fid = paraphrase_fidelity(prompts, paraphrases, collapsed_para)
        fid.to_csv(OUT_DIR / "paraphrase_fidelity.csv", index=False)
        summary["paraphrase_fidelity"] = {
            "cos_by_pivot_field_mean": fid.groupby(["pivot", "field"])["cos"].mean().round(4).unstack().to_dict("index"),
            "cos_overall_mean": float(fid["cos"].mean()),
            "cos_overall_p10": float(fid["cos"].quantile(0.10)),
            "share_identical_by_pivot": fid.groupby("pivot")["identical"].mean().round(4).to_dict(),
        }

    hp = human_pivot_comparison(noise, collapsed_para, rng)
    if hp is not None:
        hp.to_csv(OUT_DIR / "human_pivot_comparison.csv", index=False)
        summary["extra_pivot"] = hp.groupby("pivot")[["floor", "median_abs_d", "sd_d"]].median().round(5).to_dict("index")

    write_tables(ss_mat, cells, boot, clfi)
    plot_fig1a(boot, cells)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_default)
    print(json.dumps({k: summary[k] for k in ["counts", "collapses", "cells", "floor",
                                               "bootstrap_n_ci_excludes_0", "variance_decomposition",
                                               "korean_reference", "group_clfi"]},
                     indent=1, default=_default))
    print(f"\n→ {OUT_DIR}")


if __name__ == "__main__":
    main()
