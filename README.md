# How Much of Cross-Lingual Bias Is Noise in Language Models?

Code, data, and results for a cross-lingual social-bias audit of ten
open-weights LLMs across eight languages, and for the **noise-floor test**
that is the paper's central result.

## TL;DR of the finding

We introduce two metrics — the **Deployment Fairness Gap (DFG)** and the
**Cross-Lingual Fairness Index (CLFI)** — and use them to rank ten models.
The ranking looks clean. It does not survive validation.

| Quantity | Value |
|---|---|
| Observed cross-lingual DFG (10 models) | 0.004 – 0.013 |
| Within-language noise floor (same aggregation) | median **0.0076**, range 0.0029 – 0.0210 |
| Minimum detectable DFG at 80% power, current design | **0.034** |
| Probes/language needed to resolve DFG = 0.010 | **≈4,500** (we used 390) |

The cross-lingual "signal" is the same size as the gap produced by
paraphrasing **English into English**. A power analysis agrees from an
independent direction: the design is 2.6–8.5× underpowered for the effects
it reports. We publish this as a negative result for our own rankings and a
generalizable one for the field.

> **Note on a corrected metric.** An earlier version of `05_noise_floor.py`
> computed the floor as the mean over probes of `|SS_para − SS_orig|`, while
> the reported DFG is a difference of probe-set means. These are not
> comparable — absolute-value-before-averaging prevents signed per-probe
> noise from cancelling — and the old figure was ~10× inflated. The floor is
> now computed at matching aggregation. Per-probe instability is still
> reported, under its own name, as support for the power analysis.

## Pipeline

Scripts run in numeric order. Only `02` needs a GPU.

| Script | Does | GPU |
|---|---|:--:|
| `01_build_prompts.py` | Builds the probe set: StereoSet / CrowS-Pairs / BBQ + 40 handcrafted templates (Layer A, translated to 7 languages) and 21 culturally-native probes (Layer B). Back-translation validated. | – |
| `02_run_audit.py` | Scores all 10 models over all prompts (4-bit NF4, sequential load/unload, checkpointed). | ✅ |
| `03_analyze.py` | DFG / CLFI, regressions, main figures and LaTeX tables. | – |
| `04_robustness.py` | Translation-quality confound, CLFI↔DFG redundancy, cluster-robust and mixed-effects reruns. | – |
| `05_noise_floor.py` | The noise-floor experiment: 3 round-trip paraphrase sets (en→{de,ja,fi}→en), scored identically to the main audit. | ✅ |
| `06_noise_floor_figure.py` | Figure 1a — DFG vs. noise floor, forest plot. | – |
| `07_power_analysis.py` | Power analysis + per-pivot floor breakdown. | – |
| `08_robustness_figure.py` | Figure 7 — drop-BLOOMZ group means. | – |
| `appendix_plots.py` | Figure 6 — per-category CLFI. | – |

```bash
pip install -r requirements.txt

python 01_build_prompts.py          # or --skip-translate to use the cached translations
python 02_run_audit.py --resume     # GPU; ~2-3 h for all 10 models
python 03_analyze.py
python 04_robustness.py

# noise floor: build paraphrases anywhere (no GPU/torch needed), score on GPU
python 05_noise_floor.py --build-paraphrases-only
python 05_noise_floor.py --resume

python 06_noise_floor_figure.py
python 07_power_analysis.py
python 08_robustness_figure.py
python appendix_plots.py
```

`05_noise_floor.py` deletes each model's HuggingFace cache after scoring it
(`--keep-model-cache` to disable); ten models otherwise need ~150–200 GB of
disk.

## Layout

```
MultilingualBiasAudit/
├── config.py                  models, languages, paths, inference settings
├── 01…08, appendix_plots.py   pipeline (above)
├── data/
│   ├── prompts.json                 3,117 prompt instances (8 languages)
│   ├── translation_cache.json       cached MT, for --skip-translate
│   ├── noise_floor_paraphrases.json cached round-trip paraphrases
│   └── validation_results.json      back-translation similarity scores
├── results/
│   ├── all_results.csv              31,170 scored assessments (10 models × 3,117)
│   ├── <model>_checkpoint.csv       per-model raw scores
│   ├── noise_floor/                 per-model noise-floor scorings (15,480 rows)
│   ├── noise_floor_summary.csv      matched floor + per-probe instability
│   ├── noise_floor_by_pivot.csv     floor per pivot language
│   ├── power_analysis.csv           probes/language vs. detectable δ
│   ├── regression_*.json            OLS, cluster-robust, mixed-effects variants
│   └── robustness_final_verdict.json
└── paper/{figures,tables}/     every figure and table in the paper
```

Every figure and table in the paper is regenerable from `results/` by a
script in this repo.

## Setup

10 instruction-tuned models, 7–10.7 B params, in three provenance groups:

- **A — English-Centric:** Llama 3.1-8B, Mistral-7B-v0.3, OLMo-2-7B, Gemma-2-9B
- **B — Multilingual-Native:** Aya-23-8B, Aya-Expanse-8B, BLOOMZ-7B
- **C — Regional-Centric:** Qwen 2.5-7B, Yi-1.5-9B, SOLAR-10.7B

8 languages, 5 scripts: English, French, Spanish, Chinese, Korean (high),
Arabic, Hindi (mid), Swahili (low).

Requires a HuggingFace token with access to the gated models (Llama, Gemma).

## Caveats

- **Do not use the per-model rankings** in `results/` as a deployment or
  procurement signal. That is the paper's point: they do not clear the
  noise floor.
- The floor is built from *machine-translated* paraphrases. It agrees to
  within 15% across typologically distant pivots (de / fi / ja), but
  cross-validation against human-authored paraphrases and multi-seed
  inference remains future work.
- SS is an intrinsic measure; its relationship to downstream generative
  bias is contested.
- Layer B is 21 probes — a proof-of-concept, far below the power threshold
  established in the paper. Treat it as a direction, not a result.
