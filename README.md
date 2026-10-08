# Lost in Translation: Why Lower Bias Scores Don't Mean Fairer Models

Code, data, and results for a cross-lingual social-bias audit of ten
open-weights LLMs across eight languages.

## TL;DR of the finding

Cross-language bias audits usually compare a model's average stereotype score
in English with its average in another language and read the difference as a
difference in fairness. We test such gaps against a **noise floor** (the gap
produced by merely rewording English), with paired permutation tests and
false-discovery correction, item-level tests, and an injected-shift check that
the test can detect real effects.

| Quantity | Value |
|---|---|
| Model–language pairs scoring below English | **70 / 70** |
| …of which reliable after FDR correction (soft / binary score) | 28 / 39 |
| Reworded-English comparisons reliable (false alarms) | 0 / 30 |
| Item-level drop beyond rewording, clear (q ≤ 0.008) | Hindi, Korean, Swahili |
| Drop beyond rewording, all vs. fully preserved translations | 0.020 vs. 0.018 |
| English → changed-item correlation: reworded / translated | **0.48 / 0.23** |
| Within-language rewording correlation (Hindi / Spanish) | 0.84 / 0.82 |
| Injected shift detected (≈360 items/language): 0.02 / 0.04 / 0.06 | 19% / 70% / 94% |

The drops are real, mostly on StereoSet items, and do not come from
identifiable translation errors (every translation was judged for contrast
preservation). But they do **not** mean the models are fairer there:
item-level preferences stay as strong and are reliable within Hindi and
Spanish, yet only weakly follow the English ones, so averages drift toward
"no preference."

> **Three pipeline checks** (see `label_fix.py` and
> `02_run_audit.continuation_span`): StereoSet's Hugging Face `gold_label` 0 is
> *anti*-stereotype; BBQ stereotype roles must come from metadata and question
> polarity, not answer order; and tokenizing `context + " "` separately skips
> the first continuation token for 9/10 tokenizers. Scores in `results/v2/` use
> the offset-based scorer; files directly in `results/` were scored with the
> off-by-one and are used only to show its effect (`14_pipeline_ablation.py`). The paper's figures and numbers come from
> `09_reanalysis.py`, `13_paper_figures.py` and `14`–`22`.

## Pipeline

Scripts run in numeric order. Older scripts and file names use DFG / CLFI, the earlier names of the paper's CSS / CLCI. `02`, `05`, `10`–`12` need a GPU (see `MultilingualBiasAudit/GPU_RUNBOOK.md`).

| Script | Does | GPU |
|---|---|:--:|
| `01_build_prompts.py` | Builds the item set: StereoSet / BBQ + 37 handcrafted templates (translated to 7 languages; the CrowS-Pairs loader yields no items) and 21 items written in the target languages. Back-translation check on contexts. | – |
| `02_run_audit.py` | Scores all 10 models over all prompts (4-bit NF4, sequential load/unload, checkpointed). | ✅ |
| `03_analyze.py` | v1 analysis (superseded by `09`–`22`). | – |
| `04_robustness.py` | v1 robustness checks (superseded). | – |
| `05_noise_floor.py` | The noise-floor experiment: 3 round-trip paraphrase sets (en→{de,ja,fi}→en), scored identically to the main audit. | ✅ |
| `06_noise_floor_figure.py` | Figure 1a — DFG vs. noise floor, forest plot. | – |
| `07_power_analysis.py` | Power analysis + per-pivot floor breakdown. | – |
| `08_robustness_figure.py` | Figure 7 — drop-BLOOMZ group means. | – |
| `appendix_plots.py` | Figure 6 — per-category CLFI. | – |
| `09_reanalysis.py` | Revision reanalysis: collapsed-pair audit, Layer-A-only paired DFG, sign-flip permutation + BH-FDR per cell, analytic null, bootstrap DFG − own floor, binary SS, probe-level variance decomposition, paraphrase fidelity, regression refit. Regenerates Tables 1/6/9 and Figure 1a. | – |
| `10_positive_control.py` | `--synthetic`: spike-in power curve (CPU). Default: stereotype-priming prefix positive control (en, fr). | ✅ |
| `11_add_language.py` | Adds a language post hoc (default Amharic) with all-field back-translation and collapse checks. | ✅ (score) |
| `12_human_paraphrase.py` | 100-item LLM-written rewording set as a 4th noise-floor pivot (labelled `llm`). | ✅ (score) |
| `13_paper_figures.py` | Every number, figure and main table in the paper (`results/reanalysis/paper_numbers.json`). | – |
| `14_pipeline_ablation.py` | Reruns the analysis under each combination of the three pipeline errors (paper's pipeline table). | – |
| `15_robustness_checks.py` | Summed vs. per-token log-probability, length controls, per-language filters, 200-run injected-shift check. | – |
| `16_controls_and_sources.py` | Crossed mixed model (item × model), nested faithful-translation filter with paired bootstrap, results by source, length + word-frequency controls. Run after `13`. | – |
| `17_sources_and_contrast_sample.py` | Results by source (incl. BBQ + written only), mixed model testing each translation against rewording, and the 210-item contrast-preservation sample (`data/contrast_judgment_sample.csv`, LLM-judged). Run after `16`. | – |
| `18_within_language_gpu_runs.py` + `run_gpu_within_language.sh` | GPU: rewords Hindi/Spanish items within the language (Aya-Expanse-32B, validated) and scores them plus StereoSet without its blank context. | ✅ |
| `19_within_language_analysis.py` | Within-language vs cross-language carry-over; StereoSet without context. | – |
| `20_item_level_tests.py` | Item-level inference (models averaged within item; permutation + bootstrap over items) per language, also on BBQ + written only; how much the within-language rewordings changed; robustness without the Aya models. Run last. | – |
| `21_association_and_dimensions.py` | Chance baseline for same choice; absolute gap vs absolute floor; BBQ `unknown` selection; reversal vs neutralization; results by social dimension; item-flow table. | – |
| `22_translation_quality.py` | Contrast preservation for all 2,497 translations (`data/contrast_judgment_full.csv`, LLM-judged); item-level test and association on fully preserved items; item-level power; within-language reliability; StereoSet without gender. Run after `21`. | – |

GPU steps for the revision: see [`GPU_RUNBOOK.md`](MultilingualBiasAudit/GPU_RUNBOOK.md) and `run_gpu_extensions.sh`.

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

- **Do not use the per-model numbers** in `results/` as a deployment or
  procurement signal: CSS and CLCI measure consistency with English, not
  fairness, and most per-model gaps are within reach of the noise floor.
- The floor is built from machine round-trip rewordings (de / fi / ja) and
  100 LLM-written rewordings (labelled `llm`, not human-written).
- Translation-quality judgments (`data/contrast_judgment_*.csv`) were made by
  an LLM, not native speakers.
- SS is an intrinsic measure; its relationship to downstream generative
  bias is contested.
- The 21 items written in the target languages are a proof of concept, far
  below the sample size needed; they enter no comparison in the paper.
