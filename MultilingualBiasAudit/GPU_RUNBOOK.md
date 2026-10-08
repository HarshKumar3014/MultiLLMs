# GPU runbook

**Use the offset-based scorer.** Locating the continuation by tokenizing
`context + " "` separately ("v1" below) never scores the first continuation
token for 9 of the 10 tokenizers (only BLOOMZ is unaffected), so for "He/She …"
probes the pronoun itself is never compared, and single-token answers score
−inf. `02_run_audit.continuation_span` ("v2") locates tokens by character
offsets (verified on all 10 tokenizers and numerically against a token-by-token
reference). Its scores go to `results/v2/`; the v1 files in `results/` are
used only for the pipeline ablation (`14_pipeline_ablation.py`).

| Job | Script | Forward passes / model |
|---|---|---|
| **Re-score main audit (v2)** | `02_run_audit.py --no-completions` | 3,117 × 3 ≈ 9.4k |
| **Re-score noise floor (v2)** | `05_noise_floor.py` | 387 × 4 × 3 ≈ 4.6k |
| Amharic (2nd low-resource language) | `11_add_language.py --lang am` | 387 × 3 ≈ 1.2k |
| Positive control (priming prefix, en + fr, 4 conditions) | `10_positive_control.py` | 387 × 2 × 4 × 3 ≈ 9.3k |
| Human-paraphrase 4th pivot (after you fill the CSV) | `12_human_paraphrase.py` | 100 × 3 = 300 |

Rough wall-clock for all 10 models: **~4–5 h on an A100**, ~12–15 h on a T4
(plus ~5 min download per model). Prefer Vast.ai A100 — Colab free-tier
disconnects will force repeated model downloads, though every step resumes.

## Before you start (on your laptop)

1. Fill the human paraphrases, if doing them now:
   `cp data/human_paraphrases_template.csv data/human_paraphrases.csv`, fill the
   `para_*` columns (rules at the top of `12_human_paraphrase.py`), then
   `python 12_human_paraphrase.py --check`. Skip this step to run the other two
   jobs first; re-run the script later and it only scores the human set.
2. Accept the HF licences for Llama-3.1-8B-Instruct, gemma-2-9b-it, aya-23-8B,
   aya-expanse-8b with the account whose token you will use.

## Vast.ai (A100 40/80 GB, PyTorch image, ≥ 80 GB disk)

```bash
git clone <repo-url> && cd MultiLLMs/MultilingualBiasAudit
# simplest: `git checkout` the branch with these scripts here. Otherwise copy
# every changed file: config.py, 02_run_audit.py, 05_noise_floor.py, 09–12,
# label_fix.py, run_gpu_extensions.sh, data/label_fix.json,
# data/positive_control_prefixes.json, data/human_paraphrases.csv
pip install -r requirements.txt
export HF_TOKEN=hf_...
nohup bash run_gpu_extensions.sh > gpu.log 2>&1 &
tail -f gpu.log
```

Afterwards copy back `data/prompts_am.json`, `data/validation_am.json` and the
whole `results/v2/` folder, then locally run `python 09_reanalysis.py`.

## Colab (T4)

```python
!git clone <repo-url>
%cd MultiLLMs/MultilingualBiasAudit
# upload the same extra files as above (Files pane), then:
!pip -q install -r requirements.txt
import os; os.environ["HF_TOKEN"] = "hf_..."
!bash run_gpu_extensions.sh
```

Mount Drive and symlink `results/` into it if you want checkpoints to survive
a runtime reset. Run a subset per session with
`!MODELS="llama3.1-8b mistral-7b" bash run_gpu_extensions.sh`.

## What "pass" looks like

- `results/extra_lang/`: 10 files, `data/validation_am.json` shows few collapsed
  probes (Amharic marks subject gender on the verb, so he/she pairs should survive).
- `results/positive_control/positive_control_tests.csv`: `prime` and `counter`
  detected (q < .05) with opposite signs, `neutral` not detected.
- `results/reanalysis/summary.json` gains `extra_languages: ["am"]` and, with
  human paraphrases, a `human_pivot` block.

## Within-language rewording + StereoSet without context

`run_gpu_within_language.sh` (~1.5–2 h on an A100 80 GB, ~$2–3):

1. Rewords every Hindi and Spanish item in the same language with
   **Aya-Expanse-32B** (not one of the audited models), validates each
   rewording (pair still differs, contrast words kept, meaning close), and
   retries failures once.
2. Scores the rewordings and StereoSet sentences *without* their blank
   context with all ten models.
3. Runs `19_within_language_analysis.py` → `paper/tables/tab_within_language.tex`.

Before starting:
- Accept the licence of `CohereLabs/aya-expanse-32b` on Hugging Face.
- Rent an instance with **≥ 150 GB disk** (the 32B model is a ~65 GB download).
  With a 40 GB GPU the rewriter loads in 4-bit automatically.
- Smaller fallback if the 32B model is a problem:
  `REWRITER=Qwen/Qwen2.5-14B-Instruct bash run_gpu_within_language.sh` (weaker Hindi).

```bash
git clone <repo-url> && cd MultiLLMs/MultilingualBiasAudit
pip install -r requirements.txt
export HF_TOKEN=hf_...
bash run_gpu_within_language.sh 2>&1 | tee within_language.log
zip -r within_language_results.zip results/v2/within_language_runs data/target_rewordings.json within_language.log
```
Copy `within_language_results.zip` back and unzip it in `MultilingualBiasAudit/`.
