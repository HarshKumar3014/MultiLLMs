#!/usr/bin/env bash
# Re-scores the main audit + noise floor with the fixed scorer (v2, into
# results/v2/), then runs every new GPU job for the NAACL revision, resumably, in the order that
# gives usable results earliest. Safe to re-run after a disconnect: every step
# resumes from its per-model checkpoint.
#
#   export HF_TOKEN=hf_...        # needs accepted licences: Llama-3.1, Gemma-2, Aya-23, Aya-Expanse
#   bash run_gpu_extensions.sh            # all 10 models
#   MODELS="llama3.1-8b qwen2.5-7b" bash run_gpu_extensions.sh   # subset
set -euo pipefail
cd "$(dirname "$0")"

MODEL_ARGS=()
if [[ -n "${MODELS:-}" ]]; then MODEL_ARGS=(--models $MODELS); fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
fi

# 1. Amharic: translate + validate (CPU, ~25 min, needs internet) if not done yet
if [[ ! -f data/validation_am.json ]]; then
  python 11_add_language.py --lang am --build
fi

# 2. Human paraphrases, if the filled CSV has been uploaded
HUMAN=0
if [[ -f data/human_paraphrases.csv ]]; then python 12_human_paraphrase.py --check; HUMAN=1; fi

# 3. Score. One model at a time through all three jobs, so each model is
#    downloaded once (the jobs share the HF cache) and deleted afterwards.
for m in ${MODELS:-llama3.1-8b mistral-7b olmo2-7b gemma2-9b aya-23-8b aya-expanse-8b bloomz-7b qwen2.5-7b yi-1.5-9b solar-10.7b}; do
  echo "════════ $m ════════"
  # re-score the main audit and the noise floor with the fixed scorer (v2)
  python 02_run_audit.py --resume --no-completions --models "$m"
  python 05_noise_floor.py --resume --keep-model-cache --models "$m"
  python 11_add_language.py --lang am --score --resume --models "$m"
  python 10_positive_control.py --resume --models "$m" --langs en fr
  if [[ $HUMAN == 1 ]]; then python 12_human_paraphrase.py --score --resume --models "$m"; fi
  python - "$m" <<'EOF'
import sys
from config import MODEL_REGISTRY
from importlib import import_module
import_module("05_noise_floor")._free_model_cache(MODEL_REGISTRY[sys.argv[1]]["hf_id"])
EOF
done

# 4. Analysis (CPU)
python 10_positive_control.py --analyze-only
python 09_reanalysis.py
echo "done → results/reanalysis/summary.json, results/positive_control/positive_control_tests.csv"
