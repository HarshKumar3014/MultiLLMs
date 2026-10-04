#!/usr/bin/env bash
# Review round 3 GPU runs: target-language rewording (Hindi, Spanish) and
# StereoSet scored without its blank context. Resumable; ~1.5-2 h on an A100 80GB.
#
#   export HF_TOKEN=hf_...   # also accept the licence of CohereLabs/aya-expanse-32b
#   bash run_gpu_review.sh
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${HF_TOKEN:-}" ]]; then
  python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
fi

# 1. reword Hindi + Spanish items with an LLM that is not audited (~20-30 min)
python 18_review_gpu_runs.py --reword ${REWRITER:+--rewriter "$REWRITER"}
python - <<'PY'
from importlib import import_module
from config import DATA_DIR
import json
d = json.load(open(DATA_DIR / "target_rewordings.json"))
print(f"rewordings ok: {sum('context' in v for v in d.values())}/{len(d)}")
PY
# free the rewriter's disk space
python - <<'PY'
from importlib import import_module
import_module("05_noise_floor")._free_model_cache("CohereLabs/aya-expanse-32b")
PY

# 2. score with the ten audited models, one at a time
for m in ${MODELS:-llama3.1-8b mistral-7b olmo2-7b gemma2-9b aya-23-8b aya-expanse-8b bloomz-7b qwen2.5-7b yi-1.5-9b solar-10.7b}; do
  echo "════════ $m ════════"
  python 18_review_gpu_runs.py --score --resume --models "$m"
  python - "$m" <<'PY'
import sys
from config import MODEL_REGISTRY
from importlib import import_module
import_module("05_noise_floor")._free_model_cache(MODEL_REGISTRY[sys.argv[1]]["hf_id"])
PY
done

# 3. analysis (CPU)
python 19_review_runs_analysis.py || echo "⚠ analysis failed (fine for a partial run)"
echo "done → results/v2/review_runs/, data/target_rewordings.json"
