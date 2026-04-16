"""
Multilingual LLM Bias Audit — Central Configuration
====================================================
All shared constants: model registry, languages, paths, inference settings.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
TABLES_DIR = PROJECT_ROOT / "paper" / "tables"

for _d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Languages ─────────────────────────────────────────────────────────────────
# 8 languages: 4 high-resource, 2 mid-resource, 2 low-resource
# Spanning 5 scripts and 6 cultural regions
LANGUAGES = {
    "en": {"name": "English",  "resource": "high", "script": "Latin",      "deepl": "en"},
    "fr": {"name": "French",   "resource": "high", "script": "Latin",      "deepl": "fr"},
    "es": {"name": "Spanish",  "resource": "high", "script": "Latin",      "deepl": "es"},
    "zh-CN": {"name": "Chinese",  "resource": "high", "script": "CJK",     "deepl": "zh"},
    "ko": {"name": "Korean",   "resource": "high", "script": "Hangul",     "deepl": "ko"},
    "ar": {"name": "Arabic",   "resource": "mid",  "script": "Arabic",     "deepl": "ar"},
    "hi": {"name": "Hindi",    "resource": "mid",  "script": "Devanagari", "deepl": "hi"},
    "sw": {"name": "Swahili",  "resource": "low",  "script": "Latin",      "deepl": "sw"},
}

LANG_CODES = list(LANGUAGES.keys())
NON_EN_CODES = [c for c in LANG_CODES if c != "en"]

# ── Model Registry ────────────────────────────────────────────────────────────
# 10 models in 3 provenance groups
MODEL_REGISTRY = {
    # ── Group A: English-Centric ──────────────────────────────────────────────
    "llama3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "group": "A_english_centric",
        "group_label": "English-Centric",
        "org": "Meta",
        "params": "8B",
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "group": "A_english_centric",
        "group_label": "English-Centric",
        "org": "Mistral AI",
        "params": "7B",
    },
    "olmo2-7b": {
        "hf_id": "allenai/OLMo-2-0425-7B-Instruct",
        "group": "A_english_centric",
        "group_label": "English-Centric",
        "org": "AI2",
        "params": "7B",
    },
    "gemma2-9b": {
        "hf_id": "google/gemma-2-9b-it",
        "group": "A_english_centric",
        "group_label": "English-Centric",
        "org": "Google",
        "params": "9B",
    },
    # ── Group B: Multilingual-Native ──────────────────────────────────────────
    "aya-23-8b": {
        "hf_id": "CohereForAI/aya-23-8B",
        "group": "B_multilingual_native",
        "group_label": "Multilingual-Native",
        "org": "Cohere For AI",
        "params": "8B",
    },
    "aya-expanse-8b": {
        "hf_id": "CohereForAI/aya-expanse-8b",
        "group": "B_multilingual_native",
        "group_label": "Multilingual-Native",
        "org": "Cohere For AI",
        "params": "8B",
    },
    "bloomz-7b": {
        "hf_id": "bigscience/bloomz-7b1",
        "group": "B_multilingual_native",
        "group_label": "Multilingual-Native",
        "org": "BigScience",
        "params": "7B",
    },
    # ── Group C: Regional-Centric ─────────────────────────────────────────────
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "group": "C_regional_centric",
        "group_label": "Regional-Centric",
        "org": "Alibaba",
        "params": "7B",
    },
    "yi-1.5-9b": {
        "hf_id": "01-ai/Yi-1.5-9B-Chat",
        "group": "C_regional_centric",
        "group_label": "Regional-Centric",
        "org": "01.AI",
        "params": "9B",
    },
    "solar-10.7b": {
        "hf_id": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "group": "C_regional_centric",
        "group_label": "Regional-Centric",
        "org": "Upstage",
        "params": "10.7B",
    },
}

MODEL_NAMES = list(MODEL_REGISTRY.keys())

# ── Inference Settings ────────────────────────────────────────────────────────
INFERENCE = {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "num_completions": 3,        # free-text completions per prompt
    "batch_size": 4,             # prompts per batch for log-prob computation
    "use_4bit": True,            # BitsAndBytes 4-bit quantization
    "device_map": "auto",
}

# ── Bias Template Categories ─────────────────────────────────────────────────
BIAS_CATEGORIES = [
    "gender_profession",
    "gender_trait",
    "race_trait",
    "age_competence",
    "religion",
]

# ── Benchmark Datasets (loaded from HuggingFace) ─────────────────────────────
BENCHMARK_DATASETS = {
    "stereoset": {
        "hf_id": "McGill-NLP/stereoset",
        "config": "intrasentence",
        "split": "validation",
        "max_samples": 200,          # balanced subset across bias types
        "description": "StereoSet intrasentence (Nadeem et al., 2021)",
    },
    "crows_pairs": {
        "hf_id": "nyu-mll/crows_pairs",
        "config": None,
        "split": "test",
        "max_samples": 150,          # balanced subset across bias types
        "description": "CrowS-Pairs (Nangia et al., 2020)",
    },
    "bbq": {
        "hf_id": "HiTZ/bbq",
        "configs": [                 # each category is a separate config
            "Age_ambig",
            "Gender_identity_ambig",
            "Race_ethnicity_ambig",
            "Religion_ambig",
            "Nationality_ambig",
        ],
        "split": "test",
        "max_per_category": 30,
        "description": "BBQ — Bias Benchmark for QA (Parrish et al., 2022)",
    },
}

# ── Analysis Settings ─────────────────────────────────────────────────────────
ANALYSIS = {
    "bootstrap_n": 1000,         # bootstrap iterations for CIs
    "ci_level": 0.95,            # confidence interval level
    "significance": 0.05,        # p-value threshold
    "backtranslation_threshold": 0.80,  # cosine sim threshold for validation
}
