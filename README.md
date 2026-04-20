# MultilingualBiasAudit: Cross-Lingual Fairness Auditing Pipeline

This repository contains the complete pipeline for executing a large-scale, cross-lingual bias audit on modern open-weights Large Language Models (LLMs). It quantifies the parity (or disparity) of social bias manifestations across multiple languages, introducing the **Cross-Lingual Fairness Index (CLFI)** and the **Deployment Fairness Gap (DFG)**.

The framework is composed of a three-stage pipeline: **Prompt Generation**, **Bias Auditing (Inference)**, and **Statistical Analysis**.

---

## 🛠 Project Structure

```text
MultilingualBiasAudit/
│
├── config.py                 # Core configuration: models, languages, HF Auth tokens
├── 01_build_prompts.py       # Stage 1: Builds and structures the dual-layer probe dataset
├── 02_run_audit.py           # Stage 2: Executes model inference over the generated prompts
├── 03_analyze.py             # Stage 3: Statistical analysis, regressions, and LaTeX generation
├── appendix_plots.py         # Visualizations: Generates base heatmaps, radars, distributions
├── generate_new_plots.py     # Visualizations: Generates categorical CLFI and robustness plots
│
├── data/                     # Output directory for generated prompt sets (.jsonl)
├── results/                  # Output directory for inference log-probs and results (.csv)
├── paper/                    # Assorted graphical and draft outputs
└── icml2026/                 # Final manuscript and LaTeX outputs (Excluded from this doc)
```

---

## 🚀 Pipeline Overview

### Stage 1: Dataset Generation (`01_build_prompts.py`)
This script generates the 31,170-probe dataset. It utilizes a **Dual-Layer Probing Architecture**:
*   **Layer A (Translated):** Bootstraps existing Anglocentric benchmarks (BBQ, StereoSet), translates them via an API into the target typological languages, and formats them into a standardized schema context.
*   **Layer B (Culturally-Native):** Injects localized, culturally relevant diagnostic probes (e.g., regional caste systems, localized stigmas) that cannot be constructed via direct English translation.

**Output:** `data/prompts_dataset.jsonl`

### Stage 2: Auditing Inference (`02_run_audit.py`)
This script systematically queries the specified 10 Large Language Models across the generated dataset.
*   Loads models sequentially in 4-bit NormalFloat (NF4) quantization to fit within single-GPU hardware bottlenecks (e.g., RTX 5090 / 24GB VRAM).
*   Extracts Next-Token Log-Probabilities to compute the raw **Stereotype Score (SS)** for each individual prompt.
*   Includes checkpointing and fault tolerance for uninterrupted evaluation.

**Output:** `results/all_results.csv` (Contains all 31,170 individual bias assessments).

### Stage 3: Statistical Analysis (`03_analyze.py`)
Consumes the raw `.csv` results and computes the localized fairness metrics. 
*   **Deployment Fairness Gap (DFG):** Computes the absolute deviation of a model's non-English SS relative to its English baseline.
*   **Cross-Lingual Fairness Index (CLFI):** Aggregates DFG into a systemic `[0,1]` index.
*   Runs Ordinary Least Squares (OLS) regressions to isolate language-resource level effects and Kruskal-Wallis $H$-tests for model provenance variance.
*   Automatically outputs formatted `\begin{table}` LaTeX components directly to `tables/` for immediate inclusion in the manuscript.

---

## ⚙️ Configuration & Setup

All high-level variables are managed in `config.py`. To customize the audit for your hardware or research needs, modify this file:
*   `MODELS`: Dictionary of HuggingFace repository IDs governing which models are pulled.
*   `LANGUAGES`: Ordered list of language codes evaluated.
*   `BATCH_SIZE`: Inference batching (adjust based on VRAM).
*   `QUANTIZATION`: Hardware configurations (leave as 4-bit NF4 for standard consumer hardware).

### Dependencies
Ensure you have the requisite ML packages installed:
```bash
pip install -r requirements.txt
```
Key libraries: `torch`, `transformers`, `bitsandbytes` (for NF4 quantization), `scipy`, `statsmodels`, `pandas`, `seaborn`.

---

## 📉 Visualizations

Two scripts handle data visualizations for reports:
1.  **`appendix_plots.py`**: Builds the foundation visuals mapping out SS distribution variance across layers and resource levels (Figures 1-5).
2.  **`generate_new_plots.py`**: Produces our most updated findings, including the Per-Category Bias CLFI bar chart (Figure 6) and the Group-level Cross-Lingual Consistency visual accounting for the removal of scaling outliers (Figure 7).

---

## 📝 Usage

To replicate the study on an RTX-capable machine:
```bash
# 1. Prepare Data
python3 01_build_prompts.py

# 2. Run Audit (Requires HuggingFace Authentication matching Config)
python3 02_run_audit.py

# 3. Analyze Results
python3 03_analyze.py

# 4. Generate Visualizations
python3 appendix_plots.py
python3 generate_new_plots.py
```
