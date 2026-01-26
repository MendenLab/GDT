# GDT - CGDB Forecasting

This folder contains scripts for evaluating blood biomarker forecasting on the Flatiron Health-Foundation Medicine Clinico-Genomic Database (CGDB) across 20 cancer indications.

## Overview

The evaluation pipeline assesses GDT's ability to:
- **Forecast blood biomarkers** over a 13-week horizon across 93,054 patients
Results demonstrate that GDT achieves a median MASE of 0.87 for forecasting and an average C-index of 0.703 for event prediction, significantly outperforming baseline methods.

## Directory Structure

### 1. Data Generation (`1_data_generation/`)
Scripts for preprocessing and creating evaluation datasets.

### 2. Forecasting Evaluation Utils (`2_forecasting_eval_utils/`)
- `utils_forecasting_eval.py` - MASE calculation and forecasting metrics
- `generate_train_data_stats.py` - Dataset statistics and variable volatility analysis

### 3. Baseline Models

#### Copy Forward (`3_baselines/1_copy_forward/`)
Naive baseline that carries forward the last observed value.

#### Chronos (`3_baselines/2_chronos/`)
Foundation model baselines pretrained on 700k+ time series:
- `chronos_zero_shot.py` - Zero-shot inference
- `chronos_bolt_zero_shot.py` - Chronos Bolt variant
- `chronos_fine_tune_and_eval.py` - Fine-tuning on CGDB data

#### TiDE (`3_baselines/3_tide/`)
State-of-the-art time-series model with multivariate capabilities:
- `tide_train.py` - Training and evaluation

#### Llama (`3_baselines/4_llama/`)
Llama 3.1 8B baseline for comparison:
- `llama_eval.py` - Zero-shot forecasting with base LLM

### 4. GDT Evaluation (`4_gdt/`)
- `gdt_eval.py` - Main evaluation script for GDT
- `utils_call_vllm.py` - vLLM inference utilities
- `utils_gdt.py` - GDT-specific preprocessing and evaluation functions


## Requirements

See `requirements.txt` for dependencies. Key packages include:
- AutoGluon (Chronos, TiDE)
- vLLM (GDT inference)



