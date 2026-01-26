# GDT - Clinical Trials Event Prediction

This folder contains scripts for evaluating clinical event prediction on out-of-distribution clinical trial data (POPLAR and IMpower130) to assess GDT's generalization capabilities in cold-start scenarios.


## Directory Structure

### 1. Data Generation (`1_data_generation/`)
Scripts for preprocessing clinical trial data and creating evaluation datasets.

### 2. Evaluation Tools (`2_eval_tools/`)
- `utils_landmark_eval.py` - IPCW C-Index calculation and landmark event metrics

### 3. Baseline Models

#### Majority Classifier (`3_baselines/3_majority_classifier/`)
Simple baseline predicting majority class:
- `landmark/majority_classifier.py` - Majority class prediction for landmark events

#### CLIMBR-T (`3_baselines/4_climbr_t/`)
EHR foundation model baseline:
- `generate_climbr_representations.py` - Extract patient embeddings
- `landmark_train_heads_and_eval_coxph.py` - Train Cox proportional hazards model

#### Standard Survival (`3_baselines/5_standard_survival/`)
Classical survival analysis:
- `train_and_eval_survival_forest.py` - Random Survival Forest baseline

### 4. GDT Evaluation (`4_gdt/`)
- `gdt_landmark_eval_probability.py` - Main evaluation script for zero-shot GDT
- `utils_call_vllm.py` - vLLM inference utilities
- `utils_genie_dt.py` - GDT preprocessing and evaluation functions

#### Supervised Fine-Tuning (`4_gdt/sft/`)
Fine-tuning on clinical trial training data (OAK, IMpower131):
- `gdt_sft.py` - Supervised fine-tuning script
- `utils_call_vllm.py` - Inference utilities for fine-tuned model
- `utils_genie_dt.py` - Data processing for fine-tuned GDT

## Key Results

- **Zero-shot GDT** achieves C-index of 0.656, competitive with trained baselines
- **Fine-tuned GDT** reaches C-index of 0.672, surpassing RSF (0.648) across all time points
- Demonstrates strong generalization despite differences in data collection and patient populations

## Requirements

See `requirements.txt` for dependencies. Key packages include:
- scikit-survival (Random Survival Forest)
- femr (CLIMBR-T)
- vLLM (GDT inference)
- transformers, trl (fine-tuning)