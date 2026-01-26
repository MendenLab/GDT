# GDT - CGDB Event Prediction

This folder contains scripts for evaluating clinical event prediction on the Flatiron Health-Foundation Medicine Clinico-Genomic Database (CGDB) across 20 cancer indications.

## Overview

The evaluation pipeline assesses GDT's ability to predict landmark clinical events including survival, disease progression, therapy switching, and metastasis. 

## Directory Structure

### 1. Data Generation (`1_data_generation/`)
Scripts for preprocessing and creating evaluation datasets.

### 2. Evaluation Tools (`2_eval_tools/`)
- `utils_events_eval.py` - IPCW C-Index calculation and event prediction metrics

### 3. Baseline Models

#### CLIMBR-T (`3_baselines/1_climbr_t/`)
EHR foundation model baseline pretrained on 2.5M patients:
- `generate_climbr_representations.py` - Extract patient embeddings
- `train_heads_and_eval.py` - Train Cox proportional hazards head

#### Standard Survival (`3_baselines/2_standard_survival/`)
Classical survival analysis baseline:
- `train_and_eval_survival_forest.py` - Random Survival Forest training and evaluation

#### Majority Classifier (`3_baselines/3_majority_classifier/`)
Simple baseline for comparison:
- `majority_classifier.py` - Majority class prediction

### 4. GDT Evaluation (`4_gdt/`)
- `gdt_landmark_eval_probability.py` - Main evaluation script for landmark event prediction
- `utils_call_vllm.py` - vLLM inference utilities
- `utils_genie_dt.py` - GDT-specific preprocessing and evaluation functions


## Requirements

See `requirements.txt` for dependencies. Key packages include:
- scikit-survival (RSF)
- femr (CLIMBR-T)
- vLLM (GDT inference)