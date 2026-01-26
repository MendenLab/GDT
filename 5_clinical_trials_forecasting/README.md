# GDT - Clinical Trials Forecasting

This folder contains scripts for evaluating GDT's generalization to out-of-distribution clinical trial data for forecasting.


## Dataset

Evaluation uses four NSCLC clinical trials:
- **Training**: OAK (n=1,126), IMpower131 (n=949)  
- **Testing**: POPLAR (n=263), IMpower130 (n=680)

## Directory Structure

### 1. Data Generation (`1_data_generation/`)
Scripts for preprocessing clinical trial data into the required format.

### 2. Forecasting Evaluation Utils (`2_forecasting_eval_utils/`)
- `utils_forecasting_eval.py` - MASE calculation and forecasting metrics

### 3. Baseline Models

#### Copy Forward (`3_baselines/1_copy_forward/`)
Naive baseline carrying forward the last observed value.

#### TiDE (`3_baselines/2_tide/`)
State-of-the-art time-series model:
- `tide_train.py` - Training and evaluation
- `utils_tide.py` - TiDE-specific preprocessing
- `utils.py` - General utilities

#### Chronos (`3_baselines/3_chronos/`)
Foundation model baselines:
- `chronos_zero_shot.py` - Zero-shot inference
- `chronos_bolt_zero_shot.py` - Chronos Bolt variant
- `chronos_fine_tune_and_eval.py` - Fine-tuning on trial data
- `utils_chronos.py` - Chronos-specific utilities
- `utils.py` - General utilities

#### Llama (`3_baselines/4_llama/`)
Llama 3.1 8B baseline:
- `llama_eval.py` - Zero-shot forecasting with base LLM
- `utils_call_vllm.py` - vLLM inference utilities
- `utils_llama.py` - Llama-specific preprocessing

### 4. GDT Evaluation

#### Zero-Shot (`4_gdt/zero_shot/`)
Out-of-the-box evaluation without trial-specific training:
- `gdt_eval.py` - Main evaluation script
- `utils_call_vllm.py` - vLLM inference utilities
- `utils_gdt.py` - GDT-specific preprocessing and evaluation

#### Supervised Fine-Tuning (`4_gdt/sft/`)
Fine-tuned on training trials (OAK, IMpower131):
- `gdt_sft_and_eval.py` - Fine-tuning and evaluation pipeline
- `utils_call_vllm.py` - vLLM inference utilities
- `utils_gdt.py` - GDT-specific utilities



## Requirements

See `requirements.txt` for dependencies. Key packages include:
- AutoGluon (Chronos, TiDE)
- vLLM (GDT inference)