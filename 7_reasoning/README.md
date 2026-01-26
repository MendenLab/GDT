# GDT - Reasoning Extension

This folder contains scripts for extending GDT with interpretable clinical reasoning capabilities. The reasoning extension generates structured chains-of-thought alongside numerical forecasts, demonstrating how LLM-based digital twins can provide transparent, biologically grounded explanations.

## Overview

The reasoning pipeline demonstrates GDT's capability to:
- **Generate clinical rationales** that explain neutrophil trajectory predictions
- **Align reasoning with domain knowledge** through keyword analysis and mechanistic validation
- **Maintain forecast accuracy** while providing interpretability (MASE 0.862 vs 0.828 for base model)

This extension uses knowledge distillation from a teacher model (Qwen3 Next 80B-A3B) followed by reinforcement learning (GRPO) to ground reasoning in empirical accuracy.

## Directory Structure

### 1. Data Generation (`1_generate_basic_data/`)
Scripts for preprocessing and creating reasoning datasets:
- `process_data.py` - Preprocess raw clinical events
- `convert_to_text.py` - Convert patient histories to text format using TwinWeaver
- `generate_sft_dataset.py` - Generate synthetic reasoning chains using teacher model
- `post_process_sft_datasets.py` - Clean and validate generated reasoning chains

### 2. Evaluation (`2_evaluate/`)
- `eval.py` - Evaluate reasoning-enabled GDT on forecasting and reasoning quality
- `utils_call_vllm.py` - vLLM inference utilities
- `utils.py` - Helper functions for evaluation metrics

### 3. Supervised Fine-Tuning (`3_sft/`)
- `gdt_sft_pred_then_cot.py` - Fine-tune GDT on synthetic reasoning chains (prediction-then-reasoning format)
- `utils.py` - Training utilities and prompts

### 4. Reinforcement Learning (`4_grpo/`)
- `grpo_on_sft_reward_only_mae_with_norm.py` - Apply GRPO with MAE-based reward to align reasoning with accurate predictions
- `utils.py` - GRPO-specific utilities

## Pipeline

The reasoning extension follows a four-stage pipeline:

1. **Data Preparation**: Process NSCLC neutrophil forecasting data and convert to text format
2. **Knowledge Distillation**: Use teacher model to generate reasoning chains conditioned on ground truth
3. **Supervised Fine-Tuning**: Train GDT to generate predictions followed by structured reasoning
4. **Reinforcement Learning**: Optimize with GRPO using negative MAE as reward to ground reasoning in accuracy

## Key Features

- **Structured Reasoning Format**: Generates patient summaries, identifies key predictive factors, provides mechanistic analysis
- **Clinical Alignment**: Keywords analysis confirms alignment with therapy types (chemotherapy → marrow suppression, immunotherapy → immune activation)
- **Trajectory Stratification**: Reasoning concepts correlate with observed neutrophil trajectories

## Requirements

See `requirements.txt` for dependencies. Key packages include:
- PyTorch, Transformers, TRL (training)
- vLLM (inference)
- wandb (experiment tracking)
- TwinWeaver framework (from `../1_twinweaver`)

## Notes

- This extension focuses on neutrophil forecasting in NSCLC (N=2,385 train patients)
- Teacher model: Qwen3 Next 80B-A3B for synthetic reasoning generation
- Student model: Fine-tuned GDT (Llama 3.1 8B base)
- Reward function: Negative Mean Absolute Error (MAE) for numerical accuracy