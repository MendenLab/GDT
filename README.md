# Genie Digital Twin (GDT): Training and Evaluation Scripts

This repository contains the training and evaluation code for **Genie Digital Twin (GDT)**, a pan-cancer longitudinal foundation model presented in our preprint ["TwinWeaver: An LLM-Based Foundation Model Framework for Pan-Cancer Digital Twins."](https://www.arxiv.org/abs/2601.20906).

This project was a collaboration between Roche and Helmholtz Munich, as part of the Munich School of Data Science (MUDS) program.




> **⚠️ IMPORTANT NOTE:** This repository provides GDT training and evaluation scripts using an **early version of TwinWeaver** (included for reference in `1_twinweaver_reference_only/`). For full TwinWeaver framework development and application to new datasets, please use the **complete TwinWeaver package** available at [MendenLab/TwinWeaver](https://github.com/MendenLab/TwinWeaver).

>**NOTE:** This repository is provided as a reference, requiring the CGDB dataset to run.

## Overview

**GDT** is a fine-tuned Llama 3.1 8B Instruct model trained on 93,054 patients across 20 cancer indications from the Flatiron Health-Foundation Medicine Clinico-Genomic Database (CGDB). The model demonstrates state-of-the-art performance on:

- **Blood Biomarker Forecasting**: Predicting continuous time-series values up to 13 weeks ahead (median MASE 0.87 vs 0.97 for best baseline)
- **Clinical Event Prediction**: Landmark prediction for survival, progression, therapy switching, and metastasis (average C-index 0.703 vs 0.662 for best baseline)
- **Out-of-Distribution Generalization**: Zero-shot and fine-tuned performance on clinical trials (POPLAR, IMpower130)
- **Interpretable Reasoning**: Extension with structured clinical rationales for forecasts

### Key Features

- **Multi-Modal Input**: Integrates demographics, diagnoses, laboratory measurements, genetic mutations (300+ genes), treatment history, and ECOG status
- **Unified Framework**: Jointly models continuous biomarker trajectories and discrete clinical events
- **Pan-Cancer Capabilities**: Trained across 20 cancer types, enabling transfer learning for low-data indications
- **Clinical Trial Generalization**: Matches or exceeds trained baselines in cold-start scenarios
- **Sample Efficiency**: Requires only 64 samples per variable to outperform fully trained baselines

## Repository Structure

```
├── 1_twinweaver_reference_only/     # Early TwinWeaver version (reference only)
├── 2_gdt_training/                  # GDT model training pipeline
├── 3_cgdb_forecasting/              # Real-world data forecasting evaluation
├── 4_cgdb_events/                   # Real-world data event prediction evaluation
├── 5_clinical_trials_forecasting/   # Clinical trial forecasting evaluation
├── 6_clinical_trials_events/        # Clinical trial event prediction evaluation
└── 7_reasoning/                     # Interpretable reasoning extension
```

### 1. TwinWeaver Framework (`1_twinweaver_reference_only/`)

Contains the early version of TwinWeaver used to train GDT, provided for reference. This includes:
- Multi-modal data serialization to text
- Forecasting and landmark event task construction
- Prompt engineering and data preprocessing utilities

**For new projects**, use the full TwinWeaver package instead of this reference implementation.

### 2. GDT Training (`2_gdt_training/`)

Training pipeline for the GDT model including:
- Full fine-tuning configuration for Llama 3.1 8B (8x H100 GPUs, ~7 days)
- Data preprocessing for 93,054 patients across 20 cancer indications
- Multi-task training setup (forecasting + event prediction)
- Model checkpointing and distributed training utilities

See [`2_gdt_training/README.md`](2_gdt_training/README.md) for detailed training configuration.

### 3. CGDB Forecasting Evaluation (`3_cgdb_forecasting/`)

Evaluation of blood biomarker forecasting on real-world data including:
- GDT evaluation across 20 cancer indications
- Baselines: Copy Forward, TiDE, Chronos, Chronos Bolt, Llama 3.1
- MASE (Mean Absolute Scaled Error) computation
- Per-indication and per-variable analysis

**Key Results**: GDT achieves median MASE 0.87 (top 30 variables: 0.83) vs best baseline 0.97 (p<0.001).

See [`3_cgdb_forecasting/README.md`](3_cgdb_forecasting/README.md) for evaluation details.

### 4. CGDB Event Prediction Evaluation (`4_cgdb_events/`)

Evaluation of clinical event prediction on real-world data including:
- GDT landmark prediction for survival, progression, therapy switching, metastasis
- Baselines: Random Survival Forest, CLMBR-T (EHR foundation model), Majority Classifier
- IPCW C-Index calculation for risk stratification
- Multiple landmark time points (1-104 weeks)

**Key Results**: GDT achieves average C-index 0.703 vs 0.662 for Random Survival Forest across survival, progression, and therapy switching tasks.

See [`4_cgdb_events/README.md`](4_cgdb_events/README.md) for evaluation details.

### 5. Clinical Trial Forecasting Evaluation (`5_clinical_trials_forecasting/`)

Out-of-distribution evaluation on NSCLC clinical trials:
- **Training**: OAK (n=1,126), IMpower131 (n=949)
- **Testing**: POPLAR (n=263), IMpower130 (n=680)
- Zero-shot and fine-tuned GDT evaluation
- Cold-start prediction from baseline measurements only

**Key Results**: Fine-tuned GDT achieves median MASE 0.75-0.88, outperforming baselines. Zero-shot GDT matches trained baseline performance.

See [`5_clinical_trials_forecasting/README.md`](5_clinical_trials_forecasting/README.md) for evaluation details.

### 6. Clinical Trial Event Prediction Evaluation (`6_clinical_trials_events/`)

Out-of-distribution event prediction on clinical trials with identical structure to forecasting evaluation.

**Key Results**: Fine-tuned GDT achieves average C-index 0.672 vs 0.648 for best baseline.

See [`6_clinical_trials_events/README.md`](6_clinical_trials_events/README.md) for evaluation details.

### 7. Reasoning Extension (`7_reasoning/`)

Interpretable extension providing structured clinical reasoning alongside predictions:
- Knowledge distillation from Qwen3 Next 80B-A3B teacher model
- Supervised fine-tuning on synthetic reasoning chains
- Group Relative Policy Optimization (GRPO) with MAE-based reward
- Clinical alignment validation through keyword analysis

**Key Results**: Reasoning chains align with clinical expectations (e.g., chemotherapy → marrow suppression, immunotherapy → immune activation) with modest forecasting accuracy trade-off (MASE 0.862 vs 0.828).

See [`7_reasoning/README.md`](7_reasoning/README.md) for implementation details.

## Data Access

The data used in this study were originated by and are the property of Flatiron Health, Inc. and Foundation Medicine, Inc. This repository contains **evaluation and training code only**—no patient data is included.

Requests for data sharing by license or permission for replicating results can be submitted to [PublicationsDataAccess@flatiron.com](PublicationsDataAccess@flatiron.com) and [cgdb-fmi@flatiron.com](cgdb-fmi@flatiron.com).

## Citation

If you use this code or the GDT model in your research, please cite our paper:
```
@misc{makarov2026twinweaver,
      title={TwinWeaver: An LLM-Based Foundation Model Framework for Pan-Cancer Digital Twins}, 
      author={Nikita Makarov and Maria Bordukova and Lena Voith von Voithenberg and Estrella Pivel-Villanueva and Sabrina Mielke and Jonathan Wickes and Hanchen Wang and Mingyu Derek Ma and Keunwoo Choi and Kyunghyun Cho and Stephen Ra and Raul Rodriguez-Esteban and Fabian Schmich and Michael Menden},
      year={2026},
      eprint={2601.20906},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.20906}, 
}
```

## Requirements

Each subdirectory contains its own `requirements.txt`. Core dependencies include:
- PyTorch
- Transformers (Hugging Face)
- vLLM (for efficient inference)
- AutoGluon (for Chronos and TiDE baselines)
- scikit-survival (for survival analysis baselines)
- femr (for CLMBR-T baseline)

## License

TwinWeaver is licensed under the Apache License 2.0. See [LICENSE](license.txt) for details.

## Contact

For questions or issues, please contact [nikita.makarov@roche.com](nikita.makarov@roche.com) or [michael.menden@unimelb.edu.au](michael.menden@unimelb.edu.au).

