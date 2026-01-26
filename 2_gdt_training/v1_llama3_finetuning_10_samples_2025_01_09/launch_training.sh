#!/bin/bash


# NOTE: this is provided for reference, as using hard coded paths.

# Main setup
SPLITS_TO_USE=10


# Hyperparameters
NUM_GPUS=8
BATCH_SIZE=1
BATCHING_STRATEGY="padding"
CONTEXT_LENGTH=8000
NUM_EPOCHS=1
LR=1e-5
WEIGHT_DECAY=0.1
GRADIENT_CLIPPING=1.0
USE_WANDB=TRUE
WANDB_PROJECT_NAME="Genie-DT-Finetuning-V1"
WANDB_GROUP="llama_splits_"$SPLITS_TO_USE
CHECKPOINT_INTERVAL=5000  
NUM_CHECKPOINTS_TO_KEEP=4
NUM_WORKERS_DATALOADER=16
SAVING_STRATEGY="StateDictType.SHARDED_STATE_DICT"


# Setup paths
# User must clone llama-recipes or set LLAMA_RECIPES_PATH
LLAMA_RECIPES_PATH=${LLAMA_RECIPES_PATH:-"../../llama-recipes"}
PYTHON_SCRIPT="$LLAMA_RECIPES_PATH/recipes/quickstart/finetuning/finetuning.py"

# Verify script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Training script not found at $PYTHON_SCRIPT"
    echo "Please clone the llama-recipes repository into 2_gdt_training/ or set LLAMA_RECIPES_PATH environment variable."
    exit 1
fi

DIST_ROOT_FOLDER_CHECKPOINTS="/fine_tuning_experiments/genie_dt_v1/checkpoints/"
TOKENIZER_NAME="/tokenizer/llama3/"
DIST_FOLDER_CHECKPOINTS="llama3-8.1-8b-"$SPLITS_TO_USE"x-1-epoch-raw"
MODEL_PATH="/models/llama3.1-8b/"

# Setup dataset script
DATASET_PATH="load_fh_instruct_dataset_10.py"


# Launch
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS $PYTHON_SCRIPT --enable_fsdp --model_name $MODEL_PATH --save_model --dist_checkpoint_root_folder $DIST_ROOT_FOLDER_CHECKPOINTS --dist_checkpoint_folder $DIST_FOLDER_CHECKPOINTS --fsdp_config.pure_bf16 --use_fast_kernels --tokenizer_name $TOKENIZER_NAME  --batch_size_training $BATCH_SIZE --batching_strategy $BATCHING_STRATEGY --context_length $CONTEXT_LENGTH --num_epochs $NUM_EPOCHS --lr $LR --gamma 1.0 --use_wandb $USE_WANDB --wandb_config.project $WANDB_PROJECT_NAME --wandb_config.group $WANDB_GROUP --dataset "custom_dataset" --custom_dataset.file $DATASET_PATH --custom_dataset.splits_per_patient_therapy $SPLITS_TO_USE --gradient_clipping True --gradient_clipping_threshold $GRADIENT_CLIPPING --weight_decay $WEIGHT_DECAY --train_config.checkpoint_interval $CHECKPOINT_INTERVAL --train_config.max_checkpoints_to_keep $NUM_CHECKPOINTS_TO_KEEP --num_workers_dataloader $NUM_WORKERS_DATALOADER --fsdp_config.checkpoint_type $SAVING_STRATEGY


