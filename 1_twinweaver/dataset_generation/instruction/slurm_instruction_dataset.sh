#!/bin/bash


# NOTE: this script is provided as a reference, using hard coded paths and parameters.


# Define the base command and paths
PYTHON_PATH="python"
SCRIPT_PATH="gdt-release/1_twinweaver/digital_twin_converter/instruction/jsonl_converter_instruction.py"
SAVE_PATH="/flatiron_cgdb/instruction/combined/2024_11_14_"
WANDB_GROUP="2024_11_14_instruction_samples_"

NR_TOKENS_BUDGET=8192
MAX_NUM_SAMPLES_PER_PATIENT_FORECASTING=4
MAX_NUM_SAMPLES_PER_PATIENT_EVENTS=4
MAX_LEN_WEEKS_TO_SAMPLE=104
MAX_NUM_SAMPLES_PER_LOT=10

# Loop to submit jobs with different indication_id
for INDICATION_ID in {0..19}
do

    # Create a unique job name
    JOB_NAME="job_indication_${INDICATION_ID}_${MAX_NUM_SAMPLES_PER_LOT}"

    FINAL_SAVE_PATH=${SAVE_PATH}${MAX_NUM_SAMPLES_PER_LOT}'_lots_per_patient'
    FINAL_WANDB_GROUP=${WANDB_GROUP}${MAX_NUM_SAMPLES_PER_LOT}

    echo ${PYTHON_PATH} ${SCRIPT_PATH} --indication_id ${INDICATION_ID} --save_path ${FINAL_SAVE_PATH} --wandb_group ${FINAL_WANDB_GROUP} --nr_tokens_budget ${NR_TOKENS_BUDGET} --max_num_samples_per_patient_forecasting ${MAX_NUM_SAMPLES_PER_PATIENT_FORECASTING} --max_num_samples_per_patient_events ${MAX_NUM_SAMPLES_PER_PATIENT_EVENTS} --max_length_of_weeks_to_sample ${MAX_LEN_WEEKS_TO_SAMPLE} --max_num_samples_per_lot ${MAX_NUM_SAMPLES_PER_LOT}

    # SLURM submission command
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=./outputs/${JOB_NAME}_%j.out
#SBATCH --error=./outputs/${JOB_NAME}_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH -p cpu
#SBATCH --time=7-00:00:00

# Execute the command
${PYTHON_PATH} ${SCRIPT_PATH} --indication_id ${INDICATION_ID} --save_path ${FINAL_SAVE_PATH} --wandb_group ${FINAL_WANDB_GROUP} --nr_tokens_budget ${NR_TOKENS_BUDGET} --max_num_samples_per_patient_forecasting ${MAX_NUM_SAMPLES_PER_PATIENT_FORECASTING} --max_num_samples_per_patient_events ${MAX_NUM_SAMPLES_PER_PATIENT_EVENTS} --max_length_of_weeks_to_sample ${MAX_LEN_WEEKS_TO_SAMPLE} --max_num_samples_per_lot ${MAX_NUM_SAMPLES_PER_LOT}
EOT
    done
done
