import pandas as pd
import numpy as np
import wandb
import asyncio
import argparse
from datasets import Dataset
from trl import (
    SFTTrainer,
    SFTConfig,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import datetime
import os
import subprocess
import time


from utils_gdt import (setup_imports_nb,
                       make_nr_of_copies,
                       setup_all_data_managers_and_converters,
                       process_raw_data_to_list,
                       process_empty_targets_from_raw_data,
                       get_last_observed_values,
                       convert_all_results_back_to_df,
                       save_individual_trajectories,
                       aggregate_results_on_patientid_level,
                       fill_in_missing_values_with_copy_forward,
                       convert_to_eval_format)
from utils_call_vllm import run_across_all_patients, SYSTEM_PROMPT

setup_imports_nb()
from utils_forecasting_eval import ForecastingEval




DEBUG = False


# Training parameters
# Update this path to your local model path
MODEL_NAME = "chkpt/llama_3_1_8b_10x_280k_release_1"
TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
MAX_LEN = 8192
SAVE_FOLDER = "genie-dt-cit-baselines-forecasting/0_data/genie_dt_sft/runs/"
INSTRUCTION_TEMPLATE = "<|start_header_id|>system<|end_header_id|>"
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>"
PATH_TO_TRAIN_DF = "genie-dt-cit-baselines-forecasting/0_data/train_data/text/text_table_train.csv"
PATH_TO_VALIDATION_DF = "genie-dt-cit-baselines-forecasting/0_data/validation_data/text/text_table_validation.csv"
EVAL_SAVE_STEPS = 500
PROPORTION_OF_VAL_SET_TO_USE = 0.1  # Use only 10% of the validation set for training, to speed up the process
num_gpus_to_use = 3
total_gpus = 8

# Inference parameters
NR_OF_COPIES_TO_GENERATE = 10   # Running 10 for now - if needed can increase to 30
TEMPERATURE = 0.9  
TOP_P = 0.85
MAX_TOKENS = 600
base_path_text = "genie-dt-cit-baselines-forecasting/0_data/test_data/text/"
base_path_eval = "genie-dt-cit-baselines-forecasting/0_data/test_data/splits_only/"



def eval(prediction_url, model_name, base_path_text_arg=base_path_text, base_path_eval_arg=base_path_eval):

    indication = "cit"
    split = "test"
    wandb.init(project="genie-dt-cit-baselines-forecasting", mode="offline" if DEBUG else "online", group="genie-dt-sft")

    wandb.run.name = f"Genie DT - Eval - {split} - {indication}"
    wandb.config.update({
        "split": split,
        "base_path_text": base_path_text_arg,
        "eval_base_path": base_path_eval_arg,
        "indication": indication,
        "nr_of_copies_to_generate": NR_OF_COPIES_TO_GENERATE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "model": "genie-dt",
    }, allow_val_change=True)


    #: set up all converters, data managers etc.
    config, dm, converter = setup_all_data_managers_and_converters(indication)

    path_to_load = base_path_text_arg + "text_table_"  + str(split) + ".csv"
    raw_data = pd.read_csv(path_to_load)    
    
    # Clean edge case
    raw_data = raw_data[raw_data["sampled_variables"] != "[]"]

    # Setup prompts
    data_with_prompts_nr_of_copies = make_nr_of_copies(raw_data, NR_OF_COPIES_TO_GENERATE)
    data_with_prompts_ready_for_vllm = process_raw_data_to_list(data_with_prompts_nr_of_copies)
    print(f"Number of samples to process: {len(data_with_prompts_ready_for_vllm)}")
    wandb.log({"nr_of_samples_to_process": len(data_with_prompts_ready_for_vllm)})


    seeds = [98716 + i for i in range(len(data_with_prompts_ready_for_vllm))]

    returned_results = asyncio.run(run_across_all_patients(data_with_prompts_ready_for_vllm, 
                                                           prediction_url=prediction_url,
                                                           temperature=TEMPERATURE, 
                                                           top_p=TOP_P, 
                                                           max_tokens=MAX_TOKENS, 
                                                           prediction_model=model_name,
                                                           seed=seeds))


    print("Returned results length:", len(returned_results))
    wandb.log({"nr_predicted_trajectories": len(returned_results)})

    #: setup basics
    empty_target_df = process_empty_targets_from_raw_data(raw_data)
    last_observed_values = get_last_observed_values(raw_data)

    #: convert each individual back to DF
    all_results_converted_individually = convert_all_results_back_to_df(returned_results, raw_data, converter, dm)

    #: save the individual trajectories somewhere
    save_location = save_individual_trajectories(all_results_converted_individually,
                                                folder="genie-dt-cit-baselines-forecasting/0_data/genie_dt_meta/raw_trajectories/",
                                                indication=indication) 

    print(f"Saved individual trajectories to: {save_location}")
    wandb.config.update({"save_location_individual_trajectories": save_location}, allow_val_change=True)

    #: aggregate results on patientid level
    aggregated_results_df = aggregate_results_on_patientid_level(all_results_converted_individually, converter)

    #: fill in any missing values
    filled_in_results, nr_missing_values = fill_in_missing_values_with_copy_forward(aggregated_results_df, last_observed_values, empty_target_df)
    print(f"Number of missing values filled in: {nr_missing_values}")  
    wandb.log({"nr_missing_values_filled_in": nr_missing_values})

    #: convert to eval format - need to see results for this
    final_results = convert_to_eval_format(filled_in_results)

    #: run evaluation
    evaluator = ForecastingEval(indication=indication,
                                split=split,
                                data_loading_path=base_path_eval_arg)

    result = evaluator.evaluate(final_results)

    
    
    
    wandb.finish()




def main(port, num_train_epochs, learning_rate, run_eval, save_folder=SAVE_FOLDER, 
         path_to_train_df=PATH_TO_TRAIN_DF, path_to_validation_df=PATH_TO_VALIDATION_DF,
         base_path_text_arg=base_path_text, base_path_eval_arg=base_path_eval):


    #: setup basics
    indication = "cit"
    split = "train"
    wandb.init(project="genie-dt-cit-baselines-forecasting", mode="offline" if DEBUG else "online", group="genie-dt-updated-sft")

    wandb.run.name = f"Genie DT - Train - {indication} - {num_train_epochs} epochs - {learning_rate} lr"
    wandb.config.update({
        "split": split,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "model": "genie-dt-sft",
    }, allow_val_change=True)

    
    #: setup folders, based on datetime str
    curr_output_dir = save_folder + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/"
    os.makedirs(curr_output_dir, exist_ok=True)
    output_dir = curr_output_dir + "training_run/"
    save_folder_final = curr_output_dir + "saves/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_folder_final, exist_ok=True)
    
    #: load in training data
    train_df = pd.read_csv(path_to_train_df)
    validation_df = pd.read_csv(path_to_validation_df)

    # Clean edge case
    train_df = train_df[train_df["sampled_variables"] != "[]"]
    validation_df = validation_df[validation_df["sampled_variables"] != "[]"]

    # If running on a small subset of the validation set, take a random sample
    if PROPORTION_OF_VAL_SET_TO_USE < 1.0:
        validation_df = validation_df.sample(frac=PROPORTION_OF_VAL_SET_TO_USE, random_state=42)
        print(f"Using {len(validation_df)} samples from the validation set for training.")
    
    # Parse the 'prompt' column from JSON strings to dictionaries if necessary
    def prep_df(df):

        df = df.copy()

        def add_input_in_format(row):
            input_str = row["instruction"]
            #: add system message and convert to dictionary
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": input_str}]
            return messages
        df['prompt'] = df.apply(add_input_in_format, axis=1)
        
        #: add "final_response" to the prompt as assistant message
        def add_final_response(row):
            # Add the final response as an assistant message
            messages = [{"role": "assistant", "content": row['target']}]
            return messages
        df['completion'] = df.apply(add_final_response, axis=1)
        df = df[["prompt", "completion"]]
        return df

    # Prepare the DataFrame
    train_df = prep_df(train_df)
    validation_df = prep_df(validation_df)

    # Print one example to check the format
    print("Example training data:")
    print(train_df.iloc[0]['prompt'][1]["content"])
    print("Example final response:")
    print(train_df.iloc[0]['completion'][0]['content'])


    #: setup SFT
    print("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"  # Found in the tokenizer config
    tokenizer.model_max_length = MAX_LEN


    #: convert to dataset (using chat template from llama3.1 8b), 
    # make sure to use DataCollatorForCompletionOnlyLM, with the appropriate response template, so only the response is used for loss calculation
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)


    #: setup SFTconfig
    train_params = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        eval_strategy="steps",
        eval_steps=EVAL_SAVE_STEPS,
        save_strategy="steps",
        save_steps=EVAL_SAVE_STEPS,
        optim="adamw_torch",
        logging_steps=1,
        learning_rate=learning_rate,
        bf16=True,
        num_train_epochs=num_train_epochs,
        group_by_length=True,
        report_to="wandb",
        seed=42,
        max_length=MAX_LEN,             
        packing=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        completion_only_loss=True,
    )


    #: load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    

    #: setup trainer
    trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            args=train_params,
            eval_dataset=validation_dataset,
    )
    print("Trainer setup complete, starting training")


    #: run SFT
    trainer.train()

    #: save model
    model.save_pretrained(save_folder_final)

    #: clean up GPU memory
    print("Training complete, cleaning up")
    del trainer
    del model
    del tokenizer
    del train_dataset
    del validation_dataset
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(60)  # Wait for a minute to ensure memory is cleared


    #: launch vllm server in background
    if run_eval:
        prediction_url = f"http://0.0.0.0:{port}/v1/"
        vllm_command = ["python", "-m", "vllm.entrypoints.openai.api_server", "--port", str(port),
                        "--model", save_folder_final, "--enable-prefix-caching",
                        "--tokenizer", TOKENIZER]
        print(f"Launching vLLM server with command: {' '.join(vllm_command)}")
        process = subprocess.Popen(vllm_command)

        #: wait a few mins to load up
        time.sleep(5 * 60)

    #: finish wandb
    print("Finishing train run and launching vllm")
    wandb.finish()

    #: launch eval
    if run_eval:
        eval(prediction_url, save_folder_final, base_path_text_arg, base_path_eval_arg)

    print("Finished")






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SFT and evaluation for Genie DT")
    parser.add_argument("--port", type=int, default=9668, help="Port for the vLLM server")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--run_eval", action="store_true", help="Whether to run evaluation after training")
    parser.add_argument("--save_folder", type=str, default=SAVE_FOLDER, help="Folder to save the model outputs")
    parser.add_argument("--path_to_train_df", type=str, default=PATH_TO_TRAIN_DF, help="Path to the training data CSV")
    parser.add_argument("--path_to_validation_df", type=str, default=PATH_TO_VALIDATION_DF, help="Path to the validation data CSV")
    parser.add_argument("--base_path_text", type=str, default=base_path_text, help="Base path for text data files")
    parser.add_argument("--base_path_eval", type=str, default=base_path_eval, help="Base path for evaluation data files")
    args = parser.parse_args()

    main(args.port, args.num_train_epochs, args.learning_rate, args.run_eval,
         args.save_folder, args.path_to_train_df, args.path_to_validation_df,
         args.base_path_text, args.base_path_eval)
