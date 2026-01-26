import pandas as pd
import numpy as np
import wandb
import asyncio
import argparse
import os
import subprocess
import time
import sys
import requests
from requests.exceptions import ConnectionError
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

from utils_call_vllm import run_across_all_patients
from utils import (get_dataframe_of_results_and_targets_for_llms,
                   get_target, generate_llm_prompt_selection, END_OF_TEXT_TOP_5,
                   prepare_prompts_for_vllm, post_process_responses, BEGINNING_OF_TEXT)



DEBUG = False



PYTHON_PATH = "miniforge3/envs/mamba_vllm_b200/bin/python"
# Update this path to your local model path
MODEL_NAME = "chkpt/llama_3_1_8b_10x_280k_release_1"
DEFAULT_WANDB_GROUP = "genie_dt"

VALIDATION_DATA_PATH = "genie-dt-grpo-forecasting/0_data/converted_data/2025_03_18_converted_235_validation.pkl"
TEST_DATA_PATH = "genie-dt-grpo-forecasting/0_data/converted_data/2025_03_18_converted_244_test.pkl"


NUM_SAMPLES_PER_PATIENT = 10    # Standard number of copies to make

TEMPERATURE = 0.9       # Based on previous eval settings
TOP_P = 0.85             # Based on previous eval settings
MAX_CONCURRENT_REQUESTS = 100
MAX_TOKENS = 1500



def launch_vllm_server(model, prediction_url, port, python_path=PYTHON_PATH):

    print("Launching vLLM server...")
    env = os.environ.copy()
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    vllm_command = [python_path, "-m", "vllm.entrypoints.openai.api_server", "--port", str(port),
                    "--model", model, "--enable-prefix-caching",
                    "--tokenizer", model,
                    "--enforce-eager"]
    
    print(f"Launching vLLM server with command: {' '.join(vllm_command)}")

    # Launch the subprocess and capture its stdout and stderr
    process = subprocess.Popen(
        vllm_command,
        env=env,
        text=True, # Decode output as text
        bufsize=1, # Line-buffered
        universal_newlines=True
    )

    # Start a thread to watch and print the subprocess output
    # Poll the server for up to 15 minutes (900 seconds)
    server_ready = False
    start_time = time.time()
    print("\nWaiting 15 minutes for the vLLM server to load...")
    while time.time() - start_time < 900:
        try:
            # Check if the API endpoint is responding
            response = requests.get(prediction_url + "models") 
            if response.status_code == 200:
                print("✅ vLLM server is ready.")
                server_ready = True
                break
        except ConnectionError:
            # Server is not up yet, wait a bit
            time.sleep(5)

    if not server_ready:
        print("❌ vLLM server failed to start in 10 minutes. Terminating.")
        if process:
            process.terminate()
        sys.exit(1) # Exit the script with an error code




def compute_metrics(llm_predicted_df, target_df, prompts_df):

    assert len(llm_predicted_df) == len(target_df), f"Length mismatch: {len(llm_predicted_df)} vs {len(target_df)}"

    #: extract last observed value per patient from prompts_df
    last_observed_df = prompts_df[["patientid", "last_observed_values"]].copy()
    def get_last_observed_value(row):
        last_observed_neutrophil_value = float(row["last_observed_values"]["event_value"])
        return last_observed_neutrophil_value
    last_observed_df["last_observed_neutrophil_value"] = last_observed_df.apply(get_last_observed_value, axis=1)
    last_observed_df = last_observed_df[["patientid", "last_observed_neutrophil_value"]]
    target_patientid_and_dates = target_df[["patientid", "date"]].copy()
    last_observed_df = pd.merge(last_observed_df, target_patientid_and_dates, on="patientid", how="left")
    last_observed_df["event_value_copy_forward"] = last_observed_df["last_observed_neutrophil_value"]
    last_observed_df = last_observed_df[["patientid", "date", "event_value_copy_forward"]]

    #: merge on patientid
    merged_df = pd.merge(llm_predicted_df, target_df, on=["patientid", "date"], suffixes=("_pred", "_target"))
    merged_df = pd.merge(merged_df, last_observed_df, on=["patientid", "date"], suffixes=("", "_copy_forward"))

    #: get y_true, y_pred, y_copy_forward
    y_true = merged_df["event_value_target"].values
    y_pred = merged_df["event_value_pred"].values
    y_copy_forward = merged_df["event_value_copy_forward"].values

    #: compute metrics (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    mae_copy_forward = mean_absolute_error(y_true, y_copy_forward)

    mse = mean_squared_error(y_true, y_pred)
    mse_copy_forward = mean_squared_error(y_true, y_copy_forward)

    mase = mae / mae_copy_forward
    rmse = np.sqrt(mse)
    rmse_copy_forward = np.sqrt(mse_copy_forward)
    rrmse = rmse / rmse_copy_forward

    #: print & log all to W&B
    print(f"MAE: {mae}, MAE Copy Forward: {mae_copy_forward}")
    print(f"MSE: {mse}, MSE Copy Forward: {mse_copy_forward}")
    print(f"MASE: {mase}")
    print(f"RMSE: {rmse}, RMSE Copy Forward: {rmse_copy_forward}")
    print(f"RRMSE: {rrmse}")

    wandb.log({
        "MAE": mae,
        "MAE_Copy_Forward": mae_copy_forward,
        "MSE": mse,
        "MSE_Copy_Forward": mse_copy_forward,
        "MASE": mase,
        "RMSE": rmse,
        "RMSE_Copy_Forward": rmse_copy_forward,
        "RRMSE": rrmse,
        "num_predictions": len(y_pred),
    })

    # Log also as tables merged_df
    if wandb.run is not None:
        predictions_wandb = merged_df.copy()
        predictions_wandb = predictions_wandb.fillna(value='')
        wandb_table = wandb.Table(dataframe=predictions_wandb)
        wandb.log({
            "full_table": wandb_table,
        })








def main(model, port, skip_vllm_launch, wandb_group, path_to_eval_data, standard_gdt, pred_then_cot, reasoning_save_path=None,
         wandb_run_name_predix="", python_path=PYTHON_PATH):

    # Setup W&B
    wandb.init(project="genie-dt-grpo-forecasting-paper", mode="offline" if DEBUG else "online",
               group=wandb_group)
    wandb.run.name = wandb_run_name_predix + wandb_group + " - Eval: " + str(path_to_eval_data.split("/")[-1])
    wandb.config.update({
        "model": model,
        "port": port,
        "skip_vllm_launch": skip_vllm_launch,
        "wandb_group": wandb_group,
        "path_to_eval_data": path_to_eval_data,
        "standard_gdt": standard_gdt,
        "pred_then_cot": pred_then_cot,
    }, allow_val_change=True)


    #: optionally, launch model via vllm
    prediction_url = f"http://0.0.0.0:{port}/v1/"
    if skip_vllm_launch:
        print("Skipping vLLM server launch as per argument. Assuming it's already running.")
    else:
        launch_vllm_server(model, prediction_url, port, python_path)
    
    #: load in eval data
    with open(path_to_eval_data, 'rb') as f:
        raw_data = pickle.load(f)

    eval_target_df, eval_prompts_df = get_dataframe_of_results_and_targets_for_llms(raw_data)

    #: generate target prompt and df!!
    print("Generating targets...")
    eval_targets = get_target(eval_target_df, eval_prompts_df)

    #: setup prompts and targets, each entry with columns "patientid", "input_prompt", "target_prompt"
    print("Setting up prompts and targets...")
    if standard_gdt:
        eval_data_input_and_target = generate_llm_prompt_selection(eval_prompts_df, eval_targets, 
                                                                   system_prompt_override=None)
    else:
        if pred_then_cot:
            # First, in case of pred-then-cot, we need to generate the prompts with the different systemp prompt
            eval_data_input_and_target = generate_llm_prompt_selection(eval_prompts_df, eval_targets,
                                                                       system_prompt_override=BEGINNING_OF_TEXT)
        else:
            # Then we need to generate the prompts with the standard reasoning
            eval_data_input_and_target = generate_llm_prompt_selection(eval_prompts_df, eval_targets,
                                                                    system_prompt_override=END_OF_TEXT_TOP_5)

    eval_final_prompts = prepare_prompts_for_vllm(eval_data_input_and_target, num_copies=NUM_SAMPLES_PER_PATIENT)

    print("Sample final prompt:")
    print(eval_final_prompts.iloc[0]["prompt"])

    #: make calls to vllm
    seeds = [68971 + i for i in range(len(eval_final_prompts))]

    eval_responses = asyncio.run(run_across_all_patients(eval_final_prompts[["patientid", "prompt"]].values.tolist(), 
                                                        TEMPERATURE, TOP_P,
                                                        prediction_url, model,
                                                        max_concurrent_requests = MAX_CONCURRENT_REQUESTS,
                                                        max_tokens = MAX_TOKENS,
                                                        seed = seeds))

    
    #: save to CSV in case provided, for eval of reasoning steps later
    if reasoning_save_path is not None:
        eval_responses_df = pd.DataFrame(eval_responses, columns=["patientid", "response", "logprobs"])
        eval_responses_df = eval_responses_df[["patientid", "response"]]
        eval_responses_df.to_csv(reasoning_save_path, index=False)

    #: convert outputs back to DFs and aggregate
    eval_converted_and_aggregated_df = post_process_responses(eval_responses, eval_target_df, pred_then_cot=pred_then_cot)

    #: compute metrics
    compute_metrics(eval_converted_and_aggregated_df, eval_target_df, eval_prompts_df)
   
    #: end W&B run
    wandb.finish()






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SFT and evaluation for Genie DT")
    parser.add_argument("--port", type=int, default=9668, help="Port for the vLLM server")
    parser.add_argument("--model_to_eval", type=str, default=MODEL_NAME, help="Path to the model to evaluate")
    parser.add_argument("--python_path", type=str, default=PYTHON_PATH, help="Path to the Python executable for vLLM")
    parser.add_argument("--wandb_group", type=str, default=DEFAULT_WANDB_GROUP, help="W&B group")
    parser.add_argument("--wandb_run_name_prefix", type=str, default="", help="Prefix for W&B run name")
    parser.add_argument("--skip_vllm_launch", action="store_true", help="If set, skips launching the vLLM server (assumes it's already running)")
    parser.add_argument("--path_to_eval_data", type=str, default=VALIDATION_DATA_PATH, help="Path to the evaluation data (pickle file)")
    parser.add_argument("--standard_gdt", action="store_true", help="If set, uses standard GDT prompts without reasoning")
    parser.add_argument("--reasoning_save_path", type=str, default=None, help="If provided, saves the reasoning steps to this path")
    parser.add_argument("--pred_then_cot", action="store_true", help="")

    args = parser.parse_args()

    main(model=args.model_to_eval,
         port=args.port,
         skip_vllm_launch=args.skip_vllm_launch,
         wandb_group=args.wandb_group,
         path_to_eval_data=args.path_to_eval_data,
         standard_gdt=args.standard_gdt,
         pred_then_cot=args.pred_then_cot,
         reasoning_save_path=args.reasoning_save_path,
        wandb_run_name_predix=args.wandb_run_name_prefix,
        python_path=args.python_path)



