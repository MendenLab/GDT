import pandas as pd
import numpy as np
import wandb
import asyncio
import argparse
import scipy.special
import os
import subprocess
import time
import sys
from transformers import AutoTokenizer
import requests
from requests.exceptions import ConnectionError

from utils_call_vllm import run_across_all_patients_to_get_probs_for_three_states, LABEL_OCCURRED, LABEL_NOT_OCCURRED, LABEL_CENSORED
from utils_genie_dt import (setup_all_data_managers_and_converters,
                            process_raw_data_to_list)


def setup_imports_nb():
    """Sets up sys.path to import from sibling directories."""
    try:
        notebook_parent_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        notebook_parent_dir = os.getcwd()
    
    # Navigate up to the project root and then to the eval_tools directory
    project_root_dir = os.path.abspath(os.path.join(notebook_parent_dir, "../2_eval_tools/"))
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)

setup_imports_nb()

from utils_events_eval import EventsEval





DEBUG = False

# Core parameters
EVALUATION_TIMELINES = [8, 26, 52, 104] 


# Evaluation parameters
# Update these paths to your local model path
MODEL_NAME = "chkpt/llama_3_1_8b_10x_280k_release_1"
TOKENIZER = "chkpt/llama_3_1_8b_10x_280k_release_1"
PYTHON_PATH = "miniforge3/envs/mamba_vllm_b200/bin/python"
DEFAULT_WANDB_GROUP = "genie_dt"



all_indications = [
    'enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
    'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
    'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
    'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
    'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma'
]



def eval(test_targets_path, test_text_path_raw, prediction_url, model_name, tokenizer_path, wandb_group, indications_to_process):

    for indication in indications_to_process:

        #: setup basics
        split = "test"
        wandb.init(project="genie-dt-cgdb-baselines-events-probabilities", mode="offline" if DEBUG else "online", 
                    group=wandb_group)

        test_text_path = test_text_path_raw + "text_table_" + indication + "_" + str(split) + "_num_samples_per_lot_1.csv"


        #: set name and group correctly
        wandb.run.name = f"Eval - Landmark - Probability - " + str(indication) + " - Model: " + wandb_group
        wandb.config.update({
            "indication": indication,
            "model_to_eval": model_name,
            "test_targets_path": test_targets_path,
            "test_text_path": test_text_path,
            "evaluation_timelines": EVALUATION_TIMELINES,
        }, allow_val_change=True)

        print(f"Running evaluation for {indication} on {split} split with model: {model_name}")
        print(f"Prediction URL: {prediction_url}")

        #: set up all converters, data managers etc.
        config, dm, converter = setup_all_data_managers_and_converters(indication)
        raw_data = pd.read_csv(test_text_path)

        #: adjust to only have the EVALUATION_TIMELINES
        raw_data['week_to_predict'] = raw_data['patientid'].str.split('_week_').str[1].astype(int)
        raw_data = raw_data[raw_data['week_to_predict'].isin(EVALUATION_TIMELINES)].copy()

        print(f"Number of samples in raw data for evaluation: {len(raw_data)}")
        print(f"Unique weeks in raw data: {raw_data['week_to_predict'].unique()}")

        # Process the raw data to prepare it for vLLM
        data_with_prompts_ready_for_vllm = process_raw_data_to_list(raw_data)
        print("Number of samples ready for vLLM:", len(data_with_prompts_ready_for_vllm))

        # get tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Run vllm cals
        returned_results = asyncio.run(run_across_all_patients_to_get_probs_for_three_states(data_with_prompts_ready_for_vllm,
                                                                                            tokenizer,
                                                                                            prediction_model=model_name,
                                                                                            prediction_url=prediction_url))

        print("Returned results length:", len(returned_results))
        wandb.log({"nr_predicted_trajectories": len(returned_results)})

        # Post-process into a DataFrame
        results_df = pd.DataFrame(returned_results)
        results_df = results_df[["patientid", LABEL_OCCURRED, LABEL_NOT_OCCURRED, LABEL_CENSORED]]


        # Combine lists into scores by averaging the log probabilities
        def convert_list_to_score(list_entry):
            """Mean a list of log probabilities (length normalized)."""
            # Do length normalized sum of logprobs
            return np.mean(list_entry)

        results_df["avg_logprob_" + LABEL_OCCURRED] = results_df[LABEL_OCCURRED].apply(convert_list_to_score)
        results_df["avg_logprob_" + LABEL_NOT_OCCURRED] = results_df[LABEL_NOT_OCCURRED].apply(convert_list_to_score)
        results_df["avg_logprob_" + LABEL_CENSORED] = results_df[LABEL_CENSORED].apply(convert_list_to_score)


        # Apply softmax to get probability scores
        def apply_softmax(occurred, not_occurred, censored):
            """Applies the softmax function to the three log probability sums."""
            sm_occurred, sm_not_occurred, sm_censored = scipy.special.softmax([occurred, not_occurred, censored])
            return sm_occurred, sm_not_occurred, sm_censored

        # : Use .apply() with axis=1 to operate on each row.
        # A lambda function passes the row's values as arguments to your function.
        # 'result_type="expand"' splits the returned tuple into new DataFrame columns.
        logprob_cols = ["avg_logprob_" + LABEL_OCCURRED, "avg_logprob_" + LABEL_NOT_OCCURRED, "avg_logprob_" + LABEL_CENSORED]
        softmax_cols = ["softmax_" + LABEL_OCCURRED, "softmax_" + LABEL_NOT_OCCURRED, "softmax_" + LABEL_CENSORED]

        results_df[softmax_cols] = results_df[logprob_cols].apply(
            lambda row: apply_softmax(row[0], row[1], row[2]),
            axis=1,
            result_type='expand'
        )


        # Determine the final event label based on the max probability
        def get_max_probability_event(prob_occurred, prob_not_occurred, prob_censored):
            """Determines the outcome based on the highest softmax probability."""
            selection_list = [LABEL_OCCURRED, LABEL_NOT_OCCURRED, LABEL_CENSORED]
            max_index = np.argmax([prob_occurred, prob_not_occurred, prob_censored])
            selected_max_case = selection_list[max_index]
            
            # Using more descriptive local variable names to avoid confusion with inputs
            is_occurred = False
            is_censored = False

            if selected_max_case == LABEL_OCCURRED:
                is_occurred = True
            elif selected_max_case == LABEL_NOT_OCCURRED:
                # Both are False for this case
                pass
            elif selected_max_case == LABEL_CENSORED:
                is_censored = True
            else:
                raise ValueError("Invalid case selected from probability comparison.")

            return is_censored, is_occurred

        #: Again, apply the function row-wise using axis=1 and expand the results.
        results_df[["censored", "occurred"]] = results_df[softmax_cols].apply(
            lambda row: get_max_probability_event(row[0], row[1], row[2]),
            axis=1,
            result_type='expand'
        )


        # Format the final output and rename columns
        final_predictions_df = results_df[["patientid", "censored", "occurred", "softmax_" + LABEL_OCCURRED, "softmax_" + LABEL_NOT_OCCURRED]].copy()

        #: Rename
        final_predictions_df = final_predictions_df.rename(
            columns={"softmax_" + LABEL_OCCURRED: "probability_occurrence",
                    "softmax_" + LABEL_NOT_OCCURRED: "probability_no_occurrence"}
        )

        # Split the patientid column into components
        patientid_parts = final_predictions_df["patientid"].str.split('_var_', expand=True)
        main_id_parts = patientid_parts[1].str.split('_week_', expand=True)

        # Assign the correct parts to new columns
        final_predictions_df["sampled_category"] = main_id_parts[0]
        final_predictions_df["week_to_predict"] = main_id_parts[1].astype(int)


        # 7. Evaluate the filtered predictions
        print("Running evaluator...")
        evaluator = EventsEval(
            indication=indication,
            data_loading_path=test_targets_path,
            split=split
        )
        results = evaluator.evaluate(final_predictions_df)
        
        # 8. Log results and wrap up
        print("✅ Evaluation complete. Logging results.")
        if "death" in results and 52 in results["death"]:
            print("\nSample Result (Death @ 52 weeks):")
            print(results["death"][52])
        
        # Wrap up wandb
        wandb.finish()
    
    




def main(port, test_targets_path, test_text_path, model_to_eval, wandb_group, skip_vllm_launch, indications_str, tokenizer=TOKENIZER, python_path=PYTHON_PATH):


    #: setup basics
    wandb.init(project="genie-dt-cgdb-baselines-events-probabilities", mode="offline" if DEBUG else "online", 
                group=wandb_group)


    #: set name and group correctly
    wandb.run.name = f"Genie DT - Eval - Landmark - Probability - setup"
    wandb.config.update({
        "model_to_eval": model_to_eval,
        "test_targets_path": test_targets_path,
        "test_text_path": test_text_path,
        "evaluation_timelines": EVALUATION_TIMELINES,
    }, allow_val_change=True)

    # Check if tokenizer also in model path
    if os.path.exists(os.path.join(model_to_eval, "tokenizer.model")) or os.path.exists(os.path.join(model_to_eval, "tokenizer_config.json")):
        tokenizer_path =  model_to_eval
        print(f"Using tokenizer from model path: {tokenizer_path}")
    else:
        tokenizer_path = tokenizer
        print(f"Using provided tokenizer path: {tokenizer_path}")


    #: launch vllm server in background
    prediction_url = f"http://0.0.0.0:{port}/v1/"

    if skip_vllm_launch:
        print("Skipping vLLM server launch as per argument. Assuming it's already running.")
    else:
        print("Launching vLLM server...")
        env = os.environ.copy()
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
        vllm_command = [python_path, "-m", "vllm.entrypoints.openai.api_server", "--port", str(port),
                        "--model", model_to_eval, "--enable-prefix-caching",
                        "--tokenizer", tokenizer_path]
        
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
    
    print("\nStarting evaluation...")
    #: finish wandb
    wandb.finish()

    indications_to_process = [item.strip() for item in indications_str.split(',')]
    eval(test_targets_path, test_text_path, prediction_url, model_to_eval, tokenizer_path, wandb_group, indications_to_process)

    print("Finished evaluation. Terminating vLLM server.")
    process.terminate() # Cleanly shut down the server
    print("Finished")






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SFT and evaluation for Genie DT")
    parser.add_argument("--port", type=int, default=9668, help="Port for the vLLM server")
    parser.add_argument("--test_targets_path", type=str, default="genie-dt-cgdb-eval-events/0_data/splits_only/", help="Path to the test targets file for evaluation")
    parser.add_argument("--test_text_path", type=str, default="genie-dt-cgdb-eval-events/0_data/splits_text_test/", help="Path to the test text file for evaluation")
    parser.add_argument("--model_to_eval", type=str, default=MODEL_NAME, help="Path to the model to evaluate")
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER, help="Path to the tokenizer")
    parser.add_argument("--python_path", type=str, default=PYTHON_PATH, help="Path to the Python executable for vLLM")
    parser.add_argument("--wandb_group", type=str, default=DEFAULT_WANDB_GROUP, help="W&B group")
    parser.add_argument("--skip_vllm_launch", action="store_true", help="If set, skips launching the vLLM server (assumes it's already running)")
    parser.add_argument("--indications", type=str, required=True, help="Comma-separated list of indications to process")
    args = parser.parse_args()

    main(args.port, args.test_targets_path, args.test_text_path, args.model_to_eval, args.wandb_group, args.skip_vllm_launch, args.indications, args.tokenizer, args.python_path)
