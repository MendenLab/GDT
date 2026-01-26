
import pandas as pd
import numpy as np
import wandb
import asyncio
import argparse


from utils_llama import (setup_imports_nb,
                         process_raw_data_to_list,
                         add_extra_prompt_to_instruction,
                         make_nr_of_copies,
                         parse_all_results_into_df,
                         process_empty_targets_from_raw_data,
                         average_results_by_week,
                         convert_to_dates_and_match_to_closest_target_date,
                         match_to_closest_true_names,
                         fill_in_missing_values_with_copy_forward,
                         match_to_event_name_and_adjust_patientid)
from utils_call_vllm import run_across_all_patients

setup_imports_nb()
from utils_forecasting_eval import ForecastingEval



DEBUG = False
NR_OF_COPIES_TO_GENERATE = 10   # Running 10 for now - if needed can increase to 30
EXTRA_PROMPT = "Only answer the previous task to forecast blood lab values only for the previously specified weeks (which you should always write as well)! Do not write anything except the previously stated task, with specific, numeric blood values! Do not specify ranges or categories - only single numeric values, the week number and the name of the variable! Each line should be exactly the prediction for one variable for one specific week!"
TEMPERATURE = 0.9  # Same parameters we use for Genie DT
TOP_P = 0.85  # Same parameters we use for Genie DT
MAX_TOKENS = 600   # To prevent long hallucinations that balloon the length of the eval

split = "test"
base_path_text = "genie-dt-cit-baselines-forecasting/0_data/test_data/text/"
eval_base_path = "genie-dt-cit-baselines-forecasting/0_data/test_data/splits_only/"



def main(prediction_url, base_path_text_arg=base_path_text, eval_base_path_arg=eval_base_path):

    indication = "cit"
    wandb.init(project="genie-dt-cit-baselines-forecasting", mode="offline" if DEBUG else "online", group="llama-3.1-8b")

    wandb.run.name = f"Llama3.1 - Eval - {split} - {indication}"
    wandb.config.update({
        "split": split,
        "base_path_text": base_path_text_arg,
        "eval_base_path": eval_base_path_arg,
        "indication": indication,
        "nr_of_copies_to_generate": NR_OF_COPIES_TO_GENERATE,
        "extra_prompt": EXTRA_PROMPT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "model": "llama-3.1-8b",
    }, allow_val_change=True)


    path_to_load = base_path_text_arg + "text_table_" + str(split) + ".csv"
    raw_data = pd.read_csv(path_to_load)

    # Clean due to edge case
    raw_data = raw_data[raw_data["sampled_variables"] != "[]"]

    print("==========================================================")
    print(f"Indication: {indication}")

    data_with_prompts = add_extra_prompt_to_instruction(raw_data, EXTRA_PROMPT)
    data_with_prompts_nr_of_copies = make_nr_of_copies(data_with_prompts, NR_OF_COPIES_TO_GENERATE)
    data_with_prompts_ready_for_vllm = process_raw_data_to_list(data_with_prompts_nr_of_copies)

    print(f"Number of samples to process: {len(data_with_prompts_ready_for_vllm)}")
    wandb.log({"nr_of_samples_to_process": len(data_with_prompts_ready_for_vllm)})

    seeds = [98716 + i for i in range(len(data_with_prompts_ready_for_vllm))]

    returned_results = asyncio.run(run_across_all_patients(data_with_prompts_ready_for_vllm, 
                                                           prediction_url=prediction_url,
                                                           temperature=TEMPERATURE, 
                                                           top_p=TOP_P, 
                                                           max_tokens=MAX_TOKENS, 
                                                           seed=seeds))

    print(len(returned_results))

    # Parse results
    empty_target_df = process_empty_targets_from_raw_data(raw_data)
    parsed_results = parse_all_results_into_df(returned_results)

    # Match to closest true names
    results_adjusted_names = match_to_closest_true_names(parsed_results, empty_target_df)

    # Convert results to dates and match to closest target date
    results_adjusted_dates = convert_to_dates_and_match_to_closest_target_date(results_adjusted_names, empty_target_df, raw_data)

    # aggregate results by averaging
    averaged_results = average_results_by_week(results_adjusted_dates)

    # Apply filling in
    filled_results = fill_in_missing_values_with_copy_forward(averaged_results, raw_data, empty_target_df)

    # Bring it into the correct format
    final_results_for_eval = match_to_event_name_and_adjust_patientid(filled_results, empty_target_df)

    # Setup evaluation
    evaluator = ForecastingEval(indication=indication, split=split, data_loading_path=eval_base_path_arg)

    result = evaluator.evaluate(final_results_for_eval)

    
    
    
    wandb.finish()




if __name__ == "__main__":

    # Example usage: python llama_eval.py --prediction_url http://0.0.0.0:8067/v1/ 

    # Parse arguments, 
    parser = argparse.ArgumentParser(description="Run Llama evaluation for a specific indication.")
    parser.add_argument("--prediction_url", type=str, default="http://0.0.0.0:8067/v1/",
                        help="URL for the prediction service, e.g. 'http://0.0.0.0:8067/v1/'.")
    parser.add_argument("--base_path_text", type=str, default=base_path_text,
                        help="Base path for text data files")
    parser.add_argument("--eval_base_path", type=str, default=eval_base_path,
                        help="Base path for evaluation data files")
    args = parser.parse_args()

    main(args.prediction_url, args.base_path_text, args.eval_base_path)
    


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     