import pandas as pd
import numpy as np
import wandb
import asyncio
import argparse

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
from utils_call_vllm import run_across_all_patients

setup_imports_nb()
from utils_forecasting_eval import ForecastingEval




DEBUG = False
NR_OF_COPIES_TO_GENERATE = 10   # Running 10 for now - if needed can increase to 30
TEMPERATURE = 0.9  
TOP_P = 0.85
MAX_TOKENS = 1200   # Setting high, but not necessary - fine-tuned models knows to stop early

split = "test"
base_path_text = "/0_data/3_samples_per_lot_text/"
eval_base_path = "/0_data/3_samples_per_lot/"
trajectory_save_folder = "/0_data/genie_dt_meta/raw_trajectories/"






def main(indication, prediction_url, base_path_text_arg=base_path_text, eval_base_path_arg=eval_base_path, 
         trajectory_save_folder_arg=trajectory_save_folder):

    wandb.init(project="genie-dt-cgdb-baselines-forecasting", mode="offline" if DEBUG else "online", group="genie-dt")

    wandb.run.name = f"Genie DT - Eval - {split} - {indication}"
    wandb.config.update({
        "split": split,
        "base_path_text": base_path_text_arg,
        "eval_base_path": eval_base_path_arg,
        "indication": indication,
        "nr_of_copies_to_generate": NR_OF_COPIES_TO_GENERATE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "model": "genie-dt",
    }, allow_val_change=True)


    #: set up all converters, data managers etc.
    config, dm, converter = setup_all_data_managers_and_converters(indication)

    path_to_load = base_path_text_arg + "text_table_" + indication + "_" + str(split) + "_num_samples_per_lot_3.csv"
    raw_data = pd.read_csv(path_to_load)    

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
                                                folder=trajectory_save_folder_arg,
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
                                data_loading_path=eval_base_path_arg)

    result = evaluator.evaluate(final_results)

    
    
    
    wandb.finish()




if __name__ == "__main__":

    # Example usage: python gdt_eval.py --indication_list enhanced_multiplemyeloma enhanced_rcc --prediction_url http://0.0.0.0:8067/v1/ 

    # Parse arguments, 
    parser = argparse.ArgumentParser(description="Run Genie DT evaluation for a specific indication.")

    #: add indication_list arg being a list of strings which we then for loop over, and call the main function for each indication
    parser.add_argument("--indication_list", type=str, nargs='+', required=True,
                        help="List of indications to evaluate, e.g. 'enhanced_multiplemyeloma,enhanced_rcc'.")
    parser.add_argument("--prediction_url", type=str, required=True,
                        help="URL for the prediction service, e.g. 'http://0.0.0.0:8067/v1/'.")
    parser.add_argument("--base_path_text", type=str, default=base_path_text,
                        help="Base path for text data files")
    parser.add_argument("--eval_base_path", type=str, default=eval_base_path,
                        help="Base path for evaluation data files")
    parser.add_argument("--trajectory_save_folder", type=str, default=trajectory_save_folder,
                        help="Folder to save individual trajectory results")
    args = parser.parse_args()

    print(f"Indications to evaluate: {args.indication_list}")
    print(f"Prediction URL: {args.prediction_url}")

    # Call main for each indication
    for indication in args.indication_list:
        main(indication, args.prediction_url, args.base_path_text, args.eval_base_path, args.trajectory_save_folder)
    


