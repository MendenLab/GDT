import time
import pandas as pd
import numpy as np
from utils_chronos import (preprocess_input_to_autogluon_format, 
                           predict_chronos_model_on_input, map_predictions_to_original_dates, 
                           post_process_predictions_back_into_original_format, 
                           combine_dfs_into_one_for_training, finetune_chronos_model, 
                           apply_3_sigma_filtering_and_standardization)

from utils import setup_imports_nb
setup_imports_nb()
from utils_forecasting_eval import ForecastingEval
import wandb
import argparse
import os
import shutil



DEBUG = False
VAL_BASE_PATH = "genie-dt-cit-baselines-forecasting/0_data/validation_data/splits_only/"
WANDB_DEFAULT_GROUP = "chronos_standard_fine_tuned"




def eval_model(split, base_path, model, eval_prediction_length):

    indication = "cit"

    train_prediction_length = 13
    eval_prediction_length = 13
    indication = "cit"
    wandb_name = WANDB_DEFAULT_GROUP
    wandb_name += f"_pred_len_{train_prediction_length}"
    
    wandb.init(project="genie-dt-cit-baselines-forecasting", mode="offline" if DEBUG else "online",
                group=wandb_name)

    # Setting up the model parameters

    wandb.run.name = f"Chronos - Fine-Tuned - {split} - {indication}"
    wandb.config.update({
        "split": split,
        "base_path": base_path,
        "indication": indication,
        "model": "chronos_fine_tuning",
    }, allow_val_change=True)

    print("==========================================================")
    print(f"Indication: {indication}")

    # Load data
    constant_df_path = base_path + f"constant_df_{split}.csv"
    input_df_path = base_path + f"input_df_{split}.csv"
    target_df_path = base_path + f"target_df_{split}.csv"
    meta_data_path = base_path + f"meta_data_{split}.csv"

    constant_df = pd.read_csv(constant_df_path)
    input_df = pd.read_csv(input_df_path)
    target_df = pd.read_csv(target_df_path)
    meta_data = pd.read_csv(meta_data_path)

    # to datetime
    input_df["date"] = pd.to_datetime(input_df["date"])
    target_df["date"] = pd.to_datetime(target_df["date"])

    # Since now with covariates, skip them, since Chronos can't deal with them
    skip_cols_all = ["patientid", "date", "event_name", "event_value", "source",
                "event_descriptive_name", "meta", "imputed", "event_category",
                "target", "timestamp", "item_id"]
    actual_cols_to_get_input = [x for x in input_df.columns if x in skip_cols_all]
    input_df = input_df[actual_cols_to_get_input]
    actual_cols_to_get_target = [x for x in target_df.columns if x in skip_cols_all]
    target_df = target_df[actual_cols_to_get_target]


    # Clip at 3 sigma to ensure outliers don't destabilize the model
    input_df_filtered, target_df_filtered = apply_3_sigma_filtering_and_standardization(raw_input_events=input_df, 
                                                                    indication=indication, 
                                                                    raw_targets=target_df, 
                                                                    verbose=True,
                                                                    standardize=True)

    # Load the model
    print("Loading fine-tuned model...")

    # Do all Chronos preprocessing
    autogluon_input_df, patient_offsets_map, date_mapping_df = preprocess_input_to_autogluon_format(constant_df, input_df_filtered)

    predictions = predict_chronos_model_on_input(prediction_length=eval_prediction_length, 
                                        train_ag_data=autogluon_input_df, 
                                        chronos_model=model)

    mapped_predictions = map_predictions_to_original_dates(predictions, patient_offsets_map)

    target_dates_patientid_variable = target_df[["patientid", "date", "event_name"]].copy()
    processed_predictions = post_process_predictions_back_into_original_format(mapped_predictions, meta_data,
                                                                                target_dates_patientid_variable,
                                                                                indication=indication,
                                                                                destandardize=True)
    
    evaluator = ForecastingEval(indication=indication,
                        split=split,
                        data_loading_path=base_path)

    result = evaluator.evaluate(processed_predictions)
    
    
    

    wandb.finish()


def main(train_base_path, model_save_path, eval_split, eval_base_path):

    train_prediction_length = 13
    eval_prediction_length = 13
    indication = "cit"
    wandb_name = WANDB_DEFAULT_GROUP
    wandb_name += f"_pred_len_{train_prediction_length}"

    wandb.init(project="genie-dt-cit-baselines-forecasting", mode="offline" if DEBUG else "online",
               group=wandb_name)

    # Setting up the model parameters
    split = "train"
    chronos_model = "autogluon/chronos-t5-large"

    wandb.run.name = f"Chronos - Fine-Tuned - Training Job"
    wandb.config.update({
        "split": split,
        "train_base_path": train_base_path,
        "eval_base_path": eval_base_path,
        "indication": indication,
        "model": "chronos_fine_tuning",
        "chronos_model": chronos_model,
        "train_prediction_length": train_prediction_length,
        "eval_prediction_length": eval_prediction_length,
        "model_save_path": model_save_path,
    }, allow_val_change=True)

    # First delete all files in the path to save, if they exist
    if os.path.exists(model_save_path):
        print(f"Previous files exist in {model_save_path}. Deleting them.")
        shutil.rmtree(model_save_path)
    os.makedirs(model_save_path, exist_ok=True)
    
    # Load data
    all_input_target_and_meta_dfs = []

    # Load data
    constant_df_path = train_base_path + f"constant_df_{split}.csv"
    input_df_path = train_base_path + f"input_df_{split}.csv"
    target_df_path = train_base_path + f"target_df_{split}.csv"
    meta_data_path = train_base_path + f"meta_data_{split}.csv"

    constant_df = pd.read_csv(constant_df_path)
    input_df = pd.read_csv(input_df_path)
    target_df = pd.read_csv(target_df_path)
    meta_data = pd.read_csv(meta_data_path)
    print(f"Loaded data from {train_base_path}. Shapes: constant_df: {constant_df.shape}, input_df: {input_df.shape}, target_df: {target_df.shape}, meta_data: {meta_data.shape}")

    # to datetime
    input_df["date"] = pd.to_datetime(input_df["date"])
    target_df["date"] = pd.to_datetime(target_df["date"])
    
    # Clip at 3 sigma to ensure outliers don't destabilize the model
    input_df_filtered, target_df_filtered = apply_3_sigma_filtering_and_standardization(raw_input_events=input_df, 
                                                                                        indication=indication, 
                                                                                        raw_targets=target_df, 
                                                                                        verbose=True,
                                                                                        standardize=True)

    all_input_target_and_meta_dfs.append((input_df_filtered, target_df_filtered, constant_df, meta_data))

    #: combine into one dataframe
    combined_input_df, combined_constant_df = combine_dfs_into_one_for_training(all_input_target_and_meta_dfs)

    print("Total number of events: ", combined_input_df.shape)
    print("Total nr of time series: ", combined_constant_df.shape)

    # Get correct data format
    train_ag_data, patient_offsets_map, date_mapping_df = preprocess_input_to_autogluon_format(combined_constant_df, combined_input_df)

    # Finetune model
    max_nr_fine_tune_steps = 30000   # Some arbitrary high number, and then best model is selected
    eval_steps = 500
    
    print(f"Max nr of fine-tune steps: {max_nr_fine_tune_steps}")
    finetuned_model, best_model_path = finetune_chronos_model(chronos_model=chronos_model, 
                                                    train_ag_data=train_ag_data, 
                                                    prediction_length=train_prediction_length, 
                                                    log_path=model_save_path, 
                                                    time_limit=3600 * 5,  # 5 hours max training
                                                    max_nr_fine_tune_steps=max_nr_fine_tune_steps,
                                                    lr_for_single_run=1e-3,  # This is what they used in the paper
                                                    eval_steps=eval_steps,
                                                    report_to_wandb=True,
                                                    fine_tune_batch_size=32,
                                                    save_steps=eval_steps) 

    print(f"Finetuned model and saved here:\n" + str(best_model_path))
    wandb.finish()

    ############################# EVAL ##################################################

    eval_model(eval_split, eval_base_path, finetuned_model,
               eval_prediction_length=eval_prediction_length)







if __name__ == "__main__":

    # Set argparse with defaults 
    parser = argparse.ArgumentParser(description='Evaluation')
    #: set to correct path
    parser.add_argument('--train_base_path', type=str, default="genie-dt-cit-baselines-forecasting/0_data/train_data/splits_only/", help='Base path to data')
    parser.add_argument('--model_save_path', type=str, default="genie-dt-cit-baselines-forecasting/0_data/models/chronos/temp/", help='Path to save the model')
    parser.add_argument('--eval_split', type=str, default="test", help='Split to evaluate on (default: test)')
    parser.add_argument('--eval_base_path', type=str, default="genie-dt-cit-baselines-forecasting/0_data/test_data/splits_only/", help='Base path to data for evaluation')
    args = parser.parse_args()

    # Call main function with arguments
    main(args.train_base_path, args.model_save_path, args.eval_split, args.eval_base_path)




