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
WANDB_DEFAULT_GROUP = "chronos_standard_zero_shot"





def eval_model(split, base_path):

    indication = "cit"

    train_prediction_length = 13
    eval_prediction_length = 13
    indication = "cit"
    wandb_name = WANDB_DEFAULT_GROUP
    wandb_name += f"_pred_len_{train_prediction_length}"
    
    wandb.init(project="genie-dt-cit-baselines-forecasting", mode="offline" if DEBUG else "online",
                group=wandb_name)

    # Setting up the model parameters
    model = "autogluon/chronos-t5-large"   # Zero shot large model

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
    #chronos_model = TimeSeriesPredictor.load(model_load_path)

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
    
    evaluator = ForecastingEval(indication=indication, split=split, data_loading_path=base_path)

    result = evaluator.evaluate(processed_predictions)
    
    
    

    wandb.finish()




if __name__ == "__main__":

    # Set argparse with defaults 
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--eval_split', type=str, default="test", help='Split to evaluate on (default: test)')
    parser.add_argument('--eval_base_path', type=str, default="genie-dt-cit-baselines-forecasting/0_data/test_data/splits_only/", help='Base path to data for evaluation')
    args = parser.parse_args()

    eval_split = args.eval_split
    eval_base_path = args.eval_base_path

    # Evaluate the model
    eval_model(split=eval_split, base_path=eval_base_path)

