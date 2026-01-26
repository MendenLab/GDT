import time
import pandas as pd
import numpy as np
from utils_tide import (preprocess_input_to_autogluon_format, 
                           predict_model_on_input, map_predictions_to_original_dates, 
                           post_process_predictions_back_into_original_format, 
                           combine_dfs_into_one_for_training, finetune_tide_model, 
                           apply_3_sigma_filtering_and_standardization)

from utils import setup_imports_nb
setup_imports_nb()
from utils_forecasting_eval import ForecastingEval
import wandb
import argparse
import os
import shutil
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


DEBUG = False



all_indications = [
        'enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
        'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
        'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
        'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
        'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma'
    ]




def eval_model(split, base_path, model_load_path, eval_prediction_length, train_prediction_length, include_all_columns):

    for indication in all_indications:

        wandb_group = "tide"
        if include_all_columns:
            wandb_group += "_all_columns"

        # Currently hard coded - should be changed to a parameter
        wandb_group += "_small_patience"

        wandb.init(project="genie-dt-cgdb-baselines-forecasting", mode="offline" if DEBUG else "online",
                   group=wandb_group)

        # Setting up the model parameters

        wandb.run.name = f"TiDE - Eval - {split} - {indication}"
        wandb.config.update({
            "split": split,
            "base_path": base_path,
            "indication": indication,
            "model": "tide",
            "eval_prediction_length": eval_prediction_length,
            "include_all_columns": include_all_columns,
        }, allow_val_change=True)

        print("==========================================================")
        print(f"Indication: {indication}")
        
        # Load data
        constant_df_path = base_path + f"constant_df_{indication}_{split}.csv"
        input_df_path = base_path + f"input_df_{indication}_{split}.csv"
        target_df_path = base_path + f"target_df_{indication}_{split}.csv"
        meta_data_path = base_path + f"meta_data_{indication}_{split}.csv"

        constant_df = pd.read_csv(constant_df_path)
        input_df = pd.read_csv(input_df_path)
        target_df = pd.read_csv(target_df_path)
        meta_data = pd.read_csv(meta_data_path)

        # to datetime
        input_df["date"] = pd.to_datetime(input_df["date"])
        target_df["date"] = pd.to_datetime(target_df["date"])

        # Clip at 3 sigma to ensure outliers don't destabilize the model
        input_df_filtered, target_df_filtered = apply_3_sigma_filtering_and_standardization(raw_input_events=input_df,
                                                                                            include_all_columns=include_all_columns, 
                                                                                            indication=indication, 
                                                                                            raw_targets=target_df, 
                                                                                            verbose=True)

        #: process for covariates
        if include_all_columns:
            empty_target_df = target_df_filtered.copy()
            empty_target_df["is_known_future_covariate"] = True  # Mark as known future covariate
            input_df_filtered["is_known_future_covariate"] = False
            input_df_filtered = pd.concat([input_df_filtered, empty_target_df], axis=0, ignore_index=True)
        
        # Do all Chronos preprocessing
        autogluon_input_df, patient_offsets_map, date_mapping_df = preprocess_input_to_autogluon_format(constant_df, input_df_filtered)

        #: post process for known covariates - extract them from autogluon_input_df
        if include_all_columns:
            
            original_input = autogluon_input_df.copy()
            autogluon_input_df = original_input[original_input["is_known_future_covariate"] == False].drop(columns=["is_known_future_covariate"])
            covariates_preliminary = original_input[original_input["is_known_future_covariate"] == True].drop(columns=["is_known_future_covariate"])
            covariates_preliminary = covariates_preliminary.drop(columns=["target"])
            covariates_preliminary = pd.DataFrame(covariates_preliminary).reset_index(drop=False)

            # Fill in any missing blanks
            predictor = TimeSeriesPredictor.load(model_load_path)
            all_actual_future_dates = predictor.make_future_data_frame(autogluon_input_df)
            covariates = all_actual_future_dates.merge(covariates_preliminary, on=["item_id", "timestamp"], how="left")
            # Then fill forward fill for some edge cases (since anywway all future are just copy forward)
            covariates = covariates.ffill()

        else:
            covariates = None

        predictions = predict_model_on_input(new_prediction_length=eval_prediction_length,
                                            train_ag_data=autogluon_input_df,
                                            predictor_path_or_object=model_load_path,
                                            train_prediction_length=train_prediction_length,
                                            covariates=covariates)

        mapped_predictions = map_predictions_to_original_dates(predictions, patient_offsets_map)

        target_dates_patientid_variable = target_df[["patientid", "date", "event_name"]].copy()
        processed_predictions = post_process_predictions_back_into_original_format(mapped_predictions, meta_data,
                                                                                   target_dates_patientid_variable,
                                                                                    indication=indication)

        evaluator = ForecastingEval(indication=indication, split=split, data_loading_path=base_path)

        result = evaluator.evaluate(processed_predictions)
        
        
        

        wandb.finish()


def main(base_path, model_save_path, eval_split, eval_base_path, include_all_columns):

    # Currently hard coded - should be changed to a parameter
    wandb_group = "tide"
    if include_all_columns:
        wandb_group += "_all_columns"
    
    wandb.init(project="genie-dt-cgdb-baselines-forecasting", mode="offline" if DEBUG else "online",
                group=wandb_group)

    # Setting up the model parameters
    train_prediction_length  = 13  # To cover any edge cases of dates (due to genetic events)
    eval_prediction_length  = 13
    split = "train"
    patience_nr_epochs = 20

    wandb.run.name = f"TiDE - Training Job"

    wandb.config.update({
        "split": split,
        "base_path": base_path,
        "model": "tide",
        "train_prediction_length": train_prediction_length,
        "eval_prediction_length": eval_prediction_length,
        "model_save_path": model_save_path,
        "include_all_columns": include_all_columns,
        "patience_nr_epochs": patience_nr_epochs,
    }, allow_val_change=True)

    # First delete all files in the path to save, if they exist
    if os.path.exists(model_save_path):
        print(f"Previous files exist in {model_save_path}. Deleting them.")
        shutil.rmtree(model_save_path)
    os.makedirs(model_save_path, exist_ok=True)

    # Load data
    all_input_target_and_meta_dfs = []

    for indication in all_indications:

        # Load data
        constant_df_path = base_path + f"constant_df_{indication}_{split}.csv"
        input_df_path = base_path + f"input_df_{indication}_{split}.csv"
        target_df_path = base_path + f"target_df_{indication}_{split}.csv"
        meta_data_path = base_path + f"meta_data_{indication}_{split}.csv"

        constant_df = pd.read_csv(constant_df_path)
        input_df = pd.read_csv(input_df_path)
        target_df = pd.read_csv(target_df_path)
        meta_data = pd.read_csv(meta_data_path)

        # to datetime
        input_df["date"] = pd.to_datetime(input_df["date"])
        target_df["date"] = pd.to_datetime(target_df["date"])
        
        # Clip at 3 sigma to ensure outliers don't destabilize the model
        input_df_filtered, target_df_filtered = apply_3_sigma_filtering_and_standardization(raw_input_events=input_df, 
                                                                                            include_all_columns=include_all_columns,
                                                                        indication=indication, 
                                                                        raw_targets=target_df, 
                                                                        verbose=True)

        all_input_target_and_meta_dfs.append((input_df_filtered, target_df_filtered, constant_df, meta_data))

    
    #: combine into one dataframe
    combined_input_df, combined_constant_df = combine_dfs_into_one_for_training(all_input_target_and_meta_dfs)

    print("Total number of events: ", combined_input_df.shape)
    print("Total nr of time series: ", combined_constant_df.shape)

    # Get correct data format
    train_ag_data, patient_offsets_map, date_mapping_df = preprocess_input_to_autogluon_format(combined_constant_df, combined_input_df)

    # Finetune model
    finetuned_model, model_path_to_return = finetune_tide_model(train_ag_data=train_ag_data, 
                                                                prediction_length=train_prediction_length, 
                                                                log_path=model_save_path, 
                                                                include_all_columns=include_all_columns,
                                                                time_limit=3600,
                                                                num_trials=5 if not DEBUG else 1,
                                                                patience_nr_epochs=patience_nr_epochs)

    print(f"Finetuned model and saved here:\n" + str(model_save_path))
    wandb.finish()

    # Eval
    eval_model(split=eval_split, base_path=eval_base_path, model_load_path=model_save_path,
               eval_prediction_length=eval_prediction_length,
               train_prediction_length=train_prediction_length,
               include_all_columns=include_all_columns)

 




if __name__ == "__main__":

    # Set argparse with defaults 
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_path', type=str, default="/0_data/3_samples_per_lot_train_100_patients/", help='Base path to data')
    parser.add_argument('--model_save_path', type=str, default="/0_data/models/tide/temp/", help='Path to save the model')
    parser.add_argument('--eval_split', type=str, default="test", help='Split to evaluate on')
    parser.add_argument('--eval_base_path', type=str, default="/0_data/3_samples_per_lot/", help='Base path to data for evaluation')
    parser.add_argument('--include_all_columns', action='store_true', help='Whether to include all columns in the input data')
    args = parser.parse_args()

    # Call main function with arguments
    main(args.base_path, args.model_save_path, args.eval_split, args.eval_base_path, include_all_columns=args.include_all_columns)




