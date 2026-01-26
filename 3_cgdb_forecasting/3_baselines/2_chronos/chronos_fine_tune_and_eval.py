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


WANDB_DEFAULT_GROUP = "chronos_standard_fine_tuned"
WANDB_LARGER_TRAIN_DATASET = "_larger_train_dataset"
LARGER_TRAIN_DATASET_PATH = "/0_data/3_samples_per_lot_train/"
WANDB_NO_STANDARDIZE = "_no_standardize"
NUM_VAL_SAMPLES = 100



DEBUG = False



all_indications = [
        'enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
        'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
        'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
        'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
        'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma'
    ]




def eval_model(split, base_path, model, larger_train_dataset, standardize, eval_prediction_length):

    for indication in all_indications:

        wandb_name = WANDB_DEFAULT_GROUP
        if larger_train_dataset:
            wandb_name += WANDB_LARGER_TRAIN_DATASET
        if not standardize:
            wandb_name += WANDB_NO_STANDARDIZE
        wandb_name += f"_pred_len_{eval_prediction_length}"
        
        wandb.init(project="genie-dt-cgdb-baselines-forecasting", mode="offline" if DEBUG else "online",
                    group=wandb_name)

        # Setting up the model parameters

        wandb.run.name = f"Chronos - Fine-Tuned - {split} - {indication}"
        wandb.config.update({
            "split": split,
            "base_path": base_path,
            "indication": indication,
            "model": "chronos_fine_tuning",
            "larger_train_dataset": larger_train_dataset,
            "standardize": standardize,
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
                                                                        standardize=standardize)

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
                                                                                   destandardize=standardize)
        
        evaluator = ForecastingEval(indication=indication,
                            split=split,
                            data_loading_path=base_path)

        result = evaluator.evaluate(processed_predictions)
        
        
        

        wandb.finish()


def main(base_path, model_save_path, standardize, eval_split, eval_base_path):

    train_prediction_length = 13
    eval_prediction_length = 13

    wandb_name = WANDB_DEFAULT_GROUP
    if base_path == LARGER_TRAIN_DATASET_PATH:
        wandb_name += WANDB_LARGER_TRAIN_DATASET
    if not standardize:
        wandb_name += WANDB_NO_STANDARDIZE
    wandb_name += f"_pred_len_{train_prediction_length}"

    wandb.init(project="genie-dt-cgdb-baselines-forecasting", mode="offline" if DEBUG else "online",
                   group=wandb_name)

    # Setting up the model parameters
    split = "train"
    chronos_model = "autogluon/chronos-t5-large"  # TOOD: trying normal chronos


    wandb.run.name = f"Chronos - Fine-Tuned - Training Job"
    wandb.config.update({
        "split": split,
        "base_path": base_path,
        "model": "chronos_fine_tuning",
        "chronos_model": chronos_model,
        "train_prediction_length": train_prediction_length,
        "eval_prediction_length": eval_prediction_length,
        "model_save_path": model_save_path,
        "standardize": standardize,
        "larger_train_dataset": base_path == LARGER_TRAIN_DATASET_PATH,
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
                                                                                            standardize=standardize,)

        all_input_target_and_meta_dfs.append((input_df_filtered, target_df_filtered, constant_df, meta_data))

    
    #: combine into one dataframe
    combined_input_df, combined_constant_df = combine_dfs_into_one_for_training(all_input_target_and_meta_dfs)

    print("Total number of events: ", combined_input_df.shape)
    print("Total nr of time series: ", combined_constant_df.shape)
    print("Example input df: ", combined_input_df.head())
    print("Example constant df: ", combined_constant_df.head())


    # Get correct data format
    train_ag_data, patient_offsets_map, date_mapping_df = preprocess_input_to_autogluon_format(combined_constant_df, combined_input_df)
    

    # Finetune model
    max_nr_fine_tune_steps = 30000   # Some arbitrary high number, and then best model is selected
    print(f"Max nr of fine-tune steps: {max_nr_fine_tune_steps}")
    finetuned_model, best_model_path = finetune_chronos_model(chronos_model=chronos_model, 
                                                    train_ag_data=train_ag_data, 
                                                    prediction_length=train_prediction_length, 
                                                    log_path=model_save_path, 
                                                    time_limit=3600 * 5,  # 5 hours max training
                                                    max_nr_fine_tune_steps=max_nr_fine_tune_steps,
                                                    lr_for_single_run=1e-3,
                                                    eval_steps=1000, 
                                                    report_to_wandb=True,
                                                    fine_tune_batch_size=32,
                                                    save_steps=1000,
                                                    nr_validation_samples=43 * NUM_VAL_SAMPLES)  # Since length 43 per sample

    print(f"Finetuned model and saved here:\n" + str(best_model_path))
    wandb.finish()

    ############################# EVAL ##################################################

    eval_model(eval_split, eval_base_path, finetuned_model, 
               larger_train_dataset=base_path == LARGER_TRAIN_DATASET_PATH, 
               standardize=standardize,
               eval_prediction_length=eval_prediction_length)







if __name__ == "__main__":

    # Set argparse with defaults 
    parser = argparse.ArgumentParser(description='Evaluation')
    #: set to correct path
    parser.add_argument('--base_path', type=str, default="/0_data/3_samples_per_lot_train_100_patients/", help='Base path to data')
    parser.add_argument('--model_save_path', type=str, default="/0_data/models/chronos/all_indications_standard_chronos/", help='Path to save the model')
    parser.add_argument('--no-standardize', dest='standardize', action='store_false', help='Disable standardization (default: True)')
    parser.add_argument('--eval_split', type=str, default="test", help='Split to evaluate on (default: test)')
    parser.add_argument('--eval_base_path', type=str, default="/0_data/3_samples_per_lot_test_100_patients/", help='Base path to data for evaluation')
    parser.set_defaults(standardize=True)
    args = parser.parse_args()

    # Call main function with arguments
    main(args.base_path, args.model_save_path, args.standardize, args.eval_split, args.eval_base_path)




