import time
import pandas as pd
import numpy as np
import wandb
import argparse
from utils_chronos import preprocess_input_to_autogluon_format, predict_chronos_model_on_input, map_predictions_to_original_dates, post_process_predictions_back_into_original_format, apply_3_sigma_filtering_and_standardization
from utils import setup_imports_nb
setup_imports_nb()
from utils_forecasting_eval import ForecastingEval



DEBUG = False


all_indications = [
        'enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
        'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
        'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
        'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
        'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma'
    ]


def main(split, base_path):

    for indication in all_indications:

        wandb.init(project="genie-dt-cgdb-baselines-forecasting", mode="offline" if DEBUG else "online",
                   group="chronos_zero_shot")

        # Setting up the model parameters
        prediction_length  = 13  # 14 weeks (98 days) will cover all possible predictions and edge cases
        chronos_model = "autogluon/chronos-t5-large"

        wandb.run.name = f"Chronos - Zero Shot - {split} - {indication}"
        wandb.config.update({
            "split": split,
            "base_path": base_path,
            "indication": indication,
            "model": "chronos_zero_shot",
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
                                                                        indication=indication, 
                                                                        raw_targets=target_df, 
                                                                        verbose=True)

        
        # Do all Chronos preprocessing
        autogluon_input_df, patient_offsets_map, date_mapping_df = preprocess_input_to_autogluon_format(constant_df, input_df_filtered)

        predictions = predict_chronos_model_on_input(prediction_length=prediction_length, 
                                            train_ag_data=autogluon_input_df, 
                                            chronos_model=chronos_model)

        mapped_predictions = map_predictions_to_original_dates(predictions, patient_offsets_map)

        target_dates_patientid_variable = target_df[["patientid", "date", "event_name"]].copy()
        processed_predictions = post_process_predictions_back_into_original_format(mapped_predictions, meta_data,
                                                                                   target_dates_patientid_variable,
                                                                                   indication=indication)

        evaluator = ForecastingEval(indication=indication,
                            split=split,
                            data_loading_path=base_path)

        result = evaluator.evaluate(processed_predictions)
        
        
        

        wandb.finish()



if __name__ == "__main__":

    # Set argparse with defaults 
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--split', type=str, default="test", help='Split to evaluate on')
    parser.add_argument('--base_path', type=str, default="/0_data/3_samples_per_lot/", help='Base path to data')
    args = parser.parse_args()

    # Call main function with arguments
    main(args.split, args.base_path)





