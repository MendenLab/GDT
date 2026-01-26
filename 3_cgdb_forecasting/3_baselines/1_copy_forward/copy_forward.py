import time
import pandas as pd
import numpy as np
import wandb
import argparse
from utils import setup_imports_nb
setup_imports_nb()
from utils_forecasting_eval import ForecastingEval, apply_last_observed_value_to_target



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
                   group="copy_forward")

        wandb.run.name = f"Copy forward - {split} - {indication}"
        wandb.config.update({
            "split": split,
            "base_path": base_path,
            "indication": indication,
            "model": "copy_forward",
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

        empty_df = target_df.copy()
        empty_df["event_value"] = pd.NA
        assert empty_df["event_value"].isna().all()

        copy_forward_prediction = apply_last_observed_value_to_target(input_df=input_df, empty_target_df=empty_df)

        evaluator = ForecastingEval(indication=indication,
                            split=split,
                            data_loading_path=base_path)

        result = evaluator.evaluate(copy_forward_prediction)
        
        
        

        wandb.finish()



if __name__ == "__main__":

    # Set argparse with defaults 
    parser = argparse.ArgumentParser(description='Copy forward evaluation')
    parser.add_argument('--split', type=str, default="test", help='Split to evaluate on')
    parser.add_argument('--base_path', type=str, default="/0_data/3_samples_per_lot/", help='Base path to data')
    args = parser.parse_args()

    # Call main function with arguments
    main(args.split, args.base_path)





