import time
import pandas as pd
import numpy as np
import wandb
import argparse
from utils import setup_imports_nb
setup_imports_nb()
from utils_forecasting_eval import ForecastingEval, apply_last_observed_value_to_target



DEBUG = False




def main(split, base_path):


    wandb.init(project="genie-dt-cit-baselines-forecasting", mode="offline" if DEBUG else "online",
                group="copy_forward")
    indication = "cit"

    wandb.run.name = f"Copy forward - {split}"
    wandb.config.update({
        "split": split,
        "base_path": base_path,
        "indication": indication,
        "model": "copy_forward",
    }, allow_val_change=True)

    print("==========================================================")
    print(f"Indication: ")
    
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

    empty_df = target_df.copy()
    empty_df["event_value"] = pd.NA
    assert empty_df["event_value"].isna().all()

    copy_forward_prediction = apply_last_observed_value_to_target(input_df=input_df, empty_target_df=empty_df)

    evaluator = ForecastingEval(indication=indication, split=split, data_loading_path=base_path)

    result = evaluator.evaluate(copy_forward_prediction)
    
    
    

    wandb.finish()



if __name__ == "__main__":

    # Set argparse with defaults 
    parser = argparse.ArgumentParser(description='Copy forward evaluation')
    parser.add_argument('--split', type=str, default="test", help='Split to evaluate on')
    parser.add_argument('--base_path', type=str, default="genie-dt-cit-baselines-forecasting/0_data/test_data/splits_only/", help='Base path to data')
    args = parser.parse_args()

    # Call main function with arguments
    main(args.split, args.base_path)





