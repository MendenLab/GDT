from utils_split_generation import SplitterGenerator
from utils_convert_to_df import ConvertToDF
import time
import pandas as pd
import wandb
import os
import shutil


DEBUG = False
RELAUNCH = True  # Set to true if relaunching the script to avoid re-generating splits



def main(path_to_save, num_samples_per_lot, split, num_weeks_lookback, num_forecast_days):

    all_indications = [
        'enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
        'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
        'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
        'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
        'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma'
    ]

    wandb.init(project="genie-dt-cgdb-baselines-forecasting", mode="offline" if DEBUG else "online")

    wandb.config.update({
        "path_to_save": path_to_save,
        "num_samples_per_lot": num_samples_per_lot,
        "split": split,
        "num_weeks_lookback": num_weeks_lookback,
        "num_forecast_days": num_forecast_days
    })
    wandb.run.name = f"Generating data -{split} -num_samples_per_lot - {num_samples_per_lot}-num_weeks_lookback-{num_weeks_lookback}-num_forecast_days-{num_forecast_days}"

    if RELAUNCH:
        # Find out which indications already exist in the path and skip those
        existing_files = [f for f in os.listdir(path_to_save) if f.startswith("input_df_")]
        existing_indications = set(f.split("input_df_")[1].split("_" + split)[0] for f in existing_files if f.endswith(".csv"))
        all_indications_to_process = [ind for ind in all_indications if ind not in existing_indications]
        if len(all_indications_to_process) == 0:
            print(f"All indications already processed in {path_to_save}. Exiting.")
            return
        else:
            print(f"Processing the following indications: {all_indications_to_process}")
        all_indications = all_indications_to_process

    else:
        # First delete all files in the path to save, if they exist
        if os.path.exists(path_to_save):
            print(f"Previous files exist in {path_to_save}. Deleting them.")
            shutil.rmtree(path_to_save)
        os.makedirs(path_to_save, exist_ok=True)
    

    for indication in all_indications:
        
        ##################### Generate splits ###############################
        splitter = SplitterGenerator(indication=indication)
        test_set_patients = splitter.get_all_split_patientids(split=split) 

        start_time = time.time()
        all_results = []
        num_patientids = []

        for patientid in test_set_patients:
            
            curr_result = splitter.convert_full_to_string_for_one_patient(patientid=patientid, 
                                                                num_samples_per_lot=num_samples_per_lot, 
                                                                generate_conversion=False)
            
            curr_result = [result for result in curr_result if len(result) > 0]
            
            if len(curr_result) == 0:
                print(f"Warning: No results for patient {patientid}")
            else:
                all_results.extend(curr_result)
                num_patientids.append(patientid)


        print(f"Time taken for all patients: {time.time() - start_time:.2f} seconds")
        print(f"Number of results: {len(all_results)}")
        print(f"Number of patients: {len(num_patientids)}")

        # Assert that all from the correct split
        assert all([splitter.dm.patient_to_split_mapping[patientid] == split for patientid in num_patientids])

        ##################### Convert to long format ###############################
        converter = ConvertToDF(num_weeks_lookback=num_weeks_lookback, num_forecast_days=num_forecast_days)

        start_time = time.time()
        all_long_dfs = []

        for res in all_results:
            t = converter.convert_split_data_to_long_df(res[0])
            all_long_dfs.extend(t)

        print(f"Time taken for all patients: {time.time() - start_time:.2f} seconds")
        print(f"Number of long dfs: {len(all_long_dfs)}")

        ##################### Save as one dataset ###############################

        all_input_dfs = []
        all_target_dfs = []
        all_constant_dfs = []
        all_meta = []

        for i, df in enumerate(all_long_dfs):
            all_input_dfs.append(df.input_df)
            all_target_dfs.append(df.target_df)
            all_constant_dfs.append(df.constant_df)
            all_meta.append(df.get_dict_of_meta_data())

        all_input_dfs = pd.concat(all_input_dfs, ignore_index=True)
        all_target_dfs = pd.concat(all_target_dfs, ignore_index=True)
        all_constant_dfs = pd.concat(all_constant_dfs, ignore_index=True)
        all_meta = pd.DataFrame(all_meta)

        all_input_dfs.to_csv(f"{path_to_save}/input_df_{indication}_{split}.csv", index=False)
        all_target_dfs.to_csv(f"{path_to_save}/target_df_{indication}_{split}.csv", index=False)
        all_constant_dfs.to_csv(f"{path_to_save}/constant_df_{indication}_{split}.csv", index=False)
        all_meta.to_csv(f"{path_to_save}/meta_data_{indication}_{split}.csv", index=False)
        
        print(f"Saved all dfs for {indication} and split {split} to {path_to_save}")
    
    wandb.finish()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate data for forecasting")
    parser.add_argument("--path_to_save", type=str, default="/0_data/test", 
                        help="Path to save the data")
    parser.add_argument("--num_samples_per_lot", type=int, default=3, help="Number of samples per lot")
    parser.add_argument("--split", type=str, default="test", help="Split to use (train/test)")
    parser.add_argument("--num_weeks_lookback", type=int, default=26, help="Number of weeks to look back")
    parser.add_argument("--num_forecast_days", type=int, default=90, help="Number of days to forecast")

    args = parser.parse_args()

    main(args.path_to_save, args.num_samples_per_lot, args.split, args.num_weeks_lookback, args.num_forecast_days)








