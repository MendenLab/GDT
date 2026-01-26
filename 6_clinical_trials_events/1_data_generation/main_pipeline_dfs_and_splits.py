from utils_split_generation import SplitterGenerator
from utils_convert_to_df import ConvertToDF
import time
import pandas as pd
import wandb
import os
import shutil
import numpy as np


DEBUG = False
DEFAULT_MAX_NUM_PATIENTS = 1000000000  # Default maximum number of patients to process
all_weeks_to_generate = [8, 26, 52, 104]
OVERRIDE_DATE = pd.Timestamp("2020-01-22")   # Artifical standard baseline date used from data processing



def main(path_to_save, num_samples_per_lot, data_split, max_num_patients):

    indication = "cit"
    path_to_save_indication = path_to_save
    wandb.init(project="genie-dt-cit-baselines-events", mode="offline" if DEBUG else "online")

    wandb.config.update({
        "path_to_save": path_to_save,
        "num_samples_per_lot": num_samples_per_lot,
        "split": data_split,
    })
    wandb.run.name = f"Generating split + DF data - {data_split}"
    
    
    ##################### Generate splits ###############################
    splitter = SplitterGenerator(indication=indication)
    test_set_patients = splitter.get_all_split_patientids(split=data_split)

    if max_num_patients < DEFAULT_MAX_NUM_PATIENTS:
        # Shuffle and limit the number of patients
        print(f"Limiting to {max_num_patients} patients for {indication} in split {data_split}")
        np.random.seed(9682)
        np.random.shuffle(test_set_patients)
        test_set_patients = test_set_patients[:max_num_patients]

    df_converter = ConvertToDF()

    start_time = time.time()
    all_ml_inputs_dfs = []
    all_full_inputs_dfs = []
    all_constant_dfs = []
    all_target_dfs = []
    num_patientids = []


    for patientid in test_set_patients:

        # Generate split            
        curr_result = splitter.convert_full_to_string_for_one_patient(patientid=patientid, 
                                                                      weeks_to_generate=all_weeks_to_generate,
                                                                      num_samples_per_lot=num_samples_per_lot, 
                                                                      split_date=OVERRIDE_DATE,
                                                                      generate_conversion=False)

        # assert that all from the same patientid
        assert len(curr_result.keys()) == 1
        assert all([curr_split["constant_data"]["patientid"].iloc[0].split("_lot")[0] == patientid for curr_split in curr_result[list(curr_result.keys())[0]]])

        # Save directly
        for split_idx in curr_result.keys():
            split = curr_result[split_idx]
            for idx, curr_split in enumerate(split):
            
                #: convert to dfs
                ml_input_df, full_input_df, constant_df, target_df = df_converter.convert_split_data_to_input_and_output_df(curr_split)

                #: save
                all_target_dfs.append(target_df)

                # Only save DFs for first since it is the same across all splits, and is data heavy
                if idx == 0:
                    all_full_inputs_dfs.append(full_input_df)
                    all_constant_dfs.append(constant_df)
                    all_ml_inputs_dfs.append(ml_input_df)

        if len(curr_result) == 0:
            print(f"Warning: No results for patient {patientid}")
        else:
            num_patientids.append(patientid)
    
    # Assert that all from the correct split
    assert all([splitter.dm.patient_to_split_mapping[patientid] == data_split for patientid in num_patientids])

    #: save
    all_target_df = pd.concat(all_target_dfs, ignore_index=True)
    all_ml_input_df = pd.concat(all_ml_inputs_dfs, ignore_index=True)
    all_full_input_df = pd.concat(all_full_inputs_dfs, ignore_index=True)
    all_constant_df = pd.concat(all_constant_dfs, ignore_index=True)

    all_target_df.to_csv(os.path.join(path_to_save_indication, f"targets_{data_split}_{indication}.csv"), index=False)
    all_ml_input_df.to_csv(os.path.join(path_to_save_indication, f"ml_input_{data_split}_{indication}.csv"), index=False)
    all_full_input_df.to_csv(os.path.join(path_to_save_indication, f"full_input_{data_split}_{indication}.csv"), index=False)
    all_constant_df.to_csv(os.path.join(path_to_save_indication, f"constant_{data_split}_{indication}.csv"), index=False)
    
    print(f"Time taken for all patients: {time.time() - start_time:.2f} seconds")
    print(f"Number of results: {len(all_target_dfs)}")
    print(f"Number of patients: {len(num_patientids)}")
    print(f"Saved all dfs for {indication} and split {data_split} to {path_to_save_indication}")

    wandb.finish()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate data for forecasting")
    parser.add_argument("--path_to_save", type=str, default="genie-dt-cit-eval-events/0_data/development/test/", 
                        help="Path to save the data")
    parser.add_argument("--num_samples_per_lot", type=int, default=1, help="Number of samples per lot")
    parser.add_argument("--split", type=str, default="train", help="Split to use (train/test)")
    parser.add_argument("--max_num_patients", type=int, default=DEFAULT_MAX_NUM_PATIENTS, help="Maximum number of patients to process")  

    args = parser.parse_args()

    main(args.path_to_save, args.num_samples_per_lot, args.split, args.max_num_patients)








