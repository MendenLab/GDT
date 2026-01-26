from utils_split_generation import SplitterGenerator
from utils_convert_to_df import ConvertToDF
import time
import pandas as pd
import wandb
import os
import shutil
import numpy as np


DEBUG = False
RELAUNCH = False  # Set to true if relaunching the script to avoid re-generating splits
DEFAULT_MAX_NUM_PATIENTS = 1000000000  # Default maximum number of patients to process


def main(path_to_save, num_samples_per_lot, data_split, max_num_patients):

    all_indications = [
        'enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
        'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
        'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
        'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
        'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma'
    ]

    wandb.init(project="genie-dt-cgdb-baselines-events-probabilities", mode="offline" if DEBUG else "online")

    wandb.config.update({
        "path_to_save": path_to_save,
        "num_samples_per_lot": num_samples_per_lot,
        "split": data_split,
    })
    wandb.run.name = f"Generating split + DF data -{data_split} -num_samples_per_lot - {num_samples_per_lot}"
    
    if RELAUNCH:
        raise NotImplementedError("Relaunching is not implemented yet. Please set RELAUNCH to False and run the script again.")

    # Manage folder
    if not RELAUNCH:
        # First delete all files in the path to save, if they exist
        path_to_save_indication = os.path.join(path_to_save, data_split)
        if os.path.exists(path_to_save_indication):
            print(f"Previous files exist in {path_to_save_indication}. Deleting them.")
            shutil.rmtree(path_to_save_indication)
        os.makedirs(path_to_save_indication, exist_ok=True)
        

    for indication in all_indications:
        
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
        all_inputs_dfs = []
        all_target_dfs = []
        num_patientids = []

        for patientid in test_set_patients:

            # Generate split            
            curr_result = splitter.convert_full_to_string_for_one_patient(patientid=patientid, 
                                                                        num_samples_per_lot=num_samples_per_lot, 
                                                                        generate_conversion=False)

            # Save directly
            for split_idx in curr_result.keys():
                split = curr_result[split_idx]
                for curr_split in split:
                
                    #: convert to dfs
                    input_df, target_df = df_converter.convert_split_data_to_input_and_output_df(curr_split)

                    #: save
                    all_inputs_dfs.append(input_df)
                    all_target_dfs.append(target_df)

            if len(curr_result) == 0:
                print(f"Warning: No results for patient {patientid}")
            else:
                num_patientids.append(patientid)

        #: save
        all_inputs_df = pd.concat(all_inputs_dfs, ignore_index=True)
        all_target_df = pd.concat(all_target_dfs, ignore_index=True)

        all_inputs_df.to_csv(os.path.join(path_to_save_indication, f"inputs_{data_split}_{indication}.csv"), index=False)
        all_target_df.to_csv(os.path.join(path_to_save_indication, f"targets_{data_split}_{indication}.csv"), index=False)
        
        print(f"Time taken for all patients: {time.time() - start_time:.2f} seconds")
        print(f"Number of results: {len(all_target_dfs)}")
        print(f"Number of patients: {len(num_patientids)}")

        # Assert that all from the correct split
        assert all([splitter.dm.patient_to_split_mapping[patientid] == data_split for patientid in num_patientids])

        print(f"Saved all dfs for {indication} and split {data_split} to {path_to_save_indication}")
    
    wandb.finish()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate data for forecasting")
    parser.add_argument("--path_to_save", type=str, default="genie-dt-cgdb-eval-events/0_data/test/", 
                        help="Path to save the data")
    parser.add_argument("--num_samples_per_lot", type=int, default=1, help="Number of samples per lot")
    parser.add_argument("--split", type=str, default="train", help="Split to use (train/test)")
    parser.add_argument("--max_num_patients", type=int, default=DEFAULT_MAX_NUM_PATIENTS, help="Maximum number of patients to process")  

    args = parser.parse_args()

    main(args.path_to_save, args.num_samples_per_lot, args.split, args.max_num_patients)








