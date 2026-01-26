from utils_split_generation import SplitterGenerator
import time
import pandas as pd
import wandb
import os
import shutil
import numpy as np
from tqdm import tqdm


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

    wandb.init(project="genie-dt-cgdb-baselines-events", mode="offline" if DEBUG else "online")

    wandb.config.update({
        "path_to_save": path_to_save,
        "num_samples_per_lot": num_samples_per_lot,
        "split": data_split,
    })
    wandb.run.name = f"Generating text data -{data_split} -num_samples_per_lot - {num_samples_per_lot}"
    
    if RELAUNCH:
        raise NotImplementedError("Relaunching is not implemented yet. Please set RELAUNCH to False and run the script again.")


    # Manage folder
    if not RELAUNCH:
        # First delete all files in the path to save, if they exist
        if os.path.exists(path_to_save):
            print(f"Previous files exist in {path_to_save}. Deleting them.")
            shutil.rmtree(path_to_save)
        os.makedirs(path_to_save, exist_ok=True)

    

    for indication in all_indications:
        
        ##################### Generate splits ###############################
        splitter = SplitterGenerator(indication=indication)
        test_set_patients = splitter.get_all_split_patientids(split=data_split)

        if max_num_patients < DEFAULT_MAX_NUM_PATIENTS:
            # Shuffle and limit the number of patients
            print(f"Limiting to {max_num_patients} patients for {indication} in split {data_split}")
            np.random.seed(42)
            np.random.shuffle(test_set_patients)
            test_set_patients = test_set_patients[:max_num_patients]

        start_time = time.time()
        all_results = []
        num_patientids = []

        for patientid in tqdm(test_set_patients):

            # Generate split            
            curr_result = splitter.convert_full_to_string_for_one_patient(patientid=patientid, 
                                                                        num_samples_per_lot=num_samples_per_lot, 
                                                                        generate_conversion=True)

            curr_result = [result for result in curr_result if len(result) > 0]

            if len(curr_result) == 0:
                print(f"Warning: No results for patient {patientid}")
            else:
                num_patientids.append(patientid)
                all_results.extend(curr_result)
        
        # Setup saving
        print(f"Number of patients processed: {len(num_patientids)}")
        result_list = []
        
        #: save everything to files
        for result in tqdm(all_results):
            #: save instruction - (no answer since we're doing inference)
            curr_entry = {
                "instruction" : result["instruction"],
            }

            #: all that is needed for reverse is split_date_included_in_input
            #: save from metadata patientid, split_date_included_in_input, sampled_variables, 
            curr_entry["patientid"] = result["meta"]["patientid"].iloc[0]
            curr_entry["split_date_included_in_input"] = result["meta"]["split_date_included_in_input"]
            curr_entry["sampled_category"] = result["meta"]["target_meta_detailed"][0]["target_category"]
            curr_entry["week_to_predict"] = result["meta"]["week_to_predict"]
            curr_entry["indication"] = indication
            curr_entry["split"] = data_split 
            

            # Add empty target as string (useful e.g. for dates for reverse conversion for llama3.1)
            empty_target = result["meta"]["target_meta_detailed"][0]["target_data_processed"].copy()
            empty_target["censoring"] = pd.NA
            empty_target["occurred"] = pd.NA
            empty_target_as_string = empty_target.to_json(orient="records")
            curr_entry["empty_target_as_string"] = empty_target_as_string

            # Append
            result_list.append(curr_entry)
        
        #: save the result_df
        curr_path = path_to_save + f"text_table_{indication}_{data_split}_num_samples_per_lot_{num_samples_per_lot}.csv"
        result_df = pd.DataFrame(result_list)
        result_df.to_csv(curr_path)

        
        print(f"Time taken for all patients: {time.time() - start_time:.2f} seconds")
        print(f"Number of results: {len(result_list)}")
        print(f"Number of patients: {len(num_patientids)}")

        # Assert that all from the correct split
        assert all([splitter.dm.patient_to_split_mapping[patientid] == data_split for patientid in num_patientids])

        print(f"Saved all dfs for {indication} and split {data_split} to {path_to_save}")
    
    wandb.finish()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate data for forecasting")
    parser.add_argument("--path_to_save", type=str, default="genie-dt-cgdb-eval-events/0_data/test/", 
                        help="Path to save the data")
    parser.add_argument("--num_samples_per_lot", type=int, default=1, help="Number of samples per lot")
    parser.add_argument("--split", type=str, default="test", help="Split to use (train/test)")
    parser.add_argument("--max_num_patients", type=int, default=DEFAULT_MAX_NUM_PATIENTS, help="Maximum number of patients to process")

    args = parser.parse_args()

    main(args.path_to_save, args.num_samples_per_lot, args.split, args.max_num_patients)








