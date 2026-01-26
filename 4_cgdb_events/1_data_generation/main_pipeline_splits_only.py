from utils_split_generation import SplitterGenerator
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

    wandb.init(project="genie-dt-cgdb-baselines-events", mode="offline" if DEBUG else "online")

    wandb.config.update({
        "path_to_save": path_to_save,
        "num_samples_per_lot": num_samples_per_lot,
        "split": data_split,
    })
    wandb.run.name = f"Generating data -{data_split} -num_samples_per_lot - {num_samples_per_lot}"
    
    if RELAUNCH:
        raise NotImplementedError("Relaunching is not implemented yet. Please set RELAUNCH to False and run the script again.")

    for indication in all_indications:

        # Manage folder
        if not RELAUNCH:
            # First delete all files in the path to save, if they exist
            path_to_save_indication = os.path.join(path_to_save, data_split + "_" + indication)
            if os.path.exists(path_to_save_indication):
                print(f"Previous files exist in {path_to_save_indication}. Deleting them.")
                shutil.rmtree(path_to_save_indication)
            os.makedirs(path_to_save_indication, exist_ok=True)
        
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

        for patientid in test_set_patients:

            # Generate split            
            curr_result = splitter.convert_full_to_string_for_one_patient(patientid=patientid, 
                                                                        num_samples_per_lot=num_samples_per_lot, 
                                                                        generate_conversion=False)

            # Save directly
            for split_idx in curr_result.keys():
                
                split = curr_result[split_idx]

                #: assert that same input_df across each split, just to make sure
                all_input_dfs = [x["events_until_split"] for x in split]
                event_value_columns = [input_df["event_value"].tolist() for input_df in all_input_dfs]
                assert all([event_value_columns[0] == event_value_columns[i] for i in range(1, len(event_value_columns))]), \
                    f"Event value columns are not the same across all input_dfs for split {split} in indication {indication}"
                data_value  = [input_df["date"].tolist() for input_df in all_input_dfs]
                assert all([data_value[0] == data_value[i] for i in range(1, len(data_value))]), \
                    f"Date columns are not the same across all input_dfs for split {split} in indication {indication}"

                #: Save only one per LoT input DF, and all corresponding targets
                first_input_df = all_input_dfs[0]

                # Set generic patientid for better splitting
                generic_patientid = first_input_df["patientid"].apply(lambda x: x.split("_var_")[0])
                first_input_df["patientid"] = generic_patientid

                # Aggregate all targets
                all_targets = []
                for sub_split in split:
                    all_targets.append({
                        "patientid": sub_split["new_patientid"],
                        "generic_patientid": generic_patientid[0],
                        "split_date_included_in_input": sub_split["split_date_included_in_input"],
                        "sampled_category": sub_split["sampled_category"],
                        "week_to_predict": sub_split["week_to_predict"],
                        "censored": sub_split["event_censored"],
                        "occurred": sub_split["event_occured"],
                        "true_censoring": sub_split["true_censoring"],
                        "true_time": sub_split["true_time"],
                    })
                    
                    # Do checks
                    assert  sub_split["sampled_category"] in sub_split["new_patientid"], "Sampled category not in new patientid!"

                all_targets_df = pd.DataFrame(all_targets)

                #: Save DFs
                base_save_path = os.path.join(path_to_save_indication, generic_patientid[0])
                all_targets_df.to_csv(f"{base_save_path}_targets.csv", index=False)
                first_input_df.to_csv(f"{base_save_path}_input_df.csv", index=False)

            if len(curr_result) == 0:
                print(f"Warning: No results for patient {patientid}")
            else:
                num_patientids.append(patientid)

            
        
        print(f"Time taken for all patients: {time.time() - start_time:.2f} seconds")
        print(f"Number of results: {len(all_results)}")
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
    parser.add_argument("--split", type=str, default="test", help="Split to use (train/test)")
    parser.add_argument("--max_num_patients", type=int, default=DEFAULT_MAX_NUM_PATIENTS, help="Maximum number of patients to process")

    args = parser.parse_args()

    main(args.path_to_save, args.num_samples_per_lot, args.split, args.max_num_patients)








