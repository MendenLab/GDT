from utils_split_generation import SplitterGenerator
from utils_convert_to_df import ConvertToDF
import time
import pandas as pd
import wandb
import os
import shutil
import numpy as np
import gc
from tqdm import tqdm


DEBUG = False



all_indications = [
        'enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
        'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
        'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
        'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
        'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma'
    ]



def main(path_to_save, num_samples_per_lot, split, num_max_patients):

    wandb.init(project="genie-dt-cgdb-baselines-forecasting", mode="offline" if DEBUG else "online")

    wandb.config.update({
        "path_to_save": path_to_save,
        "num_samples_per_lot": num_samples_per_lot,
        "split": split,
    })
    wandb.run.name = f"Generating ablation text data -{split} -num_samples_per_lot - {num_samples_per_lot}"

    # First delete all files in the path to save, if they exist
    if os.path.exists(path_to_save):
        print(f"Previous files exist in {path_to_save}. Deleting them.")
        shutil.rmtree(path_to_save)
    os.makedirs(path_to_save, exist_ok=True)
    

    for indication in all_indications:
        
        ##################### Generate splits ###############################
        splitter = SplitterGenerator(indication=indication)
        test_set_patients = splitter.get_all_split_patientids(split=split) 

        #: randomly sample subset of patients from the train set num_max_patients
        np.random.seed(7186)
        if num_max_patients < len(test_set_patients):
            test_set_patients = np.random.choice(test_set_patients, size=num_max_patients, replace=False)
        print(f"Number of patients in {split} set: {len(test_set_patients)}")
        wandb.log({"num_patients": len(test_set_patients)})


        start_time = time.time()
        all_results = []
        num_patientids = []

        for patientid in tqdm(test_set_patients):
            curr_result = splitter.convert_full_to_string_for_one_patient(patientid=patientid, 
                                                                num_samples_per_lot=num_samples_per_lot, 
                                                                generate_conversion=True)
            
            curr_result = [result for result in curr_result if len(result) > 0]
            
            if len(curr_result) == 0:
                print(f"Warning: No results for patient {patientid}")
            else:
                all_results.extend(curr_result)
                num_patientids.append(patientid)


        print(f"Time taken for all patients: {time.time() - start_time:.2f} seconds")
        print(f"Number of results: {len(all_results)}")
        print(f"Number of patients: {len(num_patientids)}")
        # Setup saving
        result_list = []
        
        #: save everything to files
        for result in all_results:
            #: save instruction and answer
            curr_entry = {
                "instruction" : result["instruction"],
                "target": result["answer"]
            }

            #: all that is needed for reverse is split_date_included_in_input
            #: save from metadata patientid, split_date_included_in_input, sampled_variables, 
            curr_entry["patientid"] = result["meta"]["patientid"].iloc[0]
            curr_entry["split_date_included_in_input"] = result["meta"]["split_date_included_in_input"]
            curr_entry["sampled_variables"] = list(result["meta"]["combined_meta"]["variable_name_mapping"].keys())
            curr_entry["indication"] = indication
            curr_entry["split"] = split 
            

            # Add empty target as string (useful e.g. for dates for reverse conversion for llama3.1)
            empty_target = result["meta"]["target_meta_detailed"][0]["target_data_processed"].copy()
            empty_target["event_value"] = pd.NA
            empty_target["patientid"] = result["meta"]["patientid"].iloc[0]
            empty_target_as_string = empty_target.to_json(orient="records")
            curr_entry["empty_target_as_string"] = empty_target_as_string

            # Add in last observed values for every target variable
            last_observed = result["meta"]["target_meta_detailed"][0]["last_observed_values"].copy()
            last_observed["patientid"] = result["meta"]["patientid"].iloc[0]
            curr_entry["last_observed_values"] = last_observed.to_json(orient="records")

            # Append
            result_list.append(curr_entry)
        
        #: save the result_df
        curr_path = path_to_save + f"text_table_{indication}_{split}_num_samples_per_lot_{num_samples_per_lot}.csv"
        result_df = pd.DataFrame(result_list)
        result_df.to_csv(curr_path)

    
    wandb.finish()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate data for forecasting")
    parser.add_argument("--path_to_save", type=str, default="/0_data/ablations/subset_train_with_text/", 
                        help="Path to save the data")
    parser.add_argument("--num_samples_per_lot", type=int, default=3, help="Number of samples per lot")
    parser.add_argument("--split", type=str, default="train", help="Split to use (train/validation/test)")
    parser.add_argument("--num_max_patients", type=int, default=5, help="Number of patients to sample from the train set")

    args = parser.parse_args()

    main(args.path_to_save, args.num_samples_per_lot, args.split, args.num_max_patients)








