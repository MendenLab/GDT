from utils_split_generation import SplitterGenerator
from utils_convert_to_df import ConvertToDF
import time
import pandas as pd
import wandb
import os
import shutil
from tqdm import tqdm


DEBUG = False
OVERRIDE_DATE = pd.Timestamp("2020-01-22")   # Artifical standard baseline date used from data processing

all_variables = ['1751_7', '6768_6', '17861_6', '1742_6', '2075_0', '26450_7', 
                 '26449_9', '2345_7', '718_7', '2532_0', '26478_8', '26484_6', 
                 '26499_4', 'nlr_no_loinc', '26515_7', '1975_2', '2885_2']





def main(path_to_save, split):

    wandb.init(project="genie-dt-cit-baselines-forecasting", mode="offline" if DEBUG else "online")

    wandb.config.update({
        "path_to_save": path_to_save,
        "split": split,
    })
    wandb.run.name = f"Generating data with text -{split} "

    # First delete all files in the path to save, if they exist
    path_to_save_directory = os.path.dirname(path_to_save)
    if os.path.exists(path_to_save_directory):
        print(f"Previous files exist in {path_to_save}. Deleting them.")
        shutil.rmtree(path_to_save_directory)
    os.makedirs(path_to_save_directory, exist_ok=True)
    

    ##################### Generate splits ###############################
    splitter = SplitterGenerator(all_variables)
    test_set_patients = splitter.get_all_split_patientids(split=split) 

    start_time = time.time()
    num_patientids = []
    all_results = []

    for patientid in tqdm(test_set_patients):
        curr_result = splitter.convert_full_to_string_for_one_patient(patientid=patientid, 
                                                             override_date=OVERRIDE_DATE,
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
    curr_path = path_to_save + f"text_table_{split}.csv"
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(curr_path)

    wandb.finish()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate data for forecasting")
    parser.add_argument("--path_to_save", type=str, default="genie-dt-cit-baselines-forecasting/0_data/3_samples_per_lot_text/", 
                        help="Path to save the data")
    parser.add_argument("--split", type=str, default="train", help="Split to use (train/test)")

    args = parser.parse_args()

    main(args.path_to_save, args.split)








