import pandas as pd
import re
from io import StringIO
import os
import sys
import traceback
import wandb
from tqdm import tqdm


from digital_twin_converter.instruction.data_splitter_forecasting import DataSplitterForecasting
from digital_twin_converter.common.data_manager import SingleIndicationDataManager
from digital_twin_converter.instruction.data_splitter_events import DataSplitterEvents
from digital_twin_converter.instruction.converter_manual_instruction import ConverterManualInstruction
from digital_twin_converter.common.config import Config



PATH_TO_SPLITS = "cit_data/subset_data/2025_07_21_nsclc_with_fmi/"
COL_CATEGORY_LOT = "lot"
PATH_TO_CONSTANT_DESCRIPTION = "/flatiron_cgdb/2024_07_05/enhanced_nsclc_master_constant_descriptions.csv"
PATH_TO_SPLIT_JSON = "cit_data/subset_data/2025_07_21_nsclc_with_fmi/patient_split_mapping.json"
PATH_TO_RAW_SURVIVAL_DATA = "cit_data/raw_cit_data/survival_data.csv"



def setup_imports_nb():
    try:
        # In case you run this cell from a .py file later via an IDE that defines __file__
        notebook_parent_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ is not defined in a standard Jupyter notebook cell, so use os.getcwd()
        # os.getcwd() usually points to the directory where the notebook file is located,
        # or where the Jupyter server was started.
        # Be sure this is the correct directory for your structure.
        notebook_parent_dir = os.getcwd()

    project_root_dir = os.path.abspath(os.path.join(notebook_parent_dir, "../2_eval_tools/"))
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)




def make_nr_of_copies(raw_df, nr_of_copies):
    """
    Make a specified number of copies of each row in the DataFrame.
    
    Args:
        raw_df (pandas.DataFrame): The input DataFrame to copy.
        nr_of_copies (int): The number of copies to create for each row.
        
    Returns:
        pandas.DataFrame: A new DataFrame with the specified number of copies.
    """
    adjusted = pd.concat([raw_df] * nr_of_copies, ignore_index=True)
    return adjusted



def add_extra_prompt_to_instruction(raw_df, extra_prompt):
    """
    Add extra prompt to instruction column in the dataframe.
    """
    raw_df = raw_df.copy()
    raw_df["instruction"] = raw_df["instruction"].apply(lambda x: f"{x} {extra_prompt}")
    return raw_df





def process_raw_data_to_list(raw_df):
    ret = raw_df[["patientid", "instruction"]].values.tolist()
    return ret



def setup_all_data_managers_and_converters(indication):

    config = Config()

    # Override with custom dataset from CIT
    config.override_with_custom_dataset = True
    config.override_datasets_events = PATH_TO_SPLITS + "df_events_subset.csv"
    config.override_datasets_constant = PATH_TO_SPLITS + "df_constant_subset.csv"
    config.override_datasets_molecular = PATH_TO_SPLITS + "df_molecular_subset.csv"
    config.override_constant_description = PATH_TO_CONSTANT_DESCRIPTION   # Use the standard description
    config.override_statistics = None
    config.date_cutoff = pd.to_datetime("2124-03-31")   # Just some very high number, since this is already handled by CIT data

    dm = SingleIndicationDataManager(indication, config=config)
    dm.load_indication_data()
    dm.process_indication_data()
    dm.setup_unique_mapping_of_events()
    dm.setup_dataset_splits()
    converter = ConverterManualInstruction(dm.data_frames["constant_description"],
                                            nr_tokens_budget_total=8192,
                                            config=config)

    return config, dm, converter


def process_empty_targets_from_raw_data(raw_initial_data):

    all_empty_targets = []
    for i, row in tqdm(raw_initial_data.iterrows()):
        target_as_str = row["empty_target_as_string"]
        patientid = row["patientid"]
        # parse from json str, orient="records"
        target_as_df = pd.read_json(StringIO(target_as_str), orient="records",
                                    dtype={'event_name': str})
        target_as_df["patientid"] = patientid
        all_empty_targets.append(target_as_df)
    all_empty_targets_df = pd.concat(all_empty_targets, ignore_index=True)
    return all_empty_targets_df




def convert_all_results_back_to_df(all_raw_results, raw_data, converter, dm):

    split_dates = raw_data[["patientid", "split_date_included_in_input"]].copy()
    split_dates["split_date_included_in_input"] = pd.to_datetime(split_dates["split_date_included_in_input"])

    all_results = {}
    num_failure = 0

    for i, raw_result in tqdm(all_raw_results.iterrows()):

        patientid = raw_result["patientid"]
        target_string = raw_result["response"]
        split_date = split_dates[split_dates["patientid"] == patientid]["split_date_included_in_input"].values[0]

        try:
            reverse_converted = converter.reverse_conversion(
                target_string=target_string,
                split_date=split_date,
                data_manager=dm,
                patientid=patientid,
                inference_override=True
            )
        except Exception as e:
            num_failure += 1
            print(f"Error converting result for patient {patientid}: {e}")
            print(traceback.format_exc())
            print("Original target string:")
            print(target_string)
            # Skip this result if conversion fails - will be filled up with copy forward later if needed 
            # (if all other predictions also fail, which is pretty improbable)
            continue

        if patientid not in all_results:
            all_results[patientid] = []
        all_results[patientid].append(reverse_converted)
    
    print(f"Number of failed conversions: {num_failure}")
    if wandb.run is not None:
        wandb.log({"nr_failed_conversions": num_failure})
    
    return all_results



def convert_to_correct_format(all_results, empty_target_df):
    
    all_results_df = [x[0][0]["result"] for x in all_results.values()]
    all_results_df = pd.concat(all_results_df, ignore_index=True)
    all_results_df = all_results_df.drop(columns=["target_name"], errors='ignore')

    empty_target_df = empty_target_df.copy()
    empty_target_df = empty_target_df.drop(columns=["censoring", "occurred", "target_name"], errors='ignore')
    
    predictions = empty_target_df.merge(all_results_df, on=["patientid"], how="left")
    predictions = predictions.rename(columns={"censoring": "censored"})

    # Hacky way to get weeks
    predictions["week_to_predict"] = predictions['patientid'].str.split('_week_').str[1].astype(int)
    # Due to progression edge case, get sampled_category from patientid
    predictions["sampled_category"] = predictions["patientid"].str.split('_var_').str[1].str.split('_week_').str[0]
    
    return predictions


def fill_in_missing_values(predictions_formatted):
    predictions_formatted = predictions_formatted.copy()

    print(predictions_formatted['occurred'].isna().sum(), "missing values in 'occurred' column before filling")

    majority_class = predictions_formatted['occurred'].mode()[0]  # Get the most common value
    predictions_formatted['occurred'] = predictions_formatted['occurred'].fillna(majority_class)

    return predictions_formatted



def aggregate_responses(all_results):

    #: aggregate per patientid, given the most common censored and occurred results
    ret_results = {}
    
    for patientid, values in all_results.items():
        all_censored_results = [x[0]["result"]["censoring"].iloc[0] for x in values]
        all_occurred_results = [x[0]["result"]["occurred"].iloc[0] for x in values]

        most_common_censored = pd.Series(all_censored_results).mode().iloc[0]
        most_common_occurred = pd.Series(all_occurred_results).mode().iloc[0]

        ret_df = values[0][0]["result"].copy()
        ret_df["censoring"] = most_common_censored
        ret_df["occurred"] = most_common_occurred

        ret_results[patientid] = [[{
            "patientid": patientid,
            "result": ret_df,
        }]]

    return ret_results


