import os
import sys
import pandas as pd
from io import StringIO
import datetime
import traceback
import wandb


from digital_twin_converter.instruction.data_splitter_forecasting import DataSplitterForecasting
from digital_twin_converter.common.data_manager import SingleIndicationDataManager
from digital_twin_converter.instruction.data_splitter_events import DataSplitterEvents
from digital_twin_converter.instruction.converter_manual_instruction import ConverterManualInstruction
from digital_twin_converter.common.config import Config




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

    project_root_dir = os.path.abspath(os.path.join(notebook_parent_dir, "../2_forecasting_eval_utils"))
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)



def setup_all_data_managers_and_converters(indication):

    config = Config()
    dm = SingleIndicationDataManager(indication, config=config)
    dm.load_indication_data()
    dm.process_indication_data()
    dm.setup_unique_mapping_of_events()
    dm.setup_dataset_splits()
    converter = ConverterManualInstruction(dm.data_frames["constant_description"],
                                            nr_tokens_budget_total=8192,
                                            config=config)

    return config, dm, converter



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


def process_raw_data_to_list(raw_df):
    ret = raw_df[["patientid", "instruction"]].values.tolist()
    return ret





def process_empty_targets_from_raw_data(raw_initial_data):

    all_empty_targets = []
    for i, row in raw_initial_data.iterrows():
        target_as_str = row["empty_target_as_string"]
        # parse from json str, orient="records"
        target_as_df = pd.read_json(StringIO(target_as_str), orient="records",
                                    dtype={'event_name': str})
        all_empty_targets.append(target_as_df)
    all_empty_targets_df = pd.concat(all_empty_targets, ignore_index=True)
    return all_empty_targets_df



def get_last_observed_values(raw_data):

    all_last_observed = []
    for i, row in raw_data.iterrows():
        last_observed_as_str = row["last_observed_values"]
        # parse from json str, orient="records"
        lo_as_df = pd.read_json(StringIO(last_observed_as_str), orient="records",
                                    dtype={'event_name': str})
        all_last_observed.append(lo_as_df)
    all_last_observed_df = pd.concat(all_last_observed, ignore_index=True)
    all_last_observed_df = all_last_observed_df[['patientid', 'event_name', 'event_descriptive_name', 'event_value']]
    return all_last_observed_df




def convert_all_results_back_to_df(all_raw_results, raw_data, converter, dm):

    split_dates = raw_data[["patientid", "split_date_included_in_input"]].copy()
    split_dates["split_date_included_in_input"] = pd.to_datetime(split_dates["split_date_included_in_input"])

    all_results = {}
    num_failure = 0

    for i, raw_result in enumerate(all_raw_results):

        patientid = raw_result[0]
        target_string = raw_result[1]
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




def save_individual_trajectories(all_results_converted_individually, folder, indication):
    
    all_trajectories = []

    for patientid in all_results_converted_individually.keys():
        
        for idx in range(len(all_results_converted_individually[patientid])):
            
            curr_trajectory_id = patientid + "_traj_" + str(idx)
            curr_trajectory = all_results_converted_individually[patientid][idx][0]["result"].copy()
            curr_trajectory["patientid"] = curr_trajectory_id

            all_trajectories.append(curr_trajectory)
    
    all_trajectories_df = pd.concat(all_trajectories, ignore_index=True)

    #: set the correct location, as folder + indicaiton + datetime as string
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    location = folder + indication + "_" + date_str + "_trajectories.csv"
    all_trajectories_df.to_csv(location, index=False)
    return location




def aggregate_results_on_patientid_level(all_results_converted_individually, converter):

    aggregated_results = []

    for patientid in all_results_converted_individually.keys():
        
        all_requests_reverse_converted = all_results_converted_individually[patientid]
        aggregated_reversed = converter.aggregate_multiple_responses(all_responses=all_requests_reverse_converted)

        aggregated_results.append(aggregated_reversed[0]["result"])

    aggregated_results_df = pd.concat(aggregated_results, axis=0, ignore_index=True)
    
    return aggregated_results_df




def fill_in_missing_values_with_copy_forward(results, last_observed_values, empty_target_df):

    prev_nr_of_results = results.shape[0]
    prev_nr_of_nans = results["event_value"].isna().sum()

    #: match results to empty_target_df
    empty_target_df = empty_target_df[["patientid", "event_name", "date"]].copy()
    results_with_all_target_dates = empty_target_df.merge(results, 
                                                          left_on=["patientid", "event_name", "date"], 
                                                          right_on=["patientid", "event_name", "date"],
                                                          how="left", suffixes=("_target", ""))
    results = results_with_all_target_dates.sort_values(by=["patientid", "date", "event_name"])
    all_last_observed_df = last_observed_values[['patientid', 'event_name', 'event_value']].copy()

    #: match last observd to results
    results = results.merge(all_last_observed_df, on=["patientid", "event_name"],
                            how="left", suffixes=("", "_last_observed"))

    #: for any missing values, forward fill, grouped by patientid and event_name
    results['event_value'] = results.groupby(['patientid', 'event_name'])['event_value'].ffill()

    #: for any still missing values, apply the last observed value
    results['event_value'] = results.apply(
        lambda row: row['event_value'] if pd.notna(row['event_value']) else row['event_value_last_observed'], axis=1)

    assert results["event_value"].notna().all(), "There are still missing values in the results!"
    nr_missing_values = (results.shape[0] - prev_nr_of_results) + prev_nr_of_nans

    return results, nr_missing_values



def convert_to_eval_format(filled_in_results):

    final_results = filled_in_results.copy()
    final_results["patientid"] = final_results["patientid"] + "_var_" + final_results["event_name"]

    return final_results
