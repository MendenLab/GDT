import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import wandb


CHECKSUM_PATH = "genie-dt-cit-baselines-forecasting/0_data/checksums/"
TRAIN_DATA_STATS = "genie-dt-cit-baselines-forecasting/0_data/variable_stats/variable_stats.csv"





def apply_last_observed_value_to_target(input_df, empty_target_df):

    #: for each input, get last observed value, grouped by patientid and event_name, sorted by date
    last_observed_value = input_df.groupby('patientid').apply(
        lambda x: x.sort_values('date').iloc[-1]['event_value']
    ).reset_index()
    last_observed_value = last_observed_value.rename(columns={0: 'last_observed_value'}).copy()
    
    #: merge with empty_target_df
    prediction = pd.merge(
        empty_target_df,
        last_observed_value,
        on='patientid',
        how='left'
    )

    #: fill in with last observed value
    prediction["event_value"] = prediction["last_observed_value"]

    #: return
    return prediction






class ForecastingEval:

    def __init__(self, indication, data_loading_path, split, specific_trial=None):
        
        self.indication = indication
        self.data_loading_path = data_loading_path
        self.split = split
        self.specific_trial = specific_trial

        self.constant_df = None
        self.input_df = None
        self.target_df = None
        self.meta_data = None

        self.variable_stats = None

        self.checksum_dict = {}

        self._load_data()
        self._load_checksums()

    
    def _load_data(self):
        constant_df_path = self.data_loading_path + f"constant_df_{self.split}.csv"
        input_df_path = self.data_loading_path + f"input_df_{self.split}.csv"
        target_df_path = self.data_loading_path + f"target_df_{self.split}.csv"
        meta_data_path = self.data_loading_path + f"meta_data_{self.split}.csv"

        self.constant_df = pd.read_csv(constant_df_path)
        self.input_df = pd.read_csv(input_df_path)
        self.target_df = pd.read_csv(target_df_path)
        self.meta_data = pd.read_csv(meta_data_path)

        # Extract only non imputed rows from target
        self.target_df = self.target_df[self.target_df["imputed"] == 0].copy()
        self.target_df = self.target_df.sort_values(by=["patientid", "event_name", "date"])
        self.target_df = self.target_df.reset_index(drop=True)

        # Convert date to datetime
        self.target_df["date"] = pd.to_datetime(self.target_df["date"])
        self.input_df["date"] = pd.to_datetime(self.input_df["date"])

        #: extract the correct trial
        if self.specific_trial is not None:
            self.target_df["trial_id"] = self.target_df["patientid"].apply(lambda x: x.split("_")[0])  # Extract trial_id from patientid
            self.input_df["trial_id"] = self.input_df["patientid"].apply(lambda x: x.split("_")[0])  # Extract trial_id from patientid

            self.target_df = self.target_df[self.target_df["trial_id"] == self.specific_trial].copy()
            self.input_df = self.input_df[self.input_df["trial_id"] == self.specific_trial].copy()
            print(f"Loaded data ONLY from the specified trial: {self.specific_trial}")

        # Load the variable stats
        self.variable_stats = pd.read_csv(TRAIN_DATA_STATS)
    
    def _load_checksums(self):

        checksum_json = "checksum.json"
        checksum_load_path = self.data_loading_path if not self.specific_trial else self.data_loading_path + f"trial_{self.specific_trial}/"

        # Load the checksum file
        if os.path.exists(CHECKSUM_PATH + checksum_json):
            with open(CHECKSUM_PATH + checksum_json, "r") as f:
                self.checksum_dict = json.load(f)
            
            if checksum_load_path in self.checksum_dict:
                if self.split in self.checksum_dict[checksum_load_path]:
                    if self.indication not in self.checksum_dict[checksum_load_path][self.split]:
                        # Start empty one
                        self.checksum_dict[checksum_load_path][self.split][self.indication] = {}
                else:
                    # Start empty one
                    self.checksum_dict[checksum_load_path][self.split] = {self.indication : {}}
            else:
                # Start empty one
                self.checksum_dict[checksum_load_path] = {self.split : {self.indication : {}}}

        else:
            # Start empty one
            self.checksum_dict = {checksum_load_path : {self.split : {self.indication : {}}}}


    def _save_checksums(self):
        # Save the checksum file
        with open(CHECKSUM_PATH + "checksum.json", "w") as f:
            json.dump(self.checksum_dict, f, indent=4)
        print("Saved checksum file")

    def _get_copy_forward_results(self, target_df):
        
        #: get corresponding input DF
        input_df = self.input_df[self.input_df["event_name"].isin(target_df["event_name"])].copy()
        empty_target_df = target_df.copy()
        empty_target_df["event_value"] = pd.NA
        assert empty_target_df["event_value"].isna().all()

        #: get copy forward prediction
        copy_forward_prediction = apply_last_observed_value_to_target(input_df, empty_target_df)

        # Apply 3 sigma capping to the copy forward prediction
        copy_forward_prediction_capped = self._apply_3_sigma_capping(copy_forward_prediction, None, verbose=False)

        #: get metrics
        merged = pd.merge(target_df, copy_forward_prediction_capped, on=["patientid", "date", "event_name"], suffixes=("_target", "_predicted"))
        copy_forward_metrics = self._get_metrics(merged)

        #: return
        return copy_forward_metrics


    def _get_metrics(self, merged_df):
        mae = mean_absolute_error(merged_df["event_value_target"], merged_df["event_value_predicted"])
        mape = mean_absolute_percentage_error(merged_df["event_value_target"], merged_df["event_value_predicted"])

        results = {
            "mae": mae,
            "mape": mape
        }
        return results

    def _eval_metrics(self, predicted_targets, target_df, event_name, trial_id=None):

        # If trial id is not None, filter the target_df and predicted_targets by trial_id
        if trial_id is not None:
            target_df = target_df[target_df["trial_id"] == trial_id].copy()
            predicted_targets = predicted_targets[predicted_targets["patientid"].isin(target_df["patientid"])].copy()

        if event_name is None:
            # If event_name is None, we want to calculate the metrics for all events
            target_variable = target_df.copy()
            predicted_variable = predicted_targets.copy()
        else:
            # Get the target variable
            target_variable = target_df[target_df["event_name"] == event_name].copy()
            # Get the predicted variable
            predicted_variable = predicted_targets[predicted_targets["event_name"] == event_name].copy()

        # Check that length is same before and after
        num_rows_target = target_variable.shape[0]
        num_rows_predicted = predicted_variable.shape[0]
        num_patients_target = target_variable["patientid"].nunique()
        num_patients_predicted = predicted_variable["patientid"].nunique()

        # Merge the two dataframes on patientid and date
        merged_df = pd.merge(target_variable, predicted_variable, on=["patientid", "date"], suffixes=("_target", "_predicted"))

        # Check that nothing fishy going
        assert merged_df.shape[0] == num_rows_target, f"Number of rows in merged_df ({merged_df.shape[0]}) does not match the number of rows in target_variable ({num_rows_target})"
        assert merged_df.shape[0] == num_rows_predicted, f"Number of rows in merged_df ({merged_df.shape[0]}) does not match the number of rows in predicted_variable ({num_rows_predicted})"
        assert merged_df["patientid"].nunique() == num_patients_target, f"Number of patients in merged_df ({merged_df['patientid'].nunique()}) does not match the number of patients in target_variable ({num_patients_target})"
        assert merged_df["patientid"].nunique() == num_patients_predicted, f"Number of patients in merged_df ({merged_df['patientid'].nunique()}) does not match the number of patients in predicted_variable ({num_patients_predicted})"
        assert all(merged_df["imputed_target"] == 0), f"Not all rows in merged_df are original, {merged_df[merged_df['imputed_target'] != 0]}"

        # Calculate the metrics using scikit-learn
        results = self._get_metrics(merged_df)

        # Get also the results normalized by copy forward
        copy_forward_results = self._get_copy_forward_results(target_variable)
        
        # check each for ZeroDivisionError - put np.nan if it happens
        for key in copy_forward_results.keys():
            if copy_forward_results[key] == 0:
                copy_forward_results[key] = np.nan
            # Normalize the results by copy forward
            results[key + "_copy_forward_normalized"] = results[key] / copy_forward_results[key]
        
        return results

    def _get_unique_raw_patientids(self, patientids_with_variable_name):
        # Get the raw patientids from the patientids_with_variable_name
        # This is done by removing the variable name from the patientid
        # For example, if the patientid is "patientid_1", we want to get "1"
        raw_patientids = []
        for patientid in patientids_with_variable_name:
            raw_patientids.append(patientid.split("_split_")[0])
        raw_patientids = list(set(raw_patientids))
        return raw_patientids

    def _get_unique_patient_samples(self, patientids_with_variable_name):
        # Get the unique patient samples from the patientids_with_variable_name
        # This is done by removing the variable name from the patientid
        # For example, if the patientid is "patientid_1", we want to get "1"
        raw_patientids = []
        for patientid in patientids_with_variable_name:
            raw_patientids.append(patientid.split("_var_")[0])
        raw_patientids = list(set(raw_patientids))
        return raw_patientids
    
    def _apply_3_sigma_capping(self, predictions, target_df=None, verbose=True):

        # Store original event_values for assertions
        predictions = predictions.copy()
        original_predictions_event_value = predictions["event_value"].copy()

        # Get all the unique event names
        event_names = predictions["event_name"].unique()

        # For efficiency, extract those from the data stats
        data_stats = self.variable_stats[self.variable_stats["event_name"].isin(event_names)].copy()
        data_stats = data_stats[["event_name", "mean_without_outliers", "std_without_outliers"]].copy() 
        data_stats["upper_limit"] = data_stats["mean_without_outliers"] + 3 * data_stats["std_without_outliers"]
        data_stats["lower_limit"] = data_stats["mean_without_outliers"] - 3 * data_stats["std_without_outliers"]
        
        data_stats_limits = data_stats[["event_name", "upper_limit", "lower_limit"]].copy() # Renamed for clarity

        # Merge the data stats with the predictions
        predictions_merged = predictions.merge(data_stats_limits, on="event_name", how="left")

        # Apply the capping
        predictions_over_limit = predictions_merged["event_value"] > predictions_merged["upper_limit"]
        predictions_under_limit = predictions_merged["event_value"] < predictions_merged["lower_limit"]
        
        # Store the limits for assertion
        predictions_upper_limits_for_capping = predictions_merged["upper_limit"].copy()
        predictions_lower_limits_for_capping = predictions_merged["lower_limit"].copy()

        predictions_merged["event_value"] = np.where(predictions_over_limit, predictions_merged["upper_limit"], predictions_merged["event_value"])
        predictions_merged["event_value"] = np.where(predictions_under_limit, predictions_merged["lower_limit"], predictions_merged["event_value"])

        # Assign capped values back to original dataframes (or use the merged ones)
        predictions["event_value"] = predictions_merged["event_value"]
        
        # print and log how many values were capped
        if verbose:
            print(f"Number of values capped in predictions: {predictions_over_limit.sum() + predictions_under_limit.sum()}")
            print(f"Number of too low values in predictions: {predictions_under_limit.sum()}")
            print(f"Number of too high values in predictions: {predictions_over_limit.sum()}")

        if wandb.run is not None:
            wandb.log({
                "number_of_values_capped_in_predictions": predictions_over_limit.sum() + predictions_under_limit.sum(),
                "number_of_too_low_values_in_predictions": predictions_under_limit.sum(),
                "number_of_too_high_values_in_predictions": predictions_over_limit.sum(),
            })


        # Now also apply the capping to the target_df
        if target_df is not None:
            original_target_df_event_value = target_df["event_value"].copy()
            target_df_merged = target_df.merge(data_stats_limits, on="event_name", how="left")
            targets_over_limit = target_df_merged["event_value"] > target_df_merged["upper_limit"]
            targets_under_limit = target_df_merged["event_value"] < target_df_merged["lower_limit"]

            # Store the limits for assertion
            target_upper_limits_for_capping = target_df_merged["upper_limit"].copy()
            target_lower_limits_for_capping = target_df_merged["lower_limit"].copy()

            target_df_merged["event_value"] = np.where(targets_over_limit, target_df_merged["upper_limit"], target_df_merged["event_value"])
            target_df_merged["event_value"] = np.where(targets_under_limit, target_df_merged["lower_limit"], target_df_merged["event_value"])

            # Assign capped values back to original dataframes (or use the merged ones)
            target_df["event_value"] = target_df_merged["event_value"]

            if verbose:
                print(f"Number of values capped in target_df: {targets_over_limit.sum() + targets_under_limit.sum()}")
                print(f"Number of too low values in target_df: {targets_under_limit.sum()}")
                print(f"Number of too high values in target_df: {targets_over_limit.sum()}")

            if wandb.run is not None:
                wandb.log({
                    "number_of_values_capped_in_target_df": targets_over_limit.sum() + targets_under_limit.sum(),
                    "number_of_too_low_values_in_target_df": targets_under_limit.sum(),
                    "number_of_too_high_values_in_target_df": targets_over_limit.sum(),
                })

        
        # Assertions
        # For predictions
        # 1. Values not capped should remain unchanged
        not_capped_predictions_mask = ~(predictions_over_limit | predictions_under_limit)
        assert predictions.loc[not_capped_predictions_mask, "event_value"].equals(
            original_predictions_event_value[not_capped_predictions_mask]
        ), "Assertion failed: Some non-capped prediction values were changed."

        # 2. Values capped high should be equal to their upper_limit
        # Need to be careful with floating point comparisons, using np.isclose is safer
        if predictions_over_limit.any(): # Only assert if there were values capped high
            assert np.isclose(
                predictions.loc[predictions_over_limit, "event_value"],
                predictions_upper_limits_for_capping[predictions_over_limit]
            ).all(), "Assertion failed: Some high-capped prediction values are not equal to their upper limit."


        # 3. Values capped low should be equal to their lower_limit
        if predictions_under_limit.any(): # Only assert if there were values capped low
            assert np.isclose(
                predictions.loc[predictions_under_limit, "event_value"],
                predictions_lower_limits_for_capping[predictions_under_limit]
            ).all(), "Assertion failed: Some low-capped prediction values are not equal to their lower limit."

        # For target_df
        if target_df is not None:
            # 1. Values not capped should remain unchanged
            not_capped_targets_mask = ~(targets_over_limit | targets_under_limit)
            assert (target_df.loc[not_capped_targets_mask, "event_value"] == original_target_df_event_value[not_capped_targets_mask]).all(), \
                "Assertion failed: Some non-capped target_df values were changed."

            # 2. Values capped high should be equal to their upper_limit
            if targets_over_limit.any():
                assert np.isclose(
                    target_df.loc[targets_over_limit, "event_value"],
                    target_upper_limits_for_capping[targets_over_limit]
                ).all(), "Assertion failed: Some high-capped target_df values are not equal to their upper limit."

            # 3. Values capped low should be equal to their lower_limit
            if targets_under_limit.any():
                assert np.isclose(
                    target_df.loc[targets_under_limit, "event_value"],
                    target_lower_limits_for_capping[targets_under_limit]
                ).all(), "Assertion failed: Some low-capped target_df values are not equal to their lower limit."
            
            # Return the predictions and target_df with capped values
            return predictions, target_df
        else:
            # Return only the predictions with capped values
            return predictions


    def evaluate(self, predicted_targets):

        # Do very basic assertions
        assert set(predicted_targets["patientid"].unique()) == set(self.target_df["patientid"].unique()), "predicted_targets and target_df should have the same set of patientids"
        assert set(predicted_targets["event_name"].unique()) == set(self.target_df["event_name"].unique()), "predicted_targets and target_df should have the same set of event_names"

        # Extract only the non-imputed rows
        non_imputed_rows = self.target_df[["patientid", "date", "event_name"]].copy()
        predicted_targets = predicted_targets.merge(non_imputed_rows, on=["patientid", "date", "event_name"], how="inner")

        # Add empty imputed column if needed
        if "imputed" not in predicted_targets.columns:
            predicted_targets["imputed"] = pd.NaT

        # Do basic assertions
        assert isinstance(predicted_targets, pd.DataFrame), "predicted_targets should be a pandas DataFrame"
        assert predicted_targets.shape[0] == self.target_df.shape[0], "predicted_targets should have the same number of rows as target_df"
        required_cols = ["date", "event_value", "patientid", "event_name"]
        assert set(required_cols).issubset(predicted_targets.columns), f"predicted_targets should have the columns {required_cols}"
        assert set(required_cols).issubset(self.target_df.columns), f"target_df should have the columns {required_cols}"
        assert set(predicted_targets["date"].unique()) == set(self.target_df["date"].unique()), "predicted_targets and target_df should have the same set of dates"
        assert predicted_targets["event_value"].notna().all(), "predicted_targets should not have NaN values in event_value column"
        
        # do checksum checks (so that our target df is correct)
        number_of_patients = len(predicted_targets["patientid"].unique())
        number_of_events = len(predicted_targets["event_name"].unique())
        number_of_patients_target = len(self.target_df["patientid"].unique())
        number_of_events_target = len(self.target_df["event_name"].unique())
        number_of_dates_target = len(self.target_df["date"].unique())
        list_of_dates_target = self.target_df["date"].unique()
        list_of_dates_target = sorted(list_of_dates_target)
        list_of_patient_ids = self.target_df["patientid"].unique()
        list_of_patient_ids.sort()

        # Load the checksum file
        self._load_checksums()
        checksum_load_path = self.data_loading_path if not self.specific_trial else self.data_loading_path + f"trial_{self.specific_trial}/"

        if self.checksum_dict[checksum_load_path][self.split][self.indication] == {}:
            self.checksum_dict[checksum_load_path][self.split][self.indication] = {
                "number_of_patients": number_of_patients,
                "number_of_events": number_of_events,
                "number_of_patients_target": number_of_patients_target,
                "number_of_events_target": number_of_events_target,
                "number_of_dates_target": number_of_dates_target,
                "list_of_patient_ids": list_of_patient_ids.tolist(),
                "number_of_total_target_samples" : len(self.target_df),
                "number_of_unique_event_values" : len(self.target_df["event_value"].unique()),
            }
            self._save_checksums()
        else:
            assert self.checksum_dict[checksum_load_path][self.split][self.indication]["number_of_patients"] == number_of_patients, f"Number of patients in predicted_targets ({number_of_patients}) does not match the number of patients in target_df ({self.checksum_dict[self.data_loading_path][self.split][self.indication]['number_of_patients']})"
            assert self.checksum_dict[checksum_load_path][self.split][self.indication]["number_of_events"] == number_of_events, f"Number of events in predicted_targets ({number_of_events}) does not match the number of events in target_df ({self.checksum_dict[self.data_loading_path][self.split][self.indication]['number_of_events']})"
            assert self.checksum_dict[checksum_load_path][self.split][self.indication]["number_of_patients_target"] == number_of_patients_target, f"Number of patients in target_df ({number_of_patients_target}) does not match the number of patients in target_df ({self.checksum_dict[self.data_loading_path][self.split][self.indication]['number_of_patients_target']})"
            assert self.checksum_dict[checksum_load_path][self.split][self.indication]["number_of_events_target"] == number_of_events_target, f"Number of events in target_df ({number_of_events_target}) does not match the number of events in target_df ({self.checksum_dict[self.data_loading_path][self.split][self.indication]['number_of_events_target']})"
            assert self.checksum_dict[checksum_load_path][self.split][self.indication]["number_of_dates_target"] == number_of_dates_target, f"Number of dates in target_df ({number_of_dates_target}) does not match the number of dates in target_df ({self.checksum_dict[self.data_loading_path][self.split][self.indication]['number_of_dates_target']})"
            assert np.array_equal(self.checksum_dict[checksum_load_path][self.split][self.indication]["list_of_patient_ids"], list_of_patient_ids), f"List of patient ids in target_df does not match the list of patient ids in target_df"
            assert self.checksum_dict[checksum_load_path][self.split][self.indication]["number_of_total_target_samples"] == len(self.target_df), f"Number of total target samples in target_df ({len(self.target_df)}) does not match the number of total target samples in target_df ({self.checksum_dict[self.data_loading_path][self.split][self.indication]['number_of_total_target_samples']})"
            assert self.checksum_dict[checksum_load_path][self.split][self.indication]["number_of_unique_event_values"] == len(self.target_df["event_value"].unique()), f"Number of unique event values in target_df ({len(self.target_df['event_value'].unique())}) does not match the number of unique event values in target_df ({self.checksum_dict[self.data_loading_path][self.split][self.indication]['number_of_unique_event_values']})"

        print("All checks passed")

        # Post process for 3 sigma
        predicted_targets, target_df_capped = self._apply_3_sigma_capping(predicted_targets, self.target_df.copy())

        # Sort the predicted and targets by patientid, event_name and date
        predicted_targets = predicted_targets.sort_values(by=["patientid", "event_name", "date"])
        # Reset the index
        predicted_targets = predicted_targets.reset_index(drop=True)

        # Get all unique target variables from meta and iterate over all, as well as calculate global metrics
        final_results = {}
        all_unique_targets = self.meta_data["target_variable"].unique()
        target_df_capped["trial_id"] = target_df_capped["patientid"].apply(lambda x: x.split("_")[0])  # Extract trial_id from patientid
        all_trials = target_df_capped["trial_id"].unique()

        for target_variable in all_unique_targets:
            final_results[target_variable] = {}
            for trial_id in all_trials:
                final_results[target_variable][trial_id] = {}
                final_results[target_variable][trial_id]["metrics"] = self._eval_metrics(predicted_targets, target_df_capped, target_variable, trial_id=trial_id)
                final_results[target_variable][trial_id]["trial_id"] = trial_id
                final_results[target_variable][trial_id]["target_variable"] = target_variable
                final_results[target_variable][trial_id]["target_variable_name"] = target_df_capped[target_df_capped["event_name"] == target_variable]["event_descriptive_name"].dropna().values[0]
                final_results[target_variable][trial_id]["num_samples"] = len(target_df_capped[(target_df_capped["trial_id"] == trial_id) & (target_df_capped["event_name"] == target_variable)])
                final_results[target_variable][trial_id]["num_patient_splits"] = len(self._get_unique_patient_samples(target_df_capped[(target_df_capped["trial_id"] == trial_id) & (target_df_capped["event_name"] == target_variable)]["patientid"].unique()))
                final_results[target_variable][trial_id]["num_patients"] = len(self._get_unique_raw_patientids(target_df_capped[(target_df_capped["trial_id"] == trial_id) & (target_df_capped["event_name"] == target_variable)]["patientid"].unique()))
                final_results[target_variable][trial_id]["num_unique_dates"] = len(target_df_capped[(target_df_capped["trial_id"] == trial_id) & (target_df_capped["event_name"] == target_variable)]["date"].unique())


        # Log to wandb if run is happening
        if wandb.run is not None:
           
            # Then log all the results as a dictionary (this will include the new global means)
            wandb.log({"all_results": final_results})

            # Save also the tables as artifacts
            wandb_table_predictions = wandb.Table(dataframe=predicted_targets)
            wandb_table_target = wandb.Table(dataframe=target_df_capped)
            wandb.log({"df_predictions": wandb_table_predictions})
            wandb.log({"df_target": wandb_table_target})

        return final_results











        

        



    









