import numpy as np
import pandas as pd
import os
import json
import wandb
import glob
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pickle
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
import traceback



CHECKSUM_PATH = "genie-dt-cgdb-eval-events/0_data/checksums/"

TRAIN_DATA_PATH = "genie-dt-cgdb-eval-events/0_data/split_and_dfs_train/train/"



class EventsEval:

    def __init__(self, indication, data_loading_path, split):
        self.indication = indication
        data_loading_path = data_loading_path if data_loading_path.endswith("/") else data_loading_path + "/"
        self.data_loading_path = data_loading_path
        self.split = split

        # Other things
        self.target_df = None
        self.meta_data = None
        self.variable_stats = None
        self.checksum_dict = {}

        self._load_data()
        self._load_checksums()
    

    def _load_data(self):

        #: loadd in all individual targets
        target_files = glob.glob(os.path.join(self.data_loading_path, f"{self.split}_{self.indication}", "*_targets.csv"))
        target_dfs = []
        for file in target_files:
            df = pd.read_csv(file)
            target_dfs.append(df)

        #: concatenate all targets into a single dataframe
        self.target_df = pd.concat(target_dfs, ignore_index=True)

        # sort by patientid, sampled_category, and week_to_predict
        self.target_df = self.target_df.sort_values(by=["patientid", "sampled_category", "week_to_predict"]).reset_index(drop=True)

        #: Load in training data for probability metrics
        self.training_data = {}
        train_targets = pd.read_csv(os.path.join(TRAIN_DATA_PATH, "targets_train_" + self.indication + ".csv"))
        all_variables = train_targets["sampled_category"].unique().tolist()
        for variable in all_variables:
            print(f"Processing variable: {variable}")
            
            # Filter the train_targets for the current variable
            variable_targets = train_targets[train_targets["sampled_category"] == variable].copy()

            # Drop duplicate generic_patientid
            train_variable_targets = variable_targets.drop_duplicates(subset=["generic_patientid"])

            # Prepare the event indicator and time to event
            # Flipping the censoring indicator to be true for event occurrence and false for censoring
            # This is the standard format required by the sksurv libraries
            # Note: in "true_censoring" we get explicitly true/false, whilst in "censored" we have NA for no censoring and a string reason for censoring
            event_indicator = ~train_variable_targets["true_censoring"].astype(bool) 
            time_to_event = train_variable_targets["true_time"]
            
            # Make to final training dataframe with columns "event" and "time"
            final_train_df = pd.DataFrame({
                "event": event_indicator,
                "time": time_to_event
            })
            self.training_data[variable] = final_train_df
        

    def _load_checksums(self):
        checksum_json = "checksum.json"
        # Load the checksum file
        if os.path.exists(CHECKSUM_PATH + checksum_json):
            with open(CHECKSUM_PATH + checksum_json, "r") as f:
                self.checksum_dict = json.load(f)
            
            if self.data_loading_path in self.checksum_dict:
                if self.split in self.checksum_dict[self.data_loading_path]:
                    if self.indication not in self.checksum_dict[self.data_loading_path][self.split]:
                        # Start empty one
                        self.checksum_dict[self.data_loading_path][self.split][self.indication] = {}
                else:
                    # Start empty one
                    self.checksum_dict[self.data_loading_path][self.split] = {self.indication : {}}
            else:
                # Start empty one
                self.checksum_dict[self.data_loading_path] = {self.split : {self.indication : {}}}

        else:
            # Start empty one
            self.checksum_dict = {self.data_loading_path : {self.split : {self.indication : {}}}}


    def _save_checksums(self):
        # Save the checksum file
        with open(CHECKSUM_PATH + "checksum.json", "w") as f:
            json.dump(self.checksum_dict, f, indent=4)
        print("Saved checksum file")

    

    def _get_probability_metrics(self, predictions_subset, target_subset, variable, week):
        """
        Calculates time-dependent probability metrics for survival predictions at a landmark time.

        This function uses the scikit-survival library to compute metrics that account for censoring.
        - Brier Score: Measures the accuracy of probabilistic predictions.
        - C-index (IPCW): Measures overall discriminative power, adjusted for censoring.

        Args:
            predictions_subset (pd.DataFrame): DataFrame with model predictions.
                                            Must contain 'probability_occurrence'.
            target_subset (pd.DataFrame): DataFrame with ground truth labels.
                                        Must contain 'true_time' and 'true_censoring'.
            variable (str): The name of the event type being evaluated.
            week (int): The landmark week at which predictions are being evaluated.

        Returns:
            dict: A dictionary containing the calculated 'brier_score', and 'c_index'.
                Values will be np.nan if a metric cannot be computed.
        """
        # --- 1. Handle edge cases ---
        # Return NaN if there's no data to evaluate
        if target_subset.empty or predictions_subset.empty:
            return {"brier_score": np.nan, "c_index": np.nan}

        # --- 2. Prepare data in scikit-survival's expected format ---
        # Training data for estimating censoring distribution (IPCW)
        train_df = self.training_data[variable]
        survival_train = np.array(
            list(zip(train_df["event"], train_df["time"])),
            dtype=[('event', bool), ('time', float)]
        )


        # ---- Deal with edge case where test times exceed training times ----
        # Find the maximum time from the training data used for the censoring model
        max_train_time = train_df["time"].max()
        max_test_time = target_subset["true_time"].max()
        if max_test_time > max_train_time:

            # Filter the test set to only include times within the training data's range.
            # This prevents the "censoring survival function is zero" error.
            # Following best practises from: https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
            # Since this seems to be an open issue with the sksurv library: https://github.com/sebp/scikit-survival/issues/511
            mask = target_subset["true_time"] <= max_train_time
            target_subset = target_subset[mask]
            predictions_subset = predictions_subset[mask]

            print("=============================================================================")
            print(f"Warning: Test data for variable {variable} at week {week} contains times beyond the training data's maximum time.")
            print(f"Max training time: {max_train_time}, Max test time: {max_test_time}")
            print("Filtering test data to only include times within the training range to avoid errors in metric calculations.")
            print(f"Number of samples after filtering: {len(target_subset)}")
            print(f"Number of samples removed: {len(mask) - np.sum(mask)}")
            print("=============================================================================")

            # If filtering removed all data, return NaNs
            if target_subset.empty:
                return {"brier_score": np.nan, "c_index": np.nan}

        # Test data from the current evaluation subset (true for eent occurrence and false for censoring)
        event_indicator_test = ~target_subset["true_censoring"].astype(bool)
        time_to_event_test = target_subset["true_time"]
        survival_test = np.array(
            list(zip(event_indicator_test, time_to_event_test)),
            dtype=[('event', bool), ('time', float)]
        )

        # Predicted risk scores (higher probability of occurrence = higher risk)
        risk_scores = predictions_subset["probability_occurrence"].values

        # Predicted survival probabilities at the landmark time (higher means lower risk)
        survival_probs =  predictions_subset["probability_no_occurrence"].values
        
        # Set time point
        timepoint = week

        # --- 3. Calculate metrics with error handling ---
        # Note, this isn't very efficient, but it works for our scale.
        
        # Brier Score (requires survival probabilities)
        try:
            # Reshape survival_probs to (n_samples, 1) for a single evaluation time
            _, brier_scores = brier_score(
                survival_train=survival_train,
                survival_test=survival_test,
                estimate=survival_probs.reshape(-1, 1),
                times=[timepoint]
            )
            brier = brier_scores[0]
        except (ValueError, IndexError) as e:
            # ValueError may occur with insufficient data; IndexError if no subjects are at risk at 'week'
            print("=============================================================================")
            print(f"Warning: Brier score calculation failed for variable {variable} at week {week}: {e}")
            traceback.print_exc()
            brier = np.nan

        # Concordance Index (IPCW)
        try:
            # tau=timepoint limits the evaluation of pairs to the landmark time
            c_index, _, _, _, _ = concordance_index_ipcw(
                survival_train=survival_train,
                survival_test=survival_test,
                estimate=risk_scores,
                tau=timepoint
            )
        except (ValueError, ZeroDivisionError) as e:
            print("=============================================================================")
            print(f"Warning: C-index calculation failed for variable {variable} at week {week}: {e}")
            traceback.print_exc()
            c_index = np.nan
            
        # --- 4. Return results ---
        metrics = {
            "brier_score": brier,
            "c_index": c_index
        }
        return metrics


    def _get_metrics(self, predictions_subset, target_subset, variable, week):
        """
        Calculates metrics
        """
        
        #: get probability_metrics
        probability_metrics = self._get_probability_metrics(predictions_subset, target_subset, variable, week)

        return probability_metrics



    def evaluate(self, predictions, log=True):
        
        predictions = predictions.copy()

        #: do assertions of correct format of prediction
        assert all([x in predictions.columns for x in ["patientid", "sampled_category", "week_to_predict", 
                                                       "censored", "occurred", "probability_occurrence", "probability_no_occurrence"]]), "Predictions do not match target columns"
        assert set(predictions["patientid"]) == set(self.target_df["patientid"]), "Predictions not equal to targets in patientid"
        assert self.target_df.shape[0] == predictions.shape[0], "Predictions and targets do not have the same number of rows"
        #: assert no missing values in prediction
        assert predictions["occurred"].isna().sum() == 0, "Predictions contain NaN values in 'occurred' column"
        assert self.target_df["occurred"].isna().sum() == 0, "Targets contain NaN values in 'occurred' column"
        assert predictions["probability_occurrence"].isna().sum() == 0, "Predictions probability have NaNs in probability_occurrence column"

        # Sort by patientid
        predictions = predictions.sort_values(by=["patientid", "sampled_category", "week_to_predict"]).reset_index(drop=True)

        #: load and do checksums on target
        self._load_checksums()

        number_of_patients = self.target_df["patientid"].nunique()
        number_of_event_types = self.target_df["sampled_category"].nunique()
        number_of_weeks = self.target_df["week_to_predict"].nunique()
        list_of_patientids = self.target_df["patientid"].unique()
        list_of_event_types = self.target_df["sampled_category"].unique()
        list_of_occurred_values = self.target_df["occurred"].tolist()
        list_of_weeks = self.target_df["week_to_predict"].unique()

        if self.checksum_dict[self.data_loading_path][self.split][self.indication] == {}:
            # If empty, make checksum
            self.checksum_dict[self.data_loading_path][self.split][self.indication] = {
                "number_of_patients": number_of_patients,
                "number_of_event_types": number_of_event_types,
                "number_of_weeks": number_of_weeks,
                "list_of_patientids": list_of_patientids.tolist(),
                "list_of_event_types": list_of_event_types.tolist(),
                "list_of_occurred_values": list_of_occurred_values,
            }
            self._save_checksums()
        else:
            # Assert that the checksums match
            assert self.checksum_dict[self.data_loading_path][self.split][self.indication]["number_of_patients"] == number_of_patients, "Number of patients does not match"
            assert self.checksum_dict[self.data_loading_path][self.split][self.indication]["number_of_event_types"] == number_of_event_types, "Number of event types does not match"
            assert self.checksum_dict[self.data_loading_path][self.split][self.indication]["number_of_weeks"] == number_of_weeks, "Number of weeks does not match"
            assert np.array_equal(self.checksum_dict[self.data_loading_path][self.split][self.indication]["list_of_patientids"], list_of_patientids), "List of patientids does not match"
            assert np.array_equal(self.checksum_dict[self.data_loading_path][self.split][self.indication]["list_of_event_types"], list_of_event_types), "List of event types does not match"
            assert np.array_equal(self.checksum_dict[self.data_loading_path][self.split][self.indication]["list_of_occurred_values"], list_of_occurred_values), "List of occurred values does not match"

        #: evaluate for each variable and time point individually
        all_results = {}

        for event_type in list_of_event_types:
            
            all_results[event_type] = {}

            for week in list_of_weeks:
                # Filter the target dataframe for the current event type and week
                target_subset_all = self.target_df[(self.target_df["sampled_category"] == event_type) & (self.target_df["week_to_predict"] == week)]
                
                # Filter the predictions dataframe for the current event type and week
                predictions_subset_all = predictions[(predictions["sampled_category"] == event_type) & (predictions["week_to_predict"] == week)]
                
                #: calculate metrics
                metrics = self._get_metrics(predictions_subset_all, target_subset_all, variable=event_type, week=week)

                # Log to W&B
                if wandb.run is not None and log:
                    wandb.log({
                        f"{event_type}_week_{week}_brier_score": metrics["brier_score"],
                        f"{event_type}_week_{week}_c_index": metrics["c_index"],
                    })
                
                # Store the results in the all_results dictionary
                all_results[event_type][week] = metrics

        #: save predictions and targets DF to W&B
        if wandb.run is not None and log:
            predictions_df = predictions.copy()
            targets_df = self.target_df.copy()
            predictions_df_cleaned = predictions_df.fillna(value='')
            targets_df_cleaned = targets_df.fillna(value='')
            wandb_table_predictions = wandb.Table(dataframe=predictions_df_cleaned)
            wandb_table_targets = wandb.Table(dataframe=targets_df_cleaned)
            wandb.log({
                "predictions": wandb_table_predictions,
                "targets": wandb_table_targets
            })

        return all_results

        







