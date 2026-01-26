import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.common import space
import os



TRAIN_DATA_STATS = "genie-dt-cit-baselines-forecasting/0_data/variable_stats/variable_stats.csv"



def combine_dfs_into_one_for_training(all_input_target_and_meta_dfs):

    all_input_dfs = []
    all_constant_dfs = []

    for input_df, target_df, constant_df, meta_data in all_input_target_and_meta_dfs:
        
        # Add both input and target dataframes to the list since training
        all_input_dfs.append(input_df)
        all_input_dfs.append(target_df)

        # Add constant
        all_constant_dfs.append(constant_df)

    train_df = pd.concat(all_input_dfs, ignore_index=True)
    constant_df_combined = pd.concat(all_constant_dfs, ignore_index=True)

    # Sort train by patientid, event_name and then date
    train_df = train_df.sort_values(by=['patientid', 'event_name', 'date']).reset_index(drop=True)

    return train_df, constant_df_combined



def apply_3_sigma_filtering_and_standardization(raw_input_events, indication, raw_targets=None, verbose=True, standardize=True, train_data_stats_path=TRAIN_DATA_STATS):

    #: load in the stats CSV
    variable_stats = pd.read_csv(train_data_stats_path)

    # Store original event_values for assertions
    raw_input_events = raw_input_events.copy()
    original_input_value = raw_input_events["event_value"].copy()

    # Get all the unique event names
    event_names = raw_input_events["event_name"].unique()

    # For efficiency, extract those from the data stats
    data_stats = variable_stats[variable_stats["event_name"].isin(event_names)].copy()
    data_stats = data_stats[["event_name", "mean", "std", "mean_without_outliers", "std_without_outliers"]].copy()
    data_stats["upper_limit"] = data_stats["mean"] + 3 * data_stats["std"]
    data_stats["lower_limit"] = data_stats["mean"] - 3 * data_stats["std"]
    data_stats_limits = data_stats[["event_name",  "mean_without_outliers", "std_without_outliers", "upper_limit", "lower_limit"]].copy() # Renamed for clarity

    # Merge the data stats with the predictions
    raw_input_events_merged = raw_input_events.merge(data_stats_limits, on="event_name", how="left")

    # Apply the capping
    input_over_limit = raw_input_events_merged["event_value"] > raw_input_events_merged["upper_limit"]
    input_under_limit = raw_input_events_merged["event_value"] < raw_input_events_merged["lower_limit"]

    raw_input_events_merged["event_value"] = np.where(input_over_limit, raw_input_events_merged["upper_limit"], raw_input_events_merged["event_value"])
    raw_input_events_merged["event_value"] = np.where(input_under_limit, raw_input_events_merged["lower_limit"], raw_input_events_merged["event_value"])

    # Apply standardization
    if standardize:
        # Note, mean_without_outliers is the identical mean on the train data after removing outliers with 3 sigma
        # so its correct to use here.
        raw_input_events_merged["event_value"] = (raw_input_events_merged["event_value"] - raw_input_events_merged["mean_without_outliers"]) / raw_input_events_merged["std_without_outliers"]  

    # Assign capped values back to original dataframes (or use the merged ones)
    raw_input_events["event_value"] = raw_input_events_merged["event_value"]
    
    # print and log how many values were capped
    if verbose:
        print(f"Number of values capped in input_df: {input_over_limit.sum() + input_under_limit.sum()}")
        print(f"Number of too low values in input_df: {input_under_limit.sum()}")
        print(f"Number of too high values in input_df: {input_over_limit.sum()}")


    # Now also apply the capping to the target_df
    if raw_targets is not None:
        target_df = raw_targets.copy()
        original_target_df_event_value = target_df["event_value"].copy()
        target_df_merged = target_df.merge(data_stats_limits, on="event_name", how="left")
        targets_over_limit = target_df_merged["event_value"] > target_df_merged["upper_limit"]
        targets_under_limit = target_df_merged["event_value"] < target_df_merged["lower_limit"]

        target_df_merged["event_value"] = np.where(targets_over_limit, target_df_merged["upper_limit"], target_df_merged["event_value"])
        target_df_merged["event_value"] = np.where(targets_under_limit, target_df_merged["lower_limit"], target_df_merged["event_value"])

        # Apply standardization
        if standardize:
            target_df_merged["event_value"] = (target_df_merged["event_value"] - target_df_merged["mean_without_outliers"]) / target_df_merged["std_without_outliers"]

        # Assign capped values back to original dataframes (or use the merged ones)
        target_df["event_value"] = target_df_merged["event_value"]

        if verbose:
            print(f"Number of values capped in target_df: {targets_over_limit.sum() + targets_under_limit.sum()}")
            print(f"Number of too low values in target_df: {targets_under_limit.sum()}")
            print(f"Number of too high values in target_df: {targets_over_limit.sum()}")


    # returns
    if raw_targets is not None:
        return raw_input_events, target_df
    else:
        return raw_input_events






def finetune_chronos_model(chronos_model, train_ag_data, prediction_length, log_path,
                           time_limit=3600, max_nr_fine_tune_steps=10_000,
                           eval_steps=1000, report_to_wandb=True, 
                           lr_for_single_run=1e-5,
                           fine_tune_batch_size=32, save_steps=1000,
                           val_ag_data=None):

    

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq="W-MON",
        path=log_path,  # This is the base path for the predictor
        target="target",
        eval_metric="MAE",
        verbosity=4,  # For debugging
    )

    model_path_to_return = None

    
    model_hyperparameters = {
        "Chronos": {
            "model_path": chronos_model,
            "fine_tune": True,
            "ag_args": {"name_suffix": "FineTuned"},
            "target_scaler": None,
            "fine_tune_batch_size": fine_tune_batch_size,
            "fine_tune_steps": max_nr_fine_tune_steps,
            "fine_tune_lr": lr_for_single_run,
            "eval_during_fine_tune": True,
            "fine_tune_trainer_kwargs": {
                "evaluation_strategy": "steps",
                "eval_steps": eval_steps,
                "save_strategy": "steps",
                "save_steps": save_steps,
                "load_best_model_at_end": True,
                "report_to": "wandb" if report_to_wandb else [],
            }
        },
    }

    model_name_suffix = "FineTuned"

    predictor = predictor.fit(
        train_data=train_ag_data,
        hyperparameters=model_hyperparameters,
        time_limit=time_limit,
        enable_ensemble=False,
        tuning_data=val_ag_data,
        num_val_windows=1,
    )

    # Due to Chronos implementation, need to manually construct the path to the fine-tuned model
    # AutoGluon constructs the model name, e.g., "Chronos_FineTuned"
    # We can get the name of the single trained model from the leaderboard.
    leaderboard = predictor.leaderboard(silent=True)
    if not leaderboard.empty:
        # The model name is in the 'model' column of the leaderboard
        trained_model_name = leaderboard.iloc[0]['model']
        model_path_to_return = os.path.join(predictor.path, "models", trained_model_name, "fine-tuned-ckpt")
    else:
        # Fallback: construct a potential name, though leaderboard is more reliable
        print("Warning: Leaderboard was empty. Model path might be incorrect.")
        base_model_name = "Chronos"
        full_model_name = f"{base_model_name}_{model_name_suffix}" if model_name_suffix else base_model_name
        # Check if this directory exists
        potential_path = os.path.join(predictor.path, "models", full_model_name, "fine-tuned-ckpt")
        if os.path.isdir(potential_path): # AutoGluon >= 0.8 stores models under "models/" subdir
                model_path_to_return = potential_path
        else: # Older versions might store directly under predictor.path
                potential_path_old = os.path.join(predictor.path, full_model_name)
                if os.path.isdir(potential_path_old):
                    model_path_to_return = potential_path_old
                else:
                    model_path_to_return = predictor.path # Default to base path


    # The actual model weights are usually within a subdirectory of model_path_to_return
    # For Chronos, it's often a 'pytorch_model.bin' or similar within that model's directory.
    # The path returned here is the directory containing the model's artifacts.
    return predictor, model_path_to_return







def align_series_start_day(input_df, target_start_day_name="Monday"):
    """
    Aligns time series for each patientid to start on a specific day of the week.

    Args:
        input_df (pd.DataFrame): DataFrame with 'patientid' and 'date' columns.
                                 'date' column should be datetime objects.
        target_start_day_name (str): The target start day name (e.g., "Monday", "Tuesday").

    Returns:
        pd.DataFrame: DataFrame with aligned dates.
        dict: A map of {patientid: offset_timedelta} used for alignment.
        pd.DataFrame: A mapping of {patientid, original_date, new_date}.
    """
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    if target_start_day_name not in day_map:
        raise ValueError(f"target_start_day_name must be one of {list(day_map.keys())}")

    target_start_day_int = day_map[target_start_day_name]
    
    aligned_df = input_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(aligned_df['date']):
        aligned_df['date'] = pd.to_datetime(aligned_df['date'])

    patient_offsets_map = {}
    date_mappings = []

    for patient_id, group in aligned_df.groupby('patientid'):
        if group.empty:
            continue
        
        original_min_date = group['date'].min()
        current_start_day_int = original_min_date.weekday() # Monday=0, Sunday=6
        
        # Calculate days to subtract to reach the target_start_day_int
        # We want to move the date backward or keep it if it's already the target day
        days_to_subtract = (current_start_day_int - target_start_day_int + 7) % 7
        offset = pd.Timedelta(days=days_to_subtract)
        
        patient_offsets_map[patient_id] = offset
        
        # Store original and new dates for mapping
        for _, row in group.iterrows():
            original_date = row['date']
            new_date = original_date - offset
            date_mappings.append({
                'patientid': patient_id,
                'original_date': original_date,
                'new_date': new_date
            })
        
        # Apply offset to the group
        aligned_df.loc[group.index, 'date'] = group['date'] - offset
        
    date_mapping_df = pd.DataFrame(date_mappings)
    if not date_mapping_df.empty:
        date_mapping_df.sort_values(by=['patientid', 'original_date'], inplace=True)

    print(f"Date alignment complete. All series now effectively start on a {target_start_day_name}.")
    return aligned_df, patient_offsets_map, date_mapping_df



def preprocess_input_to_autogluon_format(constant_df, input_df):
    """
    Preprocesses input dataframes into AutoGluon TimeSeriesDataFrame format,
    including date alignment. Note, each patientid is unique for each time series.
    """
    # 1. Align dates so all series start on the same day of the week
    # Ensure 'date' is datetime before alignment
    input_df['date'] = pd.to_datetime(input_df['date'])
    target_start_day_name="Monday"
    aligned_input_df, patient_offsets_map, date_mapping_df = align_series_start_day(
        input_df, target_start_day_name=target_start_day_name
    )

    # Convert 'event_value' to numeric and handle potential NaNs
    aligned_input_df['event_value'] = pd.to_numeric(aligned_input_df['event_value'], errors='coerce')
    aligned_input_df.dropna(subset=['event_value'], inplace=True)

    # Prepare static features
    static_features_df = constant_df.rename(columns={'patientid': 'item_id'})

    # Convert string columns in static features to categorical
    for col in static_features_df.select_dtypes(include=['object']).columns:
        static_features_df[col] = static_features_df[col].astype('category')

    # Select relevant columns for the main time series data from the aligned_input_df
    train_data_ts = aligned_input_df[['patientid', 'date', 'event_value']].copy()
    train_data_ts.rename(columns={'patientid': 'item_id', 'date': 'timestamp', 'event_value': 'target'}, inplace=True)

    # Ensure consistency of item_ids between time series data and static features
    valid_item_ids = set(static_features_df["item_id"]).intersection(train_data_ts['item_id'].unique())
    
    if not valid_item_ids:
        raise ValueError("No common item_ids found between time series data and static features after processing. "
                         "This could be due to all data being filtered out or mismatched IDs.")

    train_data_ts = train_data_ts[train_data_ts['item_id'].isin(valid_item_ids)]
    static_features_df = static_features_df[static_features_df['item_id'].isin(valid_item_ids)]
    
    if train_data_ts.empty:
        raise ValueError("Time series data is empty after filtering for valid item_ids.")
    if static_features_df.empty and not constant_df.empty: # if constant_df was provided but now static_features is empty
        print("Warning: Static features DataFrame became empty after filtering by valid item_ids present in time series data.")


    # Create TimeSeriesDataFrame
    train_ag_data = TimeSeriesDataFrame.from_data_frame(
        train_data_ts,
        id_column="item_id",
        timestamp_column="timestamp",
        static_features_df=static_features_df if not static_features_df.empty else None
    )

    # Sort
    train_ag_data = train_ag_data.sort_index()

    assert len(train_ag_data) > 0, "The TimeSeriesDataFrame is empty after processing."
    print(f"Length of train_ag_data (time series points): {len(train_ag_data)}")
    print(f"Number of unique items in train_ag_data: {train_ag_data.num_items}")
    if not static_features_df.empty:
        print(f"Length of static_features_df: {len(static_features_df)}")
    else:
        print("No static features were provided or they became empty after filtering.")


    return train_ag_data, patient_offsets_map, date_mapping_df
        



def predict_chronos_model_on_input(prediction_length, train_ag_data, chronos_model):

    if isinstance(chronos_model, str):

        print("Loading pre-trained model from path...")

        predictor = TimeSeriesPredictor(
            prediction_length=prediction_length,
            freq="W-MON",
            target="target",
            eval_metric="MAE",
            verbosity=4, # For debugging
        )

        # Fit the predictor (for zero-shot/inference with Chronos, this loads the pre-trained model)
        # We explicitly tell it to use only Chronos.
        # Provide a short time_limit as we are not really "training" from scratch.
        # However, for Chronos, the fit call mainly prepares data and ensures model context.
        # AutoGluon may still perform some minimal adjustments or internal processing.

        predictor.fit(
            train_ag_data,
            hyperparameters={
                "Chronos": {
                    "model_path": chronos_model,
                    "target_scaler": None,
                },
            },
        )
    else:
        assert isinstance(chronos_model, TimeSeriesPredictor), "chronos_model should be a TimeSeriesPredictor object!"
        predictor = chronos_model
        predictor.prediction_length = prediction_length
        print("Using fine-tuned TimeSeriesPredictor object for predictions...")

    # Predict
    # The `predict` method will use the historical data in `train_ag_data` as context
    print(f"\nMaking predictions for the next {predictor.prediction_length} time steps...")
    predictions = predictor.predict(train_ag_data, use_cache=False)

    return predictions







def map_predictions_to_original_dates(predictions_df, patient_offsets_map):
    """
    Maps the 'timestamp' column of the predictions DataFrame back to their original date scale.

    Args:
        predictions_df (pd.DataFrame): DataFrame of predictions from AutoGluon.
                                       Expected to have 'item_id' and 'timestamp'.
        patient_offsets_map (dict): A map of {patientid: offset_timedelta}
                                    where offset_timedelta is what was *subtracted* originally.
    
    Returns:
        pd.DataFrame: Predictions DataFrame with 'timestamp' adjusted to original scale.
    """

    # Reset index
    predictions_df = predictions_df.copy().reset_index()

    if 'item_id' not in predictions_df.columns or 'timestamp' not in predictions_df.index.names:
         # If timestamp is not in index, try getting it from columns
        if 'timestamp' not in predictions_df.columns:
            raise ValueError("Predictions DataFrame must have 'item_id' column and 'timestamp' in index or as a column.")
        
        # If timestamp is a column, set it as index to perform mapping, then reset
        predictions_df_copy = predictions_df.set_index('timestamp', append=True) # append if item_id is already index
        is_timestamp_col = True
    else: # timestamp is already part of the index
        predictions_df_copy = predictions_df.copy()
        is_timestamp_col = False

    # If 'item_id' is part of MultiIndex, it's level 0 by default from TimeSeriesDataFrame
    # If 'timestamp' is part of MultiIndex, it's level 1 by default
    
    original_timestamps = []

    # predictions_df_copy has a MultiIndex (item_id, timestamp)
    for idx, row in predictions_df_copy.iterrows():
        item_id = row["item_id"] # Assuming item_id is the first level of the index
        predicted_aligned_date = idx[1] # Assuming timestamp is the second level

        offset_to_add = patient_offsets_map.get(item_id)
        if offset_to_add is None:
            print(f"Warning: No offset found for item_id {item_id}. Timestamp will not be changed.")
            original_timestamps.append(predicted_aligned_date)
        else:
            original_timestamps.append(predicted_aligned_date + offset_to_add)
            
    # Create a new DataFrame with the original timestamps
    # We need to preserve all other columns and the item_id structure
    
    # If predictions_df_copy has MultiIndex (item_id, timestamp)
    new_multi_index = pd.MultiIndex.from_tuples(
        [(idx[0], ots) for idx, ots in zip(predictions_df_copy.index, original_timestamps)],
        names=predictions_df_copy.index.names
    )
    
    predictions_original_dates_df = predictions_df_copy.copy()
    predictions_original_dates_df.index = new_multi_index
    
    if is_timestamp_col: # If timestamp was originally a column, reset index
        predictions_original_dates_df.reset_index(inplace=True)

    print("\nTimestamps in predictions mapped back to original scale.")
    # print("Predictions (with original dates) sample:")
    # print(predictions_original_dates_df.head())
    return predictions_original_dates_df



def post_process_predictions_back_into_original_format(predictions_df, meta_data, target_dates_patientid_variable, indication, 
                                                       column_for_predictions="mean", destandardize=True, override_checks=False, train_data_stats_path=TRAIN_DATA_STATS):

    assert "event_value" not in target_dates_patientid_variable.columns, "event_value should not be in target_dates_patientid_variable!"
    assert all([x in target_dates_patientid_variable.columns for x in ["patientid", "date", "event_name"]]), \
        "target_dates_patientid_variable should contain patientid, date, and event_name columns!"

    predictions = predictions_df.copy()
    predictions = pd.DataFrame(predictions)

    predictions = predictions.reset_index()
    predictions = predictions[['item_id', 'timestamp', column_for_predictions]]
    predictions.rename(columns={column_for_predictions: 'event_value'}, inplace=True)
    predictions['event_value'] = predictions['event_value'].astype(float)
    predictions.rename(columns={'item_id': 'patientid', 'timestamp': 'date'}, inplace=True)

    # Add in event name
    specific_meta_data = meta_data[["patientid", "target_variable"]]
    specific_meta_data = specific_meta_data.rename(columns={"target_variable": "event_name"})
    predictions = pd.merge(predictions, specific_meta_data, on="patientid", how="left")


    def _check_missing_values_from_two_dfs(df_to_ensure_all_values_present, df_to_check):
        merged_df = pd.merge(df_to_ensure_all_values_present, df_to_check, on=["patientid", "date", "event_name"], 
                         how="outer", indicator=True)
        missing = merged_df[merged_df['_merge'] == "left_only"]
        return missing
    
    # Destandardize
    if destandardize:

        assert predictions["event_value"].notnull().all(), "Some event_values are null in predictions!"

        # Load in the stats CSV
        variable_stats = pd.read_csv(train_data_stats_path)
        data_stats = variable_stats[["event_name",  "mean_without_outliers", "std_without_outliers"]].copy()

        # Merge the data stats with the predictions
        predictions_merged = predictions.merge(data_stats, on="event_name", how="left")

        # Apply destandardization
        predictions_merged["event_value"] = (predictions_merged["event_value"] * predictions_merged["std_without_outliers"]) + predictions_merged["mean_without_outliers"]

        # Assign destandardized values back to original dataframe
        predictions["event_value"] = predictions_merged["event_value"]

    if override_checks:
        print("Overriding automatic checks! Only do this for debugging purposes!!!!")
        return predictions

    # In some very rare cases << 1%, due to the implementation of Chronos (which cannot handle
    # irregular sampling here), we may miss some predictions, so we forward fill. The proportion
    # is so low that we can ignore it, and we ensure that its low (< 0.1%).
    entries = predictions[["patientid", "date", "event_name"]]
    entries_missed_in_predictions = _check_missing_values_from_two_dfs(target_dates_patientid_variable, entries)
    if entries_missed_in_predictions.shape[0] > 0:
        print(f"Warning: {entries_missed_in_predictions.shape[0]} entries were missed in predictions. "
              "This is due to irregular sampling.")
        assert entries_missed_in_predictions.shape[0] < 50, "More than 50 entries were missed in predictions. Check the data!"
        assert  entries_missed_in_predictions.shape[0] / predictions.shape[0] < 0.001, "More than 0.1% of entries were missed in predictions. Check the data!"
        original_preds = predictions.copy()

        #: Add new entries to predictions
        new_entries = entries_missed_in_predictions[["patientid", "date", "event_name"]].copy()
        new_entries['event_value'] = np.nan
        new_predictions = pd.concat([predictions, new_entries], ignore_index=True)
        
        #: group by patientid and event_name, sort by date and forward fill
        new_predictions_sorted = new_predictions.sort_values(by=['patientid', 'event_name', 'date'])
        new_predictions_filled = new_predictions_sorted.groupby(['patientid', 'event_name'], group_keys=False).apply(lambda group: group.ffill())
        new_predictions = new_predictions_filled

        # Merge back to predictions
        predictions = new_predictions

        # Do final check
        entries_missed_in_predictions = _check_missing_values_from_two_dfs(target_dates_patientid_variable, 
                                                                           predictions[["patientid", "date", "event_name"]])
        assert entries_missed_in_predictions.shape[0] == 0, "Some entries are still missing in predictions after forward fill!"
        assert all(predictions['event_value'].notnull()), "Some event_values are still null after forward fill!"

        #: assert that all original predictions are still there
        assert pd.merge(
            original_preds[original_preds['event_value'].notna()],  # Filter for original non-NaN predictions
            predictions,                                            # The final predictions DataFrame
            on=['patientid', 'date', 'event_name', 'event_value'],  # Merge on keys AND the event_value itself
            how='inner'                                             # Inner merge means only exact matches are kept
        ).shape[0] == original_preds['event_value'].notna().sum(), \
            "AssertionError: Some original non-NaN predictions were altered or are missing from the final predictions after processing."
    

    return predictions






