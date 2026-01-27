import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import logging
from datetime import timedelta
from datetime import datetime

from digital_twin_converter.instruction.data_splitter import BaseDataSplitter
from digital_twin_converter.common.data_manager import SingleIndicationDataManager
from digital_twin_converter.common.config import Config





class DataSplitterForecasting(BaseDataSplitter):
    """
    Generates forecasting tasks by splitting patient time-series data.

    This class identifies suitable points in a patient's timeline (split dates)
    and selects specific variables (e.g., lab results) to forecast. It calculates
    statistics on variable predictability using a simple baseline (copy-forward)
    to inform variable sampling. It handles filtering outliers and ensures
    splits meet criteria like minimum data points before and after the split date.

    Attributes:
        variable_stats (pd.DataFrame | None): Statistics calculated for each
            potential variable (e.g., R², NRMSE based on copy-forward baseline).
            Computed by `setup_statistics`.
        min_num_samples_for_statistics (int): Minimum data points required per
            variable across the training set to compute statistics.
        sampling_score_to_use (str): The column name in `variable_stats` used
            as a score for weighted sampling of variables to forecast.
        min_nr_variable_seen_previously (int): Minimum occurrences of a variable
            required within the lookback period before a split date.
        min_nr_variable_seen_after (int): Minimum occurrences of a variable
            required within the forecast period after a split date.
        list_of_valid_categories (list): Event categories (e.g., ['LABS']) to
            consider for forecasting tasks.
        save_path_for_variable_stats (str | None): Optional path to save the
            computed `variable_stats` DataFrame.
        min_nr_variables_to_sample (int): Minimum number of distinct variables
            to include in a single forecasting task sample.
        max_nr_variables_to_sample (int): Maximum number of distinct variables
            to include in a single forecasting task sample.
        filtering_stategy (str): Strategy name ('3-sigma') used to filter or
            clip outlier values in the target data.
        _filtering_methods (dict): Maps filtering strategy names to methods.
    """

    def __init__(self,
                 config : Config,
                 data_manager: SingleIndicationDataManager,
                 max_length_days_after_lot : int = 90,
                 max_lookback_for_value : int = 90,
                 max_forecast_for_value : int = 90,
                 min_num_samples_for_statistics : int = 50,
                 sampling_score_to_use : str = 'score_log_nrmse_n_samples',
                 min_nr_variable_seen_previously : int = 1,
                 min_nr_variable_seen_after : int = 1,
                 list_of_valid_categories : list = None,
                 save_path_for_variable_stats : str = None,
                 min_nr_variables_to_sample : int = 3,
                 max_nr_variables_to_sample : int = 3,
                 filtering_stategy : str = '3-sigma'):
        """
        Initializes the DataSplitterForecasting instance.

        Args:
            config: Configuration object containing shared settings like column names.
            data_manager: Provides access to patient data for a single indication.
            max_length_days_after_lot: Max days after LoT start to consider for split dates. Defaults to 90.
            max_lookback_for_value: Max days before a split date to look for past variable occurrences. Defaults to 90.
            max_forecast_for_value: Max days after a split date to look for future variable occurrences (target data). Defaults to 90.
            min_num_samples_for_statistics: Minimum total occurrences of a variable across the training set needed to calculate statistics. Defaults to 50.
            sampling_score_to_use: Column name in the computed statistics table used for weighted sampling of variables. Defaults to 'score_log_nrmse_n_samples'.
            min_nr_variable_seen_previously: Min occurrences of a variable required in the lookback window for a split to be valid for that variable. Defaults to 1.
            min_nr_variable_seen_after: Min occurrences of a variable required in the forecast window for a split to be valid for that variable. Defaults to 1.
            list_of_valid_categories: List of event categories to consider for forecasting (e.g., ['LABS']). Defaults to `config.event_category_labs`.
            save_path_for_variable_stats: Optional file path to save the calculated variable statistics CSV. Defaults to None.
            min_nr_variables_to_sample: The minimum number of distinct variables to attempt to sample for each forecasting task. Defaults to 3.
            max_nr_variables_to_sample: The maximum number of distinct variables to attempt to sample for each forecasting task. Defaults to 3.
            filtering_stategy: The strategy for handling outliers in target variable values ('3-sigma'). Defaults to '3-sigma'.
        """
        super().__init__(data_manager, config, max_length_days_after_lot, max_lookback_for_value, max_forecast_for_value)

        self.variable_stats = None
        self.min_num_samples_for_statistics = min_num_samples_for_statistics
        self.sampling_score_to_use = sampling_score_to_use

        self.min_nr_variable_seen_previously = min_nr_variable_seen_previously
        self.min_nr_variable_seen_after = min_nr_variable_seen_after
        self.list_of_valid_categories = (list_of_valid_categories if
                                         list_of_valid_categories is not None else [ self.config.event_category_labs])
        self.save_path_for_variable_stats = save_path_for_variable_stats
        self.min_nr_variables_to_sample = min_nr_variables_to_sample
        self.max_nr_variables_to_sample = max_nr_variables_to_sample
        self.filtering_stategy = filtering_stategy

        self._filtering_methods = {
            '3-sigma': self._filter_3_sigma
        }


    def setup_statistics(self):
        """
        Calculates baseline performance statistics for variables.

        Iterates through all patients in the training set and potential split
        dates within specified ranges around Lines of Therapy (LoTs). For each
        numeric variable (typically labs), it calculates metrics like R², NRMSE,
        MAPE, mean, standard deviation, etc., based on a simple "copy forward"
        prediction baseline (predicting the next value as the previous one).
        These statistics quantify the inherent predictability/variability of each
        variable and are stored in the `self.variable_stats` DataFrame. This
        dataframe is used later for filtering variables and weighted sampling
        during split generation.
        """

        #: setup labs
        events = self.dm.data_frames[self.config.event_table_name]
        all_labs = events[events[self.config.event_category_col] ==  self.config.event_category_labs][self.config.event_name_col].unique()

        #: setup all possible split dates by looping through all patients
        all_possible_split_dates = []
        relevant_events = events[[self.config.patient_id_col, self.config.date_col, self.config.event_category_col, self.config.event_name_col]].copy()
        relevant_events = relevant_events.sort_values([self.config.patient_id_col, self.config.date_col])
        grouped_events = relevant_events.groupby(self.config.patient_id_col)

        for idx, (patientid, event_data) in enumerate(grouped_events):
            if idx % 1000 == 0:
                logging.info(f"Processing patient ({idx+1}/{len(self.dm.all_patientids)})")
            temp_patient_data = {"events": event_data}

            temp_splits = self._get_all_dates_within_range_of_lot(temp_patient_data,
                                                                  days_before_lot_start=self.max_lookback_for_value,
                                                                  days_after_start_lot=self.max_forecast_for_value)
            temp_splits[self.config.patient_id_col] = patientid
            temp_splits = temp_splits[[self.config.date_col, self.config.patient_id_col]]
            all_possible_split_dates.append(temp_splits)
            del temp_patient_data

        all_possible_split_dates = pd.concat(all_possible_split_dates, axis=0, ignore_index=True)
        all_possible_split_dates = all_possible_split_dates.drop_duplicates()

        #: filter to only train patients
        all_train_patientids = [patientid for patientid, split in self.dm.patient_to_split_mapping.items()
                                if split == self.config.train_split_name]
        rows_to_select = all_possible_split_dates[self.config.patient_id_col].isin(all_train_patientids)
        all_possible_split_dates = all_possible_split_dates[rows_to_select]

        # Setup status
        self.variable_stats = {}

        #: loop through every lab variable
        for idx, lab_var in enumerate(all_labs):

            if idx % 20 == 0:
                logging.info(f"Processing lab variable {lab_var} ({idx+1}/{len(all_labs)})")

            # Get corresponding events, sorted by date and patientid
            curr_events = events[events[self.config.event_name_col] == lab_var].copy().sort_values([self.config.patient_id_col, self.config.date_col])
            descriptive_name = curr_events[self.config.event_descriptive_name_col].iloc[0]
            curr_events = curr_events[[self.config.patient_id_col, self.config.date_col, self.config.event_value_col]].drop_duplicates()

            # : extract only those dates which are given in self._get_all_possible_split_dates
            # by doing inner join with all_possible_split_dates
            curr_events = curr_events.merge(all_possible_split_dates, on=[self.config.patient_id_col, self.config.date_col], how='inner')

            # Turn into numeric
            curr_events[self.config.event_value_col] = pd.to_numeric(curr_events[self.config.event_value_col], errors='coerce')

            # Shift values by one for copy forward
            curr_events['predicted_value'] = curr_events.groupby(self.config.patient_id_col)[self.config.event_value_col].shift(1)

            # Drop rows where predicted_value is NaN (first value for each patient)
            valid_events = curr_events.dropna(subset=['predicted_value'])

            # Need at least 2 samples for R^2, else, we should ignore the variable anyway
            if valid_events.shape[0] >= self.min_num_samples_for_statistics:

                # Calculate R² across all
                r2 = r2_score(valid_events[self.config.event_value_col], valid_events['predicted_value'])

                # Calculate NRMSE
                mse = np.mean((valid_events[self.config.event_value_col] - valid_events['predicted_value']) ** 2)
                rmse = np.sqrt(mse)
                nrmse = rmse / (valid_events[self.config.event_value_col].std())

                # Calculate mape
                mape = np.mean(np.abs((valid_events[self.config.event_value_col] - valid_events['predicted_value']) /
                                      valid_events[self.config.event_value_col])) * 100

                # Calculate score
                score_nrmse_n_samples = nrmse * valid_events.shape[0]
                score_log_nrmse_n_samples = np.log2(score_nrmse_n_samples)

                # Calculate bucekts
                _, bin_5_edges = pd.qcut(valid_events[self.config.event_value_col], q=5,
                                         retbins=True, labels=False, duplicates='drop')

                # Calculate mean and std after removing over 3 standard deviations
                mean = valid_events[self.config.event_value_col].mean()
                std = valid_events[self.config.event_value_col].std()
                valid_events = valid_events.copy()

                valid_events[self.config.event_value_col + "_cleaned"] = valid_events[self.config.event_value_col].apply(lambda x: x if
                            (x > mean - 3 * std) and
                            (x < mean + 3 * std) else
                            np.nan)

                mean_without_outliers = np.nanmean(valid_events[self.config.event_value_col + "_cleaned"].values)
                std_without_outliers = np.nanstd(valid_events[self.config.event_value_col + "_cleaned"].values)


                # Record
                self.variable_stats[lab_var] = {
                    "event_descriptive_name": descriptive_name,
                    'r2': r2,
                    'nrmse': nrmse,
                    'mape': mape,
                    'score_nrmse_n_samples': score_nrmse_n_samples,
                    'score_log_nrmse_n_samples': score_log_nrmse_n_samples,
                    'std': std,
                    'mean': mean,
                    'range': valid_events[self.config.event_value_col].max() - valid_events[self.config.event_value_col].min(),
                    'num_samples': valid_events.shape[0],
                    '5_equal_sized_bins': bin_5_edges.tolist(),
                    "mean_without_outliers": mean_without_outliers,
                    "std_without_outliers": std_without_outliers,
                }

        #: turn into a pandas dataframe
        self.variable_stats = pd.DataFrame(self.variable_stats).T
        self.variable_stats = self.variable_stats.reset_index(drop=False, names=self.config.event_name_col)

        # Print some statistics
        logging.info(f"Number of variables included in selection: {self.variable_stats.shape[0]}")
        logging.info(f"Mean of score used for sampling: {self.variable_stats[self.sampling_score_to_use].mean()}")
        logging.info(f"Std of score used for sampling: {self.variable_stats[self.sampling_score_to_use].std()}")
        logging.info(f"Min of score used for sampling: {self.variable_stats[self.sampling_score_to_use].min()}")
        logging.info(f"Max of score used for sampling: {self.variable_stats[self.sampling_score_to_use].max()}")

        assert self.variable_stats.shape[0] > 0, "Error - for some reason no variables have been included in the statistics table. Check your data & setup."

        # Save if requested
        if self.save_path_for_variable_stats is not None:
            self.variable_stats.to_csv(self.save_path_for_variable_stats)


    def _sample_proportionally(self, possible_variables : list, num_samples : int) -> list:
        """
        Samples variables based on a pre-calculated score.

        Given a list of variable names deemed valid for a specific split point,
        this method samples a subset of them without replacement. The sampling is
        weighted proportionally to the score specified by `self.sampling_score_to_use`
        (calculated in `setup_statistics`), making variables with higher scores
        (e.g., lower NRMSE * num_samples) more likely to be chosen.

        Args:
            possible_variables: A list of variable names (event_name) eligible
                for sampling at a particular split date.
            num_samples: The desired number of variables to sample. The actual
                number returned will be min(num_samples, len(possible_variables)).

        Returns:
            A list of sampled variable names, or None if no variables could be
            sampled (e.g., if `possible_variables` is empty or scores result in NaN probabilities).
        """

        #: get all variables
        curr_vars = self.variable_stats[self.config.event_name_col].isin(possible_variables)
        all_variables = self.variable_stats[curr_vars]

        if all_variables.shape[0] == 0:
            return None

        #: get all scores
        all_scores = all_variables[self.sampling_score_to_use].values
        all_scores = all_scores.astype(np.float64)

        #: get all probabilities
        all_probs = all_scores / all_scores.sum()

        # if any nans in probs, then return None
        if np.isnan(all_probs).any():
            return None

        #: sample
        var_choice = all_variables[self.config.event_name_col].tolist()
        real_num_samples = min(num_samples, len(var_choice))
        sampled_var = np.random.choice(var_choice, size=real_num_samples, replace=False, p=all_probs)

        return sampled_var

    def _filter_3_sigma(self, events : pd.DataFrame) -> pd.DataFrame:
        """
        Filters or clips event values based on the 3-sigma rule.

        For each unique event name in the input DataFrame, this method uses the
        pre-calculated mean and standard deviation (from `self.variable_stats`)
        for that event type. It then clips values lying outside the
        [mean - 3*std, mean + 3*std] range to the boundary of this interval.
        Rows where the value was originally NaN or becomes NaN after potential
        numeric conversion issues are dropped.

        Args:
            events: DataFrame containing event data, including columns for
                    event name and value. Expected to contain target data.

        Returns:
            A DataFrame with the event values clipped based on the 3-sigma rule,
            and potentially fewer rows if NaNs were present or introduced.
        """

        #: group by event name
        events = events.copy()
        grouped_events = events.groupby(self.config.event_name_col)

        #: loop through every group
        for event_name, event_data in grouped_events:

            #: get the mean and std
            stats = self.variable_stats[self.variable_stats[self.config.event_name_col] == event_name]
            mean_val = stats['mean'].values[0]
            std_val = stats['std'].values[0]

            #: filter
            event_data[self.config.event_value_col] = event_data[self.config.event_value_col].apply(lambda x: x if
                                                                            (x > mean_val - 3 * std_val) and
                                                                            (x < mean_val + 3 * std_val) else
                                                                            np.clip(x, mean_val - 3 * std_val,
                                                                                    mean_val + 3 * std_val))

            #: update, convert to float the event value column
            events[self.config.event_value_col] = events[self.config.event_value_col].astype(float)
            events.loc[event_data.index, self.config.event_value_col] = event_data[self.config.event_value_col]

        # Drop nan values in value column
        events = events.dropna(subset=[self.config.event_value_col])

        return events


    def _get_all_possible_splits(self, patient_data_dic : dict,
                                 min_nr_variable_seen_previously : int = 1,
                                 min_nr_variable_seen_after : int = 1,
                                 list_of_valid_categories : list = None,
                                 subselect_random_within_lot : int = False,
                                 max_num_samples_per_lot : int = 1) -> pd.DataFrame:
        """
        Identifies all potential (date, variable) pairs for forecasting tasks.

        This method scans a patient's event data to find all combinations of
        dates and variables that meet the criteria for forming a valid forecasting
        split. A split is valid for a specific variable at a specific date if:
        1. The date falls within the allowed range relative to a Line of Therapy (LoT) start.
        2. The variable belongs to the `list_of_valid_categories`.
        3. The variable has at least `min_nr_variable_seen_previously` occurrences
           within the `max_lookback_for_value` period before the date.
        4. The variable has at least `min_nr_variable_seen_after` occurrences
           within the `max_forecast_for_value` period after the date.

        If `subselect_random_within_lot` is True, it first identifies all potential
        split dates per LoT using `_get_all_dates_within_range_of_lot` and then
        randomly selects up to `max_num_samples_per_lot` dates from those associated
        with each LoT before checking variable validity.

        Args:
            patient_data_dic: Dictionary containing the patient's dataframes
                (e.g., 'events', 'constant').
            min_nr_variable_seen_previously: Minimum required past occurrences.
            min_nr_variable_seen_after: Minimum required future occurrences.
            list_of_valid_categories: Categories of variables to consider.
            subselect_random_within_lot: If True, randomly sample dates per LoT
                before checking variable validity.
            max_num_samples_per_lot: Max dates to sample per LoT if
                `subselect_random_within_lot` is True.

        Returns:
            A tuple containing two DataFrames:
            1. `return_splits`: DataFrame listing all valid split combinations.
               Columns include 'date' (split date), 'event_name' (variable),
               'event_category', and 'lot_date' (associated LoT start date).
               If no variables are valid for a sampled date, a row with None
               values for date/event_name/category is added for that lot_date.
            2. `all_possible_dates`: DataFrame containing the potential split dates
               considered, along with their associated 'lot_date'. This reflects
               the output of `_get_all_dates_within_range_of_lot`, potentially
               filtered by `select_random_splits_within_lot`. Columns: 'date', 'lot_date'.
        """

        #: setup data
        events = patient_data_dic["events"]

        #: go over all possible events
        all_events = events[[self.config.event_name_col, self.config.event_category_col]].copy()
        all_events = all_events.drop_duplicates().values.tolist()
        all_events.sort()

        #: get all starting LoTs dates
        all_possible_dates = self._get_all_dates_within_range_of_lot(patient_data_dic,
                                                                    days_before_lot_start=0,
                                                                    days_after_start_lot=self.max_length_days_after_lot)

        return_splits = []

        # If needed, select only those within an lot
        if subselect_random_within_lot:
            all_possible_dates = self.select_random_splits_within_lot(all_possible_dates,
                                                                      max_num_samples_per_lot=max_num_samples_per_lot)

        # Go over all dates and check all variables with which are eligible for a split
        if list_of_valid_categories is not None:
            events_category = events[events[self.config.event_category_col].isin(list_of_valid_categories)]
            all_events = [(var, cat) for var, cat in all_events if cat in list_of_valid_categories]

        # Pre-compute date ranges for lookback and forecast
        lookback_range = timedelta(days=self.max_lookback_for_value)
        forecast_range = timedelta(days=self.max_forecast_for_value)

        # Initialize the return_splits list
        return_splits = []

        # Iterate over all possible dates
        for row in all_possible_dates.itertuples(index=False):

            curr_date, lot_date = row
            num_added = 0

            # Filter events within the lookback and forecast ranges
            lookback_events = events_category[(events_category[self.config.date_col] <= curr_date) &
                                              (events_category[self.config.date_col] >= curr_date - lookback_range)]
            forecast_events = events_category[(events_category[self.config.date_col] > curr_date) &
                                              (events_category[self.config.date_col] <= curr_date + forecast_range)]

            # Iterate over all events
            for curr_var, curr_event_category in all_events:

                # Filter events by current variable
                prev_events = lookback_events[lookback_events[self.config.event_name_col] == curr_var]
                future_events = forecast_events[forecast_events[self.config.event_name_col] == curr_var]

                # Count events
                prev_events_count = prev_events.shape[0]
                future_events_count = future_events.shape[0]

                # Check conditions and add to return_splits if valid
                if (prev_events_count >= min_nr_variable_seen_previously and
                    future_events_count >= min_nr_variable_seen_after):

                    return_splits.append({
                        "date": curr_date,
                        "event_name": curr_var,
                        "event_category": curr_event_category,
                        "lot_date" : lot_date,
                    })
                    num_added += 1

            #: if nothing has been added for current lot_date, then add a none event
            if num_added == 0:
                return_splits.append({
                    "date": None,
                    "event_name": None,
                    "event_category": None,
                    "lot_date" : lot_date,
                })

        #: transform to pandas dataframe
        return_splits = pd.DataFrame(return_splits)

        #: drop duplicates
        return_splits = self.drop_duplicates_except_na_for_date_col(return_splits)

        #: return splits list
        return return_splits, all_possible_dates


    def _generate_variable_splits_for_date(self, curr_date, nr_samples,
                                           override_variables_to_predict,
                                           events, all_possible_split_dates,
                                           apply_filtering, override_split_dates,
                                           patient_data, lot_date):
        """
        Generates specific forecasting task samples for a given split date.

        For a single potential split date (`curr_date`), this method creates
        `nr_samples` forecasting tasks. Each task involves:
        1. Determining the set of variables valid for forecasting at `curr_date`
           (based on `all_possible_split_dates`).
        2. Sampling a subset of these variables (between `min_nr_variables_to_sample`
           and `max_nr_variables_to_sample`), weighted by their score, unless
           `override_variables_to_predict` is provided.
        3. Creating the actual data split: 'events_until_split' (history) and
           'target_events_after_split' (future values of sampled variables within
           the forecast window, ensuring no overlap with the next LoT).
        4. Optionally applying filtering (e.g., 3-sigma) to the target values.
        5. Bundling the split data along with metadata ('constant_data',
           'split_date_included_in_input', 'sampled_variables', 'lot_date').

        It also updates `all_possible_split_dates` by removing the variable/date
        combinations used in the generated samples to avoid reuse.

        Args:
            curr_date: The specific date to generate splits for. Can be None.
            nr_samples: The number of variable sets to sample for this date.
            override_variables_to_predict: If provided, forces these variables
                to be used instead of sampling (checks if they are valid first).
            events: The full event history for the patient.
            all_possible_split_dates: DataFrame mapping valid dates to valid variables.
            apply_filtering: Whether to filter target event values.
            override_split_dates: List of externally provided split dates (used to check
                if filtering should be skipped even if target is empty).
            patient_data: Dictionary containing patient's 'events' and 'constant' data.
            lot_date: The Line of Therapy start date associated with `curr_date`.

        Returns:
            A tuple containing:
            - `date_splits`: A list of dictionaries, each dictionary is a complete
              forecasting sample {'events_until_split', 'target_events_after_split', ...}.
              Empty if no valid samples could be generated for this date.
            - `valid_sample_date`: Boolean, True if `curr_date` was not None or NaN.
            - `date_splits_meta`: A single-row DataFrame containing the `curr_date`
              and `lot_date` used for this attempt.
            - `all_possible_split_dates`: The input DataFrame, updated to remove
              the date/variable combinations that were successfully used.
        """

        # Get current date -> can be multiple dates per lot
        possible_variables = all_possible_split_dates[all_possible_split_dates[self.config.date_col] == curr_date]
        possible_variables = possible_variables[self.config.event_name_col].tolist()
        date_splits = []
        valid_sample_date = False

        if curr_date is None or pd.isna(curr_date):

            # Generate empty meta and return
            date_splits_meta = [{self.config.date_col: curr_date, self.config.lot_date_col: lot_date}]
            date_splits_meta = pd.DataFrame(date_splits_meta)

            return date_splits, valid_sample_date, date_splits_meta, all_possible_split_dates

        # Note that generally valid date
        valid_sample_date = True

        # Try generating samples
        for _ in range(nr_samples):

            #: uniformly sample nr of variables to sample in
            # range(min_nr_variables_to_sample, max_nr_variables_to_sample)
            max_nr_variables_to_sample = min(len(possible_variables), self.max_nr_variables_to_sample)
            min_nr_variables_to_sample = min(len(possible_variables), self.min_nr_variables_to_sample)
            if max_nr_variables_to_sample > min_nr_variables_to_sample:
                nr_variables_to_sample = np.random.randint(min_nr_variables_to_sample, max_nr_variables_to_sample)
            else:
                nr_variables_to_sample = min_nr_variables_to_sample

            # If we have less variables than the minimum, we skip this sample
            if nr_variables_to_sample == 0:
                continue

            #: sample which variables via _sample_proportionally or manual override
            if override_variables_to_predict is None:
                sampled_variables = self._sample_proportionally(possible_variables, nr_variables_to_sample)
            else:
                # Skip if not all override variables are in possible_variables
                if not all([var in possible_variables for var in override_variables_to_predict]):
                    continue
                sampled_variables = override_variables_to_predict

            #: if no variables sampled, skip
            if sampled_variables is None:
                continue

            #: remove only sampled variables at current date from all_possible_split_dates
            rows_to_remove = ((all_possible_split_dates[self.config.date_col] == curr_date) &
                            all_possible_split_dates[self.config.event_name_col].isin(sampled_variables))
            all_possible_split_dates = all_possible_split_dates[~rows_to_remove]

            #: get the splits for the given patient data
            events_before_split = events[events[self.config.date_col] <= curr_date]
            events_after_split = events[events[self.config.date_col] > curr_date]
            events_after_split = events_after_split[events_after_split[self.config.date_col] <= curr_date +
                                                    pd.Timedelta(days=self.max_forecast_for_value)]
            events_after_split = events_after_split[events_after_split[self.config.event_name_col].isin(sampled_variables)]

            #: filter so that we do not overlap with next LoT, since that will invalidate the results
            lots = events[events[self.config.event_category_col] == self.config.event_category_lot]
            lots = lots[lots[self.config.date_col] > curr_date]
            lots = lots.sort_values(self.config.date_col)
            if lots.shape[0] > 0 and not self.config.skip_future_lot_filtering:
                date_of_next_lot = lots[self.config.date_col].iloc[0]
                events_after_split = events_after_split[events_after_split[self.config.date_col] < date_of_next_lot]

            #: if apply_filtering, apply 3-sigma filtering (only to target) and drop any bad rows
            if apply_filtering:
                events_after_split[self.config.event_value_col] = pd.to_numeric(events_after_split[self.config.event_value_col],
                                                                    errors='coerce')
                events_after_split = self._filtering_methods[self.filtering_stategy](events_after_split)

            #: check if still valid samples (i.e. values are not nan in output),
            # but only if no override (e.g. in inference)
            if events_after_split.shape[0] == 0 and override_split_dates is None:
                continue

            #: save to a list
            date_splits.append({
                "events_until_split": events_before_split,
                "target_events_after_split": events_after_split,
                "constant_data" : patient_data["constant"].copy(),
                "split_date_included_in_input": curr_date,
                "sampled_variables": sampled_variables,
                "lot_date": lot_date,
            })

        # Turn into 1 row dataframe
        date_splits_meta = [{self.config.date_col: curr_date, self.config.lot_date_col: lot_date}]
        date_splits_meta = pd.DataFrame(date_splits_meta)

        return date_splits, valid_sample_date, date_splits_meta, all_possible_split_dates


    def get_splits_from_patient(self,
                                patient_data : dict,
                                nr_samples : int,
                                apply_filtering : bool,
                                return_splits : bool = False,
                                override_variables_to_predict : list[str] = None,
                                override_split_dates : list[datetime] = None,
                                max_num_samples_per_lot : int = 1) -> list:
        """
        Generates multiple forecasting splits for a patient.

        This is the main method for creating forecasting tasks for a single patient.
        It first identifies potential split dates, typically by randomly selecting
        up to `max_num_samples_per_lot` valid dates associated with each Line of
        Therapy (LoT) using `_get_all_possible_splits`. Alternatively, specific
        dates can be provided via `override_split_dates`.

        For each selected date, it calls `_generate_variable_splits_for_date` to
        generate `nr_samples` distinct forecasting tasks by sampling different
        sets of variables to predict (unless `override_variables_to_predict` is set).
        Filtering of target values can be applied.

        Args:
            patient_data: Dictionary containing the patient's 'events' and 'constant' data.
            nr_samples: The number of variable sets to sample per selected split date.
            apply_filtering: If True, applies filtering (e.g., 3-sigma) to the target event values.
            return_splits: If True, returns both the generated splits and a DataFrame
                of the split dates used. Defaults to False.
            override_variables_to_predict: If provided, forces prediction of these specific
                variables for all generated splits, bypassing proportional sampling. Requires
                `override_split_dates` to also be set for typical use cases (like inference).
            override_split_dates: If provided, uses these specific dates instead of discovering
                and sampling dates based on LoTs.
            max_num_samples_per_lot: The maximum number of dates to randomly sample per LoT
                when `override_split_dates` is None. Defaults to 1.

        Returns:
            If `return_splits` is False:
                A list of lists. Each inner list corresponds to one selected split date
                (one per LoT typically) and contains `nr_samples` dictionaries, where each
                dictionary represents a full forecasting task sample
                (e.g., {'events_until_split': df, 'target_events_after_split': df, ...}).
                Returns `[[None]]` if no valid splits are found.
            If `return_splits` is True:
                A tuple: (`ret_splits`, `all_possible_split_dates_return`)
                - `ret_splits`: The list of lists of split dictionaries as described above.
                - `all_possible_split_dates_return`: A DataFrame containing the actual split
                  dates and their associated LoT dates that were successfully used to
                  generate the samples in `ret_splits`. Columns: ['date', 'lot_date'].
        """

        # Setup basics
        events = patient_data["events"]

        if override_split_dates is None:
            #: get all possible splits via _get_all_possible_splits, randomly selecting one split date per LoT
            all_possible_split_dates, all_possible_split_dates_no_vars = self._get_all_possible_splits(patient_data,
                                                                    min_nr_variable_seen_previously=self.min_nr_variable_seen_previously,
                                                                    min_nr_variable_seen_after=self.min_nr_variable_seen_after,
                                                                    list_of_valid_categories=self.list_of_valid_categories,
                                                                    subselect_random_within_lot=True,
                                                                    max_num_samples_per_lot=max_num_samples_per_lot)

            if all_possible_split_dates.shape[0] == 0:
                logging.info("No possible forecasting splits found for patient: " +
                            str(patient_data["constant"]["patientid"].iloc[0]))
                ret = [None], None if return_splits else None
                return ret



        else:
            assert override_variables_to_predict is not None, ("If you override split dates, "
                                                               "you must also override variables to predict")

            #: create all_possible_split_dates, with override_split_dates for date and nr of rows
            #: then for each row, we add: None for LoT date, and override_variables_to_predict for variables

            all_possible_split_dates = []
            for split_date in override_split_dates:
                for variable_to_predict in override_variables_to_predict:
                    all_possible_split_dates.append({
                        self.config.date_col: split_date,
                        self.config.event_name_col: variable_to_predict,
                        self.config.lot_date_col: "override"
                    })
            all_possible_split_dates = pd.DataFrame(all_possible_split_dates)
            all_possible_split_dates_no_vars = all_possible_split_dates.copy()
            all_possible_split_dates_no_vars = all_possible_split_dates_no_vars[[self.config.date_col,
                                                                                 self.config.lot_date_col]].drop_duplicates()


        #: loop through 1 to nr_samples
        all_lots_dates = all_possible_split_dates_no_vars[[self.config.date_col, self.config.lot_date_col]]

        ret_splits = []
        ret_split_dates = []

        for lot_date in all_lots_dates[self.config.lot_date_col].unique():

            all_dates_in_lot = all_lots_dates[all_lots_dates[self.config.lot_date_col] == lot_date][self.config.date_col]

            for curr_date in all_dates_in_lot:

                date_splits = []

                # Try generating date splits for current date
                (date_splits, valid_date, date_split_meta,
                 all_possible_split_dates) = self._generate_variable_splits_for_date(
                    curr_date, nr_samples,
                    override_variables_to_predict, events,
                    all_possible_split_dates, apply_filtering,
                    override_split_dates, patient_data, lot_date
                )


                # In case we didn't add any splits, due to issues with the timeline (and not invalid date),
                # then try with another date in the current lot
                # A bit hacky, and slow, but should work in case there is an option
                if len(date_splits) == 0 and valid_date:

                    # Try earlier dates (since often those have more success due to future LoTs blocking)
                    other_dates_in_lot = pd.Series(
                        all_dates_in_lot[all_dates_in_lot != curr_date]
                    ).sort_values().unique()

                    for other_date in other_dates_in_lot:

                        # Generate data from another date
                        (date_splits, valid_date, date_split_meta,
                         all_possible_split_dates) = self._generate_variable_splits_for_date(
                            other_date, nr_samples,
                            override_variables_to_predict, events,
                            all_possible_split_dates, apply_filtering,
                            override_split_dates, patient_data, lot_date
                        )

                        if len(date_splits) > 0:
                            break

                    # If nothing found, then append empty list and meta
                    if len(date_splits) == 0:
                        date_splits = []
                        date_split_meta = [{self.config.date_col: curr_date, self.config.lot_date_col: lot_date}]
                        date_split_meta = pd.DataFrame(date_split_meta)


                #: append to return_splits, so to randomly subselect which variables to use
                ret_splits.append(date_splits)
                ret_split_dates.append(date_split_meta)


        #: return list
        if return_splits:

            #: get all possible split dates from what was actually used
            all_possible_split_dates_return = pd.concat(ret_split_dates, axis=0, ignore_index=True)

            # Return
            return ret_splits, all_possible_split_dates_return
        else:
            return ret_splits
