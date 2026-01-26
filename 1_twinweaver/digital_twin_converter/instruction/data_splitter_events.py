import numpy as np
import pandas as pd

from digital_twin_converter.instruction.data_splitter import BaseDataSplitter
from digital_twin_converter.common.data_manager import SingleIndicationDataManager
from digital_twin_converter.common.config import Config





class DataSplitterEvents(BaseDataSplitter):


    def __init__(self,
                 data_manager: SingleIndicationDataManager,
                config: Config,
                 max_length_of_weeks_to_sample : int,
                 max_length_days_after_lot : int = 90,
                 max_lookback_for_value : int = 90,
                 max_forecast_for_value : int = 90,
                 manual_variables_category_mapping : dict = None,):
        """
                Initialize the DataSplitterEvents class.

        Parameters
        ----------
        data_manager : SingleIndicationDataManager
            The data manager to handle data operations.
        config : Config
            Configuration object holding constants.
        max_length_of_weeks_to_sample : int
            The maximum number of weeks into the future to sample for event prediction.
        max_length_days_after_lot : int, optional
            The maximum number of days after the line of therapy to consider for split points.
        max_lookback_for_value : int, optional
            The maximum number of days to look back for a value (inherited but not directly used here).
        max_forecast_for_value : int, optional
            The maximum number of days to forecast a value (inherited but not directly used here).
        manual_variables_category_mapping : dict, optional
            A dictionary mapping event categories to descriptive names. Overrides defaults.
            Since events operates on category level, this provides the naming for the events.
        """
        super().__init__(data_manager, config, max_length_days_after_lot, max_lookback_for_value, max_forecast_for_value)
        self.max_length_of_weeks_to_sample = max_length_of_weeks_to_sample


        if manual_variables_category_mapping is None:
            # Use config constants for keys where available
            self.manual_variables_category_mapping = {
                self.config.event_category_death : "death",
                "progression" : "next progression", # "progression" not in provided config
                self.config.event_category_lot : "next line of therapy",
                "metastasis" : "next metastasis", # "metastasis" not in provided config
            }
        else:
            self.manual_variables_category_mapping = manual_variables_category_mapping



    def setup_variables(self):
        """
        Setup the variables to be used for sampling.
        """

        #: get all categories in indication
        all_categories = self.dm.data_frames["events"][self.config.event_category_col].unique().tolist()

        #: first look at the manual variables
        self.manual_variables_category_mapping = {x : self.manual_variables_category_mapping[x]
                                                  for x in self.manual_variables_category_mapping.keys()
                                                  if x in all_categories}


    def _sample_manual_variables(self, events_after_split : pd.DataFrame, override_category : str) -> tuple:
        """
        Sample manual variables from the events occurring after the split date.

        Parameters
        ----------
        events_after_split : pd.DataFrame
            The dataframe containing events that occur after the split date.

        Returns
        -------
        tuple
            A tuple containing the sampled variable, the category of the sampled variable,
            and the descriptive name of the sampled variable.
        """

        if override_category is None:
            #: we need to uniformly sample the exact variable based on category
            category = np.random.choice(list(self.manual_variables_category_mapping.keys()))
        else:
            category = override_category

        # Also return the descriptive name based on category
        next_var_descriptive = self.manual_variables_category_mapping[category]

        #: in case of progression, try alternatively death, since it is also a progression evne
        if category not in events_after_split[self.config.event_category_col].unique():
            if category == "progression":
                if "death" in events_after_split[self.config.event_category_col].unique():
                    category = "death"

        #: return exact variable
        return category, next_var_descriptive




    def get_splits_from_patient(self,
                                patient_data : dict,
                                max_nr_samples : int,
                                preselected_split_dates : pd.DataFrame = None,
                                override_split_dates : list = None,
                                override_category : str = None,
                                override_end_week_delta : int = None,
                                 max_num_samples_per_lot : int = 1) -> list:
        """
        Generates event prediction tasks (splits) for a given patient.

        For each unique Line of Therapy (LoT) start date in the patient's history,
        this function potentially selects one or more random split points within a defined
        window after the LoT start (`max_length_days_after_lot`). The number of
        split points selected per LoT is controlled by `max_num_samples_per_lot`.

        If `preselected_split_dates` (typically generated by a parallel forecasting
        splitter for consistency) is provided, those exact split dates are used instead
        of random sampling based on LoT. If `override_split_dates` is provided
        (e.g., for inference), those specific dates are used. Only one of
        `preselected_split_dates` or `override_split_dates` can be used.

        For each chosen split date (`curr_date`), this method generates multiple event
        prediction tasks (up to `max_nr_samples`). Each task involves predicting
        a specific event category (e.g., 'death', 'next line of therapy') within a
        randomly determined future time window (`end_week_delta`, up to
        `max_length_of_weeks_to_sample`). The function handles censoring based on
        subsequent events (like next LoT start or death) or end of available data.

        Parameters
        ----------
        patient_data : dict
            A dictionary containing the patient's data. Expected keys:
            'events': pd.DataFrame with patient event history, including columns defined
                      in `self.config` (e.g., date, event category, LoT date).
            'constant': pd.DataFrame with static patient information.
        max_nr_samples : int
            The maximum number of distinct event prediction tasks (different event
            categories or prediction windows) to generate for *each* selected split date.
            The actual number might be less if fewer unique categories are available
            to sample after the split date.
        preselected_split_dates : pd.DataFrame, optional
            A DataFrame containing specific split dates to use, typically generated by
            another data splitter (e.g., DataSplitterForecasting) to ensure alignment
            between different task types. Must contain the columns specified in
            `self.config.date_col` and `self.config.lot_date_col`. If provided,
            `override_split_dates` must be None. Defaults to None.
        override_split_dates : list, optional
            A list of specific datetime objects to use as split dates, typically for
            inference scenarios. If provided, `preselected_split_dates` must be None.
            Defaults to None.
        override_category : str, optional
            If provided, forces the sampling process to only consider this specific
            event category for prediction, instead of randomly sampling from available
            categories. Defaults to None.
        override_end_week_delta : int, optional
            If provided, forces the prediction window to be exactly this many weeks,
            instead of randomly sampling a window duration. Defaults to None.
        max_num_samples_per_lot: int, optional
            When split dates are *not* overridden, this determines the maximum number
            of random split dates to select per unique LoT start date during the
            initial candidate selection. Defaults to 1.

        Returns
        -------
        list[list[dict]]
            A list where each element corresponds to one of the selected split dates.
            Each element is itself a list containing multiple dictionaries (up to
            `max_nr_samples`). Each dictionary represents a single event prediction
            task (split) and contains:
                - 'events_until_split': pd.DataFrame of events up to the split date.
                - 'constant_data': pd.DataFrame of constant patient data.
                - 'event_occured': bool indicating if the sampled event occurred within
                                   the prediction window before censoring.
                - 'event_censored': str or None, indicating the reason for censoring
                                    ('lot', 'end') or None if not censored.
                - 'date_event_occured': datetime, the date the event occurred or the
                                        end date of the prediction window if it didn't.
                - 'split_date_included_in_input': datetime, the split date used.
                - 'sampled_category': str, the event category being predicted.
                - 'sampled_variable_name': str, descriptive name for the category.
                - 'end_date': datetime, the end of the prediction window.
                - 'lot_date': datetime or pd.NA, the LoT start date associated with
                              this split point.

        Raises
        ------
        ValueError
            If both `preselected_split_dates` and `override_split_dates` are provided.
            If required columns are missing in `patient_data['events']`.
        AssertionError
            If internal checks fail, e.g., when using `preselected_split_dates` and
            consistency checks with potential dates fail.
        TypeError
            If input arguments have incorrect types.
        """

        # --- Assertions ---

        # Input Type Assertions
        assert isinstance(patient_data, dict), "patient_data must be a dictionary."
        assert isinstance(max_nr_samples, int) and max_nr_samples > 0, "max_nr_samples must be a positive integer."
        assert isinstance(max_num_samples_per_lot, int) and max_num_samples_per_lot > 0, "max_num_samples_per_lot must be a positive integer."
        assert preselected_split_dates is None or isinstance(preselected_split_dates, pd.DataFrame), "preselected_split_dates must be None or a pandas DataFrame."
        assert override_split_dates is None or isinstance(override_split_dates, list), "override_split_dates must be None or a list."
        assert override_category is None or isinstance(override_category, str), "override_category must be None or a string."
        assert override_end_week_delta is None or isinstance(override_end_week_delta, int), "override_end_week_delta must be None or an integer."

        # Input Data Structure and Content Assertions
        assert "events" in patient_data, "patient_data dictionary must contain the key 'events'."
        assert "constant" in patient_data, "patient_data dictionary must contain the key 'constant'."
        assert isinstance(patient_data["events"], pd.DataFrame), "patient_data['events'] must be a pandas DataFrame."
        assert isinstance(patient_data["constant"], pd.DataFrame), "patient_data['constant'] must be a pandas DataFrame."

        # Check for required columns in the events dataframe
        required_event_cols = [self.config.date_col, self.config.event_category_col, self.config.event_name_col]
        missing_cols = [col for col in required_event_cols if col not in patient_data["events"].columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in patient_data['events']: {missing_cols}")

        # Mutual Exclusivity Assertion for Split Date Sources
        assert preselected_split_dates is None or override_split_dates is None, \
            "Cannot provide both preselected_split_dates and override_split_dates."

        #: get all possible splits
        events = patient_data["events"]
        events = events.sort_values(self.config.date_col)

        #: get unique dates, if needed
        pot_all_possible_split_dates = self._get_all_dates_within_range_of_lot(patient_data,
                                                                               0,
                                                                               self.max_length_days_after_lot)
        pot_all_possible_split_dates = self.select_random_splits_within_lot(pot_all_possible_split_dates,
                                                                      max_num_samples_per_lot=max_num_samples_per_lot)

        if preselected_split_dates is None and override_split_dates is None:
            all_possible_split_dates = pot_all_possible_split_dates

        elif preselected_split_dates is not None:

            # Set to the preselected split dates, and do some assertions
            all_possible_split_dates = preselected_split_dates.copy()
            all_possible_split_dates = all_possible_split_dates.reset_index(drop=True)
            assert all_possible_split_dates.shape[0] == pot_all_possible_split_dates.shape[0], "# rows don't match"
            assert (set(all_possible_split_dates[self.config.lot_date_col].unique()) ==
                    set(pot_all_possible_split_dates[self.config.lot_date_col].unique())), "The unique LoT dates don't match"
            assert all_possible_split_dates[self.config.date_col].isna().sum() == 0, "Still missing dates"


        elif override_split_dates is not None:

            # If we're overriding the split dates, then we need to create a new dataframe
            all_possible_split_dates = pd.DataFrame({
                self.config.date_col : override_split_dates,
                self.config.lot_date_col : [pd.NA] * len(override_split_dates)
            })

        else:
            raise ValueError("Invalid split dates provided")


        ret_splits = []

        for curr_sample_index in range(len(all_possible_split_dates)):

            #: get current data
            curr_date, lot_date = all_possible_split_dates.iloc[curr_sample_index, :].tolist()

            #: get the input & output data
            events_before_split = events[events[self.config.date_col] <= curr_date]
            events_after_split = events[events[self.config.date_col] > curr_date]

            prev_sampled_category = []
            ret_split_lot = []

            #: loop through 1 to max_nr_samples
            for _ in range(max_nr_samples):

                #: sample variables
                sampled_cateogry, sampled_var_name = self._sample_manual_variables(events_after_split,
                                                                                   override_category)

                #: check if we sampled the same category as before
                if sampled_cateogry in prev_sampled_category:
                    continue
                prev_sampled_category.append(sampled_cateogry)

                # Determine how many weeks to predict into the future
                if override_end_week_delta is None:
                    #: randomly sample end date -> so that we also get random values in between for consistency
                    # This is so that the model can learn different time values for the same variable
                    #: To not bias the model, we select a random nr of weeks as max end date
                    end_week_delta = np.random.randint(1, self.max_length_of_weeks_to_sample + 1)
                else:
                    end_week_delta = override_end_week_delta

                # Process the actual end date
                end_date = curr_date + pd.Timedelta(days=end_week_delta * 7)
                end_date = max(end_date, events_after_split[self.config.date_col].min())
                end_date_within_data = end_date <= events[self.config.date_col].max()
                events_limited_after_split = events_after_split[events_after_split[self.config.date_col] <= end_date]

                # Get the events
                diagnosis_after_split = events_limited_after_split[events_limited_after_split[self.config.event_category_col] ==
                                                                sampled_cateogry]
                lot_after_split = events_limited_after_split[events_limited_after_split[self.config.event_category_col] ==
                                                            self.config.event_category_lot]
                death_after_split = events_limited_after_split[events_limited_after_split[self.config.event_name_col] ==
                                                                self.config.event_category_death]

                #: apply censoring using next_lot_date
                next_lot_date = lot_after_split[self.config.date_col].min() if len(lot_after_split) > 0 else None
                next_death_date = death_after_split[self.config.date_col].min() if len(death_after_split) > 0 else None

                #: determine whether occured, censored & if so, which date
                occurred = None
                censored = None
                date_occured = end_date


                if diagnosis_after_split.shape[0] > 0:

                    # Event occured within end date
                    occurred = True

                    # If an lot occurred first though, then we're censored
                    if next_lot_date is not None and diagnosis_after_split[self.config.date_col].min() > next_lot_date:
                        censored = "lot"
                        occurred = False

                else:
                    # Event did not occur
                    occurred = False

                    if next_lot_date is not None:
                        # If we were censored by the next lot date
                        censored = "lot"

                    elif next_death_date is not None:
                        # If death occured then not censored, since this is the only time we
                        # actually know event didn't occur
                        # In case we're sampling for death var, and it occured, then it wouldn't trigger this logic
                        censored = None

                    elif end_date_within_data:
                        # Event did not occur within the given time frame
                        censored = None

                    else:
                        # If we were censored by the end of the data, but not death
                        censored = "end"

                # Check for data cutoff as a final safeguard
                if censored is None and occurred is False and end_date > self.config.date_cutoff:

                    # Check if outside of date cutoff
                    # if occurred is False and not censored, then we know event didn't occur in the mean time
                    occurred = False
                    censored = "data_cutoff"

                #: add to return list
                ret_split_lot.append({
                    "events_until_split": events_before_split,
                    "constant_data" : patient_data["constant"].copy(),
                    "event_occured": occurred,
                    "event_censored": censored,
                    "date_event_occured": date_occured,
                    "split_date_included_in_input": curr_date,
                    "sampled_category": sampled_cateogry,
                    "sampled_variable_name": sampled_var_name,
                    "end_date": end_date,
                    "lot_date": lot_date,
                })

            # Add for current LoT possible splits
            ret_splits.append(ret_split_lot)

        #: return list
        return ret_splits
