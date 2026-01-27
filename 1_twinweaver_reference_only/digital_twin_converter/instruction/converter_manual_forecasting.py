import pandas as pd
from transformers import AutoTokenizer
from datetime import datetime

from digital_twin_converter.common.converter_manual_base import ConverterManualBase
from digital_twin_converter.common.converter_manual_base import round_and_strip
from digital_twin_converter.common.config import Config # Assuming Config class is in this path


class ConverterManualForecasting(ConverterManualBase):
    """
    Handles the conversion between structured patient data splits and
    text-based formats suitable for forecasting language models.

    This class focuses specifically on generating prompts that ask a model
    to predict future values of specific variables (e.g., lab results) at
    given future time points (weeks relative to a split date). It also
    formats the actual future data (target) into a corresponding text string
    and handles the reverse conversion from model-generated text back to a
    structured DataFrame.

    Attributes:
        constant_description (pd.DataFrame): DataFrame holding descriptions for constant variables.
        nr_tokens_budget_total (int): The target token budget for the combined input and output strings.
        forecasting_prompt_start (str): The initial text segment of the forecasting prompt.
        forecasting_prompt_var_time (str): The text segment used in the prompt to link a variable to its prediction times.
        forecasting_prompt_summarized_start (str): Starting text for summarized prompts (not used in core methods here).
        forecasting_prompt_summarized_genetic (str): Text for genetic info in summarized prompts (not used here).
        forecasting_prompt_summarized_lot (str): Text for LoT info in summarized prompts (not used here).
        tokenizer (AutoTokenizer): Tokenizer instance for calculating token counts.
        nr_tokens_budget_padding (int): Padding added to token budget calculations.
        always_keep_first_visit (bool): Flag indicating if the first visit's data should always be kept during token budget trimming.
    """

    def __init__(self,
                 constant_description : pd.DataFrame,
                 nr_tokens_budget_total : int,
                 config : Config) -> None:
        """
        Initializes the ConverterManualForecasting instance.

        Sets up the converter with necessary configurations, including descriptions
        for constant patient features, the total token budget for generated text,
        and various prompt templates defined in the Config object.

        Args:
            constant_description: DataFrame containing descriptions for constant
                patient attributes (e.g., 'Sex: Male'). Used potentially by base
                class methods for input string generation.
            nr_tokens_budget_total: The target maximum number of tokens for the
                combined input (history) and output (forecast) text.
            config: A Config object containing shared configuration settings like
                prompt templates, column names, tokenizer details, etc.
        """
        # Initialize base class with config and potential overrides
        super().__init__(config)

        # Store specific attributes for this class
        self.constant_description = constant_description
        self.nr_tokens_budget_total = nr_tokens_budget_total

        # Set forecasting prompts, using overrides or config defaults
        self.forecasting_prompt_start = self.config.forecasting_fval_prompt_start
        self.forecasting_prompt_var_time = self.config.forecasting_prompt_var_time
        self.forecasting_prompt_summarized_start = self.config.forecasting_prompt_summarized_start
        self.forecasting_prompt_summarized_genetic =self.config.forecasting_prompt_summarized_genetic
        self.forecasting_prompt_summarized_lot = self.config.forecasting_prompt_summarized_lot

        # Initialize tokenizer and budget padding
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_to_use)
        self.nr_tokens_budget_padding = self.config.nr_tokens_budget_padding
        self.always_keep_first_visit = self.config.always_keep_first_visit


    def _generate_target_string(self, patient_split : dict) -> tuple[str, dict]:
        """
        Generates the target forecast string and associated metadata.

        Takes a patient data split containing future events, processes them,
        and formats them into a structured text string representing the forecast
        target. It also computes metadata detailing which variables are forecasted
        at which future weeks relative to the split date, and includes the raw
        and processed target data.

        Args:
            patient_split: A dictionary containing the data for a single split.
                Expected keys include:
                - "target_events_after_split": DataFrame of events to be forecasted.
                - "split_date_included_in_input": The reference date marking the end
                  of the input history.
                - "events_until_split": DataFrame of historical events (used to find
                  last observed values).
                - "lot_date": The start date of the Line of Therapy associated with
                  this split (optional).

        Returns:
            A tuple containing:
            - target_str (str): The formatted string representing the target forecast
              (e.g., "[2 weeks later]\n\tLab X: 10.5\n[4 weeks later]\n\tLab Y: 2.1").
            - target_meta (dict): A dictionary containing metadata about the target,
              including: 'dates_to_forecast', 'future_weeks_to_forecast',
              'dates_per_variable', 'future_weeks_per_variable',
              'variable_name_mapping', 'target_data_raw', 'target_data_processed',
              'lot_date', 'last_observed_values' (DataFrame of last known values
              for forecasted variables from the input history).
        """

        #: preprocess:
        target_data = patient_split["target_events_after_split"].copy()
        target_cleaned = self._preprocess_events(target_data.copy())

        #: get delta between split and first target
        target_first_day = target_cleaned[self.config.date_col].min()
        split_date = patient_split["split_date_included_in_input"]
        delta_days = (target_first_day - split_date).days / 7.0

        #: convert to string using default approach
        target_str = self._get_event_string(target_cleaned,
                                            events_delta_0 = delta_days,
                                            use_accumulative_dates = False,
                                            add_first_day_preamble = False)

        #: get meta, i.e. which dates to forecast and the relative future weeks grouped by variable
        dates_per_variable = target_cleaned.groupby(self.config.event_name_col)[self.config.date_col].unique()
        future_weeks_per_variable = {}
        variable_name_mapping = {}
        for variable, dates in dates_per_variable.items():

            # Get weeks to predict (relative to split date)
            # Ensure dates is a pd.Series or pd.DatetimeIndex for subtraction
            if not isinstance(dates, (pd.Series, pd.DatetimeIndex)):
                dates = pd.to_datetime(dates)
            future_weeks_per_variable[variable] = (dates - split_date).days / 7.0

            # Get descriptive name
            curr_var = target_cleaned[target_cleaned[self.config.event_name_col] == variable]
            # Handle case where variable might not be found or has no descriptive name
            if not curr_var.empty and self.config.event_descriptive_name_col in curr_var.columns:
                 descriptive_name = curr_var[self.config.event_descriptive_name_col].iloc[0]
            else:
                 descriptive_name = variable # Fallback to variable name itself
            variable_name_mapping[variable] = descriptive_name


        dates_to_forecast = target_cleaned[self.config.date_col].unique()
        # Ensure dates_to_forecast is pd.DatetimeIndex for subtraction
        if not isinstance(dates_to_forecast, pd.DatetimeIndex):
            dates_to_forecast = pd.to_datetime(dates_to_forecast)
        future_weeks_to_forecast = (dates_to_forecast - split_date).days / 7.0

        #: add last observed values of each variable from input history
        input_history = patient_split["events_until_split"]
        last_observed_values_dict = {}
        for variable in future_weeks_per_variable.keys():

            # Check if variable is in input history by event_name
            matches = input_history[input_history[self.config.event_name_col] == variable]
            if not matches.empty:
                last_observed = matches.sort_values(by=self.config.date_col).iloc[-1]
            else:
                 # Fallback to looking at descriptive name if event_name not found
                 descriptive_name_to_match = variable_name_mapping.get(variable, variable)
                 matches = input_history[input_history[self.config.event_descriptive_name_col] == descriptive_name_to_match]
                 if not matches.empty:
                      last_observed = matches.sort_values(by=self.config.date_col).iloc[-1]
                 else:
                      last_observed = None # Variable not found in history


            #: add to dict if found
            if last_observed is not None:
                last_observed_values_dict[variable] = last_observed

        # Convert dict of Series to DataFrame
        if last_observed_values_dict:
            last_observed_values = pd.DataFrame(last_observed_values_dict).T
        else:
             # Create an empty DataFrame with expected columns if nothing was observed
             expected_cols = [self.config.date_col, self.config.event_category_col,
                              self.config.event_name_col, self.config.event_descriptive_name_col,
                              self.config.event_value_col, self.config.source_col]
             last_observed_values = pd.DataFrame(columns=expected_cols)


        #: setup return metadata dictionary
        target_meta = {
            "dates_to_forecast": dates_to_forecast,
            "future_weeks_to_forecast": future_weeks_to_forecast,
            "dates_per_variable": dates_per_variable,
            "future_weeks_per_variable": future_weeks_per_variable,
            "variable_name_mapping": variable_name_mapping,
            "target_data_raw": target_data,
            "target_data_processed": target_cleaned,
            "lot_date": patient_split.get("lot_date"), # Use .get for safety
            "last_observed_values": last_observed_values,
        }

        return target_str, target_meta


    def _generate_prompt(self, target_meta : dict) -> str:
        """
        Generates the forecasting prompt based on target metadata.

        Constructs the text prompt that instructs the language model what to
        forecast. It uses the metadata about which variables need prediction
        at which future weeks.

        Args:
            target_meta: A dictionary containing metadata about the target forecast,
                as generated by `_generate_target_string`. Key required elements are:
                - "future_weeks_per_variable": Dict mapping variable names to lists
                  of future weeks (relative to split date) they should be predicted at.
                - "variable_name_mapping": Dict mapping internal variable names to
                  their descriptive names for use in the prompt.

        Returns:
            The fully constructed prompt string asking the model to perform the
            specified forecast.
        """

        # start ret
        ret_prompt = ""

        #: make sure to round weeks from future_weeks_per_variable to predict to specified precision
        future_weeks_per_variable = target_meta["future_weeks_per_variable"]
        future_weeks_per_variable_rounded = {}
        for k, v in future_weeks_per_variable.items():
            # Check if v is iterable, handle potential scalar values if necessary
            if hasattr(v, '__iter__'):
                 future_weeks_per_variable_rounded[k] = [round_and_strip(v2, self.decimal_precision) for v2 in v]
            else:
                 # Handle non-iterable case, maybe log a warning or convert to list
                 future_weeks_per_variable_rounded[k] = [round_and_strip(v, self.decimal_precision)]


        #: add base prompt start text
        ret_prompt += self.forecasting_prompt_start

        #: sort alphabetically by variable name for consistent prompt order
        future_weeks_per_variable_sorted = dict(sorted(future_weeks_per_variable_rounded.items(), key=lambda item: item[0]))

        #: create prompt for which variables to predict and when
        for variable, weeks in future_weeks_per_variable_sorted.items():

            #: need event_descriptive_name from mapping
            variable_desc_name = target_meta["variable_name_mapping"].get(variable, variable) # Fallback to variable name

            # Create prompt line for this variable
            ret_prompt += "\n" + "\t" + variable_desc_name
            ret_prompt += self.forecasting_prompt_var_time
            # Format the list of weeks nicely
            ret_prompt += ', '.join(map(str, weeks)) # Use map for clean conversion to string

        #: return the fully constructed prompt
        return ret_prompt


    def forward_conversion(self, patient_split : dict) -> tuple[str, str, dict]:
        """
        Performs the primary conversion from a data split to prompt and target strings.

        This method orchestrates the generation of the target string (what the model
        should predict) and the corresponding prompt string (the instruction asking
        for the prediction) based on a patient data split.

        Note: This method generates the *prompt* and *target* strings. The generation
        of the *input* string (patient history) is typically handled separately, often
        by the base class's `_get_input_string` method, using the "events_until_split"
        and "constant_data" from the `patient_split`.

        Args:
            patient_split: A dictionary containing the patient data split, including
                input events ("events_until_split"), target events
                ("target_events_after_split"), the split date
                ("split_date_included_in_input"), and potentially constant data
                and LoT date.

        Returns:
            A tuple containing:
            - prompt_str (str): The generated prompt instructing the model on the
              forecasting task.
            - target_str (str): The formatted string representing the target forecast values.
            - target_meta (dict): Metadata associated with the generated target string
              (output from `_generate_target_string`).
        """

        #: generate target string and associated metadata
        target_str, target_meta = self._generate_target_string(patient_split)

        #: generate prompt string based on the target metadata
        prompt_str = self._generate_prompt(target_meta)

        # Note: The input string (patient history text) generation is handled by the base class's
        # `_get_input_string` method, which would typically be called elsewhere
        # in the pipeline that uses this forward_conversion result along with the input events.
        # This method focuses on generating the *prompt* and *target* based on the split.

        return prompt_str, target_str, target_meta


    def forward_conversion_inference(self,
                                     patient_split : dict,
                                     future_weeks_per_variable : dict) -> tuple[str, dict]:
        """
        Generates only the prompt string for inference scenarios.

        Used when the goal is to get a prediction from the model, not for training.
        It takes the patient's historical data and a specific set of variables and
        future weeks to predict, then constructs the appropriate prompt string.
        It does not generate a target string, as the target is what the model
        is expected to produce.

        Args:
            patient_split: Dictionary containing patient split data, primarily requiring
                "events_until_split" (for context and variable name lookups) and
                "split_date_included_in_input" (the reference time point).
            future_weeks_per_variable: Dictionary explicitly defining the forecasting task.
                Format: {<variable_name>: [<future_week_1>, <future_week_2>, ...], ...}
                These weeks are relative to the `split_date_included_in_input`.

        Returns:
            A tuple containing:
            - prompt_str (str): The generated prompt for the inference task.
            - target_pseudo_meta (dict): A dictionary containing the metadata constructed
              specifically for generating this inference prompt (includes the provided
              `future_weeks_per_variable`, derived `dates_per_variable`, and looked-up
              `variable_name_mapping`).
        """

        #: generate target_pseudo meta data needed for prompt generation
        target_pseudo_meta = {}

        #: Use the provided future_weeks_per_variable directly
        target_pseudo_meta["future_weeks_per_variable"] = future_weeks_per_variable

        # Make dates per variable based on the provided weeks and split date
        dates_per_variable = {}
        split_date = patient_split["split_date_included_in_input"]
        for variable, weeks in future_weeks_per_variable.items():
             # Ensure weeks is iterable
             if not hasattr(weeks, '__iter__'):
                 weeks = [weeks]
             dates_per_variable[variable] = [split_date + pd.Timedelta(days=float(w)*7) for w in weeks]
        target_pseudo_meta["dates_per_variable"] = dates_per_variable

        #: make target_meta["variable_name_mapping"] by looking up in input events
        variable_name_mapping = {}
        input_events = patient_split["events_until_split"]
        for variable in future_weeks_per_variable.keys():
             # Get descriptive name from input history
             curr_var = input_events[input_events[self.config.event_name_col] == variable]
             if not curr_var.empty and self.config.event_descriptive_name_col in curr_var.columns:
                  descriptive_name = curr_var[self.config.event_descriptive_name_col].iloc[0]
             else:
                 # Fallback: If not found by event_name, try descriptive name directly,
                 # or just use the variable name itself. This part might need refinement
                 # depending on how variables are guaranteed to be present/identifiable.
                 descriptive_name = variable # Simple fallback
             variable_name_mapping[variable] = descriptive_name

        target_pseudo_meta["variable_name_mapping"] = variable_name_mapping

        #: generate prompt (including when to generate what) based on constructed metadata
        prompt_str = self._generate_prompt(target_pseudo_meta)

        return prompt_str, target_pseudo_meta

    def generate_target_manual(self, target_events_after_split : pd.DataFrame,
                               split_date_included_in_input : datetime,
                               events_until_split : pd.DataFrame,
                               lot_date : datetime = None): # Added optional lot_date param
        """
        Manually generates the target string and metadata from explicit components.

        Provides an alternative way to generate the target string and metadata
        by directly passing the necessary DataFrames and the split date, rather
        than relying on a pre-packaged `patient_split` dictionary. This might be
        useful in scenarios where the data components are sourced differently.

        Args:
            target_events_after_split: DataFrame containing the future events
                that constitute the target forecast.
            split_date_included_in_input: The reference date marking the end of
                the input history and the start of the forecast period.
            events_until_split: DataFrame containing the historical events up to
                the `split_date_included_in_input`. Used to find the last observed
                values for context in the metadata.
            lot_date: The start date of the relevant Line of Therapy, if applicable.
                Defaults to None.

        Returns:
            A tuple containing:
            - target_str (str): The formatted string representing the target forecast.
            - target_meta (dict): Metadata associated with the generated target string
              (same structure as output from `_generate_target_string`).
        """
        # Construct the dictionary expected by _generate_target_string
        patient_dic = {
            "target_events_after_split": target_events_after_split,
            "split_date_included_in_input": split_date_included_in_input,
            "events_until_split": events_until_split,
            "lot_date": lot_date, # Pass the provided lot_date
        }
        # Call the internal method to generate target string and metadata
        return self._generate_target_string(patient_dic)


    def aggregate_multiple_responses(self, responses_dfs : list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
        """
        Aggregates structured data from multiple model responses (e.g., trajectories).

        Takes a list of DataFrames, where each DataFrame represents the structured
        output converted from a single model response (e.g., using `reverse_conversion`).
        It concatenates these DataFrames and calculates the mean of numeric event
        values, grouping by date and event identifiers. This is useful for combining
        results from multiple stochastic samples from a model.

        Args:
            responses_dfs: A list of DataFrames, each converted from a separate
                model forecast output string. Expected to have consistent columns
                as defined in the config (date, event name, value, etc.).

        Returns:
            A tuple containing:
            - resulting_df (pd.DataFrame): A DataFrame containing the aggregated
              results, typically with the event values averaged per date/event.
            - meta (dict): A dictionary containing metadata, currently includes
              "all_trajectory_data" which holds the original list of input DataFrames.
        """

        #: concat all response DataFrames together
        if not responses_dfs: # Handle empty list case
             return pd.DataFrame(), {"all_trajectory_data": []}

        concatenated = pd.concat(responses_dfs, axis=0, ignore_index=True)

        # Convert event value column to numeric, coercing errors to NaN
        concatenated[self.config.event_value_col] = pd.to_numeric(concatenated[self.config.event_value_col], errors='coerce')

        #: get average of values per date/event identifier, keeping relevant columns
        # Define columns to group by
        grouping_cols = [self.config.date_col, self.config.event_name_col,
                         self.config.event_descriptive_name_col,
                         self.config.event_category_col, self.config.patient_id_col,
                         self.config.source_col]

        # Ensure all grouping columns exist, handle potential missing ones gracefully
        valid_grouping_cols = [col for col in grouping_cols if col in concatenated.columns]
        if not valid_grouping_cols:
             # Cannot group if no valid columns are present
             # Consider logging a warning or raising an error
             return pd.DataFrame(), {"all_trajectory_data": responses_dfs.copy()}


        # Perform the groupby and aggregation
        # Use observed=True for potentially better performance with categorical data
        resulting = concatenated.groupby(valid_grouping_cols, observed=True)[self.config.event_value_col].mean().reset_index()

        #: prepare metadata dictionary
        meta = {
            "all_trajectory_data": responses_dfs.copy(), # Keep original list
        }
        return resulting, meta




    def reverse_conversion(self,
                           text_to_convert: str,
                           unique_events: pd.DataFrame,
                           split_date : datetime) -> pd.DataFrame:
        """
        Converts a formatted forecast text string back into a structured DataFrame.

        Parses a text string (assumed to be generated by a forecasting model in
        a specific format, e.g., using "[X] weeks later..." markers) and extracts
        the forecasted event data. It uses the `split_date` as the reference point
        for calculating absolute dates from relative week offsets in the text.
        Relies on the base class's `_extract_event_data` method for the core parsing logic.

        Args:
            text_to_convert: The text string generated by the model containing the forecast.
            unique_events: A DataFrame defining known event types (names, categories)
                to help with parsing and validation within the base class method.
            split_date: The date relative to which the week offsets (e.g., "[X] weeks later")
                in the `text_to_convert` should be interpreted.

        Returns:
            A DataFrame containing the structured event data parsed from the input text,
            sorted by date, category, and event name for consistency.
        """

        #: Prepend the event day preamble if it's missing at the start, as the base
        #  method might expect it for splitting days/visits.
        if self.event_day_preamble and not text_to_convert.startswith(self.event_day_preamble):
            text_to_convert = self.event_day_preamble + text_to_convert

        #: Call the base class's extraction method. Assuming the forecast output format
        #  is compatible with the general event extraction logic (e.g., uses the
        #  same "[X] weeks later..." structure). Set `only_contains_events=True`
        #  as we don't expect demographic data in the forecast output.
        converted_data = self._extract_event_data(text_to_convert,
                                                  unique_events,
                                                  init_date = split_date,
                                                  only_contains_events = True)

        #: sort for consistency based on config columns
        sort_columns = [col for col in [self.config.date_col, self.config.event_category_col, self.config.event_name_col] if col in converted_data.columns]
        if sort_columns:
            converted_data = converted_data.sort_values(by=sort_columns).reset_index(drop=True)

        return converted_data
