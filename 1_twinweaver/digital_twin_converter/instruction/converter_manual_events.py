import pandas as pd
from transformers import AutoTokenizer
from collections import Counter

from digital_twin_converter.common.converter_manual_base import ConverterManualBase
from digital_twin_converter.common.converter_manual_base import round_and_strip
from digital_twin_converter.common.config import Config  # Assuming Config is imported

class ConverterManualEvents(ConverterManualBase):
    """
    Manages the conversion between structured patient event data and formatted
    strings suitable for Time-To-Event (TTE) forecasting tasks with language models.

    This class specializes `ConverterManualBase` to handle event-based forecasting.
    It uses specific prompt templates defined in a `Config` object to generate
    input prompts (conditioning on a time duration and event) and target strings
    (describing event occurrence and censoring status). It also provides methods
    for reverse conversion (parsing model output strings back to structured data)
    and utility functions for comparing and aggregating potentially noisy model outputs.
    """

    def __init__(self,
                 config : Config,
                 constant_description : pd.DataFrame,
                 nr_tokens_budget_total : int) -> None:
        """
        Initializes the ConverterManualEvents class.

        Sets up the converter with configuration, constant descriptions, token budget,
        and initializes the tokenizer and specific prompt templates for TTE forecasting
        tasks using values from the provided Config object.

        Parameters
        ----------
        config : Config
            Configuration object containing settings like tokenizer name, prompt templates
            (e.g., `forecasting_tte_prompt_start`, `target_prompt_start`), token budget padding, etc.
        constant_description : pd.DataFrame
            DataFrame containing descriptions for constant patient features (potentially used
            in base class or future extensions, currently unused in this subclass's methods).
        nr_tokens_budget_total : int
            Total number of tokens budgeted for the input sequence (prompt + context).
            Used potentially in conjunction with padding settings from config.
        """

        super().__init__(config)

        self.constant_description = constant_description
        self.nr_tokens_budget_total = nr_tokens_budget_total

        # Use config defaults if overrides are None
        self.forecasting_prompt_start = self.config.forecasting_tte_prompt_start
        self.forecasting_prompt_mid = self.config.forecasting_tte_prompt_mid
        self.forecasting_prompt_end = self.config.forecasting_tte_prompt_end
        self.forecasting_prompt_summarized_start = self.config.forecasting_prompt_summarized_start
        self.forecasting_prompt_summarized_genetic = self.config.forecasting_prompt_summarized_genetic
        self.forecasting_prompt_summarized_lot = self.config.forecasting_prompt_summarized_lot

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_to_use)
        self.nr_tokens_budget_padding = self.config.nr_tokens_budget_padding
        self.always_keep_first_visit = self.config.always_keep_first_visit


    def _generate_target_string(self, patient_split : dict) -> tuple:
        """
        Generates the target output string and associated metadata for a TTE task.

        Constructs a string describing the outcome of the event being predicted,
        including whether it was censored and whether it occurred, based on predefined
        templates from the config object (e.g., `config.target_prompt_start`). Also
        compiles metadata about the target outcome.

        Parameters
        ----------
        patient_split : dict
            A dictionary containing information about the sampled event outcome for
            a patient, requiring keys like 'sampled_variable_name', 'event_censored',
            'event_occured', 'split_date_included_in_input', 'date_event_occured',
            'sampled_category'.

        Returns
        -------
        tuple
            - str: The formatted target string (e.g., "Outcome for event (Event A): Not censored. Event occurred.\n").
            - dict: Metadata dictionary containing details like the raw target string,
                    censoring status (boolean and detail), occurrence status (boolean),
                    target event name/category, relevant dates, and a small DataFrame
                    summarizing the key outcome components ('censoring', 'occurred', 'target_name').
        """

        # This is structured this way to minimize bias
        # 1. Censoring
        # 2. Event occured
        # This way we can condition the LLM for different scenarios

        #: setup base prompt using config
        ret_prompt = self.config.target_prompt_start.format(event_name=patient_split["sampled_variable_name"])

        #: add censoring using config
        censoring = patient_split["event_censored"]
        if censoring is not None:
            ret_prompt += self.config.target_prompt_censor_true
        else:
            ret_prompt += self.config.target_prompt_censor_false

        #: add whether occured or not using config
        ret_prompt += self.config.target_prompt_before_occur
        event_occur = patient_split["event_occured"]
        if event_occur is True:
            ret_prompt += self.config.target_prompt_occur
        else:
            ret_prompt += self.config.target_prompt_not_occur
        ret_prompt += "\n"

        #: make meta
        target_meta = {
            "target_string": ret_prompt,
            "censoring_detail" : censoring,
            "censoring": censoring is not None,
            "occurred" : event_occur,
            "split_date_included_in_input" : patient_split["split_date_included_in_input"],
            "date_event_occured" : patient_split["date_event_occured"],
            "target_category": patient_split["sampled_category"],
            "target_name": patient_split["sampled_variable_name"],
        }

        #: make it as dataframe
        # Use string constants for column names here as they define the output structure
        target_meta["target_data_processed"] = pd.DataFrame([target_meta])[["censoring", "occurred", "target_name"]]

        #: return
        return ret_prompt, target_meta


    def _generate_prompt(self, patient_split : dict) -> tuple:
        """
        Generates the input prompt string for a TTE forecasting task.

        Constructs a prompt asking the language model to predict the time until a
        specific event occurs. It calculates the time difference between the patient's
        split date (last date included in input) and the actual event date, converts
        it to weeks, rounds it, and formats it into the prompt string using templates
        from the config (e.g., `self.forecasting_prompt_start`).

        Parameters
        ----------
        patient_split : dict
            A dictionary containing information about the sampled event outcome for
            a patient, requiring keys like 'sampled_variable_name', 'date_event_occured',
            and 'split_date_included_in_input'.

        Returns
        -------
        tuple
            - str: The formatted prompt string (e.g., "Predict the time in weeks until event Event A occurs: 12.3 weeks. Input data:\n").
            - float: The calculated time difference in weeks (numeric, before rounding/formatting).
        """

        #: Get event name descriptive
        curr_event_name = patient_split["sampled_variable_name"]

        #: get delta in time in weeks, rounded using round_and_strip
        delta_time_numeric = patient_split["date_event_occured"] - patient_split["split_date_included_in_input"]
        delta_time_numeric = delta_time_numeric.days / 7
        delta_time = round_and_strip(delta_time_numeric, self.decimal_precision)

        #: construct prompt using config attributes accessed via self
        ret_prompt = self.forecasting_prompt_start + str(delta_time)
        ret_prompt += self.forecasting_prompt_mid + curr_event_name
        ret_prompt += self.forecasting_prompt_end

        #: return
        return ret_prompt, delta_time_numeric


    def forward_conversion(self, patient_split : dict) -> dict:
        """
        Performs the complete forward conversion from structured patient data to prompt/target strings.

        This method orchestrates the generation of both the input prompt and the target
        output string for a given patient's event prediction scenario, using the
        `_generate_prompt` and `_generate_target_string` helper methods. It combines
        the outputs and associated metadata.

        Parameters
        ----------
        patient_split : dict
            A dictionary containing the structured data for a patient's sampled
            event outcome scenario (required keys depend on helper methods).

        Returns
        -------
        tuple
             - str: The generated input prompt string.
             - str: The generated target output string.
             - dict: A metadata dictionary containing combined information from
                     prompt and target generation (including numeric time delta,
                     target details, etc.).
        """

        #: generate target string
        target_str, target_meta = self._generate_target_string(patient_split)

        #: generate prompt (including when to generate what)
        prompt_str, delta_time_numeric = self._generate_prompt(patient_split)
        target_meta["delta_time_numeric"] = delta_time_numeric

        # Return prompt_str, target_str, target_meta (as per function signature hint)
        return prompt_str, target_str, target_meta


    def forward_conversion_inference(self,
                                     patient_split : dict) -> dict:
        """
        Performs forward conversion suitable for inference time.

        Generates only the input prompt string and associated metadata, omitting the
        target string generation. This is useful when preparing input for a model
        prediction task where the target is unknown or not needed.

        Parameters
        ----------
        patient_split : dict
            A dictionary containing the structured data for a patient's scenario
            (required keys depend on `_generate_prompt`).

        Returns
        -------
        tuple
            - str: The generated input prompt string.
            - dict: The metadata dictionary associated with the prompt generation
                    (contains numeric time delta, target name/category from input, etc.).
        """
        prompt, _, meta = self.forward_conversion(patient_split)
        # Return prompt, meta (as per function signature hint)
        return prompt, meta


    def generate_target_manual(self, target_name : str,
                                 event_censored : str, # Note: type hint was str, assuming it can be None or some indicator
                                 event_occurred : bool) -> tuple: # Changed return type hint to tuple based on implementation
        """
        Manually generates a target string and metadata from specified outcome components.

        This allows creating a target string representation without needing the full
        `patient_split` dictionary, by directly providing the key outcome details. Useful
        for testing or specific generation scenarios.

        Parameters
        ----------
        target_name : str
            The descriptive name of the target event.
        event_censored : str or None
            The censoring status detail. `None` typically indicates not censored,
            while a string value might provide details if censored.
        event_occurred : bool
            Boolean indicating whether the event occurred.

        Returns
        -------
        tuple
            - str: The formatted target string.
            - dict: The associated metadata dictionary.
        """

        patient_dic = {
            "sampled_variable_name": target_name,
            "event_censored": event_censored,
            "event_occured": event_occurred,
            "split_date_included_in_input": None,
            "date_event_occured": None,
            "sampled_category": None,
        }
        # Return type should be tuple as per _generate_target_string
        return self._generate_target_string(patient_dic)


    def reverse_conversion(self, target_string):
        """
        Parses a target string to extract structured event outcome information.

        Attempts to reconstruct the censoring status, occurrence status, and target
        event name from a formatted target string (presumably generated by an LLM or
        following the format created by `_generate_target_string`). It uses the
        predefined prompt template strings from the config as delimiters/markers.

        Parameters
        ----------
        target_string : str
            The formatted target string to parse.

        Returns
        -------
        pd.DataFrame
            A single-row DataFrame containing the extracted information with columns:
            'censoring' (bool or None), 'occurred' (bool or None), 'target_name' (str or None).
            Returns None for fields that cannot be reliably extracted.

        Raises
        ------
        ValueError
            If no structured data (all fields are None) can be extracted from the string.
        """

        # Initialize the dictionary to store the extracted data
        # Using string keys as these define the structure of the output DataFrame
        extracted_data = {
            "censoring": None,
            "occurred": None,
            "target_name" : None
        }

        # Check for sampled_variable_name using "(" and ")"
        if "(" in target_string and ")" in target_string:
            try:
                sampled_var_name = target_string.split("(")[1].split(")")[0]
                extracted_data["target_name"] = sampled_var_name
            except IndexError:
                # Handle cases where split might fail if format is unexpected
                 pass # Keep target_name as None

        # Check for censoring information using config constants
        if self.config.target_prompt_censor_false.strip() in target_string:
            extracted_data["censoring"] = False
        elif self.config.target_prompt_censor_true.strip() in target_string:
            extracted_data["censoring"] = True

        # Check for event occurrence information using config constants
        # Note: Added check for potential old prompt constant TARGET_PROMPT_OCCUR_OLD
        # Assuming TARGET_PROMPT_OCCUR_OLD was meant to be handled, added it here.
        # If TARGET_PROMPT_OCCUR_OLD is not defined or needed, remove the check.
        if self.config.target_prompt_occur.strip() in target_string: # Removed check for TARGET_PROMPT_OCCUR_OLD as it's not in Config
            extracted_data["occurred"] = True
        elif self.config.target_prompt_not_occur.strip() in target_string:
            extracted_data["occurred"] = False

        # In the case where the model hallucinates the event, make it none
        # Using hardcoded string as this checks for a specific hallucination pattern
        if "did not occur/occured" in target_string:
            extracted_data["occurred"] = None

        # Convert the extracted data to a DataFrame
        structured_data = pd.DataFrame([extracted_data])

        # Throw error if only nans
        if structured_data.isna().all().all():
            raise ValueError("No structured data could be extracted from the target string.")

        return structured_data


    def get_difference_in_event_dataframes(self, df1, df2):
        """
        Compares two single-row DataFrames representing event outcomes and identifies differences.

        Designed to compare DataFrames generated by `reverse_conversion`. It checks for
        discrepancies in the 'censoring', 'occurred', and 'target_name' columns between
        the two DataFrames.

        Parameters
        ----------
        df1 : pd.DataFrame
            The first single-row DataFrame (columns: 'censoring', 'occurred', 'target_name').
        df2 : pd.DataFrame
            The second single-row DataFrame (columns: 'censoring', 'occurred', 'target_name').

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the differing values. If the inputs are identical,
            an empty DataFrame is returned. The output DataFrame has columns like
            'df1_censoring', 'df2_censoring', etc., showing the differing values side-by-side.

        Raises
        ------
        ValueError
            If input DataFrames are missing expected columns or do not have exactly one row.
        """
        # Define expected columns using strings, as these relate to the structure created by reverse_conversion
        cols_to_compare = ['censoring', 'occurred', "target_name"]

        # Ensure that the columns are in the same order for both DataFrames
        try:
            df1 = df1[cols_to_compare]
            df2 = df2[cols_to_compare]
        except KeyError as e:
             raise ValueError(f"Input DataFrames are missing expected columns: {e}. Expected: {cols_to_compare}")


        # Check if both DataFrames have the same shape
        if df1.shape != df2.shape:
            # Consider if shape mismatch should raise error or be handled differently (e.g., return info about mismatch)
            raise ValueError("DataFrames do not have the same shape and cannot be compared.")

        # Find rows that are different
        # Handle potential NaNs in comparison gracefully
        diff_mask = (df1.ne(df2) & ~(df1.isna() & df2.isna())).any(axis=1)


        # If there are no differences, return an empty DataFrame
        if not diff_mask.any():
            return pd.DataFrame()

        # Create a DataFrame to hold differences, using string keys for new column names
        differences = pd.DataFrame({
            'df1_censoring': df1.loc[diff_mask, 'censoring'],
            'df2_censoring': df2.loc[diff_mask, 'censoring'],
            'df1_occurred': df1.loc[diff_mask, 'occurred'],
            'df2_occurred': df2.loc[diff_mask, 'occurred'],
            'df1_target_name': df1.loc[diff_mask, 'target_name'],
            'df2_target_name': df2.loc[diff_mask, 'target_name']
        })

        # Reset index to make it more readable
        differences.reset_index(drop=True, inplace=True)

        return differences

    def aggregate_multiple_responses(self, responses_dfs : list[pd.DataFrame]) -> tuple: # Changed return type hint to tuple
        """
        Aggregates multiple single-row event outcome DataFrames by majority vote.

        Takes a list of DataFrames (presumably from multiple `reverse_conversion` calls
        on model outputs for the same input) and determines the most common combination
        of 'censoring', 'occurred', and 'target_name' values. Ties are broken arbitrarily
        by `collections.Counter`.

        Parameters
        ----------
        responses_dfs : list[pd.DataFrame]
            A list of single-row pandas DataFrames, each expected to have columns
            'censoring', 'occurred', and 'target_name'.

        Returns
        -------
        tuple
            - pd.DataFrame: A single-row DataFrame representing the most common response.
            - dict: Metadata containing the distribution (percentage) of all unique
                    responses observed, stored under the key 'distribution_of_responses'
                    as a DataFrame.

        Raises
        ------
        ValueError
            If the input list `responses_dfs` is empty or if any DataFrame within the
            list does not conform to the expected structure (single row, required columns).
        """
        if not responses_dfs:
             raise ValueError("Input list `responses_dfs` cannot be empty.")

        # Use string column names consistent with reverse_conversion output
        original_cols = ['censoring', 'occurred', 'target_name']
        try:
             # Ensure all DFs have the expected columns before processing
             responses_as_list = [df[original_cols].values.tolist() for df in responses_dfs]
        except KeyError as e:
             raise ValueError(f"One or more input DataFrames are missing expected columns: {e}. Expected: {original_cols}")


        # Flatten list and count occurrences of each unique response tuple
        # Handle potential nested lists if DataFrames have more than one row (though typically they shouldn't here)
        element_counts = Counter(tuple(row[0]) for row in responses_as_list if row) # Ensure row is not empty


        if not element_counts:
            # This case might occur if all input DFs were empty or had only NaNs that didn't parse correctly.
            # Return an empty DataFrame or handle as appropriate.
             # For now, return empty DF and empty meta, but consider logging a warning.
            empty_df = pd.DataFrame(columns=original_cols)
            return empty_df, {"distribution_of_responses": pd.DataFrame()}


        #: pick the one with highest occurence, or random if equal (Counter.most_common handles ties arbitrarily)
        most_common_element = element_counts.most_common(1)[0][0]

        #: transform into dictionary using string keys
        ret_dict = {
            "censoring": most_common_element[0],
            "occurred": most_common_element[1],
            "target_name": most_common_element[2],
        }

        #: return as dataframe
        ret_df = pd.DataFrame([ret_dict])

        #: get distribution of responses
        total_responses = sum(element_counts.values())
        distribution_dict = {k: round((v / total_responses) * 100, 2) for k, v in element_counts.items()}
        distribution_list = []
        for k, v in distribution_dict.items():
            dist_dict = dict(zip(original_cols, k))
            dist_dict["distribution_percentage"] = v # Use string key
            distribution_list.append(dist_dict)

        distribution_df = pd.DataFrame(distribution_list)

        # Use string key for metadata dictionary
        meta = {
            "distribution_of_responses": distribution_df
        }

        return ret_df, meta
