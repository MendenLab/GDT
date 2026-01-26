import pandas as pd
import re

from digital_twin_converter.common.converter_manual_base import ConverterManualBase
from digital_twin_converter.common.config import Config



class ConverterManualTemplate(ConverterManualBase):
    """
    Implements bidirectional conversion between structured patient data and a textual representation.

    This class provides the core logic for transforming pandas DataFrames containing patient
    events and constant information into a human-readable text format (`forward_conversion`)
    and parsing this text format back into DataFrames (`reverse_conversion`). It inherits
    base functionalities and configuration handling from `ConverterManualBase`.

    Attributes:
        config (Config): Configuration object storing settings like column names, date formats, etc.
        preable_text (str): Inherited/configured text added at the beginning of the output string.
        constant_text (str): Inherited/configured text marking the start of the constant data section.
        first_day_text (str): Inherited/configured text marking the start of the event data section.
        # Other attributes related to formatting and separators inherited from base class.
    """


    def __init__(self,
                 config : Config) -> None:
        """
        Initializes the ConverterManualTemplate instance.

        Sets up the converter using the provided configuration object, primarily by
        calling the initializer of the base class `ConverterManualBase`.

        Parameters
        ----------
        config : Config
            A configuration object containing necessary settings for data processing,
            such as standard column names, text separators, and formatting details.
        """
        super().__init__(config)


    def forward_conversion(self, events : pd.DataFrame, constant : pd.DataFrame,
                           constant_description : pd.DataFrame) -> dict:
        """
        Converts structured patient data (events, constant info) into a textual representation.

        This method takes patient data as DataFrames, preprocesses them (e.g., calculating age
        from birthdate), formats the constant information and time-series events into predefined
        textual structures, and combines them into a single string. It returns the generated text
        along with metadata containing both the original and processed DataFrames.

        Parameters
        ----------
        events : pd.DataFrame
            DataFrame containing the time-series event data for the patient. Expected columns
            are defined in the `config` object (e.g., date, category, name, value).
        constant : pd.DataFrame
            DataFrame containing constant (non-time-varying) information for the patient.
            Expected to have a single row. Columns represent different attributes.
        constant_description : pd.DataFrame
            DataFrame describing the columns in the `constant` DataFrame. Typically contains
            columns like 'variable' (column name) and 'comment' (human-readable description).
            Used to generate descriptive text for constant attributes. Retrieved via
            `DataManager.get_constant_descriptive_mapping()`.

        Returns
        -------
        dict
            A dictionary containing the conversion results:
            {
                "text": str,  # The full textual representation of the patient's data.
                "meta": {
                    "raw_constant": pd.DataFrame,  # Original input constant DataFrame.
                    "processed_constant": pd.DataFrame,  # Constant DataFrame after preprocessing.
                    "raw_events": pd.DataFrame,  # Original input events DataFrame.
                    "events": pd.DataFrame,  # Events DataFrame after preprocessing.
                    "constant_description": pd.DataFrame  # The constant description used.
                }
            }
        """
        # Run assertions on input data
        self._run_input_assertions(events, constant, constant_description)

        # NOTE: most functions here are coming from the base class
        #: preprocess constant data into appropriate format (e.g. birthyear to age)
        raw_constant = constant.copy()
        raw_events = events.copy()
        constant, constant_description = self._preprocess_constant_date(events, constant, constant_description)

        #: add in preamble
        master_string = self.preable_text

        #: add in constant
        constant_string = self._get_constant_string(constant, constant_description)
        master_string += constant_string

        # preprocess events
        events = self._preprocess_events(events)

        #: Convert events
        events_string = self._get_event_string(events)
        master_string += events_string

        #: setup return dictionary with meta data
        ret_dict = {
            "text": master_string,
            "meta": {
                "raw_constant" : raw_constant,
                "processed_constant": constant.copy(),
                "raw_events" : raw_events,
                "events": events.copy(),
                "constant_description": constant_description.copy()
            }
        }

        #: return
        return ret_dict


    def reverse_conversion(self, text: str,
                        meta_data: dict,
                        unique_events: pd.DataFrame) -> dict:
        """
        Converts a textual representation of patient data back into structured DataFrames.

        Parses the input text string, attempting to extract constant patient information and
        time-series events based on the formatting conventions used by `forward_conversion`.
        Uses helper methods (`_extract_constant_data`, `_extract_event_data`) and metadata
        (like constant descriptions and original event structure hints) to reconstruct the
        DataFrames.

        Note: The original implementation notes mention this process can be "hacky", may have
        known issues, and was developed with AI assistance. Its primary purpose might be
        validation rather than robust, general-purpose parsing.

        Parameters
        ----------
        text : str
            The textual representation of patient data, as generated by `forward_conversion`.
        meta_data : dict
            The metadata dictionary associated with the text, typically containing necessary
            context like 'constant_description' and potentially 'raw_events' or other hints.
            This usually comes from the 'internal_meta' part of the `_convert_patient` output.
        unique_events : pd.DataFrame
            A DataFrame listing all possible unique events (e.g., names, categories) in the dataset.
            Used to help identify and parse events correctly from the text.

        Returns
        -------
        dict
            A dictionary containing the reconstructed data:
            {
                "constant": pd.DataFrame,  # DataFrame of the extracted constant information.
                "events": pd.DataFrame     # DataFrame of the extracted event data.
            }
        """
        # Extract constant data
        constant_data = self._extract_constant_data(text, meta_data["constant_description"])

        # Extract event data
        event_data = self._extract_event_data(text, unique_events, meta_data["raw_events"])

        # Combine constant and event data
        ret_dict = {
            "constant": constant_data,
            "events": event_data
        }

        return ret_dict

    def _run_input_assertions(self, events: pd.DataFrame, constant: pd.DataFrame,
                              constant_description : pd.DataFrame) -> None:
        """
        Validates the types and basic structure of input DataFrames for `forward_conversion`.

        Checks if the inputs `events`, `constant`, and `constant_description` are pandas
        DataFrames. Verifies that `constant` has exactly one row. Checks if `events`
        (if not empty) and `constant_description` contain essential columns as expected
        (based on `self.config` for events, and fixed names like 'variable', 'comment'
        for `constant_description`). Raises an AssertionError if any validation fails.

        Parameters
        ----------
        events : pd.DataFrame
            The events DataFrame to validate.
        constant : pd.DataFrame
            The constant DataFrame to validate.
        constant_description : pd.DataFrame
            The constant description DataFrame to validate.

        Raises
        ------
        AssertionError
            If any of the input validation checks fail.
        """

        assert isinstance(events, pd.DataFrame), f"Input 'events' must be a pandas DataFrame, but got {type(events)}"
        assert isinstance(constant, pd.DataFrame), f"Input 'constant' must be a pandas DataFrame, but got {type(constant)}"
        assert isinstance(constant_description, pd.DataFrame), f"Input 'constant_description' must be a pandas DataFrame, but got {type(constant_description)}"

        # Check constant DataFrame structure (should represent one patient)
        assert constant.shape[0] == 1, f"Input 'constant' DataFrame should have exactly one row, but found {constant.shape[0]} rows."
        # Optional: Check for a patient ID column using config
        # assert self.config.patient_id_col in constant.columns, f"Input 'constant' DataFrame must contain the patient ID column: '{self.config.patient_id_col}'."

        # Check events DataFrame structure using self.config (allow empty events, but check columns if not empty)
        required_event_cols = [
            self.config.date_col,
            self.config.event_category_col,
            self.config.event_name_col,
            self.config.event_value_col
        ] # Modify if other columns from config are essential
        if not events.empty:
             missing_event_cols = [col for col in required_event_cols if col not in events.columns]
             assert not missing_event_cols, f"Input 'events' DataFrame is missing required columns defined in config: {missing_event_cols}"
        # Optional: Check if date column is datetime type
        # if not events.empty:
        #    assert pd.api.types.is_datetime64_any_dtype(events[self.config.date_col]), f"Column '{self.config.date_col}' in 'events' should be a datetime type."

        # Check constant_description DataFrame structure (columns 'variable' and 'comment' are likely fixed structural names, not from config)
        required_const_desc_cols = ['variable', 'comment'] # These describe the constant data structure itself
        missing_const_desc_cols = [col for col in required_const_desc_cols if col not in constant_description.columns]
        assert not missing_const_desc_cols, f"Input 'constant_description' DataFrame is missing required columns: {missing_const_desc_cols}"
        assert not constant_description.empty, "Input 'constant_description' DataFrame cannot be empty."
        # --- End Input Assertions ---

    def _extract_constant_data(self, text: str, constant_description: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts constant patient data from its textual representation.

        Uses regular expressions to locate the section of the text containing constant
        information (between `self.constant_text` and `self.first_day_text`). It then
        iterates through lines in this section, matching them against the descriptions
        in `constant_description` to identify variable names and extract their values.

        Parameters
        ----------
        text : str
            The full textual representation of the patient data.
        constant_description : pd.DataFrame
            DataFrame describing the constant variables, mapping descriptions ('comment')
            back to variable names ('variable').

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the extracted constant data, with one row and columns
            corresponding to the identified variables. Returns an empty DataFrame or one
            with missing values if extraction fails for some variables.
        """

        constant_data = {}

        # Extract constant section
        constant_section = re.search(
            re.escape(self.constant_text) + r'(.*?)' + re.escape(self.first_day_text),
            text, re.DOTALL
        )
        if constant_section:
            constant_section = constant_section.group(1).strip()

            # Split by lines and process each line
            for line in constant_section.split(",\n"):
                line = line.strip()
                if line:
                    for _, row in constant_description.iterrows():
                        if row["comment"] in line:
                            value = line.split(" is ")[1].strip()
                            constant_data[row["variable"]] = value
                            break

        # Convert to DataFrame
        constant_df = pd.DataFrame([constant_data])

        return constant_df
