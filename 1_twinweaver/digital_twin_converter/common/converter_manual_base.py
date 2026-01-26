import pandas as pd
from typing import Tuple
import re
from datetime import datetime
from digital_twin_converter.common.config import Config # Assuming config.py exists in the same directory or is installable


# Round all numeric values to decimal_precision
def round_and_strip(value, decimal_precision):
    """
    Rounds a numeric value represented as a string or number to a specified
    number of decimal places and removes trailing zeros and decimal points.

    Attempts to convert the input `value` to a float, rounds it using
    `decimal_precision`, converts it back to a string, and then strips
    any trailing '.0' or '.' to provide a clean representation (e.g., 12.0 -> '12',
    12.50 -> '12.5', 12.56 -> '12.56'). If the input cannot be converted to a float,
    the original value is returned unchanged.

    Parameters
    ----------
    value : any
        The value to be rounded. Can be a number or a string representation of a number.
    decimal_precision : int
        The number of decimal places to round to.

    Returns
    -------
    str
        The rounded and stripped string representation of the number, or the original
        value if conversion to float fails.
    """
    try:
        # Attempt to convert to float and round
        rounded_value = round(float(value), decimal_precision)
        # Convert to string and strip trailing zeros
        return str(rounded_value).rstrip('0').rstrip('.')
    except ValueError:
        # If conversion fails, return the original value
        return value


class ConverterManualBase:
    """
    Base class for converting structured patient event data into textual representations
    and vice-versa, using manually defined templates and logic.

    This class provides the core functionalities for preprocessing constant (demographic)
    and event data, generating text strings based on configured templates, parsing these
    strings back into structured data, managing token budgets for text length control,
    and comparing event datasets. It relies heavily on a `Config` object for various
    settings like column names, text snippets, and processing flags.

    Derived classes are expected to implement specific methods like tokenizer initialization
    and potentially override base methods for custom behavior.
    """

    def __init__(self,
                 config : Config) -> None:
        """
        Initialize the ConverterManualBase class with configuration settings.

        Sets up internal attributes based on the provided `Config` object, including
        text templates for conversion, mappings for special event handling (like death or drugs),
        decimal precision for rounding, and the random seed. Initializes tokenizer-related
        attributes (`tokenizer`, `nr_tokens_budget_padding`, `always_keep_first_visit`) to None,
        expecting them to be set by a derived class or later configuration.

        Parameters
        ----------
        config : Config
            A configuration object containing settings like column names, text templates,
            event category mappings, decimal precision, seed, and paths/flags.
        """
        # Set up config
        self.config = config
        self.seed = self.config.seed

        # Set decimal precision
        self.decimal_precision = self.config.decimal_precision

        # Setup all text passages using config defaults if overrides are None
        self.preable_text = self.config.preamble_text
        self.constant_text = self.config.constant_text
        self.first_day_text = self.config.first_day_text
        self.genetic_skip_text = self.config.genetic_skip_text_value
        self.event_day_preamble = self.config.event_day_preamble
        self.event_day_text = self.config.event_day_text
        self.post_event_text = self.config.post_event_text

        # Setup special mappings using config defaults if overrides are None
        self.event_category_preamble_mapping = (
            self.config.event_category_preamble_mapping_override
            if self.config.event_category_preamble_mapping_override is not None
            # Using a hardcoded default 'drug' as it appears intentional in the original code
            else {"drug": "drug"}
        )

        # Use config constant for 'death' category in the default replacement mapping
        self.event_category_and_name_replace = (
            self.config.event_category_and_name_replace_override
            if self.config.event_category_and_name_replace_override is not None
            else {
                self.config.event_category_death: { # Use config constant
                    "death": { # Assuming 'death' is the event_name associated with this category
                        "full_replacement_string": "death",
                        "reverse_string_value": "death"
                    }
                }
            }
        )

        # These should instantiated from derived class
        self.tokenizer = None
        self.nr_tokens_budget_padding = None
        self.always_keep_first_visit = None



    def _preprocess_constant_date(self, events : pd.DataFrame, constant : pd.DataFrame,
                                    constant_description : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses static patient data (e.g., demographics) for text conversion.

        Applies specific preprocessing steps based on the `config.constant_use_cgdb_preprocessing`
        flag. If True (default CGDB logic), it calculates age from birth year and the first event date,
        extracts specific columns defined in `config.constant_columns_to_use`, processes indication
        and gender values, and updates the `constant_description` DataFrame accordingly.
        If False, it simply extracts the columns specified in the config.

        Parameters
        ----------
        events : pd.DataFrame
            DataFrame containing the patient's time-series event data (used to find the first event date for age calculation in CGDB mode).
        constant : pd.DataFrame
            DataFrame containing the raw static patient data (e.g., birth year, race, gender).
        constant_description : pd.DataFrame
            DataFrame describing the variables in the `constant` DataFrame (e.g., variable name and a comment/description).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing:
            - The processed `constant` DataFrame, potentially with calculated age and filtered/modified columns.
            - The potentially updated `constant_description` DataFrame.
        """

        if self.config.constant_use_cgdb_preprocessing:

            # This is default CGDB special preprocessing

            #: convert yearofbirth to age, by taking the first events date and subtracting yearofbirth
            age = (events[self.config.date_col].min().year - constant["birthyear"]).astype(int).astype(str) + " years" # Using config for date_col

            #: extract race, gender, ethnicity
            constant = constant.copy()[self.config.constant_columns_to_use]
            constant["age"] = age
            constant["indication"] = constant["indication"].iloc[0].split("enhanced_")[1] # Specific logic

            # Convert in gender "M" to "male" and "F" to "female"
            gender_mapping = {"M": "male", "F": "female"}
            constant["gender"] = constant["gender"].map(gender_mapping) # Specific logic

            #: add age and indication to constant_description
            constant_description = constant_description.copy()
            cont_new_row = pd.DataFrame([{"variable": "age", "comment": "age of patient"},
                                        {"variable": "indication", "comment": "cancer indication of patient"}])
            constant_description = pd.concat([constant_description, cont_new_row], ignore_index=True)

        else:

            # Extracting corresponding variables
            constant = constant.copy()[self.config.constant_columns_to_use]

            # Override birthdate (or variations of it) to age
            if self.config.constant_birthdate_column is not None:

                # Check if the column contains integer ages (not birthdates)
                if pd.api.types.is_numeric_dtype(constant[self.config.constant_birthdate_column]):
                    # Handle integer ages - just format them as "X years"
                    constant[self.config.constant_birthdate_column] = constant[self.config.constant_birthdate_column].astype(int).astype(str) + " years"
                    print(f"Converted integer ages in {self.config.constant_birthdate_column} to age format")

                # Try converting the column to datetime if it is not already, if doesn't work then just keep it as is
                elif not pd.api.types.is_datetime64_any_dtype(constant[self.config.constant_birthdate_column]):
                    try:
                        constant[self.config.constant_birthdate_column] = pd.to_datetime(constant[self.config.constant_birthdate_column])

                        # Calculate age in years from the first event date
                        constant[self.config.constant_birthdate_column] = (events[self.config.date_col].min() - constant[self.config.constant_birthdate_column]).dt.days // 365
                        constant[self.config.constant_birthdate_column] = constant[self.config.constant_birthdate_column].astype(int).astype(str) + " years"
                    except Exception as e:
                        print(f"Warning: Could not convert {self.config.constant_birthdate_column} to datetime. Keeping original values. Error: {e}")

            # Assuming constant_description is stable (copying for future use)
            constant_description = constant_description.copy()


        #: return constant, constant_description
        return constant, constant_description


    def _get_constant_string(self, constant : pd.DataFrame, constant_description : pd.DataFrame) -> str:
        """
        Generates a formatted string representation of the constant (demographic) patient data.

        Iterates through the columns of the preprocessed `constant` DataFrame, retrieves the
        corresponding description from `constant_description`, and constructs a string
        listing each piece of information (e.g., "\t'age of patient' is '65 years',\n").
        Prepends the `self.constant_text` preamble and formats the final string.

        Parameters
        ----------
        constant : pd.DataFrame
            Preprocessed DataFrame containing the constant patient data (typically one row).
        constant_description : pd.DataFrame
            DataFrame containing descriptions for the variables in the `constant` DataFrame.

        Returns
        -------
        str
            A formatted string summarizing the patient's constant data.
        """

        #: create string representation of constant
        constant_string = self.constant_text

        for col in constant.columns:
            col_value = constant[col].iloc[0]
            # Keeping hardcoded column names as they seem specific to this function's input structure
            col_description = constant_description[constant_description["variable"] == col]["comment"].iloc[0]
            constant_string += f"\t{col_description} is {col_value},\n"

        # Replace last , with .
        constant_string = constant_string[:-2] + ".\n"

        return constant_string



    def _preprocess_events(self, events : pd.DataFrame) -> pd.DataFrame:
        """
        Performs initial preprocessing on the time-series event data.

        Sorts the events DataFrame primarily by date, then by event category and name, according
        to the column names specified in the `config`. It also applies the `round_and_strip`
        function to the event value column (`config.event_value_col`) using the configured
        `self.decimal_precision` to standardize numeric representations.

        Parameters
        ----------
        events : pd.DataFrame
            DataFrame containing the raw time-series event data for a patient.

        Returns
        -------
        pd.DataFrame
            The preprocessed events DataFrame, sorted and with rounded/stripped numeric values.
        """

        # First sort using config constants
        events = events.sort_values(by=[self.config.date_col, self.config.event_category_col, self.config.event_name_col])

        # Use config constant for event_value_col
        events[self.config.event_value_col] = events[self.config.event_value_col].apply(round_and_strip, args=(self.decimal_precision,))

        return events

    def _get_event_string(self,
                          events : pd.DataFrame,
                          events_delta_0 : int = 0,
                          use_accumulative_dates : bool = False,
                          add_first_day_preamble : bool = True) -> str:
        """
        Converts preprocessed time-series event data into a structured textual representation.

        Groups events by visit date, calculates the time delta (in weeks) between consecutive visits,
        and formats the output string visit by visit. Uses templates from the `config` object
        (`first_day_text`, `event_day_preamble`, `event_day_text`, `post_event_text`) to structure
        the text. Handles genetic events separately, enclosing them in tags (`config.genetic_tag_opening`,
        `config.genetic_tag_closing`) and potentially skipping the value if it matches
        `config.genetic_skip_text_value`. Applies specific formatting based on event category mappings
        and replacement overrides defined in the `config`.

        Parameters
        ----------
        events : pd.DataFrame
            Preprocessed DataFrame containing the patient's event data, sorted by date.
        events_delta_0 : int, optional
            The initial delta value (in weeks) to assume for the first visit date, by default 0.
        use_accumulative_dates : bool, optional
            If True, the reported delta for each visit will be the cumulative weeks since the
            very first visit, otherwise it's weeks since the *previous* visit, by default False.
        add_first_day_preamble : bool, optional
            If True, uses the special `self.first_day_text` preamble for the first visit,
            otherwise treats it like any other subsequent visit, by default True.

        Returns
        -------
        str
            A formatted string representing the patient's timeline of events.
        """

        #: sort by date using config constant
        events = events.sort_values(self.config.date_col).reset_index(drop=True)

        #: for every visit get delta to previous in weeks, starting with 0 using config constant
        events_delta = events[self.config.date_col].diff().dt.days / 7
        events_delta[0] = events_delta_0
        events["delta"] = events_delta

        #: get all unique pairs of dates as well as deltas, then sort by date using config constant
        all_unique_dates = pd.concat([events.iloc[0:1], events[events["delta"] > 0]], axis=0, ignore_index=True)
        all_unique_dates = all_unique_dates.drop_duplicates(subset=[self.config.date_col, "delta"])[[self.config.date_col, "delta"]]
        all_unique_dates = all_unique_dates.to_numpy().tolist()
        all_unique_dates = sorted(all_unique_dates, key=lambda x: x[0])

        #: in case of accumulative dates, accumulate
        if use_accumulative_dates:
            all_unique_dates = [(date, sum([delta for _, delta in all_unique_dates[:idx+1]])) for
                                idx, (date, _) in enumerate(all_unique_dates)]

        #: setup string
        events_string = ""

        #: replace all "-( -)*" with "-" in event_descriptive_name and event_value using config constants
        events[self.config.event_descriptive_name_col] = events[self.config.event_descriptive_name_col].str.replace(r'(\s-\s-)+', ' - ', regex=True)
        events[self.config.event_value_col] = events[self.config.event_value_col].str.replace(r'(\s-\s-)+', ' - ', regex=True)


        #: Go per event
        for idx, (date, delta) in enumerate(all_unique_dates):

            #: Get subset using config constant
            all_events_curr_date = events[events[self.config.date_col] == date]

            #: make alternative text for first event
            if idx == 0 and add_first_day_preamble:
                events_string += self.first_day_text
            else:
                #: add event day text, and delta (if possible, round so 7.0 -> 7, 7.1 -> 7.1)
                events_string += self.event_day_preamble
                events_string += f"{delta:.2f}".rstrip('0').rstrip('.')  # Max 2 post decimal
                events_string += self.event_day_text

            #: get genetic first using config constants
            genetic_subset = all_events_curr_date[all_events_curr_date[self.config.source_col] == self.config.source_genetic]

            if genetic_subset.shape[0] > 0:
                #: enclose all genetic with <genetic> </genetic> tags
                events_string += "\t"+ self.config.genetic_tag_opening +"\n"

                #: sort within genetic alphabetically using config constants
                genetic_subset = genetic_subset.sort_values(by=[self.config.event_category_col, self.config.event_name_col])

                for _, row in genetic_subset.iterrows():
                    #: convert genetic using event_descriptive_name, adding tab before & new line after using config constants
                    event_descriptive_name = row[self.config.event_descriptive_name_col]
                    event_value = row[self.config.event_value_col]

                    # Skip event_value if it is the default (e.g. "present"), using self.genetic_skip_text (initialized from config)
                    if event_value == self.genetic_skip_text:
                        events_string += "\t" + event_descriptive_name + ",\n"
                    else:
                        events_string += "\t" + event_descriptive_name + " is " + event_value + ",\n"

                #: enclose all genetic with <genetic> </genetic> tags
                events_string += "\t" + self.config.genetic_tag_closing

                # Add newline only if there are further events
                if all_events_curr_date.shape[0] > genetic_subset.shape[0]:
                    events_string += ",\n"

            #: sort rest alphabetically using config constants
            # Keeping "events" source hardcoded as it seems distinct from genetic source
            event_subset = all_events_curr_date[all_events_curr_date[self.config.source_col] == "events"]
            event_subset = event_subset.sort_values(by=[self.config.event_category_col, self.config.event_name_col])

            #: convert rest using event_descriptive_name
            for idx_inner, (_, row) in enumerate(event_subset.iterrows()): # Renamed inner loop index to avoid shadowing

                #: add tab before & new line after using config constants
                event_descriptive_name = row[self.config.event_descriptive_name_col]
                event_value = row[self.config.event_value_col]
                event_category = row[self.config.event_category_col]
                event_name = row[self.config.event_name_col]

                #: add custom preamble for event_category
                if event_category in self.event_category_preamble_mapping:
                    event_descriptive_name = (
                        self.event_category_preamble_mapping[event_category] + " " +
                        event_descriptive_name
                    )

                # Convert value to lower case, so it is easier for tokenizer to use
                event_value = event_value.lower()

                # Add to event string
                if (event_category in self.event_category_and_name_replace and
                        event_name in self.event_category_and_name_replace[event_category]):
                    # Allow override for special events for custom strings.
                    # Need to be careful that they can also do reverse translation
                    events_string += "\t"
                    events_string += (
                        self.event_category_and_name_replace[event_category][event_name]
                        ["full_replacement_string"]
                    )

                else:
                    # Default of "<name> is <value>"
                    events_string += "\t" + event_descriptive_name + " is " + event_value

                if idx_inner < event_subset.shape[0] - 1: # Use inner loop index
                    events_string += ",\n"

            #: add self.post_event_text
            events_string += self.post_event_text

        return events_string


    def _extract_event_data(self, text: str, unique_events: pd.DataFrame,
                            raw_events: pd.DataFrame = None,
                            init_date : datetime = None,
                            only_contains_events : bool = False) -> pd.DataFrame:
        """
        Parses a textual representation of patient history back into a structured event DataFrame.

        Identifies event sections within the input `text` (either the whole text if
        `only_contains_events` is True, or the part following `self.first_day_text`).
        Splits the text into visits based on date delta markers (`self.event_day_preamble`,
        `self.event_day_text`). Parses each visit's text using `_parse_visit` to extract
        individual events, calculates their absolute dates based on the deltas and an initial
        date (`init_date` derived from `raw_events` or provided directly), and reconstructs
        the event timeline.

        Requires either `raw_events` (to determine the minimum date) or `init_date` to be provided.

        Parameters
        ----------
        text : str
            The textual representation of the patient's event history.
        unique_events : pd.DataFrame
            A lookup DataFrame containing information about all possible unique events
            (mapping descriptive names to categories and original names).
        raw_events : pd.DataFrame, optional
            The original raw event DataFrame for the patient, used to determine the `init_date`
            if `init_date` is not provided directly.
        init_date : datetime, optional
            The explicit starting date for the first visit. Used if `raw_events` is None.
        only_contains_events : bool, optional
            If True, assumes the entire `text` consists of event descriptions without preamble
            or constant data sections, by default False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the events extracted from the text, with columns matching
            the standard event structure (date, category, name, descriptive_name, value, source).

        Raises
        ------
        AssertionError
            If neither `raw_events` nor `init_date` is provided.
        """

        # Setup up init date using config constant
        assert raw_events is not None or init_date is not None, "Either raw_events or init_date must be provided."
        if raw_events is not None:
            init_date = raw_events[self.config.date_col].min()

        # Initialize an empty list to store event data
        event_data = []

        # Extract the section of the text that contains events
        if only_contains_events:
            event_section = text
        else:
            # Use self.first_day_text which is initialized from config
            event_section = re.search(re.escape(self.first_day_text) + r'(.*)', text, re.DOTALL)
            event_section = event_section.group(1).strip()

        if event_section:

            # Split the event section into individual visits
            # Use self.event_day_preamble and self.event_day_text initialized from config
            visits = re.split(
                re.escape(self.event_day_preamble) + r'(\d+(\.\d+)?)' + re.escape(self.event_day_text),
                event_section
            )

            # The first visit is special and doesn't have a delta
            first_visit = visits[0].strip()
            if first_visit:
                new_events = self._parse_visit(first_visit, 0, unique_events, init_date)
                event_data.extend(new_events)
                # Use config constant for date column in the event data dictionary
                last_visit_date = event_data[-1][self.config.date_col]
            else:
                last_visit_date = init_date

            # Process subsequent visits
            for i in range(1, len(visits), 3):
                delta = float(visits[i])
                visit_text = visits[i + 2].strip()
                new_events = self._parse_visit(visit_text, delta, unique_events, last_visit_date)
                event_data.extend(new_events)
                # Use config constant for date column in the event data dictionary
                last_visit_date = event_data[-1][self.config.date_col]

        # Convert the event data to a DataFrame
        event_df = pd.DataFrame(event_data)

        return event_df

    def _parse_visit(self, visit_text: str, delta: float, unique_events: pd.DataFrame, last_visit_date):
        """
        Parses the text corresponding to a single visit into a list of event dictionaries.

        Splits the `visit_text` into individual event lines based on ",\n". Handles the removal
        of the trailing `self.post_event_text`. Detects and processes genetic events enclosed
        in genetic tags (`config.genetic_tag_opening`, `config.genetic_tag_closing`) using
        `_parse_genetic_event`. Parses standard events using `_parse_standard_event`.

        Parameters
        ----------
        visit_text : str
            The block of text describing the events that occurred during a single visit.
        delta : float
            The time delta (in weeks) from the previous visit (or `init_date` if the first visit).
        unique_events : pd.DataFrame
            Lookup DataFrame for unique event details.
        last_visit_date : pd.Timestamp
            The timestamp of the previous visit date (or `init_date` if the first visit).

        Returns
        -------
        list[dict]
            A list where each element is a dictionary representing a single parsed event,
            containing keys like date, event category, name, value, etc.
        """

        # Split the visit text into individual events
        events = visit_text.split(",\n")
        new_events = []
        genetic_mode = False

        for idx, event in enumerate(events):
            event = event.strip()

            # If last line, then remove trailing "." using self.post_event_text initialized from config
            if idx == len(events) - 1:
                # Use self.post_event_text which is initialized from config
                raw_post_event_text = self.post_event_text.replace("\n", "")
                event = event[:-len(raw_post_event_text)]

            if event:
                # Check if the event is genetic
                if event.startswith(self.config.genetic_tag_closing):
                    genetic_mode = False

                elif event.startswith(self.config.genetic_tag_opening) or genetic_mode:

                    genetic_mode = True
                    # Process out <genetic> if needed
                    if event.startswith(self.config.genetic_tag_opening):
                        event = event.split("\n")[1]

                    # Pass "unknown - genetic" as category, consistent with original logic
                    new_event = self._parse_genetic_event(event, delta, unique_events, last_visit_date)
                    if new_event: # Ensure event was parsed correctly
                         new_events.append(new_event)
                else:
                    new_event = self._parse_standard_event(event, delta, unique_events, last_visit_date)
                    if new_event: # Ensure event was parsed correctly
                        new_events.append(new_event)

        return new_events


    def _parse_genetic_event(self, genetic_event: str, delta: float, unique_events: pd.DataFrame, last_visit_date):
        """
        Parses a single line of text representing a genetic event.

        Extracts the descriptive name and value using `_extract_event_details`.
        Assigns a fixed category "unknown - genetic" (based on original implementation logic).
        Calculates the event date based on `last_visit_date` and `delta`. Uses
        `_add_event_to_data` to structure the output dictionary.

        Parameters
        ----------
        genetic_event : str
            The text line describing a genetic event (e.g., "Gene X Mutation is detected").
        delta : float
            Time delta (weeks) from the previous visit.
        unique_events : pd.DataFrame
            Lookup DataFrame for unique event details.
        last_visit_date : pd.Timestamp
            Timestamp of the previous visit date.

        Returns
        -------
        dict or None
            A dictionary representing the parsed genetic event, or None if parsing fails
            (e.g., `_extract_event_details` returns None).
        """

        # Split the genetic event into individual lines (although usually just one per entry)
        lines = genetic_event.split(",\n") # Original split logic, may need review if multi-line genetics occur differently
        for line in lines:
            line = line.strip()

            if line:
                event_descriptive_name, event_value = self._extract_event_details(line)
                # Use "unknown - genetic" as the category, consistent with original apparent logic
                return self._add_event_to_data(
                    event_descriptive_name, event_value, delta,
                    "unknown - genetic", unique_events, last_visit_date
                )
        return None # Return None if no valid line found


    def _parse_standard_event(self, event: str, delta: float, unique_events: pd.DataFrame, last_visit_date):
        """
        Parses a single line of text representing a standard (non-genetic) event.

        Extracts the descriptive name and value using `_extract_event_details`. Determines the
        event category by looking up the descriptive name in `unique_events` using
        `_get_event_category`. Calculates the event date based on `last_visit_date` and `delta`.
        Uses `_add_event_to_data` to structure the output dictionary.

        Parameters
        ----------
        event : str
            The text line describing a standard event (e.g., "Hemoglobin is 12.5").
        delta : float
            Time delta (weeks) from the previous visit.
        unique_events : pd.DataFrame
            Lookup DataFrame for unique event details.
        last_visit_date : pd.Timestamp
            Timestamp of the previous visit date.

        Returns
        -------
        dict or None
            A dictionary representing the parsed standard event, or None if parsing fails
            (e.g., `_extract_event_details` returns None).
        """

        event_descriptive_name, event_value = self._extract_event_details(event)
        if event_descriptive_name is None: # Check if extraction failed
             return None
        event_category = self._get_event_category(event_descriptive_name, unique_events)
        return self._add_event_to_data(
            event_descriptive_name, event_value, delta,
            event_category, unique_events, last_visit_date
        )

    def _extract_event_details(self, event: str) -> Tuple[str, str]:
        """
        Extracts the descriptive name and value from a single event string.

        Handles different event string formats:
        1. Checks if the string matches a predefined `full_replacement_string` from the
           `self.event_category_and_name_replace` override mapping. If so, returns the
           corresponding original event name (as descriptive name) and `reverse_string_value`.
        2. If the string contains " is ", splits it into descriptive name and value.
        3. Otherwise, assumes the entire string is the descriptive name and assigns
           `self.genetic_skip_text` (e.g., "present") as the value.
        Removes any category-specific preambles (defined in `self.event_category_preamble_mapping`)
        from the beginning of the extracted descriptive name.

        Parameters
        ----------
        event : str
            The textual representation of a single event line.

        Returns
        -------
        Tuple[str, str] or Tuple[None, None]
            A tuple containing the extracted (and cleaned) event descriptive name and event value.
            Returns (None, None) if parsing fails (e.g., due to unexpected format).
        """

        # Check if in manual override:
        all_manual_overrides = []
        for cat in self.event_category_and_name_replace:
            for event_name in self.event_category_and_name_replace[cat]:
                replace_info = self.event_category_and_name_replace[cat][event_name]
                full_replacement = replace_info["full_replacement_string"]
                reverse_value = replace_info["reverse_string_value"]
                # Store descriptive name (event_name) associated with the full replacement
                override_tuple = (cat, event_name, full_replacement, reverse_value)
                all_manual_overrides.append(override_tuple)

        # Find if the event text matches a full replacement string
        matched_override = next((ov for ov in all_manual_overrides if event == ov[2]), None)

        if matched_override:
            # Extract the original descriptive name and its associated reverse value
            event_descriptive_name = matched_override[1] # The event_name is the descriptive name here
            event_value = matched_override[3]
        elif " is " in event:
            try:
                event_descriptive_name, event_value = event.split(" is ", 1)
            except ValueError:
                 print(f"Warning: Could not parse event string '{event}' using ' is ' split.")
                 return None, None # Indicate parsing failure
        else:
            # Assume it's an event without a value (like genetic 'present')
            event_descriptive_name = event
            # Use self.genetic_skip_text which is initialized from config
            event_value = self.genetic_skip_text

        # Remove any preamble if possible
        for category, preamble in self.event_category_preamble_mapping.items():
            # Check if preamble exists at the beginning of the descriptive name
            if event_descriptive_name and event_descriptive_name.startswith(preamble + " "):
                event_descriptive_name = event_descriptive_name[len(preamble):].strip()

        return event_descriptive_name.strip(), event_value.strip()

    def _get_event_category(self, event_descriptive_name: str, unique_events: pd.DataFrame) -> str:
        """
        Determines the event category corresponding to a given descriptive name.

        Looks up the `event_descriptive_name` in the `unique_events` DataFrame (using the
        `config.event_descriptive_name_col`). If found, returns the associated category
        (`config.event_category_col`). If not found directly, it checks if the name matches
        an event name defined within the `self.event_category_and_name_replace` override;
        if so, it returns the corresponding category key. If still not found, returns "unknown".

        Parameters
        ----------
        event_descriptive_name : str
            The descriptive name of the event (after potential preamble removal).
        unique_events : pd.DataFrame
            Lookup DataFrame mapping descriptive names to categories and original names.

        Returns
        -------
        str
            The determined event category string (e.g., "lab", "diagnosis", "lot", "unknown").
        """

        # Note: Preamble removal is done in _extract_event_details now.

        # Try matching using config constant for descriptive name column
        matches_in_unique_events = unique_events[unique_events[self.config.event_descriptive_name_col] == event_descriptive_name]

        if matches_in_unique_events.shape[0] == 0:
            # Check if it was a manually replaced event (like death) where descriptive name might be the event_name
             manual_match = None
             for cat, name_map in self.event_category_and_name_replace.items():
                  if event_descriptive_name in name_map:
                       manual_match = cat
                       break
             if manual_match:
                   return manual_match
             else:
                 #print(f"Warning: Event descriptive name '{event_descriptive_name}' not found in unique events or manual overrides.")
                 return "unknown" # Default if not found
        else:
             # Use config constant for category column
            event_category = matches_in_unique_events[self.config.event_category_col].iloc[0]
            return event_category

    def _add_event_to_data(self, event_descriptive_name: str, event_value: str,
                           delta: float, event_category: str, unique_events: pd.DataFrame,
                           prev_date : pd.Timestamp): # Changed type hint from pd.DatetimeIndex to pd.Timestamp
        """
        Constructs a dictionary representing a single event record with standardized keys.

        Calculates the event date by adding the `delta` (in days, rounded) to the `prev_date`.
        Looks up the original `event_name` in `unique_events` based on the `event_descriptive_name`,
        falling back to the descriptive name itself or checking manual overrides if not found.
        Determines the `source` based on whether the `event_category` is "unknown - genetic"
        (source = `config.source_genetic`) or not (source = "events"). Populates a dictionary
        using column names defined in the `config` object (`date_col`, `event_category_col`, etc.).

        Parameters
        ----------
        event_descriptive_name : str
            The descriptive name of the event.
        event_value : str
            The value associated with the event.
        delta : float
            Time delta (in weeks) from the previous visit date.
        event_category : str
            The category assigned to the event.
        unique_events : pd.DataFrame
            Lookup DataFrame for unique event details.
        prev_date : pd.Timestamp
            The timestamp of the previous visit date (or `init_date`).

        Returns
        -------
        dict
            A dictionary containing the structured event data with keys corresponding to
            the configured standard column names (e.g., `config.date_col`, `config.event_name_col`).
        """

        # Get event name, falling back to using the event_descriptive_name as event_name
        # Use config constant for descriptive name column
        unique_events_data_matched = unique_events[unique_events[self.config.event_descriptive_name_col] == event_descriptive_name]

        if unique_events_data_matched.shape[0] == 0:
             # Check manual overrides again for event_name mapping
             event_name = event_descriptive_name # Default fallback
             for cat, name_map in self.event_category_and_name_replace.items():
                   if cat == event_category and event_descriptive_name in name_map:
                        # This assumes the key in name_map is the desired event_name
                        event_name = event_descriptive_name
                        break
        else:
             # Use config constant for event name column
            event_name = unique_events_data_matched[self.config.event_name_col].iloc[0]

        # Determine source: Use config constant for genetic source. Keep 'events' literal for other source.
        # The logic `event_category == "unknown - genetic"` seems more correct based on _parse_genetic_event
        source_value = self.config.source_genetic if event_category == "unknown - genetic" else "events"

        new_event = {
            self.config.date_col: prev_date + pd.to_timedelta(round(delta * 7), unit='D'),
            self.config.event_category_col: event_category,
            self.config.event_name_col: event_name,
            self.config.event_descriptive_name_col: event_descriptive_name,
            self.config.event_value_col: event_value,
            self.config.source_col: source_value
        }
        return new_event


    def _estimate_budget_per_variable(self, events : pd.DataFrame) -> pd.DataFrame:
        """
        Estimates the number of tokens required for each event row in a DataFrame.

        Requires `self.tokenizer` to be set. Calculates the tokens for the descriptive name
        and value separately. Estimates base tokens needed for date preambles (`first_day_text`,
        `event_day_text`) and per-line structural tokens (" is \t ,\n"). Distributes the
        base date preamble tokens across all lines and adds them to the per-line estimate.
        Adds columns "nr_tokens_descriptive", "nr_tokens_value", and "nr_tokens_total"
        to the DataFrame.

        Parameters
        ----------
        events : pd.DataFrame
            DataFrame containing event data for which to estimate token counts.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with added columns for estimated token counts per component
            and in total for each event row.

        Raises
        ------
        ValueError
            If `self.tokenizer` has not been initialized.
        """

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before estimating budget.")

        def get_nr_tokens(curr_string):
            return len(self.tokenizer(str(curr_string))["input_ids"]) # Ensure input is string

        #: estimate base tokens
        # Using self.first_day_text and self.event_day_text initialized from config
        nr_tokens_date_first = get_nr_tokens(self.first_day_text)
        nr_tokens_date_all = get_nr_tokens(self.event_day_text) * len(events[self.config.date_col].unique())
        nr_tokens_base = nr_tokens_date_first + nr_tokens_date_all

        #: estimated tokens per line
        base_tokens_per_line = get_nr_tokens(" is \t ,\n") # Base structure tokens
        tokens_per_line = base_tokens_per_line + (nr_tokens_base / len(events)) if len(events) > 0 else base_tokens_per_line

        #: apply tokenization to each line using config constants
        events = events.copy()
        events["nr_tokens_descriptive"] = events[self.config.event_descriptive_name_col].apply(get_nr_tokens)
        events["nr_tokens_value"] = events[self.config.event_value_col].apply(get_nr_tokens)
        events["nr_tokens_total"] = events["nr_tokens_descriptive"] + events["nr_tokens_value"] + tokens_per_line

        return events

    def _get_all_most_recent_events_within_budget(self, events : pd.DataFrame,
                                                  budget_total : int) -> pd.DataFrame:
        """
        Selects the most recent events that fit within a specified total token budget.

        Requires `self.tokenizer` and `self.always_keep_first_visit` to be set.
        Estimates the token budget per event row using `_estimate_budget_per_variable`.
        Sorts events by date descending. Calculates the cumulative token sum. Selects rows
        where the cumulative sum is less than or equal to `budget_total`. If
        `self.always_keep_first_visit` is True, it ensures that all events from the
        earliest date are included, potentially exceeding the budget slightly if necessary,
        and then adds other recent events within the remaining budget. Sorts the final
        selected events by date ascending and removes token count columns before returning.

        Parameters
        ----------
        events : pd.DataFrame
            DataFrame containing event data, typically preprocessed and sorted.
        budget_total : int
            The maximum total number of tokens allowed for the selected events' textual representation.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the subset of most recent events that fit within the budget,
            sorted chronologically.

        Raises
        ------
        ValueError
            If `self.tokenizer` or `self.always_keep_first_visit` has not been set.
        """

        # Check that we have correct things
        if self.tokenizer is None:
             raise ValueError("Tokenizer must be set before calculating budget.")
        if self.always_keep_first_visit is None:
             raise ValueError("always_keep_first_visit must be set.")

        #: estimate budget per row
        events = self._estimate_budget_per_variable(events)

        #: sort by date descending using config constant
        events = events.sort_values(self.config.date_col, ascending=False)

        if self.always_keep_first_visit:
             if not events.empty: # Check if dataframe is not empty
                 #: make all events with first visit date be moved to the beginning of the dataframe using config constant
                 first_date = events[self.config.date_col].min()
                 first_date_events = events[events[self.config.date_col] == first_date]
                 other_events = events[events[self.config.date_col] != first_date]
                 # Keep original index for potential debugging, reset later if needed
                 events = pd.concat([first_date_events, other_events], ignore_index=False) # Keep original index


        # Do calculation using cumulative sum
        events["cumsum"] = events["nr_tokens_total"].cumsum()

        # Do actual selection based on budget
        events_within_budget = events[events["cumsum"] <= budget_total]

        # If always_keep_first_visit, ensure first visit events are kept even if budget is tight
        if self.always_keep_first_visit and not events.empty:
            first_date = events[self.config.date_col].min() # Get first date again
            first_date_events_original = events[events[self.config.date_col] == first_date]
            # Combine kept first visit events with other events within budget
            other_events_within_budget = events_within_budget[events_within_budget[self.config.date_col] != first_date]
            events_final = pd.concat([first_date_events_original, other_events_within_budget]).drop_duplicates()
        else:
            events_final = events_within_budget

        # Sort final selection by date ascending using config constant
        events_final = events_final.sort_values(self.config.date_col)

        #: drop nr tokens columns
        events_final = events_final.drop(columns=["nr_tokens_descriptive", "nr_tokens_value",
                                               "nr_tokens_total", "cumsum"])

        #: return events
        return events_final.reset_index(drop=True) # Reset index for clean output


    def _generate_summarized_row_string(self, input_event_data, combined_target_meta : dict) -> str:
        """
        Creates a summary string containing the most recent genetic, LoT, and target variable values.

        Extracts the latest occurrences of genetic events and the most recent Line of Therapy (LoT)
        start/name event from the `input_event_data`. Formats these using templates from the
        `config` (`forecasting_prompt_summarized_genetic`, `forecasting_prompt_summarized_lot`).
        If target variable information is present in `combined_target_meta["dates_per_variable"]`,
        it finds the last recorded value for each target variable in `input_event_data`,
        sorts them alphabetically, and adds them to the string using the
        `config.forecasting_prompt_summarized_start` template.

        Parameters
        ----------
        input_event_data : pd.DataFrame
            DataFrame containing the patient's event history (input side).
        combined_target_meta : dict
            A dictionary potentially containing target variable information under the key
            "dates_per_variable" (mapping variable names to something, structure implies keys exist)
            and optionally "variable_name_mapping" (mapping variable names to descriptive names).

        Returns
        -------
        str
            A formatted string summarizing the most recent key information for forecasting tasks.
        """

        # start ret
        ret_prompt = ""

        #: add most recent genetic info using config constants
        ret_prompt += self.config.forecasting_prompt_summarized_genetic # Using config attribute

        #: select only genetic info from input side using config constants
        genetic_info = input_event_data[input_event_data[self.config.source_col] == self.config.source_genetic]

        #: select for each genetic variable occurence the most recent value using config constants
        genetic_info_processed = genetic_info.sort_values(self.config.date_col).drop_duplicates(
             subset=[self.config.event_name_col], keep='last'
        )


        #: call _get_event_string with genetic info
        # Set add_first_day_preamble to False as this is a summary section
        if genetic_info_processed.empty:
            genetic_str = self.config.genetic_empty_text
        else:
            genetic_str = self._get_event_string(genetic_info_processed,
                                                use_accumulative_dates = False,
                                                add_first_day_preamble=False) # Don't add first day preamble here

        #: Process the genetic string: remove potential date preamble and keep only event lines
        genetic_str_lines = genetic_str.strip().split('\n')
        processed_genetic_lines = []
        # Skip lines related to date/delta preamble if present
        for line in genetic_str_lines:
             # Simple check: if line starts with a number followed by ' weeks later', skip it. Adjust regex if needed.
             if not re.match(r'^\s*\d+(\.\d+)?\s+' + re.escape(self.config.event_day_text.strip().split(' ', 1)[1]), line):
                 processed_genetic_lines.append(line)

        # Remove <genetic> tags if they are the only content on their lines
        processed_genetic_lines = [line for line in processed_genetic_lines if line.strip() not in [self.config.genetic_tag_opening, self.config.genetic_tag_closing]]

        # Re-join and ensure proper indentation (assuming tabs are used in _get_event_string)
        ret_prompt += "\n" + "\n".join(processed_genetic_lines) + "\n" # Add newlines for separation


        #: add most recent LoT info using config constants
        ret_prompt += self.config.forecasting_prompt_summarized_lot # Using config attribute
        lot_info = input_event_data[input_event_data[self.config.event_category_col] == self.config.event_category_lot]

        # Ensure lot_info is sorted by date to correctly find the last one
        lot_info = lot_info.sort_values(self.config.date_col)

        # Create selections based on event name and event value using config constants
        if self.config.lot_name_col is not None and self.config.event_value_lot_start is not None:
            lot_selection_1 = lot_info[lot_info[self.config.event_name_col] == self.config.lot_name_col] # Using config attribute
            lot_selection_2 = lot_info[lot_info[self.config.event_value_col] == self.config.event_value_lot_start] # Using config attribute
        else:
            # Just use all lot_info if no specific columns are defined
            lot_selection_1 = lot_info
            lot_selection_2 = pd.DataFrame() # Empty DataFrame if no specific selection is made

        # Get the most recent entries and their dates using config constant
        most_recent_lot_1 = lot_selection_1.iloc[-1] if not lot_selection_1.empty else None
        most_recent_lot_2 = lot_selection_2.iloc[-1] if not lot_selection_2.empty else None

        date_1 = most_recent_lot_1[self.config.date_col] if most_recent_lot_1 is not None else pd.NaT # Using config attribute, use NaT for comparison
        date_2 = most_recent_lot_2[self.config.date_col] if most_recent_lot_2 is not None else pd.NaT # Using config attribute, use NaT for comparison


        # Determine which lot to use based on the dates
        most_recent_lot = None
        if pd.notna(date_1) and pd.notna(date_2):
             most_recent_lot = most_recent_lot_2 if date_2 >= date_1 else most_recent_lot_1 # Use >= to prefer LoT start if dates are same
        elif pd.notna(date_1):
            most_recent_lot = most_recent_lot_1
        elif pd.notna(date_2):
            most_recent_lot = most_recent_lot_2


        # Append the appropriate information to ret_prompt using config constants
        if most_recent_lot is not None:

            # Override
            if self.config.lot_concatenate_descriptive_and_value:
                # Use config constant for concatenation
                ret_prompt += "\t" + most_recent_lot[self.config.event_descriptive_name_col] + self.config.lot_concatenate_string + most_recent_lot[self.config.event_value_col] + "\n"
            else:
                # Prefer showing the LoT start event's descriptive name if it was chosen
                if most_recent_lot is most_recent_lot_2:
                    ret_prompt += "\t" + most_recent_lot[self.config.event_descriptive_name_col] + "\n" # Using config attribute
                else: # Otherwise show the value associated with the line_name event
                    ret_prompt += "\t" + most_recent_lot[self.config.event_value_col] + "\n" # Using config attribute
        else:
            ret_prompt += "\tNo line of therapy start information available.\n" # Adjusted message


        #: if we have target vars, for every target variable, retrieve their last value in input
        if "dates_per_variable" in combined_target_meta and combined_target_meta["dates_per_variable"]:

            last_vals = {}

            # Ensure input data is sorted by date to get the actual last value
            input_event_data_sorted = input_event_data.sort_values(self.config.date_col)

            for target_var in combined_target_meta["dates_per_variable"].keys():
                # Use config constant for event name column
                curr_var_data = input_event_data_sorted[input_event_data_sorted[self.config.event_name_col] == target_var]
                if not curr_var_data.empty:
                    # Use config constant for event value column
                    last_value = curr_var_data[self.config.event_value_col].iloc[-1]
                    last_value_rounded = round_and_strip(last_value, self.decimal_precision)
                    last_vals[target_var] = last_value_rounded
                # else: handle case where target variable isn't in input_event_data? Maybe add a placeholder?

            #: sort alphabetically by variable name
            sorted_last_vals = sorted(last_vals.items())

            #: then transform into string
            ret_prompt += self.config.forecasting_prompt_summarized_start # Using config attribute

            for variable, value in sorted_last_vals:
                # Use descriptive name from mapping if available, else use variable name
                var_descriptive_name = combined_target_meta.get("variable_name_mapping", {}).get(variable, variable)
                ret_prompt += "\t" + var_descriptive_name + " was " + str(value) + "\n"

        #: return
        return ret_prompt

    def get_difference_in_event_dataframes(self, events_1 : pd.DataFrame, events_2 : pd.DataFrame,
                                           skip_genetic : bool = False, skip_vals_list = []):
        """
        Compares two event DataFrames and identifies rows that are not identical based on key columns.

        Compares `events_1` and `events_2` based on `date`, `event_name` (case-insensitive),
        and `event_value` (case-insensitive). Optionally filters out genetic events (based on
        category "unknown - genetic" or source `config.source_genetic`) and events whose
        descriptive name contains any substring from `skip_vals_list` (case-insensitive) before
        comparison. Uses a merge operation with an indicator to find rows present in one DataFrame
        but not the other.

        Parameters
        ----------
        events_1 : pd.DataFrame
            The first event DataFrame for comparison.
        events_2 : pd.DataFrame
            The second event DataFrame for comparison.
        skip_genetic : bool, optional
            If True, genetic events are excluded from the comparison, by default False.
        skip_vals_list : list, optional
            A list of substrings. Events whose descriptive name contains any of these substrings
            (case-insensitive) will be excluded from the comparison, by default [].

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the rows that differ between the two input DataFrames,
            including a '_merge' column indicating whether the row was unique to 'left_only' (events_1)
            or 'right_only' (events_2).
        """
        # Make both to lower on event_name and event_value
        events_1[self.config.event_name_col] = events_1[self.config.event_name_col].str.lower()
        events_1[self.config.event_value_col] = events_1[self.config.event_value_col].str.lower()
        events_2[self.config.event_name_col] = events_2[self.config.event_name_col].str.lower()
        events_2[self.config.event_value_col] = events_2[self.config.event_value_col].str.lower()

        # Skip genetic if needed
        if skip_genetic:
            events_1 = events_1[events_1[self.config.event_category_col] != "unknown - genetic"]
            events_2 = events_2[events_2[self.config.event_category_col] != "unknown - genetic"]
            if self.config.source_col in events_1:
                events_1 = events_1[events_1[self.config.source_col] != self.config.source_genetic]
            if self.config.source_col in events_2:
                events_2 = events_2[events_2[self.config.source_col] != self.config.source_genetic]

        # Skip vals list if needed
        if len(skip_vals_list) > 0:
            pattern = '|'.join(skip_vals_list)
            events_1 = events_1[~events_1[self.config.event_descriptive_name_col].str.contains(pattern, case=False, na=False)]
            events_2 = events_2[~events_2[self.config.event_descriptive_name_col].str.contains(pattern, case=False, na=False)]

        # Only keep columns date, event_name and event_value
        events_1 = events_1[[self.config.date_col, self.config.event_name_col, self.config.event_value_col]]
        events_2 = events_2[[self.config.date_col, self.config.event_name_col, self.config.event_value_col]]

        # Match then on date, event_name and event_value, and return which rows are not the same

        # Merge the two DataFrames on date, event_name, and event_value with an indicator
        merged_df = events_1.merge(events_2, on=[self.config.date_col, self.config.event_name_col, self.config.event_value_col], how='outer', indicator=True)

        # Filter rows that are not the same in both DataFrames
        difference_df = merged_df[merged_df['_merge'] != 'both']

        return difference_df



    def forward_conversion_inference(self):
        raise NotImplementedError("forward_conversion_inference not implemented in base class.")

    def generate_target_manual(self):
        raise NotImplementedError("generate_target_manual not implemented in base class.")

    def aggregate_multiple_responses(self):
        raise NotImplementedError("aggregate_multiple_responses not implemented in base class.")
