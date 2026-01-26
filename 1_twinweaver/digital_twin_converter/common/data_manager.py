import pandas as pd
import numpy as np
import os
import logging
import json
from digital_twin_converter.common.config import Config  # Assuming config.py contains the Config class definition


class AllIndicationsDataManager:
    """
    Manages the discovery and mapping of data files for multiple indications.

    This class scans a specified base directory, identifies indications based on
    file naming conventions (specifically `*_master_events.csv`), and creates
    a mapping between each indication name and the paths to its associated data
    files (constant, constant description, events, molecular, statistics).
    It relies on a `Config` object to determine the base data path.
    """


    def __init__(self, config : Config) -> None:
        """
        Initializes the AllIndicationsDataManager.

        Sets up the base data path using the provided configuration object
        and triggers the discovery of indications and their associated files.

        Parameters
        ----------
        config : Config
            A configuration object containing settings, including the
            `base_data_path` where indication data files are located.
        """

        #: setup base path
        self.config = config
        self.indication_to_file_mapping = {}
        # Use base_data_path from config
        self.base_data_path = self.config.base_data_path
        self._setup_all_indications()


    def _setup_all_indications(self) -> None:
        """
        Scans the base data directory to find indications and map them to their files.

        Identifies indications by looking for files ending with '_master_events.csv'.
        For each indication found, it constructs the expected paths for constant,
        constant description, events, molecular, and statistics CSV files within
        the `base_data_path` and stores these paths in the
        `indication_to_file_mapping` dictionary.
        """

        #: get all files in self.base_data_path folder
        # Use base_data_path from config
        all_files = os.listdir(self.base_data_path)

        #: get all files ending with _master_events.csv, then extract the first part
        # Note: Filename structure seems specific and not fully covered by config constants beyond base path
        all_events_files = [file for file in all_files if file.endswith("_master_events.csv")]
        all_indications = [file.split("_master_events")[0] for file in all_events_files]

        #: generate mapping from indication to all related files
        for indication in all_indications:

            constant_file = os.path.join(self.base_data_path, f"{indication}_master_constant.csv")
            constant_description_file = os.path.join(self.base_data_path,
                                                     f"{indication}_master_constant_descriptions.csv")
            # Construct events file path using base_data_path
            events_file = os.path.join(self.base_data_path, f"{indication}_master_events.csv")
            molecular_file = os.path.join(self.base_data_path, f"{indication}_master_molecular.csv")
            statistics_file = os.path.join(self.base_data_path, f"{indication}_master_statistics.csv")


            # Use config.event_table_name for the 'events' key if appropriate,
            # but the keys here seem logical names rather than config values.
            # Sticking to original keys for clarity of file types.
            self.indication_to_file_mapping[indication] = {
                "constant": constant_file,
                "constant_description": constant_description_file,
                # Use config.event_table_name as the key *if* it's meant to replace "events" everywhere.
                # However, the original code uses "events" as a logical key for the events file path.
                # Let's map the file path to the standard key "events" for consistency within this class.
                "events": events_file,
                "molecular": molecular_file,
                "statistics": statistics_file
            }

    @property
    def indications(self) -> list[str]:
        """
        Provides a list of all discovered indication names.

        Returns
        -------
        list[str]
            A list containing the names of all indications for which data files
            were found and mapped during initialization.
        """

        return list(self.indication_to_file_mapping.keys())


    def get_paths_of_indication(self, indication : str) -> dict:
        """
        Retrieves the dictionary of file paths associated with a specific indication.

        Parameters
        ----------
        indication : str
            The name of the indication for which to retrieve file paths.

        Returns
        -------
        dict
            A dictionary where keys are file type names (e.g., "events",
            "constant", "molecular") and values are the full paths to the
            corresponding data files for the requested indication.

        Raises
        ------
        KeyError
            If the specified indication name is not found in the mapping.
        """
        return self.indication_to_file_mapping[indication]






class SingleIndicationDataManager:
    """
    Manages data loading, processing, and splitting for a single indication.

    This class handles the lifecycle of data for one specific indication,
    including loading data from files (or using overridden dataframes),
    performing processing steps like date conversion and cleaning, ensuring
    unique event naming, and splitting the patient data into training,
    validation, and test sets based on patient IDs. It utilizes a `Config`
    object for various settings and column names.
    """

    def __init__(self, indication : str,
                 config : Config, # Added config parameter
                 train_split_min : float = 0.8,
                 validation_split_max : float = 0.1,
                 test_split_max : float = 0.1,
                 max_val_test_nr_patients : int = 500,
                 replace_special_symbols_override : list = None) -> None:
        """
        Initializes the SingleIndicationDataManager for a specific indication.

        Sets up the manager with the indication name, configuration, data split
        parameters, and options for handling special characters in event names.

        Parameters
        ----------
        indication : str
            The name of the specific indication to manage (e.g., "NSCLC").
        config : Config
            A configuration object containing paths, column names, category names,
            and other constants used throughout the data management process.
        train_split_min : float, optional
            The minimum proportion of patients to allocate to the training set.
            Defaults to 0.8. The actual number will be the remainder after
            allocating validation and test sets.
        validation_split_max : float, optional
            The maximum proportion of the total patients to allocate to the
            validation set. The actual number is capped by
            `max_val_test_nr_patients`. Defaults to 0.1.
        test_split_max : float, optional
            The maximum proportion of the total patients to allocate to the
            test set. The actual number is capped by `max_val_test_nr_patients`.
            Defaults to 0.1.
        max_val_test_nr_patients : int, optional
            The absolute maximum number of patients to include in the validation
            and test sets combined. Defaults to 500.
        replace_special_symbols_override : list, optional
            A list of tuples to override the default special character replacements
            in event descriptive names. Each tuple should be in the format
            `(event_category, (string_to_replace, replacement_string))`. If None,
            default replacements specified in the method are used. Defaults to None.
        """


        #: initialize the data manager
        self.config = config # Store config object
        self.indication = indication
        self.train_split = train_split_min
        self.validation_split = validation_split_max
        self.test_split = test_split_max
        self.max_val_test_nr_patients = max_val_test_nr_patients

        # Setup replacing of special symbol, format is event_category : (<string_to_replace>, <replacement_string>)
        if replace_special_symbols_override is not None:
            self.replace_special_symbols = replace_special_symbols_override
        else:
            # Use config constants for event categories where available
            self.replace_special_symbols = [
                (self.config.event_category_labs, ("/", " per ")),
                (self.config.event_category_labs, (".", " ")),
                ("drug", ("/", " ")), # "drug" category not explicitly in Config constants provided
                (self.config.event_category_lot, ("/", " ")), # Use config for 'lot' category
            ]

        # Setup indication
        if self.config.override_with_custom_dataset:
            self.all_indications_data_manager = None
        else:
            self.all_indications_data_manager = AllIndicationsDataManager(self.config) # Pass config
        self.data_frames = None
        self.unique_events = None
        self.patient_to_split_mapping = {}
        self.all_patientids = None


    def load_indication_data(self) -> None:
        """
        Loads the raw data tables for the specified indication.

        If `config.override_with_custom_dataset` is True, it loads data directly
        from paths specified in the `config` object (e.g.,
        `config.override_datasets_events`). Otherwise, it uses an instance of
        `AllIndicationsDataManager` to find the standard file paths for the
        indication based on the `config.base_data_path` and reads the CSV files
        into pandas DataFrames stored in `self.data_frames`.

        It also removes any columns named "Unnamed: *" from the loaded DataFrames.
        """

        if self.config.override_with_custom_dataset:
            self.data_frames = {}
            logging.info(f"Using overridden datasets for indication {self.indication}")

            assert self.config.override_datasets_events is not None, "Override datasets events cannot be None"
            assert self.config.override_datasets_constant is not None, "Override datasets constant cannot be None"
            assert self.config.override_constant_description is not None, "Override constant description cannot be None"

            self.data_frames["events"] = pd.read_csv(self.config.override_datasets_events)
            self.data_frames["constant"] = pd.read_csv(self.config.override_datasets_constant)
            self.data_frames["constant_description"] = pd.read_csv(self.config.override_constant_description)

            if self.config.override_datasets_molecular is not None:
                self.data_frames["molecular"] = pd.read_csv(self.config.override_datasets_molecular)
            else:
                # Empty df
                self.data_frames["molecular"] = pd.DataFrame(columns=[self.config.date_col,
                                                                      self.config.patient_id_col,
                                                                        self.config.biomarker_category_col,
                                                                        self.config.biomarker_event_col,
                                                                        self.config.biomarker_value_col,
                                                                        self.config.biomarker_descriptive_name_col,
                                                                        self.config.meta_data_col])

            if self.config.override_statistics is not None:
                self.data_frames["statistics"] = pd.read_csv(self.config.override_statistics)
            else:
                self.data_frames["statistics"] = None

        else:
            #: load in corresponding CSV files
            logging.info(f"Loading data for indication {self.indication} using default approach")

            indication_paths = self.all_indications_data_manager.get_paths_of_indication(self.indication)

            self.data_frames = {}
            # Use config.event_table_name for the key and filename lookup if consistent
            # Sticking to original keys for loading based on indication_paths structure
            for table_name in ["events", "molecular", "statistics", "constant", "constant_description"]:
                self.data_frames[table_name] = pd.read_csv(indication_paths[table_name])

        #: drop all "Unnamed" columns
        def remove_unnamed_columns(df):
            return df.loc[:, ~df.columns.str.contains("^Unnamed")]
        for key in self.data_frames.keys():
            if self.data_frames[key] is not None:
                self.data_frames[key] = remove_unnamed_columns(self.data_frames[key])

        logging.info(f"Data loaded for indication {self.indication}")

    def process_indication_data(self) -> None:
        """
        Performs initial processing on the loaded indication data.

        Requires `load_indication_data` to be called first.
        This method converts the date columns (specified by `config.date_col`)
        in the 'events' and 'molecular' DataFrames to datetime objects.
        It also checks for and removes rows with missing dates in these tables,
        logging a warning if any are found.

        Raises
        ------
        ValueError
            If `load_indication_data` has not been successfully called before
            this method.
        """

        # Check that we already have self.data_frames
        if not self.data_frames:
            raise ValueError("Data not loaded yet. Please load data first by calling load_indication_data()")

        # Use config.date_col and config.event_table_name
        events_table_key = self.config.event_table_name # "events"
        molecular_table_key = "molecular" # Key remains "molecular" as not defined in config
        date_col = self.config.date_col # "date"

        #: convert for all COL_DATE column in each dataset to datetime
        self.data_frames[events_table_key][date_col] = pd.to_datetime(self.data_frames[events_table_key][date_col])
        self.data_frames[molecular_table_key][date_col] = pd.to_datetime(self.data_frames[molecular_table_key][date_col])

        # Assert that all dates before cutoff date from config
        assert self.data_frames[events_table_key][self.data_frames[events_table_key][date_col] > self.config.date_cutoff].empty, (
            f"Some dates in {events_table_key} are after the config cutoff date {self.config.date_cutoff}"
        )
        assert self.data_frames[molecular_table_key][self.data_frames[molecular_table_key][date_col] > self.config.date_cutoff].empty, (
            f"Some dates in {molecular_table_key} are after the config cutoff date {self.config.date_cutoff}"
        )

        # Check and drop all rows with missing date in events and molecular, and print warning if more than 0
        missing_date_events = self.data_frames[events_table_key][date_col].isnull().sum()
        total_events = len(self.data_frames[events_table_key])
        missing_date_molecular = self.data_frames[molecular_table_key][date_col].isnull().sum()
        total_molecular = len(self.data_frames[molecular_table_key])

        def handle_missing_dates(df_key, missing_count, total_count, col_date):
            if missing_count > 0:
                warning_msg = (
                    f"Found {missing_count} out of {total_count} "
                    f"missing dates in {df_key} for indication {self.indication}"
                )
                logging.warning(warning_msg)
                self.data_frames[df_key] = self.data_frames[df_key].dropna(subset=[col_date])

        # Use table keys and config.date_col
        handle_missing_dates(events_table_key, missing_date_events, total_events, date_col)
        handle_missing_dates(molecular_table_key, missing_date_molecular, total_molecular, date_col)

        logging.info(f"Data processed for indication {self.indication}")


    def setup_unique_mapping_of_events(self) -> None:
        """
        Ensures uniqueness of descriptive event names and applies replacements.

        Requires `load_indication_data` to be called first.
        This method first identifies `event_descriptive_name` values that map to
        multiple `event_name` values within the same `event_category`. For these
        non-unique descriptive names, it appends the corresponding `event_name`
        to make them unique (e.g., "Measurement" becomes "Measurement - Systolic BP").

        Secondly, it applies predefined or overridden special character replacements
        (e.g., replacing "/" with " per " in lab results) to the
        `event_descriptive_name` column based on the `event_category`.

        Finally, it rebuilds the `self.unique_events` mapping (containing unique
        combinations of event_name, event_descriptive_name, and event_category)
        and asserts that all `event_descriptive_name` values are now unique.

        Raises
        ------
        ValueError
            If `load_indication_data` has not been successfully called before
            this method.
        AssertionError
            If, after processing, the `event_descriptive_name` column still
            contains duplicate values.
        """


        # Check that we already have self.data_frames
        if not self.data_frames:
            raise ValueError("Data not loaded yet. Please load data first by calling load_indication_data()")

        # Use config constants for column names
        event_name_col = self.config.event_name_col
        event_desc_name_col = self.config.event_descriptive_name_col
        event_cat_col = self.config.event_category_col
        events_table_key = self.config.event_table_name

        #: get all unique pairs of event_name and event_descriptive_name in self.data_frames["events"]
        self.unique_events = self.data_frames[events_table_key]
        self.unique_events = self.unique_events[[event_name_col, event_desc_name_col, event_cat_col]]
        self.unique_events = self.unique_events.copy().drop_duplicates()
        self.unique_events = self.unique_events.reset_index(drop=True)

        #: get all event_descriptive_name that are not unique
        non_unique_events = self.unique_events[event_desc_name_col].value_counts()
        non_unique_events = non_unique_events[non_unique_events > 1]

        # Extract corresponding event_name and event_category
        filtered_events = self.unique_events[event_desc_name_col]
        non_unique_events = self.unique_events[filtered_events.isin(non_unique_events.index)].copy()

        # create mapping for all non-unique descriptive names, and
        # then add event_name to those, and apply across entire dataset
        # Keep temporary column name as string literal
        non_unique_events["new_descriptive_name"] = (
            non_unique_events[event_desc_name_col] + " - " + non_unique_events[event_name_col]
        )
        # Use config constants for column names
        non_unique_events = non_unique_events[["new_descriptive_name", event_name_col, event_cat_col]]

        self.data_frames[events_table_key] = pd.merge(self.data_frames[events_table_key],
                                                      non_unique_events, how="left",
                                                      on=(event_name_col, event_cat_col)) # Use config constants
        events_df = self.data_frames[events_table_key]
        new_desc_name = "new_descriptive_name" # Keep temporary column name as string literal
        # Use config constant
        events_df[event_desc_name_col] = events_df[new_desc_name].fillna(events_df[event_desc_name_col])
        self.data_frames[events_table_key] = self.data_frames[events_table_key].drop(columns=["new_descriptive_name"])

        #: first convert special symbols in event_descriptive_name to alternatives, using self.replace_special_symbols
        for event_category, (string_to_replace, replacement_string) in self.replace_special_symbols:
            events_df = self.data_frames[events_table_key]
            # Use config constants
            category_mask = events_df[event_cat_col] == event_category
            desc_name_col = event_desc_name_col

            events_df.loc[category_mask, desc_name_col] = (
                events_df.loc[category_mask, desc_name_col]
                .astype(str) # Ensure string type before replace
                .str.replace(string_to_replace, replacement_string, regex=False) # Added regex=False for literal replacement
            )

        #: recalculate self.unique_events and ensure no more non-unique event_descriptive_name
        # Use config constants
        cols_to_select = [event_name_col, event_desc_name_col, event_cat_col]
        self.unique_events = self.data_frames[events_table_key][cols_to_select].copy().drop_duplicates()
        self.unique_events = self.unique_events.reset_index(drop=True)

        # Assert that all unique now
        # Use config constant
        assert len(self.unique_events) == len(self.data_frames[events_table_key][event_desc_name_col].unique())


    def setup_dataset_splits(self, patient_to_split_mapping_path : str = None,
                             patient_to_split_mapping_override : dict = None) -> None:
        """
        Assigns each patient to a data split (train, validation, or test).

        Requires `load_indication_data` to be called first.
        The method determines the split assignment for each patient based on the
        following priority:
        1. If `patient_to_split_mapping_override` is provided, it uses this
           dictionary directly.
        2. If `patient_to_split_mapping_path` is provided, it loads the split
           mapping from the specified JSON file.
        3. Otherwise, it calculates the splits: It retrieves all unique patient IDs
           from the 'constant' data table. It calculates the number of patients
           for validation and test sets based on the `validation_split_max`,
           `test_split_max`, and `max_val_test_nr_patients` parameters set during
           initialization. The remaining patients are assigned to the training set
           (respecting `train_split_min`). Patients are randomly shuffled
           (with a fixed seed for reproducibility) before assignment.

        The resulting mapping (patient ID to split name) is stored in
        `self.patient_to_split_mapping`. It also stores all patient IDs in
        `self.all_patientids`. Asserts are performed to ensure the mapping covers
        all patients without overlap and that the split sizes match calculations.

        Parameters
        ----------
        patient_to_split_mapping_path : str, optional
            Path to a JSON file containing a pre-defined patient ID to split name
            mapping. Defaults to None.
        patient_to_split_mapping_override : dict, optional
            A dictionary directly providing the patient ID to split name mapping.
            Defaults to None.

        Raises
        ------
        ValueError
            If `load_indication_data` has not been successfully called before
            this method.
        FileNotFoundError
            If `patient_to_split_mapping_path` is provided but the file does not exist.
        json.JSONDecodeError
            If the file specified by `patient_to_split_mapping_path` is not valid JSON.
        AssertionError
            If calculated splits do not match expected counts or if overlaps exist.
        """

        # Check that we already have self.data_frames
        if not self.data_frames:
            raise ValueError("Data not loaded yet. Please load data first by calling load_indication_data()")

        # Use config constants
        patient_id_col = self.config.patient_id_col
        constant_table_key = "constant" # Key remains "constant"
        train_split_name = self.config.train_split_name # Use config for "train" split name

        # Load from patient_to_split_mapping_path in case it exists, which is
        # a json and should load into self.patient_to_split_mapping
        if patient_to_split_mapping_path is not None:
            with open(patient_to_split_mapping_path, "r") as f:
                self.patient_to_split_mapping = json.load(f)

            all_patients = self.data_frames[constant_table_key][patient_id_col].unique()
            self.all_patientids = all_patients
            return

        if patient_to_split_mapping_override is not None:
            self.patient_to_split_mapping = patient_to_split_mapping_override
            all_patients = self.data_frames[constant_table_key][patient_id_col].unique()
            self.all_patientids = all_patients
            return

        #: get all patientids from self.data_frames["constant"]
        all_patients = self.data_frames[constant_table_key][patient_id_col].unique()
        self.all_patientids = all_patients

        #: get min(self.validation_split * num_patients, self.max_val_test_nr_patients)
        validation_nr_patients = min(int(self.validation_split * len(all_patients)), self.max_val_test_nr_patients)

        #: then the same for test
        test_nr_patients = min(int(self.test_split * len(all_patients)), self.max_val_test_nr_patients)

        #: randomly shuffle with seed 42 and split into train/val/test, using df.sample
        np.random.seed(42)
        all_patients = np.random.permutation(all_patients)
        train_nr_patients = len(all_patients) - validation_nr_patients - test_nr_patients

        #: setup mapping so that each patientid returns which split it belongs to
        self.patient_to_split_mapping = {}
        # Use config.train_split_name for the train split key/value
        # Keep "validation" and "test" as strings since not defined in config
        self.patient_to_split_mapping.update({patient: train_split_name for patient in all_patients[:train_nr_patients]})
        self.patient_to_split_mapping.update({patient: "validation" for patient in
                                              all_patients[train_nr_patients:train_nr_patients+validation_nr_patients]})
        self.patient_to_split_mapping.update({patient: "test" for patient in
                                              all_patients[train_nr_patients+validation_nr_patients:]})

        #: assert that no overlap in patient mappings
        assert len(self.patient_to_split_mapping) == len(all_patients)

        #: assert that correct lengths
        # Use config.train_split_name for checking train split length
        assert len([patient for patient, split in
                    self.patient_to_split_mapping.items() if split == train_split_name]) == train_nr_patients
        assert len([patient for patient, split in
                    self.patient_to_split_mapping.items() if split == "validation"]) == validation_nr_patients
        assert len([patient for patient, split in
                    self.patient_to_split_mapping.items() if split == "test"]) == test_nr_patients

    def get_patient_split(self, patientid: str) -> str:
        """
        Retrieves the assigned data split for a given patient ID.

        Requires `setup_dataset_splits` to be called first.

        Parameters
        ----------
        patientid : str
            The unique identifier for the patient.

        Returns
        -------
        str
            The name of the data split ("train", "validation", or "test" - note:
            the actual name for "train" might be defined in `config.train_split_name`)
            to which the patient belongs.

        Raises
        ------
        KeyError
            If the provided `patientid` is not found in the
            `patient_to_split_mapping`.
        AttributeError
            If `setup_dataset_splits` has not been called yet and the mapping
            does not exist.
        """
        # Use config constant for patient ID if needed, but here it's just a key lookup
        # patientid is the key itself.
        return self.patient_to_split_mapping[patientid]

    def get_patient_data(self, patientid: str) -> dict:
        """
        Retrieves and consolidates all data for a specific patient.

        Requires `load_indication_data` and `process_indication_data` to have
        been called. It's also recommended to call `setup_unique_mapping_of_events`
        to ensure consistent event naming.

        This method gathers data from the 'events', 'molecular', and 'constant'
        DataFrames for the specified `patientid`.
        - It filters the 'events' and 'molecular' tables for the patient.
        - It renames columns in the 'molecular' data to match the 'events' data
          schema (e.g., biomarker columns mapped to event columns using config).
        - It adds a 'source' column (`config.source_col`) to distinguish between
          original events ("events") and molecular data (`config.source_genetic`).
        - It concatenates the patient's events and molecular data, then sorts the
          combined DataFrame by date (`config.date_col`). Duplicates are dropped.
        - It filters the 'constant' table for the patient's static data.

        Parameters
        ----------
        patientid : str
            The unique identifier for the patient whose data is to be retrieved.

        Returns
        -------
        dict
            A dictionary containing the patient's data, with two keys:
            - "events": A pandas DataFrame containing all time-series events
                        (original events and molecular data combined and sorted
                        by date).
            - "constant": A pandas DataFrame containing the static (constant)
                          data for the patient.

        Raises
        ------
        ValueError
            If `load_indication_data` has not been successfully called before
            this method.
        KeyError
            If essential columns specified in the config are missing from the
            dataframes after loading.
        """

        # Check that we already have self.data_frames
        if not self.data_frames:
            raise ValueError("Data not loaded yet. Please load data first by calling load_indication_data()")

        # Use config constants for column names and table keys/sources where applicable
        patient_id_col = self.config.patient_id_col
        source_col = self.config.source_col
        date_col = self.config.date_col
        events_table_key = self.config.event_table_name # "events"
        molecular_table_key = "molecular" # Key remains "molecular"
        constant_table_key = "constant" # Key remains "constant"
        source_genetic = self.config.source_genetic # "genetic"

        biomarker_cat_col = self.config.biomarker_category_col
        biomarker_event_col = self.config.biomarker_event_col
        biomarker_val_col = self.config.biomarker_value_col
        biomarker_desc_name_col = self.config.biomarker_descriptive_name_col

        event_cat_col = self.config.event_category_col
        event_name_col = self.config.event_name_col
        event_val_col = self.config.event_value_col
        event_desc_name_col = self.config.event_descriptive_name_col


        #: get all data for a specific patient
        patient_data = {}

        #: first from events
        events = self.data_frames[events_table_key][self.data_frames[events_table_key][patient_id_col] == patientid].copy()
        # add events flag (keep "events" string as source value, not in config)
        events[source_col] = "events"

        #: next from molecular
        molecular = self.data_frames[molecular_table_key][self.data_frames[molecular_table_key][patient_id_col] == patientid].copy()

        # Rename biomarker columns to event columns using config constants
        rename_dic = {
            biomarker_cat_col: event_cat_col,
            biomarker_event_col: event_name_col,
            biomarker_val_col: event_val_col,
            biomarker_desc_name_col: event_desc_name_col
        }
        molecular = molecular.rename(columns=rename_dic)
        # add molecular flag using config constant for source value
        molecular[source_col] = source_genetic

        #: insert all entries of molecular at correct points in events, based on the column COL_DATE
        # Use config constant for date column
        patient_data["events"] = pd.concat([events, molecular], axis=0, ignore_index=True).sort_values(date_col)

        #: then from constant
        selected_patient = self.data_frames[constant_table_key][patient_id_col] == patientid
        patient_data["constant"] = self.data_frames[constant_table_key][selected_patient]

        # Remove any duplicates in case they get in events
        # Keep "events" key as string
        patient_data["events"] = patient_data["events"].drop_duplicates()

        #: return
        return patient_data
