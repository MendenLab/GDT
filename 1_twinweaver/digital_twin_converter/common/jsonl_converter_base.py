import pandas as pd
import numpy as np
from digital_twin_converter.common.data_manager import SingleIndicationDataManager
import logging
import json
import os
import wandb
from datetime import datetime
import time
from digital_twin_converter.common.config import Config
import glob
logging.basicConfig(level=logging.NOTSET)



class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy data types and datetime objects.

    Overrides the default JSONEncoder behavior to serialize common NumPy types
    (integers, floats, arrays, booleans) and datetime objects into standard
    JSON-compatible formats (int, float, list, bool, str).
    """
    def default(self, obj):
        dtypes = (np.datetime64, np.complexfloating, datetime)
        if isinstance(obj, dtypes):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            if any([np.issubdtype(obj.dtype, i) for i in dtypes]):
                return obj.astype(str).tolist()
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)






class JSONLConverterBase:
    """
    Abstract base class for converting structured patient data into JSON Lines format.

    This class provides a framework for converting patient data, managed by
    `SingleIndicationDataManager`, for a specific indication into a JSONL file where
    each line typically represents one patient record. It handles the overall workflow,
    including data loading, patient iteration, optional reverse conversion checks for
    data integrity, periodic saving, and statistics logging (optionally to Weights & Biases).

    Concrete implementations must inherit from this class and override the
    `_convert_patient` and `_assess_reverse_conversion` methods to define the
    specific conversion logic and validation checks.
    """


    def __init__(self, indication : str, save_path : str,
                 delete_previous_files : bool = True,
                 reverse_patient_ratio_assesed : float = 0.01,
                 reverse_patient_skip_list_override : list = None,
                 saving_interval : float = 0.1,
                 wandb_group : str = None,
                 indication_split_path : str = None,
                 config : Config = None) -> None:
        """
        Initializes the JSONLConverterBase.

        Sets up the converter for a specific indication, configures paths,
        conversion parameters, logging, and initializes the data manager.
        Optionally deletes pre-existing output files for the indication.

        Parameters
        ----------
        indication : str
            The name of the indication (e.g., "NSCLC") whose data is being converted.
        save_path : str
            The base directory where the output JSONL files (split by train/val/test)
            and statistics files will be saved.
        delete_previous_files : bool, optional
            If True, deletes any files in `save_path` starting with the indication name
            before starting the conversion. Defaults to True.
        reverse_patient_ratio_assesed : float, optional
            The proportion of patients (randomly selected) for whom the reverse
            conversion check (`_assess_reverse_conversion`) should be performed.
            Defaults to 0.01 (1%).
        reverse_patient_skip_list_override : list, optional
            A list of event names (or identifiers) to explicitly skip during the
            reverse conversion assessment. If None, a default list is used.
            Defaults to None.
        saving_interval : float, optional
            The fraction of total patients processed between saving intermediate
            results to the JSONL files. E.g., 0.1 means saving after every 10%
            of patients are converted. Defaults to 0.1.
        wandb_group : str, optional
            If provided, initializes Weights & Biases logging for the specified project
            (from config) and assigns this run to the given group name. Defaults to None.
        indication_split_path : str, optional
            Path to a JSON file containing a pre-defined patient ID to split mapping.
            This is passed to `SingleIndicationDataManager` to ensure consistent
            train/validation/test splits. If None, the data manager calculates splits.
            Defaults to None.
        config : Config, optional
            A configuration object. If None, a default `Config()` object is created.
            Defaults to None.
        """

        # Set up config
        if config is None:
            self.config = Config()
        else:
            self.config = config

        #: set up W&B
        self.wandb_group = wandb_group
        if wandb_group is not None:
            wandb.init(project=self.config.wandb_project, group=wandb_group)
            wandb.run.name = indication + "_conversion_" + str(pd.Timestamp.now())

        # Set basics
        self.indication = indication
        self.base_save_path = save_path
        self.reverse_patient_ratio_assesed = reverse_patient_ratio_assesed
        self.saving_interval = saving_interval
        if reverse_patient_skip_list_override is not None:
            self.reverse_patient_skip_list = reverse_patient_skip_list_override
        else:
            self.reverse_patient_skip_list = ["electrophoresis m spike"]
        self.delete_previous_files = delete_previous_files
        self.seed = self.config.seed

        #: Setup data manager & load in data
        dm = SingleIndicationDataManager(self.indication, config=self.config)
        dm.load_indication_data()
        dm.process_indication_data()
        dm.setup_unique_mapping_of_events()
        dm.setup_dataset_splits(patient_to_split_mapping_path=indication_split_path)
        self.dm = dm
        self.converter = None

        # Setup stats
        self.stats = None

        # Check if self.base_save_path + self.indication* files exists and delete all they do
        if self.delete_previous_files:
            for file_path in glob.glob(os.path.join(self.base_save_path, self.indication + '*')):
                os.remove(file_path)
                logging.info(f"File {file_path} has been deleted.")



    def convert_indication(self) -> None:
        """
        Orchestrates the conversion process for all patients of the indication.

        Retrieves all patient IDs from the data manager, shuffles them randomly
        (using the configured seed), and iterates through each patient. For each
        patient, it calls the `_convert_patient` method. For a randomly selected
        subset of patients (determined by `reverse_patient_ratio_assesed`), it
        calls `_assess_reverse_conversion`.

        The converted data (as JSON strings) is collected and periodically saved
        to the appropriate train/validation/test JSONL files using `_save_jsonl`
        based on the `saving_interval`. Finally, any remaining data is saved,
        collected statistics are saved to a JSON file, and logged to W&B if enabled.

        Raises
        ------
        AssertionError
            If `self.converter` has not been set by the subclass before calling this method.
        """

        assert self.converter is not None, "Converter not set up"

        #: setup entire indication & randomly shuffle patientids
        all_patientids = self.dm.all_patientids.copy()
        np.random.seed(self.seed)  # I like this number
        all_patientids = np.random.permutation(all_patientids)
        jsonl_to_save = []
        last_time = time.time()

        #: randomly sample patients for testing
        # This is used to randomly sample patients for reverse conversion testing
        # This is done since reverse testing makes sure that no edge cases come through
        # However, reverse testing is computationally expensive, so we only do it for a small subset
        patientids_to_test = np.random.choice(all_patientids,
                                              round(len(all_patientids) * self.reverse_patient_ratio_assesed),
                                              replace=False)


        for idx, patientid in enumerate(all_patientids):

            if idx % 100 == 0:
                logging.info((f"Converting patient {idx + 1} of {len(all_patientids)} "
                              f"taking from last message {time.time() - last_time} seconds"))
                last_time = time.time()

            #: go through all patients and convert them
            patient_data = self._convert_patient(patientid)

            #: at appropriate intervals, assess reverse conversion
            if patientid in patientids_to_test:
                self._assess_reverse_conversion(patient_data)

            #: convert to JSON and add to list
            json_patient_data = [(json.dumps(x[0], cls=NpEncoder), x[1]) for x in patient_data]
            jsonl_to_save.extend(json_patient_data)

            #: periodically save converted data to JSONL file (every x% of patients converted)
            if idx % round(len(all_patientids) * self.saving_interval) == 0:
                self._save_jsonl(jsonl_to_save)
                jsonl_to_save = []

        #: save final version
        self._save_jsonl(jsonl_to_save)

        # Save stats
        if self.stats is not None:
            assert "name" in self.stats, "Stats must have a 'name'"
            logging.info("Stats: ")
            stats_path = os.path.join(self.base_save_path, self.indication + "_" + self.stats["name"] + "_stats.json")
            with open(stats_path, "w") as f:
                json.dump(self.stats, f, cls=NpEncoder)
            logging.info(f"Saved stats to {stats_path}")

            # Log to wandb
            if self.wandb_group is not None:
                wandb.log(self.stats)

        # Wrap up job
        logging.info("Conversion completed")
        if self.wandb_group is not None:
            wandb.finish()


    def _convert_patient(self, patientid : str) -> list[tuple[dict, dict]]:
        """
        Abstract method to convert a single patient's data.

        Subclasses must implement this method to define the logic for transforming
        the raw data of a single patient (retrieved via `self.dm.get_patient_data(patientid)`)
        into the desired dictionary format suitable for JSONL output. It should also
        generate associated metadata for the patient record.

        Parameters
        ----------
        patientid : str
            The unique identifier of the patient to convert.

        Returns
        -------
        list[tuple[dict, dict]]
            A list containing one or more tuples. Each tuple represents a record
            to be written to the JSONL file and consists of:
            - dict: The converted patient data dictionary.
            - dict: A metadata dictionary for this record, which MUST include a
                    "split" key indicating the data split ('train', 'validation',
                    'test' - retrieved via `self.dm.get_patient_split(patientid)`).

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError()


    def _assess_reverse_conversion(self, converted_data : dict, internal_meta : dict) -> None:
        """
        Abstract method to assess the reversibility of the conversion for a patient.

        Subclasses should implement this method to perform checks that validate
        the conversion process. This typically involves attempting to parse or
        partially reconstruct the original data format or key information from the
        `converted_data` dictionary and comparing it against the original source
        data (which might need to be retrieved again or passed via `internal_meta`).
        This helps catch errors or inconsistencies introduced during conversion.

        Parameters
        ----------
        converted_data : dict
            The dictionary representing the patient's data after conversion by
            `_convert_patient`.
        internal_meta : dict
            The metadata dictionary associated with the converted record, which might
            contain information useful for the reverse check (e.g., original patient ID).

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        AssertionError (or other Exceptions)
            If the reverse conversion check fails, indicating a potential issue
            with the conversion logic.
        """
        raise NotImplementedError()


    def _save_jsonl(self, jsonl_and_meta_to_save : list) -> None:
        """
        Appends converted patient data (as JSON strings) to the appropriate JSONL files.

        Separates the provided list of JSON strings based on the "split" value
        found in their associated metadata dictionaries ('train', 'validation', 'test').
        It then appends each JSON string to the corresponding file within the
        `base_save_path` directory (e.g., `indication_train.jsonl`).

        Parameters
        ----------
        jsonl_and_meta_to_save : list[tuple[str, dict]]
            A list where each element is a tuple containing:
            - str: The JSON string representation of a single converted patient record.
            - dict: The metadata dictionary for that record, containing at least a
                    "split" key.
        """

        # First get the splits
        jsonl_to_save_expanded = [(jsonl, m, m["split"]) for jsonl, m in jsonl_and_meta_to_save]

        all_splits = list(set([split for _, _, split in jsonl_to_save_expanded]))

        for split in all_splits:

            # Set current save
            save_path = os.path.join(self.base_save_path, self.indication + "_" + split + ".jsonl")

            # Set jsonl to save
            jsonl_to_save = [jsonl for jsonl, m, s in jsonl_to_save_expanded if s == split]

            #: save jsonl_to_save to self.save_path
            with open(save_path, "a") as f:
                for json_patient_data in jsonl_to_save:
                    f.write(json_patient_data + "\n")

            logging.info(f"Saved {len(jsonl_to_save)} samples to {save_path}")
