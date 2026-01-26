from digital_twin_converter.common.jsonl_converter_base import JSONLConverterBase
from digital_twin_converter.pretrain.converter_manual_template import ConverterManualTemplate
from digital_twin_converter.common.config import Config
import logging
logging.basicConfig(level=logging.NOTSET)
import argparse
import json


class JSONLConverter(JSONLConverterBase):
    """
    Converts patient data for a specific medical indication to the JSON Lines (JSONL) format.

    This class is designed to be executed as a script, processing data for all patients
    associated with a single indication. It generates a JSONL file where each line
    corresponds to one patient's data, formatted for pretraining purposes. The conversion
    logic leverages the `ConverterManualTemplate` class. It inherits common functionalities
    from `JSONLConverterBase`.

    Attributes:
        indication (str): The medical indication being processed.
        save_path (str): Directory path where the output JSONL files will be saved.
        converter (ConverterManualTemplate): An instance used for the actual data-to-text conversion.
        dm (DataManager): Inherited or initialized by the base class, used for data access.
        # Other attributes from base class initialization are implicitly present.
    """


    def __init__(self, indication : str, save_path : str,
                 delete_previous_files = True,
                 reverse_patient_ratio_assesed = 0.01,
                 reverse_patient_skip_list_override = None,
                 saving_interval = 0.1,
                 wandb_group = None,
                 indication_split_path = None,
                 config : Config = None) -> None:
        """
        Initializes the JSONLConverter instance.

        Sets up the converter for a specific indication, configuring paths, conversion parameters,
        and logging options.

        Parameters
        ----------
        indication : str
            The identifier for the medical indication whose data is to be converted (e.g., 'enhanced_nsclc').
        save_path : str
            The base directory path where the output JSONL file and associated metadata (like splits) will be stored.
        delete_previous_files : bool, optional
            If True, existing files in the save path for this indication might be removed before conversion. Defaults to True.
        reverse_patient_ratio_assesed : float, optional
            The fraction of patients for whom the reverse conversion (text-to-data) process will be tested for accuracy. Defaults to 0.01 (1%).
        reverse_patient_skip_list_override : list, optional
            A list of specific event values or types to ignore when comparing original and reverse-converted data during assessment. Defaults to None.
        saving_interval : float, optional
            The fraction of total patients processed between saving intermediate results to the JSONL file. Helps prevent data loss on large datasets. Defaults to 0.1 (10%).
        wandb_group : str, optional
            A group name for organizing runs in Weights & Biases logging, if used. Defaults to None.
        indication_split_path : str, optional
            Path to a JSON file defining pre-existing train/validation/test splits for patients of this indication. Ensures consistency across runs. Defaults to None.
        config : Config, optional
            A configuration object containing settings like column names and data paths. If None, a default configuration might be loaded by the base class. Defaults to None.
        """

        # Call super
        super().__init__(indication, save_path, delete_previous_files, reverse_patient_ratio_assesed,
                        reverse_patient_skip_list_override, saving_interval, wandb_group,
                        indication_split_path, config)

        #: set up converter
        self.converter = ConverterManualTemplate(config=self.config)



    def _convert_patient(self, patientid : str) -> tuple[dict, dict]:
        """
        Converts the raw data of a single patient into the desired text format.

        Retrieves the patient's event and constant data using the DataManager (`self.dm`),
        performs the forward conversion using `self.converter`, formats the output metadata
        (converting DataFrames to JSON strings for serialization), and prepares the internal
        metadata used for potential reverse conversion assessment.

        Parameters
        ----------
        patientid : str
            The unique identifier of the patient whose data needs conversion.

        Returns
        -------
        list[tuple[dict, dict]]
            A list containing a single tuple. The tuple consists of:
            - The primary converted data dictionary, structured as:
              {
                  "text": <The textual representation of the patient's data>,
                  "meta": {
                      "patientid": <str>,
                      "indication": <str>,
                      "split": <str, e.g., 'train', 'val', 'test'>,
                      "constant": <JSON string representation of processed constant data>,
                      "events": <JSON string representation of processed event data>
                  }
              }
            - The internal metadata dictionary, containing the raw and processed DataFrames
              and other details needed internally, particularly for reverse conversion checks.
        """

        #: get patient data
        patient_data = self.dm.get_patient_data(patientid)

        #: convert patient data using ConverterManualTemplate
        p_converted = self.converter.forward_conversion(patient_data["events"],
                                                        patient_data["constant"],
                                                        self.dm.data_frames["constant_description"])

        #: convert extras into JSON using df.to_json
        internal_meta = p_converted["meta"].copy()
        internal_meta["split"] = self.dm.get_patient_split(patientid=patientid)
        p_converted["meta"] = {
            "patientid" : patientid,
            "indication" : self.indication,
            "split" : self.dm.get_patient_split(patientid=patientid),
            "constant" : p_converted["meta"]["processed_constant"].to_json(orient="split"),
            "events" : p_converted["meta"]["events"].to_json(orient="split"),
        }

        #: return as dict
        return [(p_converted, internal_meta)]


    def _assess_reverse_conversion(self, all_patient_data) -> None:
        """
        Assesses the accuracy of the reverse conversion process for a single patient.

        Takes the converted data and internal metadata for one patient (as returned by
        `_convert_patient`), performs the reverse conversion (text back to structured data)
        using `self.converter.reverse_conversion`, and compares the resulting event DataFrame
        against the original event DataFrame stored in the internal metadata. It asserts that
        there are no discrepancies, considering any specified skips (`self.reverse_patient_skip_list`).
        Logs and raises an assertion error if differences are found.

        Parameters
        ----------
        all_patient_data : list[tuple[dict, dict]]
            The list containing the tuple of converted data and internal metadata for the
            patient being assessed, as produced by `_convert_patient`.
        """

        # Split up
        converted_data, internal_meta = all_patient_data[0]

        # Log that testing patient
        logging.info("Assessing reverse conversion for patient" + str(converted_data["meta"]['patientid']))

        #: do reverse conversion using ConverterManualTemplate
        p_reverse_converted = self.converter.reverse_conversion(converted_data["text"],
                                                                internal_meta,
                                                                self.dm.unique_events)

        #: check differences appropriately
        diff = self.converter.get_difference_in_event_dataframes(internal_meta["events"], p_reverse_converted["events"],
                                                                 skip_genetic=True,
                                                                 skip_vals_list=self.reverse_patient_skip_list)

        #: assert that no differences are found, and print patientid if issues are found
        assert diff.shape[0] == 0, f"Patient {internal_meta['patientid']} has differences in reverse conversion: {diff}"



if __name__ == "__main__":

    # Usage: python JSONLConverter.py --indication indication_id --save_path /path/to/save --wandb_group example_group
    parser = argparse.ArgumentParser(description='Convert indication to JSONL.')

    # Add the arguments
    parser.add_argument('--indication_id', type=int, required=True, help='Indication ID')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output')
    parser.add_argument('--wandb_group', type=str, required=True, help='WandB group name')
    parser.add_argument('--delete_previous_files', type=bool, default=True,
                        help='Delete previous files, in case they were already generated previously')
    parser.add_argument('--indication_split_path', type=str,
                    default="/flatiron_cgdb/jsonl/2024_07_11/splits_",
                    help='Path to the indication split file, to ensure consistent train/val/test splits')
    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    indication_id = args.indication_id
    save_path = args.save_path
    wandb_group = args.wandb_group
    delete_previous_files = args.delete_previous_files
    base_indication_split_path = args.indication_split_path

    all_indications = ['enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
                       'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
                       'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
                       'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
                       'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma']

    indication = all_indications[indication_id]
    indication_split_path = base_indication_split_path + indication + ".json"

    j = JSONLConverter(indication, save_path, wandb_group=wandb_group, delete_previous_files=delete_previous_files,
                       indication_split_path=indication_split_path)
    j.convert_indication()

    # also save for the indication the get_all_patientid_splits to ensure consistent splits
    with open(save_path + "/splits_" + indication + ".json", "w") as f:
        json.dump(j.dm.patient_to_split_mapping, f)
