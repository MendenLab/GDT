import logging
logging.basicConfig(level=logging.NOTSET)
import argparse
import json
import pandas as pd
import os
from transformers import AutoTokenizer

from digital_twin_converter.instruction.data_splitter_events import DataSplitterEvents
from digital_twin_converter.instruction.data_splitter_forecasting import DataSplitterForecasting
from digital_twin_converter.instruction.converter_manual_instruction import (
    ConverterManualInstruction,
)
from digital_twin_converter.common.converter_manual_base import JSONLConverterBase
from digital_twin_converter.common.config import Config

class JSONLConverterInstruction(JSONLConverterBase):
    """
    Handles the conversion of patient data into instruction-based JSONL format for
    an entire indication.

    This class orchestrates the process of taking raw patient data, splitting it
    into historical context and future prediction targets (for both forecasting
    and event prediction tasks), converting these splits into structured
    instruction/answer pairs suitable for language models, and saving the results
    in JSONL format. It manages data splitting, token budgeting, conversion logic,
    and reverse conversion checks for data integrity.

    Attributes:
        max_num_samples_per_patient_forecasting (int): Max forecasting samples per patient.
        max_num_samples_per_patient_events (int): Max event samples per patient.
        max_num_samples_per_lot (int): Max samples per Line of Therapy (LoT).
        splitter_forecasting (DataSplitterForecasting): Instance for creating forecasting splits.
        splitter_events (DataSplitterEvents): Instance for creating event prediction splits.
        converter (ConverterManualInstruction): Instance for converting splits to instruction format.
        tokenizer (AutoTokenizer): Tokenizer used for counting tokens in generated text.
        stats (dict | None): Dictionary to store statistics about the conversion process. Initialized lazily.
    """


    def __init__(self,
                 max_num_samples_per_patient_forecasting : int,
                 max_num_samples_per_patient_events : int,
                 indication : str, save_path : str,
                 indication_split_path : str,
                 nr_tokens_budget_total : int,
                 max_length_of_weeks_to_sample : int,
                 delete_previous_files = True,
                 reverse_patient_ratio_assesed = 0.01,
                 reverse_patient_skip_list_override = None,
                 saving_interval = 0.025,
                 wandb_group = None,
                 tokenizer_to_load_for_counting = 'meta-llama/Meta-Llama-3-8B',
                 max_num_samples_per_lot = 1,
                 config : Config = None) -> None:
        """
        Initializes the JSONLConverterInstruction instance.

        Sets up the necessary components for the conversion process, including
        data splitters for forecasting and events, the instruction converter,
        and the tokenizer for token counting. It also initializes configuration
        parameters and prepares the save directory.

        Args:
            max_num_samples_per_patient_forecasting: The maximum number of
                forecasting samples to generate per patient. More splits might be
                generated initially, but only this many are randomly selected.
            max_num_samples_per_patient_events: The maximum number of event
                prediction samples to generate per patient, aligned with the
                forecasting split dates.
            indication: The specific medical indication (e.g., 'enhanced_multiplemyeloma')
                being processed.
            save_path: The directory path where the output JSONL files and
                statistics will be saved.
            indication_split_path: Path to the JSON file containing the predefined
                train/validation/test split for patient IDs for this indication.
            nr_tokens_budget_total: The target maximum number of tokens (input + output)
                for each generated instruction sample. The converter will attempt
                to stay within this budget.
            max_length_of_weeks_to_sample: The maximum duration (in weeks) of
                patient history to potentially include in the input context for
                event prediction tasks.
            delete_previous_files: If True, existing files in the 'save_path'
                for this indication will be deleted before starting the conversion.
                Defaults to True.
            reverse_patient_ratio_assesed: The fraction of patients for whom
                a reverse conversion check will be performed to ensure data integrity.
                Defaults to 0.01 (1%).
            reverse_patient_skip_list_override: An optional list of patient IDs
                to explicitly exclude from the reverse conversion check.
            saving_interval: The fraction of total patients after which intermediate
                results should be saved. Defaults to 0.025 (2.5%).
            wandb_group: An optional string specifying the Weights & Biases group name
                for logging metrics during the conversion process.
            tokenizer_to_load_for_counting: The Hugging Face model identifier for the
                tokenizer used to count input and output tokens. Defaults to
                'meta-llama/Meta-Llama-3-8B'.
            max_num_samples_per_lot: The maximum number of samples to generate per unique
                Line of Therapy (LoT) within a patient's timeline. Defaults to 1.
            config: A Config object containing shared configuration settings (e.g., seed,
                task prompts). If None, a default Config object is created.
        """

        # Call super
        super().__init__(indication, save_path, delete_previous_files, reverse_patient_ratio_assesed,
                        reverse_patient_skip_list_override, saving_interval, wandb_group,
                        indication_split_path, config)

        seed = config.seed

        self.max_num_samples_per_patient_forecasting = max_num_samples_per_patient_forecasting
        self.max_num_samples_per_patient_events = max_num_samples_per_patient_events
        self.max_num_samples_per_lot = max_num_samples_per_lot

        self.splitter_forecasting = DataSplitterForecasting(data_manager=self.dm, config=self.config)
        self.splitter_forecasting.setup_statistics()
        self.splitter_forecasting.variable_stats.to_csv(os.path.join(save_path, str(indication)
                                                                     + "_variable_stats.csv"))

        self.splitter_events = DataSplitterEvents(self.dm,
                                                  config=self.config,
                                                  max_length_of_weeks_to_sample=max_length_of_weeks_to_sample)
        self.splitter_events.setup_variables()

        self.converter = ConverterManualInstruction(self.dm.data_frames["constant_description"],
                                                    nr_tokens_budget_total = nr_tokens_budget_total,
                                                    seed=seed, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_load_for_counting)

    def _get_token_count(self, string_to_count : str) -> int:
        """
        Calculates the number of tokens in a given string using the initialized tokenizer.

        Args:
            string_to_count: The text string to tokenize and count.

        Returns:
            The number of tokens generated by the tokenizer for the input string.
        """
        return len(self.tokenizer(string_to_count)["input_ids"])

    def _convert_patient(self, patientid : str) -> tuple[dict, dict]:
        """
        Processes a single patient's data to generate instruction-based samples.

        Retrieves the patient's data, generates potential split points for
        forecasting and event prediction tasks based on the configured parameters.
        For each valid split, it converts the historical data and prediction targets
        into an instruction/answer pair using the `ConverterManualInstruction`.
        It calculates token counts, formats metadata, and collects statistics.

        Args:
            patientid: The unique identifier for the patient whose data is to be converted.

        Returns:
            A list of tuples. Each tuple represents one generated sample for the
            patient and contains:
            - A dictionary representing the final JSONL row (including 'instruction',
              'answer', and formatted 'meta' data).
            - A dictionary containing internal metadata ('internal_meta') used primarily
              for reverse conversion checks, including the original dataframes before
              serialization.
            Returns an empty list if no valid splits can be generated for the patient.
        """

        #: get patient data
        patient_data = self.dm.get_patient_data(patientid)
        ret_list = []

        # Setup stats if needed
        if self.stats is None:
            self.stats = {
                "name" : "instruction",
                "num_patients_no_splits" : 0,
                "num_patients" : 0,
                "num_total_samples": 0,
                "num_tasks_forecasting": 0,
                "num_tasks_forecasting_qa": 0,
                "num_tasks_events": 0,
                "num_tasks_forecasting_OR_qa": 0,
                "num_samples_per_patient": [],
                "num_tokens_input": [],
                "num_tokens_target": [],
            }
        num_samples = 0

        #: generate splits (already randomized)
        processed_splits_fc, split_dates = self.splitter_forecasting.get_splits_from_patient(patient_data,
                                                                                             nr_samples=self.max_num_samples_per_patient_forecasting,
                                                                                             apply_filtering=True,
                                                                                             return_splits=True,
                                                                                             max_num_samples_per_lot=self.max_num_samples_per_lot)

        processed_splits_ev = self.splitter_events.get_splits_from_patient(patient_data,
                                                                           preselected_split_dates=split_dates,
                                                                           max_nr_samples=self.max_num_samples_per_patient_events,
                                                                        max_num_samples_per_lot=self.max_num_samples_per_lot)

        # Skip if no splits available
        if processed_splits_fc == [None] and processed_splits_ev == []:
            self.stats["num_patients_no_splits"] += 1
            logging.info(f"Patient {patientid} has no splits available.")
            return []
        else:
            self.stats["num_patients"] += 1

        # Check that all correct
        assert len(processed_splits_ev) == len(processed_splits_fc), "Lengths do not match."

        #: loop over splits
        for split_idx in range(len(processed_splits_fc)):

            #: convert
            p_converted = self.converter.forward_conversion(forecasting_splits=processed_splits_fc[split_idx],
                                                            event_splits=processed_splits_ev[split_idx],
                                                            variable_stats=self.splitter_forecasting.variable_stats)

            #: add meta and statistics
            internal_meta = p_converted["meta"].copy()
            internal_meta["split"] = self.dm.get_patient_split(patientid=patientid)
            target_meta = []
            num_forecasting = 0
            num_forecasting_qa = 0
            num_events = 0
            num_input_tokens = self._get_token_count(p_converted["instruction"])
            num_target_tokens = self._get_token_count(p_converted["answer"])
            self.stats["num_tokens_input"].append(num_input_tokens)
            self.stats["num_tokens_target"].append(num_target_tokens)

            for task_meta in internal_meta["target_meta_detailed"]:

                # Add statistics
                if task_meta["task_type"] == self.config.task_prompt_events:
                    self.stats["num_tasks_events"] += 1
                    num_events += 1

                elif task_meta["task_type"] == self.config.task_prompt_forecasting:
                    self.stats["num_tasks_forecasting"] += 1
                    num_forecasting += 1

                elif task_meta["task_type"] == self.config.task_prompt_forecasting_qa:
                    self.stats["num_tasks_forecasting_qa"] += 1
                    num_forecasting_qa += 1

                # Get meta data
                new_task_meta = {}
                for k, v in task_meta.items():
                    if (not isinstance(v, pd.DataFrame) and not isinstance(v, pd.arrays.DatetimeArray)
                        and not isinstance(v, pd.Series)):
                        new_task_meta[k] = v
                    if isinstance(v, pd.DataFrame):
                        new_task_meta[k] = v.to_json(orient="split")
                    if isinstance(v, pd.Series):
                        new_task_meta[k] = v.to_json(orient="split", default_handler=str)
                    if isinstance(v, pd.arrays.DatetimeArray):
                        new_task_meta[k] = list(v.astype("string"))
                target_meta.append(new_task_meta)

            p_converted["meta"] = {
                "patientid" : patientid,
                "indication" : self.indication,
                "split" : self.dm.get_patient_split(patientid=patientid),
                "forecasting_type" : "instructions",
                "split_date_included_in_input": p_converted["meta"]["split_date_included_in_input"],
                "history_data" : p_converted["meta"]["history_data"].to_json(orient="split"),
                "constant" : p_converted["meta"]["constant_data"].to_json(orient="split"),
                "target_meta" : target_meta,
                "num_tokens_input": num_input_tokens,
                "num_tokens_target": num_target_tokens,
                "num_tokens_total": num_input_tokens + num_target_tokens,
            }

            # Add to main list
            ret_list.append((p_converted, internal_meta))

            #: record stats
            self.stats["num_total_samples"] += 1
            self.stats["num_tasks_forecasting_OR_qa"] += 1 if num_forecasting > 0 or num_forecasting_qa > 0 else 0
            num_samples += 1

        #: record stats
        self.stats["num_samples_per_patient"].append(num_samples)

        #: return as dict
        return ret_list


    def _assess_reverse_conversion(self, all_patient_data) -> None:
        """
        Performs a reverse conversion check on the generated samples for a patient.

        Iterates through the samples generated by `_convert_patient` for a specific
        patient. For each sample, it takes the generated 'answer' string and attempts
        to convert it back into structured data using the `converter.reverse_conversion`
        method. It then compares this reversed data with the original target data
        stored in the `internal_meta`. An assertion error is raised if any
        discrepancies are found, indicating a potential issue in the conversion or
        reverse conversion logic.

        Args:
            all_patient_data: A list of tuples, where each tuple contains the
                converted sample data (dict) and its corresponding internal metadata (dict),
                as returned by `_convert_patient`.

        Raises:
            AssertionError: If the reverse-converted data does not match the original
                target data for any of the provided samples.
        """

        #: loop through all samples for current patient
        for sample in all_patient_data:

            # Split up
            converted_data, internal_meta = sample

            # Log that testing patient
            logging.info("Assessing reverse conversion for patient: " + str(converted_data["meta"]['patientid']))

            #: get the reverse conversion
            reverse  = self.converter.reverse_conversion(target_string=converted_data["answer"],
                                                         data_manager=self.dm,
                                                         split_date=converted_data["meta"]["split_date_included_in_input"])

            #: get difference
            diff = self.converter.get_difference_in_event_dataframes(internal_meta["target_meta_detailed"], reverse)

            #: assert that no differences are found, and print patientid if issues are found
            assert all(curr_diff.shape[0] == 0 for curr_diff in diff), (f"Patient {internal_meta['patientid']} has "
                                                                        f"differences in reverse conversion: {diff}")



if __name__ == "__main__":

    # Usage: python JSONLConverter.py --indication indication_id --save_path /path/to/save --wandb_group example_group
    #                                 --forecasting_type [forecasting/qa] --max_num_samples_per_patient_forecasting 5
    #                                 --nr_tokens_budget_total 8192 --max_num_samples_per_patient_events 5
    parser = argparse.ArgumentParser(description='Convert indication to JSONL.')

    # Add the arguments
    parser.add_argument('--indication_id', type=int, required=True, help='Indication ID')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output')
    parser.add_argument('--wandb_group', type=str, required=True, help='WandB group name')
    parser.add_argument('--max_num_samples_per_patient_forecasting', type=int, required=True,
                        help='Max number of samples per patient for forecasting to generate (1 picked randomly)')
    parser.add_argument('--max_num_samples_per_patient_events', type=int, required=True,
                        help='Maximum number of samples per patient for events.')
    parser.add_argument('--nr_tokens_budget_total', type=int, required=True,
                        help='Total budget of tokens available for conversion')
    parser.add_argument('--delete_previous_files', type=bool, default=True,
                        help='Delete previous files, in case they were already generated previously')
    parser.add_argument('--indication_split_path', type=str,
                        default="/flatiron_cgdb/jsonl/2024_07_11/splits_",
                        help='Path to the indication split file')
    parser.add_argument('--max_length_of_weeks_to_sample', type=int, default=104,
                        help='Maximum length of weeks to sample')
    parser.add_argument('--max_num_samples_per_lot', type=int, default=1,
                        help='Max number of samples to generate per LoT')


    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    indication_id = args.indication_id
    save_path = args.save_path
    wandb_group = args.wandb_group
    delete_previous_files = args.delete_previous_files
    base_indication_split_path = args.indication_split_path
    max_num_samples_per_patient_forecasting = args.max_num_samples_per_patient_forecasting
    max_num_samples_per_patient_events = args.max_num_samples_per_patient_events
    nr_tokens_budget_total = args.nr_tokens_budget_total
    max_length_of_weeks_to_sample = args.max_length_of_weeks_to_sample
    max_num_samples_per_lot = args.max_num_samples_per_lot



    all_indications = ['enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
                       'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
                       'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
                       'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
                       'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma']
    indication = all_indications[indication_id]
    indication_split_path = base_indication_split_path + indication + ".json"

    j = JSONLConverterInstruction(max_num_samples_per_patient_forecasting=max_num_samples_per_patient_forecasting,
                                  max_num_samples_per_patient_events=max_num_samples_per_patient_events,
                                  indication=indication, save_path=save_path,
                                  nr_tokens_budget_total=nr_tokens_budget_total,
                                  wandb_group=wandb_group, delete_previous_files=delete_previous_files,
                                  indication_split_path=indication_split_path,
                                  max_length_of_weeks_to_sample=max_length_of_weeks_to_sample,
                                  max_num_samples_per_lot=max_num_samples_per_lot)
    j.convert_indication()

    # also save for the indication the get_all_patientid_splits to ensure consistent splits
    with open(save_path + "/splits_" + indication + ".json", "w") as f:
        json.dump(j.dm.patient_to_split_mapping, f)
