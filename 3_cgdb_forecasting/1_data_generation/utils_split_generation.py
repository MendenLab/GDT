import pandas as pd
from transformers import AutoTokenizer
import logging
logging.basicConfig(level=logging.NOTSET)
import numpy as np


from digital_twin_converter.instruction.data_splitter_forecasting import DataSplitterForecasting
from digital_twin_converter.common.data_manager import SingleIndicationDataManager
from digital_twin_converter.instruction.data_splitter_events import DataSplitterEvents
from digital_twin_converter.instruction.converter_manual_instruction import ConverterManualInstruction
from digital_twin_converter.common.config import Config


PATH_TO_SPLITS = "/flatiron_cgdb/instruction/combined/2024_11_14_10_lots_per_patient/"


class SplitterGenerator:

    def __init__(self,
                 indication,
                 nr_variables_to_sample=3,
                 tokenizer_to_load_for_counting = 'meta-llama/Meta-Llama-3-8B',
                 base_path_to_splits=PATH_TO_SPLITS):

        # Set basics
        self.indication = indication
        self.config = Config()

        # Set empties
        self.dm = None
        self.data_splitter_forecasting = None
        self.data_splitter_events = None
        self.converter = None
        self.forecasting_variable_stats = None
        self.min_nr_variables_to_sample = nr_variables_to_sample
        self.max_nr_variables_to_sample = nr_variables_to_sample

        # Init data managers
        path_to_splits = base_path_to_splits + "splits_" + self.indication + ".json"
        self._init_data_managers(patient_to_split_mapping_path=path_to_splits)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_load_for_counting)


    def _init_data_managers(self, patient_to_split_mapping_path):

        print(f"Preloading & saving data manager for {self.indication} - might take some minutes.")

        self.dm = SingleIndicationDataManager(self.indication, config=self.config)
        self.dm.load_indication_data()
        self.dm.process_indication_data()
        self.dm.setup_unique_mapping_of_events()
        self.dm.setup_dataset_splits(patient_to_split_mapping_path=patient_to_split_mapping_path)

        self.data_splitter_events = DataSplitterEvents(self.dm, config=self.config, max_length_of_weeks_to_sample=104)
        self.data_splitter_events.setup_variables()
        self.data_splitter_forecasting = DataSplitterForecasting(data_manager=self.dm, config=self.config,
                                                                 min_nr_variables_to_sample=self.min_nr_variables_to_sample,
                                                                 max_nr_variables_to_sample=self.max_nr_variables_to_sample)
        self.data_splitter_forecasting.setup_statistics()
        self.converter = ConverterManualInstruction(self.dm.data_frames["constant_description"],
                                                nr_tokens_budget_total=8192,
                                                config=self.config)


    def get_all_split_patientids(self, split):
        return [patient for patient, curr_split in self.dm.patient_to_split_mapping.items() if curr_split == split]
    

    def _apply_new_patientid_to_entire_dic(self, dic, new_patientid):
        # Apply to all DFs, including those in nested dicts
        new_dic = {}

        for key, value in dic.items():
            if isinstance(value, pd.DataFrame) :
                new_df = value.copy()
                if "patientid" in new_df.columns:
                    # Assuming 'patientid' is a column you want to update for all rows
                    new_df["patientid"] = new_patientid
                new_dic[key] = new_df
            elif isinstance(value, pd.Series):
                # Assuming 'patientid' is a column you want to update for all rows
                new_series = value.copy()
                if "patientid" == new_series.name:
                    new_series.iloc[0] = new_patientid
                new_dic[key] = new_series
            elif isinstance(value, dict):
                # Recursively call the function for nested dictionaries
                new_dic[key] = self._apply_new_patientid_to_entire_dic(value, new_patientid)
            else:
                # If it's neither a DataFrame nor a dict, keep the original value
                new_dic[key] = value
        return new_dic




    def convert_full_to_string_for_one_patient(self, patientid, num_samples_per_lot, generate_conversion=False):

        # Reset seed every time to ensure reproducibility
        np.random.seed(self.config.seed)

        patient_data = self.dm.get_patient_data(patientid)

        # Do only forecasting splits
        processed_splits_fc, split_dates = self.data_splitter_forecasting.get_splits_from_patient(patient_data,
                                                                                             nr_samples=1,
                                                                                             apply_filtering=True,
                                                                                             return_splits=True,
                                                                                             max_num_samples_per_lot=num_samples_per_lot)
        

        # Check if empty
        if processed_splits_fc == [[]] or processed_splits_fc == [None]:
            return [[]]
        processed_splits_fc = [split for split in processed_splits_fc if split != []]
        new_patientids = [patientid + "_split_" + str(split_idx) for split_idx in range(len(processed_splits_fc))]

        if generate_conversion:
            ret_list = []
            for split_idx in range(len(processed_splits_fc)):

                #: convert
                p_converted = self.converter.forward_conversion(forecasting_splits=processed_splits_fc[split_idx],
                                                                event_splits=[],
                                                                variable_stats=self.data_splitter_forecasting.variable_stats,
                                                                override_mode_to_select_forecasting="forecasting")

                #: Adjust patientids at all important DFs
                new_patientid = new_patientids[split_idx]
                new_dic = self._apply_new_patientid_to_entire_dic(p_converted, new_patientid)

                # Append
                ret_list.append(new_dic)

            return ret_list
        else:

            #: Adjust patientids at all important DFs
            final_ret = []

            for split_idx in range(len(processed_splits_fc)):

                # Get patientid
                new_patientid = new_patientids[split_idx]

                # Since max 1 sample for split
                curr_split = processed_splits_fc[split_idx][0]

                # Apply new patientid
                new_dic = self._apply_new_patientid_to_entire_dic(curr_split, new_patientid)

                # Get actually sample variables
                new_dic["sampled_variables"] = new_dic["target_events_after_split"]["event_name"].unique().tolist()

                # Append
                final_ret.append([new_dic])

            return final_ret

















