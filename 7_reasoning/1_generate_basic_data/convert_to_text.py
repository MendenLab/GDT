import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import pickle
import datetime


from digital_twin_converter.instruction.data_splitter_forecasting import DataSplitterForecasting
from digital_twin_converter.common.data_manager import SingleIndicationDataManager
from digital_twin_converter.instruction.data_splitter_events import DataSplitterEvents
from digital_twin_converter.instruction.converter_manual_instruction import ConverterManualInstruction

COL_CATEGORY_LOT = "lot"


class ConvertToText:

    def __init__(self, 
                 indication = "enhanced_nsclc",
                 cache_location = "/data_cache/flatiron_web_app_update_20241106/all_indications"):

        # Set basics
        self.indication = indication
        self.cache_location = cache_location

        # Set empties
        self.dm = None
        self.data_splitter_forecasting = None
        self.data_splitter_events = None
        self.converter = None
        self.forecasting_variable_stats = None

        # Init data managers
        self._init_data_managers()


    def _init_data_managers(self):
        
        save_path = os.path.join(self.cache_location, "data_manager_" + self.indication + ".pkl")

        if not os.path.exists(save_path):

            print(f"Preloading & saving data manager for {self.indication}")

            self.dm = SingleIndicationDataManager(self.indication)
            self.dm.load_indication_data()
            self.dm.process_indication_data()
            self.dm.setup_unique_mapping_of_events()
            self.dm.setup_dataset_splits()

            self.data_splitter_events = DataSplitterEvents(self.dm, max_length_of_weeks_to_sample=104)
            self.data_splitter_events.setup_variables()
            self.data_splitter_forecasting = DataSplitterForecasting(self.dm)
            self.data_splitter_forecasting.setup_statistics()
            self.converter = ConverterManualInstruction(self.dm.data_frames["constant_description"],
                                                   nr_tokens_budget_total=8192, seed=42)        

        else:

            # Load from cache
            print("Loading from cache")

            with open(save_path, "rb") as f:
                self.dm, _, self.data_splitter_events, self.data_splitter_forecasting, self.converter = pickle.load(f)

            # Preprocess the data as needed
            events_possible_variables = list(self.data_splitter_events.manual_variables_category_mapping.keys())
            self.forecasting_variable_stats = self.data_splitter_forecasting.variable_stats



    def convert_to_string_for_all_patients(self, patient_full_data, variables_to_predict):
        
        ret_list = []
        num_skipped = 0
        np.random.seed(7862)

        #: go through all patients 
        for patient_constant, curr_patient_data in tqdm(patient_full_data):
                
            # Setup basics
            patient_data = {
                "events": curr_patient_data,
                "constant": patient_constant,
            }
            patient_data["events"] = patient_data["events"].sort_values("date")
            
            #: no event split
            events_split = []

            #: generate forecasting split
            processed_splits_fc, split_dates = self.data_splitter_forecasting.get_splits_from_patient(patient_data,
                                                                                             nr_samples=1,
                                                                                             apply_filtering=True,
                                                                                             return_splits=True,
                                                                                             max_num_samples_per_lot=1,
                                                                                             override_variables_to_predict=variables_to_predict)

            # skip if empty
            processed_splits_fc = [x for x in processed_splits_fc if x is not None]
            processed_splits_fc = [x for x in processed_splits_fc if len(x) > 0]
            if len(processed_splits_fc) == 0:
                num_skipped += 1
                continue

            # Randomly select one split
            split_index = np.random.choice(range(len(processed_splits_fc)))
            split_forecasting = processed_splits_fc[split_index]
            
            # Convert to text
            converted = self.converter.forward_conversion(
                forecasting_splits=split_forecasting,
                event_splits=events_split,
                override_mode_to_select_forecasting="forecasting",
                variable_stats=None,
            )

            #: add to running list
            ret_list.append((patient_constant, curr_patient_data, converted))
    
        return ret_list, num_skipped
        

    def reverse_convert_for_all_patients(self, responses_with_meta):

        ret_list = []

        #: go through all of them
        for constant, patient_data, converted_data, response_string, logprobs in tqdm(responses_with_meta):

            #: reverse convert
            reverse_converted = self.converter.reverse_conversion(
                target_string=response_string,
                split_date = converted_data["meta"]["split_date_included_in_input"],
                data_manager=self.dm,
                patientid=constant["patientid"].iloc[0],
                inference_override=True
            )

            #: add to list
            ret_list.append((constant, patient_data, converted_data, response_string, logprobs, reverse_converted))
        
        return ret_list







