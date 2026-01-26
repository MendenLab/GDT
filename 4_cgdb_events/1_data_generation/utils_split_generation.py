import pandas as pd
from transformers import AutoTokenizer
import logging
logging.basicConfig(level=logging.NOTSET)
import numpy as np
import datetime

from digital_twin_converter.common.data_manager import SingleIndicationDataManager
from digital_twin_converter.instruction.data_splitter_events import DataSplitterEvents
from digital_twin_converter.instruction.converter_manual_instruction import ConverterManualInstruction
from digital_twin_converter.common.config import Config


PATH_TO_SPLITS = "/flatiron_cgdb/instruction/combined/2024_11_14_10_lots_per_patient/"
COL_CATEGORY_LOT = "lot"


class SplitterGenerator:

    def __init__(self,
                 indication,
                 nr_variables_to_sample=3,
                 tokenizer_to_load_for_counting = 'meta-llama/Meta-Llama-3-8B',):

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
        path_to_splits = PATH_TO_SPLITS + "splits_" + self.indication + ".json"
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



    def _get_true_censoring_and_times(self, patient_data, split_date_included_in_input, category_to_predict):

        events_after_split = patient_data["events"][patient_data["events"]["date"] > split_date_included_in_input].copy()

        # Determine exact (true) censoring (i.e. new LoT) nr of weeks and event occuring for various calculations
        exact_event_occurence_time = None
        exact_censoring = False

        if events_after_split.shape[0] == 0:
            # We explicitly set to 0 and censored in case of no events after split (i.e. no follow up)
            # This means that the patient was censored at the time of last visit (which is the split date)
            # This is a design choice, and can be changed if needed.
            # This is very rare in the data, but need to handle this edge case.
            exact_censoring = True
            exact_event_occurence_time = 0
        else:
            
            # Get all event occurences
            all_event_occurences = events_after_split[events_after_split["event_category"] == category_to_predict]["date"].tolist()

            # Get all LoTs
            all_lots = events_after_split[events_after_split["event_category"] == COL_CATEGORY_LOT]["date"].tolist()


            # If no LoTs, then set to max, or if we're predicting LoT event (as an edge case, since its never censored by LoT, only by last visit)
            if len(all_lots) == 0 or category_to_predict == COL_CATEGORY_LOT:
                min_lot = datetime.datetime.max
            else:
                min_lot = min(all_lots)
            
            # Check if event occured
            if len(all_event_occurences) == 0:
                
                # Get last visit
                last_visit = events_after_split["date"].max()

                # In case of no event occurence, check if lot occured
                if last_visit < min_lot:
                    exact_event_occurence_time = (last_visit - split_date_included_in_input).days / 7
                    exact_censoring = True
                else:
                    exact_censoring = True
                    exact_event_occurence_time = (min_lot - split_date_included_in_input).days / 7
            
            else:    
                min_event_occurence = min(all_event_occurences)

                if min_event_occurence < min_lot:
                    exact_event_occurence_time = (min_event_occurence - split_date_included_in_input).days / 7
                else:
                    exact_censoring = True
                    exact_event_occurence_time = (min_lot - split_date_included_in_input).days / 7

        # Double check data
        assert exact_event_occurence_time is not None, "Exact event occurence time should not be None."

        # Add meta
        meta = {}
        meta["true_time"] = exact_event_occurence_time
        meta["true_censoring"] = exact_censoring
        return meta




    def convert_full_to_string_for_one_patient(self, patientid, num_samples_per_lot, generate_conversion=False):

        assert num_samples_per_lot == 1, "Currently only supports 1 sample per LoT. If you want to change this, please adjust the code accordingly."

        # Reset seed every time to ensure reproducibility
        np.random.seed(self.config.seed)

        patient_data = self.dm.get_patient_data(patientid)

        #: split events, for each of the variables, for specific times
        # We use the same split date for all event types and times for each LoT (the function processes all LoTs in parallel)
        event_splits = {}
        preselected_split_dates = None
        all_vars = ["death", "progression", "lot", "metastasis"]
        actual_possible_variables = [x for x in all_vars if x in self.data_splitter_events.manual_variables_category_mapping.keys()]

        for override_week in [8, 26, 52, 104]:
            for override_category in actual_possible_variables:
                
                # Using override_split_dates (and not preselected_split_dates), since it takes into account the 
                # This generates for every LoT one sample
                curr_event_split = self.data_splitter_events.get_splits_from_patient(patient_data,
                                                                override_split_dates=preselected_split_dates if preselected_split_dates is not None else None,
                                                                max_nr_samples=1,
                                                                max_num_samples_per_lot=num_samples_per_lot,
                                                                override_category=override_category,
                                                                override_end_week_delta=override_week,)

                # Assign new patientids and meta data for targets
                for split_idx in range(len(curr_event_split)):
                    curr_event_split[split_idx][0]["new_patientid"] = f"{patientid}_lot_{split_idx}_var_{override_category}_week_{override_week}"
                    curr_event_split[split_idx][0]["week_to_predict"] = override_week

                    # In case of progression, the library sometimes will set it to death, since it is also a progression event
                    # In this cases (for future correct usage), we manually set it to progression again
                    if override_category == "progression" and curr_event_split[split_idx][0]["sampled_category"] == "death":
                        curr_event_split[split_idx][0]["sampled_category"] = "progression"

                    # Add extra meta data including true censoring and times
                    extra_meta = self._get_true_censoring_and_times(patient_data,
                                                                    split_date_included_in_input=curr_event_split[split_idx][0]["split_date_included_in_input"],
                                                                    category_to_predict=override_category)
                    curr_event_split[split_idx][0].update(extra_meta)

                #: for easier downstream usage, save each LoT in a separate list
                for idx in range(len(curr_event_split)):
                    split_date = curr_event_split[idx][0]["split_date_included_in_input"]
                    if split_date not in event_splits:
                        event_splits[split_date] = []
                    event_splits[split_date].append(curr_event_split[idx])
                
                if preselected_split_dates is None:
                    # Needs to be 1 split date per LoT
                    preselected_split_dates = [curr_event_split[idx][0]["split_date_included_in_input"] for idx in range(len(curr_event_split))]
                else:
                    assert len(curr_event_split) == len(preselected_split_dates), "The number of current event splits does not match the preselected split dates. This should not happen, please check the logic."
                    assert all([curr_event_split[idx][0]["split_date_included_in_input"] == preselected_split_dates[idx] for idx in range(len(curr_event_split))]), \
                        "Preselected split dates do not match the current event splits. This should not happen, please check the logic."
        

        # Setup
        all_split_dates = sorted(list(event_splits.keys()))

        if generate_conversion:
            
            ret_list = []
            
            for split_idx in range(len(all_split_dates)):
                for split in event_splits[all_split_dates[split_idx]]:

                    #: convert to text if needed
                    p_converted = self.converter.forward_conversion_inference(event_split=split[0])

                    #: Adjust patientids at all important DFs
                    new_patientid = split[0]["new_patientid"]
                    new_dic = self._apply_new_patientid_to_entire_dic(p_converted, new_patientid)

                    # Add extra meta
                    new_dic["meta"]["week_to_predict"] = new_patientid.split("_week_")[1]  # Hacky

                    # Append
                    ret_list.append(new_dic)

            return ret_list
        else:

            #: Adjust patientids at all important DFs
            final_ret = {}

            for split_idx in range(len(all_split_dates)):
                for split in event_splits[all_split_dates[split_idx]]:

                    # Get patientid
                    split = split[0]
                    new_patientid = split["new_patientid"]

                    # Apply new patientid
                    new_dic = self._apply_new_patientid_to_entire_dic(split, new_patientid)

                    # Append
                    if split["split_date_included_in_input"] not in final_ret:
                        final_ret[split["split_date_included_in_input"]] = []
                    final_ret[split["split_date_included_in_input"]].append(new_dic)

            return final_ret

















