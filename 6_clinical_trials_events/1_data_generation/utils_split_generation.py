import pandas as pd
from transformers import AutoTokenizer
import logging
logging.basicConfig(level=logging.NOTSET)
import numpy as np
import datetime


from digital_twin_converter.instruction.data_splitter_forecasting import DataSplitterForecasting
from digital_twin_converter.common.data_manager import SingleIndicationDataManager
from digital_twin_converter.instruction.data_splitter_events import DataSplitterEvents
from digital_twin_converter.instruction.converter_manual_instruction import ConverterManualInstruction
from digital_twin_converter.common.config import Config


PATH_TO_SPLITS = "cit_data/subset_data/2025_07_21_nsclc_with_fmi/"
COL_CATEGORY_LOT = "lot"
PATH_TO_CONSTANT_DESCRIPTION = "/flatiron_cgdb/2024_07_05/enhanced_nsclc_master_constant_descriptions.csv"
PATH_TO_SPLIT_JSON = "cit_data/subset_data/2025_07_21_nsclc_with_fmi/patient_split_mapping.json"
PATH_TO_RAW_SURVIVAL_DATA = "cit_data/raw_cit_data/survival_data.csv"


class SplitterGenerator:

    def __init__(self,
                 indication,
                 nr_variables_to_sample=3,
                 tokenizer_to_load_for_counting = 'meta-llama/Meta-Llama-3-8B',):

        # Set basics
        self.indication = "CIT"        
        self.config = Config()

        # Override with custom dataset from CIT
        self.config.override_with_custom_dataset = True
        self.config.override_datasets_events = PATH_TO_SPLITS + "df_events_subset.csv"
        self.config.override_datasets_constant = PATH_TO_SPLITS + "df_constant_subset.csv"
        self.config.override_datasets_molecular = PATH_TO_SPLITS + "df_molecular_subset.csv"
        self.config.override_constant_description = PATH_TO_CONSTANT_DESCRIPTION   # Use the standard description
        self.config.override_statistics = None
        self.config.date_cutoff = pd.to_datetime("2050-03-31")   # Some arbitrary high number since data already correctly handles end of study censoring

        # Set empties
        self.dm = None
        self.data_splitter_forecasting = None
        self.data_splitter_events = None
        self.converter = None
        self.forecasting_variable_stats = None
        self.min_nr_variables_to_sample = nr_variables_to_sample
        self.max_nr_variables_to_sample = nr_variables_to_sample
        self.raw_survival_data = pd.read_csv(PATH_TO_RAW_SURVIVAL_DATA)

        # Init data managers
        self._init_data_managers(patient_to_split_mapping_path=PATH_TO_SPLIT_JSON)
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



    def _get_true_censoring_and_times(self, patient_data, category_to_predict, override_week):

        #: get true censoring and times from raw data
        patient_raw_id = patient_data["new_patientid"].split("_lot_")[0].split("_")[1]
        trial_id =  patient_data["new_patientid"].split("_lot_")[0].split("_")[0]

        patient_surv_data = self.raw_survival_data[
            (self.raw_survival_data["UNI_ID"] == patient_raw_id) &
            (self.raw_survival_data["STUDYID"] == trial_id)
        ]

        assert len(patient_surv_data) == 1, f"Expected exactly one row for patient {patient_raw_id} in raw survival data, found {len(patient_surv_data)}."

        surv_data_column = {"death": "OS", "progression": "PFS"}[category_to_predict]
        censoring_column = surv_data_column + "_CENSORED"

        true_censoring = bool(patient_surv_data[censoring_column].values[0])
        true_time = patient_surv_data[surv_data_column].values[0]

        #: also calculate whether occured or not for specified time
        override_days = override_week * 7
        occurred = None
        censored = None

        if true_censoring:
            # If censored true
            if override_days > true_time:
                occurred = False
                censored = True
            else:
                occurred = False
                censored = False
        else:
            # If not censored true
            if override_days > true_time:
                occurred = True
                censored = False
            else:
                occurred = False
                censored = False
        
        # Set up for correct prompt
        date_event_occured = patient_data["split_date_included_in_input"] + datetime.timedelta(days=override_days)
        
        #: setup correct return dictionary
        ret_dict = {
            "true_censoring": true_censoring,
            "true_time": true_time,
            "event_occured": occurred,
            "event_censored": censored,
            "date_event_occured": date_event_occured,
        }
        return ret_dict
        


    def _make_correct_target_string(self, curr_event_split):

        # Gather all extra data
        variable_to_predict = curr_event_split["sampled_category"]
        variable_descriptive_name = self.data_splitter_events.manual_variables_category_mapping[variable_to_predict]
        curr_censoring = curr_event_split["event_censored"]
        curr_occurence = curr_event_split["event_occured"]

        #: make the correct target string manually
        base_string = "Task 1 is time to event prediction:\nHere is the prediction: the event ({variable}) was {censored} and {occurence}."

        final_string = base_string.format(
            variable=variable_descriptive_name,
            censored="censored" if curr_censoring else "not censored",
            occurence="occurred" if curr_occurence else "did not occur"
        )

        # Example 1: Here is the prediction: the event (death) was censored and did not occur.
        # Example 2: Here is the prediction: the event (next line of therapy) was not censored and occured.

        return final_string



    def convert_full_to_string_for_one_patient(self, patientid, num_samples_per_lot, weeks_to_generate, split_date, generate_conversion=False):

        assert num_samples_per_lot == 1, "Currently only supports 1 sample per LoT. If you want to change this, please adjust the code accordingly."

        # Reset seed every time to ensure reproducibility
        np.random.seed(self.config.seed)

        patient_data = self.dm.get_patient_data(patientid)

        #: split events, for each of the variables, for specific times
        # We use the same split date for all event types and times for each LoT (the function processes all LoTs in parallel)
        event_splits = {}
        all_vars = ["death", "progression"]
        actual_possible_variables = [x for x in all_vars if x in self.data_splitter_events.manual_variables_category_mapping.keys()]

        for override_week in weeks_to_generate:
            for override_category in actual_possible_variables:
                
                # Using override_split_dates (and not preselected_split_dates), since it takes into account the 
                # This generates for every LoT one sample
                curr_event_split = self.data_splitter_events.get_splits_from_patient(patient_data,
                                                                                    override_split_dates=[split_date],
                                                                                    max_nr_samples=1,
                                                                                    max_num_samples_per_lot=num_samples_per_lot,
                                                                                    override_category=override_category,
                                                                                    override_end_week_delta=override_week)

                
                for split_idx in range(len(curr_event_split)):
                    
                    # Assign new patientids and meta data for targets that include the split_index and week
                    curr_event_split[split_idx][0]["new_patientid"] = f"{patientid}_lot_{split_idx}_var_{override_category}_week_{override_week}"
                    curr_event_split[split_idx][0]["week_to_predict"] = override_week

                    # In case of progression, the library sometimes will set it to death, since it is also a progression event
                    # In this cases (for future correct usage), we manually set it to progression again
                    if override_category == "progression" and curr_event_split[split_idx][0]["sampled_category"] == "death":
                        curr_event_split[split_idx][0]["sampled_category"] = "progression"

                    # Add extra meta data including true censoring and times
                    extra_meta = self._get_true_censoring_and_times(curr_event_split[split_idx][0],
                                                                    category_to_predict=override_category,
                                                                    override_week=override_week)
                    curr_event_split[split_idx][0].update(extra_meta)


                #: for easier downstream usage, save each LoT in a separate list (should be 1 here)
                for idx in range(len(curr_event_split)):
                    split_date = curr_event_split[idx][0]["split_date_included_in_input"]
                    if split_date not in event_splits:
                        event_splits[split_date] = []
                    event_splits[split_date].append(curr_event_split[idx])
                
                assert all([curr_event_split[idx][0]["split_date_included_in_input"] == split_date for idx in range(len(curr_event_split))])

        # Setup
        all_split_dates = sorted(list(event_splits.keys()))

        if generate_conversion:
            
            ret_list = []
            
            for split_idx in range(len(all_split_dates)):
                for split in event_splits[all_split_dates[split_idx]]:

                    # Adjust to be as close as possible to CGDB
                    split[0]["constant_data"]["indication"] = "enhanced_" + split[0]["constant_data"]["indication"]

                    #: convert to text if needed
                    p_converted = self.converter.forward_conversion_inference(event_split=split[0])

                    #: adjust target string
                    p_converted["answer"] = self._make_correct_target_string(split[0])

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

















