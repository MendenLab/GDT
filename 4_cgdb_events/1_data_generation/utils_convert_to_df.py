import pandas as pd 
import numpy as np



class ConvertToDF:


    def __init__(self,
                 ropro_missing_value=-1.0,
                 ):


        # Missing value for ropro variables
        self.ropro_missing_value = ropro_missing_value
                
        # Format: name_used_in_final_df: (event_category, event_name)
        self.ropro_mapping = {
            "weight": ("vitals", "body_weight"),
            "height": ("vitals", "body_height"),
            "oxygen_saturation": ("vitals", "oxygen_saturation"),
            "systolic_blood_pressure": ("vitals", "systolic_blood_pressure"),
            "ecog": ("ecog", "ecog"),
            "hemoglobin": ["lab", "718_7"],
            "urea_nitrogen": ["lab", "3094_0"],
            "platelets": ["lab", "26515_7"],
            "calcium": ["lab", "17861_6"],
            "glucose": ["lab", "2345_7"],
            "lymphocytes_pct": ["lab", "26478_8"],
            "alkaline_phosphatase": ["lab", "6768_6"],
            "total_protein": ["lab", "2885_2"],
            "alanine_aminotransferase": ["lab", "1742_6"],
            "albumin": ["lab", "1751_7"],
            "total_bilirubin": ["lab", "1975_2"],
            "chloride": ["lab", "2075_0"],
            "monocytes_num": ["lab", "26485_3"],
            "eosinophils_pct": ["lab", "26450_7"],
            "lactate_dehydrogenase": ["lab", "2532_0"],
        }
    

    def _get_last_observed_value(self, patient_history, event_category, event_name):
        """
        Get the last observed value of a specific event category and event name
        """
        # Filter the patient history for the specific event category and event name
        filtered_history = patient_history[
            (patient_history["event_category"] == event_category) &
            (patient_history["event_name"] == event_name)
        ]

        # Order by date
        filtered_history = filtered_history.sort_values(by="date")

        # Get the last observed value
        if not filtered_history.empty:
            last_observed_value = filtered_history.iloc[-1]["event_value"]
            return last_observed_value
        else:
            return None


    def _get_last_observed_values_of_ropro(self, patient_history):

        #: iterate over self.ropro_mapping and get last observed values
        ropro_return = {}

        for var, (event_category, event_name) in self.ropro_mapping.items():
            last_observed_value = self._get_last_observed_value(patient_history, event_category, event_name)
            if last_observed_value is not None:
                # Try to first convert to float, backup use string
                try:
                    last_observed_value = float(last_observed_value)
                except ValueError:
                    last_observed_value = str(last_observed_value)
                ropro_return[var] = last_observed_value
            else:
                # If no last observed value, set to missing value
                ropro_return[var] = self.ropro_missing_value
            
        return ropro_return


    def _get_nr_diagnoses_in_history(self, patient_history):
        """
        Get the number of diagnoses in the patient history
        """
        diagnoses = patient_history[patient_history["event_category"] == "diagnosis"]
        num_diagnoses = len(diagnoses)
        return num_diagnoses

    def _get_nr_genetic_events_in_history(self, patient_history):
        """
        Get the number of genetic events in the patient history
        """
        genetic_events = patient_history[patient_history["source"] == "genetic"]
        num_genetic_events = len(genetic_events)
        return num_genetic_events


    def convert_split_data_to_input_and_output_df(self, data):

        #: create constant data
        #: add age, gender, indication
        #: get last observed values of ropro variables (_get_last_observed_values_of_ropro)
        new_constant = {}

        constant = data["constant_data"]
        curr_patientid = constant["patientid"].iloc[0]
        new_constant["age"] = data["split_date_included_in_input"].year - constant["birthyear"].iloc[0]
        indication = constant["indication"].iloc[0]
        new_constant["indication"] = indication
        new_constant["gender"] = constant["gender"].iloc[0]
        new_constant["nr_previous_diagnoses"] = self._get_nr_diagnoses_in_history(data["events_until_split"])
        new_constant["nr_genetic_events"] = self._get_nr_genetic_events_in_history(data["events_until_split"])
        ropro_constants = self._get_last_observed_values_of_ropro(data["events_until_split"])
        new_constant.update(ropro_constants)

        #: add therapy name & line number
        therapy_name = self._get_last_observed_value(data["events_until_split"], "lot", "line_name")
        therapy_line_number = self._get_last_observed_value(data["events_until_split"], "lot", "line_number")
        new_constant["therapy_name"] = therapy_name if therapy_name is not None else "unknown"
        new_constant["therapy_line_number"] = therapy_line_number if therapy_line_number is not None else -1
        
        # Add basics in case we build a single model
        new_constant["sampled_category"] = data["sampled_category"]
        new_constant["week_to_predict"] = data["week_to_predict"]
        
        # Add meta
        new_constant["patientid"] = curr_patientid
        new_constant = pd.DataFrame(new_constant, index=[0])

        #: generate corresponding target DF
        target = [{
                    "patientid": data["new_patientid"],
                    "generic_patientid": data["new_patientid"].split("_var_")[0],
                    "split_date_included_in_input": data["split_date_included_in_input"],
                    "sampled_category": data["sampled_category"],
                    "week_to_predict": data["week_to_predict"],
                    "censored": data["event_censored"],
                    "occurred": data["event_occured"],
                    "true_censoring": data["true_censoring"],
                    "true_time": data["true_time"],
        }]
        target_df = pd.DataFrame(target)

        return new_constant, target_df
        










