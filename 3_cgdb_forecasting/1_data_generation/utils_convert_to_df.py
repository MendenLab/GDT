import pandas as pd 
import numpy as np
import traceback



# Converting the split data from before, into pandas DF ready for the model
# Generally, following the GluonTS multivariate data format



class ConvertedDFData:

    def __init__(self, constant_df, input_df, target_df, 
                 split_date_included_in_input,
                 target_variable, patient_id,
                 split_date_changed=False, old_split_date=None):
        """
        Class to hold the converted data in a format ready for the model
        """
        self.constant_df = constant_df
        self.input_df = input_df
        self.target_df = target_df
        self.split_date_included_in_input = split_date_included_in_input
        self.target_variable = target_variable
        self.patientid = patient_id
        self.split_date_changed = split_date_changed
        self.old_split_date = old_split_date

    def get_dict_of_meta_data(self):
        """
        Get the meta data of the data
        """
        meta_data = {
            "patientid": self.patientid,
            "split_date_included_in_input": self.split_date_included_in_input,
            "target_variable": self.target_variable,
            "split_date_changed": self.split_date_changed,
            "old_split_date": self.old_split_date
        }
        return meta_data





class ConvertToDF:


    def __init__(self, num_weeks_lookback,
                 num_forecast_days,
                 ropro_missing_value=-1.0,
                 train_data_stats_folder="/0_data/train_data_stats/",
                 ):


        self.num_weeks_lookback = num_weeks_lookback
        self.num_forecast_days = num_forecast_days
        self.train_data_stats_folder = train_data_stats_folder
        self.train_data_stats = {}
        self.variable_means = {}


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
            "lymphocytes_pct": ["lab", "2345_7"],
            "alkaline_phosphatase": ["lab", "6768_6"],
            "total_protein": ["lab", "2885_2"],
            "alanine_aminotransferase": ["lab", "1742_6"],
            "albumin": ["lab", "1751_7"],
            "total_bilirubin": ["lab", "1975_2"],
            "chloride": ["lab", "2075_0"],
            "monocytes_num": ["lab", "26485_3"],
            "eosinophils_pct": ["lab", "26485_3"],
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


    def convert_split_data_to_long_df(self, data):

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
        new_constant = pd.DataFrame(new_constant, index=[0])

        # Load in stats of the train data stats
        if indication not in self.train_data_stats:
            self.train_data_stats[indication] = pd.read_csv(
                f"{self.train_data_stats_folder}/{indication}_train_data_stats.csv"
            )
            self.variable_means[indication] = self.train_data_stats[indication][["event_name", "mean_without_outliers"]]

        #: iterate over every variable to predict
        ret_list = []   # List of ConvertedDFData

        for variable_to_predict in data["sampled_variables"]:

            # Basics
            split_date_changed = False
            old_split_date = None

            all_variables_to_select = all_overlapping_lab_vars.copy() + [variable_to_predict]
            all_variables_to_select = list(set(all_variables_to_select))
            
            #: for each, get history data for current var
            curr_history = data["events_until_split"].copy()
            curr_history = curr_history[
                (curr_history["event_name"].isin(all_variables_to_select))
            ]
            curr_history = curr_history.sort_values(by="date")

            # In incredibly rare cases, there are duplicates in the history data (only a few in the whole dataset), so we randomly select one
            duplicates = curr_history[curr_history.duplicated(subset=["date", "event_category", "event_name"], keep=False)]
            if len(duplicates) > 0:
                print("Current history has duplicates, dropping them - patientd: ", curr_patientid, " variable: ", variable_to_predict)
                print("Duplicates: ", duplicates)
                curr_history = curr_history.drop_duplicates(subset=["date", "event_category", "event_name"], keep="last").copy()

            #: extract the history data from date_min_history
            split_date = data["split_date_included_in_input"]
            date_min_history = split_date - pd.Timedelta(days=self.num_weeks_lookback * 7.0)
            curr_history = curr_history[curr_history["date"] >= date_min_history].copy()

            #: do some post conversions
            curr_history["event_value"] = curr_history["event_value"].astype(float)
            old_history_curr_var = curr_history[curr_history["event_name"] == variable_to_predict].copy()
            old_history_dates = old_history_curr_var["date"].unique().tolist()
            old_history_values = old_history_curr_var["event_value"].unique().tolist()
            curr_descriptive_name = old_history_curr_var["event_descriptive_name"].iloc[0]

            #: get target data for current var
            curr_target = data["target_events_after_split"].copy()
            curr_target = curr_target[
                (curr_target["event_name"] == variable_to_predict)
            ]
            curr_target.loc[:, "event_value"] = curr_target.loc[:, "event_value"].astype(float)
            curr_target = curr_target.sort_values(by="date")
            old_target_dates = curr_target["date"].unique().tolist()
            old_target_values = curr_target["event_value"].unique().tolist()
            target_number_of_days_delta = np.floor(self.num_forecast_days / 7.0) * 7.0

            # Skip if empty target for this variable
            if len(curr_target) == 0:
                print("No target data available for variable: ", variable_to_predict)
                continue


            #: in edge cases, the split date might be on a biomarker date, which due to a bug, might not be on the week
            # level cycle. So in this case we adjust the split date to the previous week
            potential_date_range = pd.date_range(start=date_min_history, end=split_date, freq='7D')
            if not set(old_history_dates).issubset(set(potential_date_range)):
                #: adjust split date to the previous week
                print("Split date is not on the week level cycle, adjusting to previous week")
                old_split_date = split_date
                raw_events = data["events_until_split"]
                curr_history_adjusted_dates = raw_events[raw_events["source"] != "genetic"].copy()
                curr_history_adjusted_dates = curr_history_adjusted_dates["date"].sort_values()
                new_split_date = curr_history_adjusted_dates.iloc[-1]
                split_date_changed = True
                split_date = new_split_date
                curr_history = curr_history[curr_history["date"] <= new_split_date]

                #: adjust the max number of forecast days, so that we don't exclude anything
                if curr_target["date"].iloc[-1] > new_split_date + pd.Timedelta(days=target_number_of_days_delta):
                    target_number_of_days_delta = (curr_target["date"].iloc[-1] - new_split_date).days
                    print("Adjusted target number of days delta to: ", target_number_of_days_delta)

                #: adjust the old history dates
                old_history_curr_var = curr_history[curr_history["event_name"] == variable_to_predict].copy()
                old_history_dates = old_history_curr_var["date"].unique().tolist()
                old_history_values = old_history_curr_var["event_value"].unique().tolist()

                #: adjust date_min_history
                date_min_history = split_date - pd.Timedelta(days=self.num_weeks_lookback * 7.0)

            else:
                split_date = data["split_date_included_in_input"]


            #: generate wide format dataset for the history data
            curr_history = curr_history.pivot_table(
                index='date',
                columns='event_name',
                values='event_value',
                aggfunc='first'
            ).reset_index()

            #: add in missing columns as pd.NA
            missing_cols = set(all_overlapping_lab_vars) - set(curr_history.columns)
            curr_history = curr_history.reindex(columns=curr_history.columns.tolist() + list(missing_cols), fill_value=pd.NA)
            curr_history = curr_history.reindex(sorted(curr_history.columns), axis=1)
            curr_history = curr_history.sort_values(by='date')
            curr_history[all_overlapping_lab_vars] = curr_history[all_overlapping_lab_vars].apply(pd.to_numeric, errors='raise')

            # Calculate appropriate date ranges
            date_max_future = split_date + pd.Timedelta(days=target_number_of_days_delta)
            date_range = pd.date_range(start=date_min_history, end=split_date, freq='7D')
            all_dates_df = pd.DataFrame({'date': date_range})
            curr_history = pd.merge(all_dates_df, curr_history, on='date', how='left')


            #: impute history (linear interpolation + forward/backward fill)
            curr_history['event_category'] = 'lab'
            curr_history['event_name'] = variable_to_predict
            curr_history["event_descriptive_name"] = curr_descriptive_name
            curr_history.loc[:, 'event_value'] = curr_history[variable_to_predict].astype(float).copy()
            curr_history.loc[:, 'event_value'] = curr_history['event_value'].interpolate(method='linear')
            curr_history.loc[:, 'event_value'] = curr_history['event_value'].ffill().bfill()

            #: also interpolate for the other overlapping variables
            curr_history.loc[:, all_overlapping_lab_vars] = curr_history[all_overlapping_lab_vars].interpolate(method='linear')
            curr_history.loc[:, all_overlapping_lab_vars] = curr_history[all_overlapping_lab_vars].ffill().bfill()

            # Drop if variable_to_predict not in overlapping lab vars
            if variable_to_predict not in all_overlapping_lab_vars:
                curr_history = curr_history.drop(columns=[variable_to_predict], errors='ignore')

            #: fill in any NaNs with train set means
            for var in all_overlapping_lab_vars:
                if var in self.variable_means[indication].event_name.values:
                    mean_value = self.variable_means[indication].loc[
                        self.variable_means[indication]["event_name"] == var, "mean_without_outliers"
                    ].values[0]
                    curr_history[var] = curr_history[var].fillna(mean_value)
                else:
                    # For the vitals, we don't have the means readily available, so we just fill them with 0
                    if var in ['body_height', 'body_weight', 'oxygen_saturation', 'systolic_blood_pressure',
                               'diastolic_blood_pressure', 'body_temperature', 'body_surface_area']:
                        curr_history[var] = curr_history[var].fillna(0.0)
                    else:
                        raise ValueError(f"Variable {var} not found in variable means for indication {indication}")


            #: generate wide format dataset for the target data
            curr_target = curr_target.pivot_table(
                index='date',
                columns='event_name',
                values='event_value',
                aggfunc='first'
            ).reset_index()

            #: add in missing columns as pd.NA
            missing_cols_target = set(all_overlapping_lab_vars) - set(curr_target.columns)
            curr_target = curr_target.reindex(columns=curr_target.columns.tolist() + list(missing_cols_target), fill_value=pd.NA)
            curr_target = curr_target.reindex(sorted(curr_target.columns), axis=1)
            curr_target = curr_target.sort_values(by='date')
            curr_target[all_overlapping_lab_vars] = curr_target[all_overlapping_lab_vars].apply(pd.to_numeric, errors='raise')

            #: apply last observed to non-target variables
            original_targets_values = curr_target[variable_to_predict].copy().values
            last_observed_history = curr_history.iloc[-1].copy()
            curr_target.loc[:, all_overlapping_lab_vars] = last_observed_history[all_overlapping_lab_vars].values
            curr_target.loc[:, "event_value"] = original_targets_values.copy()
            
            # drop if target column not in all_overlapping_lab_vars
            if variable_to_predict not in all_overlapping_lab_vars:
                curr_target = curr_target.drop(columns=[variable_to_predict])

            #: impute target (use last observed value of history + forward fill)
            date_range_future = pd.date_range(start=split_date + pd.Timedelta(days=7.0),
                                               end=date_max_future, freq='7D')
            all_dates_future_df = pd.DataFrame({'date': date_range_future})
            curr_target = pd.merge(all_dates_future_df, curr_target, on='date', how='left')
            curr_target['event_category'] = 'lab'
            curr_target['event_name'] = variable_to_predict
            curr_target["event_descriptive_name"] = curr_descriptive_name
            
            if pd.isna(curr_target["event_value"].iloc[0]):
                curr_target.loc[curr_target['date'] == curr_target['date'].min(), 
                                'event_value'] = self._get_last_observed_value(curr_history, "lab", variable_to_predict)
            curr_target['event_value'] = curr_target['event_value'].interpolate(method='linear').ffill()

            #: also interpolate for the other overlapping variables
            curr_target.loc[:, all_overlapping_lab_vars] = curr_target[all_overlapping_lab_vars].interpolate(method='linear')
            curr_target.loc[:, all_overlapping_lab_vars] = curr_target[all_overlapping_lab_vars].ffill().bfill()

            #: add which events are imputed as second column to input & target (dynamic feat)
            curr_history["imputed"] = 1
            curr_history.loc[curr_history["date"].isin(old_history_dates), "imputed"] = 0
            curr_target["imputed"] = 1
            curr_target.loc[curr_target["date"].isin(old_target_dates), "imputed"] = 0

            #: make assertions to check that everything is correct
            try:
                assert len(curr_history) == len(date_range)
                assert len(curr_target) == len(date_range_future)
                assert curr_history["date"].iloc[0] == date_min_history
                assert curr_history["date"].iloc[-1] == split_date
                assert curr_target["date"].iloc[0] > split_date
                assert curr_target["date"].iloc[-1] == date_max_future
                assert curr_target["date"].iloc[0] == split_date + pd.Timedelta(days=7.0)
                assert all(curr_target["date"] > curr_history["date"].max())
                assert all(curr_target["date"].diff().dt.days[1:] == 7.0)
                assert all(curr_history["date"].diff().dt.days[1:] == 7.0)
                assert all([x in curr_target["date"].tolist() for x in old_target_dates])
                assert all([x in curr_history["date"].tolist() for x in old_history_dates])
                assert all(curr_history["event_value"].notnull())
                assert all(curr_target["event_value"].notnull())
                assert all([x in curr_target["event_value"].tolist() for x in old_target_values])
                assert all([x in curr_history["event_value"].tolist() for x in old_history_values])
                if variable_to_predict in all_overlapping_lab_vars:
                    assert all([x == curr_history[variable_to_predict].iloc[-1] for x in curr_target[variable_to_predict]])
                else:
                    assert variable_to_predict not in curr_history.columns
                    assert variable_to_predict not in curr_target.columns
            except AssertionError as e:
                print("Assertion error: ", e)
                print("curr_history: ", curr_history)
                print("curr_target: ", curr_target)
                print("date_min_history: ", date_min_history)
                print("split_date: ", split_date)
                print("date_max_future: ", date_max_future)
                print("old_split_date: ", old_split_date)
                print("old_history_dates: ", old_history_dates)
                print("old_target_dates: ", old_target_dates)
                print("old_history_values: ", old_history_values)
                print("old_target_values: ", old_target_values)
                print("History dates: ", curr_history["date"].unique().tolist())
                print("Target dates: ", curr_target["date"].unique().tolist())
                print("Patientid: ", curr_patientid)
                print("Variable to predict: ", variable_to_predict)
                
                traceback.print_exc()
                raise e


            #: add correct patientid (with appended variable) to all DFs
            new_patientid = curr_patientid + "_var_" + str(variable_to_predict)
            curr_history["patientid"] = new_patientid
            curr_target["patientid"] = new_patientid
            curr_const = new_constant.copy()
            curr_const["patientid"] = new_patientid
            curr_const["variable_to_predict"] = str(variable_to_predict)

            #: make ConvertedDFData object
            curr_converted_data = ConvertedDFData(
                constant_df=curr_const,
                input_df=curr_history,
                target_df=curr_target,
                split_date_included_in_input=data["split_date_included_in_input"],
                target_variable=variable_to_predict,
                patient_id=new_patientid,
                split_date_changed=split_date_changed,
                old_split_date=old_split_date,
            )
            ret_list.append(curr_converted_data)

        #: return
        return ret_list





# We also include overlapping vitals
all_overlapping_lab_vars = ['14804_9',
 '14979_9',
 '1742_6',
 '1751_7',
 '17861_6',
 '19023_1',
 '19123_9',
 '1920_8',
 '1968_7',
 '1971_1',
 '1975_2',
 '2028_9',
 '20482_6',
 '20570_8',
 '2075_0',
 '2160_0',
 '2276_4',
 '2324_2',
 '2345_7',
 '2532_0',
 '26444_0',
 '26449_9',
 '26450_7',
 '26453_1',
 '26464_8',
 '26474_7',
 '26478_8',
 '26484_6',
 '26485_3',
 '26499_4',
 '26505_8',
 '26507_4',
 '26511_6',
 '26515_7',
 '2823_3',
 '2885_2',
 '2947_0',
 '2951_2',
 '3016_3',
 '30180_4',
 '30394_1',
 '30395_8',
 '30451_9',
 '3084_1',
 '3094_0',
 '35591_7',
 '4544_3',
 '48642_3',
 '48643_1',
 '5902_2',
 '5905_5',
 '6298_4',
 '6690_2',
 '6768_6',
 '69405_9',
 '704_7',
 '706_2',
 '707_0',
 '718_7',
 '731_0',
 '736_9',
 '742_7',
 '744_3',
 '751_8',
 '770_8',
 '777_3',
 '789_8',
 '98979_8',
 'body_height',
 'body_surface_area',
 'body_temperature',
 'body_weight',
 'diastolic_blood_pressure',
 'oxygen_saturation',
 'systolic_blood_pressure']









