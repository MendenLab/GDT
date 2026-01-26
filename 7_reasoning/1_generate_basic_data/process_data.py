import pandas as pd
import numpy as np
from tqdm import tqdm

# Column names used throughout, as defined in the original data (see README for details)
COL_EVENT_DESCRIPTIVE_NAME = "event_descriptive_name"
COL_EVENT_NAME = "event_name"
COL_EVENT_CATEGORY = "event_category"
COL_DATE = "date"
COL_PATIENTID = "patientid"
COL_EVENT_VALUE = "event_value"
COL_SOURCE = "source"
COL_BIOMARKER_CATEGORY = "biomarker_category"
COL_BIOMARKER_EVENT = "biomarker_event"
COL_BIOMARKER_VALUE = "biomarker_value"
COL_BIOMARKER_DESCRIPTIVE_NAME = "biomarker_descriptive_name"
COL_CATEGORY_LOT = "lot"
COL_CATEGORY_DEATH = "death"




def preprocess_events(events):

    replace_special_symbols = [
        ("lab", ("/", " per ")),
        ("lab", (".", " ")),
        ("drug", ("/", " ")),
        ("lot", ("/", " ")),
    ]

    #: get all unique pairs of event_name and event_descriptive_name in events
    unique_events = events
    unique_events = unique_events[[COL_EVENT_NAME, COL_EVENT_DESCRIPTIVE_NAME, COL_EVENT_CATEGORY]]
    unique_events = unique_events.copy().drop_duplicates()
    unique_events = unique_events.reset_index(drop=True)

    #: get all event_descriptive_name that are not unique
    non_unique_events = unique_events[COL_EVENT_DESCRIPTIVE_NAME].value_counts()
    non_unique_events = non_unique_events[non_unique_events > 1]

    # Extract corresponding event_name and event_category
    filtered_events = unique_events[COL_EVENT_DESCRIPTIVE_NAME]
    non_unique_events = unique_events[filtered_events.isin(non_unique_events.index)].copy()

    # create mapping for all non-unique descriptive names, and
    # then add event_name to those, and apply across entire dataset
    non_unique_events["new_descriptive_name"] = non_unique_events["new_descriptive_name"] = (
        non_unique_events[COL_EVENT_DESCRIPTIVE_NAME] + " - " + non_unique_events[COL_EVENT_NAME]
    )
    non_unique_events = non_unique_events[["new_descriptive_name", COL_EVENT_NAME, COL_EVENT_CATEGORY]]

    events = pd.merge(events, non_unique_events, how="left",
                      on=(COL_EVENT_NAME, COL_EVENT_CATEGORY))
    events_df = events
    new_desc_name = "new_descriptive_name"
    events_df[COL_EVENT_DESCRIPTIVE_NAME] = events_df[new_desc_name].fillna(events_df[COL_EVENT_DESCRIPTIVE_NAME])
    events = events.drop(columns=["new_descriptive_name"])

    #: first convert special symbols in event_descriptive_name to alternatives, using replace_special_symbols
    for event_category, (string_to_replace, replacement_string) in replace_special_symbols:
        events_df = events
        category_mask = events_df[COL_EVENT_CATEGORY] == event_category
        desc_name_col = COL_EVENT_DESCRIPTIVE_NAME

        events_df.loc[category_mask, desc_name_col] = (
            events_df.loc[category_mask, desc_name_col]
            .str.replace(string_to_replace, replacement_string)
        )

    #: recalculate unique_events and ensure no more non-unique event_descriptive_name
    cols_to_select = [COL_EVENT_NAME, COL_EVENT_DESCRIPTIVE_NAME, COL_EVENT_CATEGORY]
    unique_events = events[cols_to_select].copy().drop_duplicates()
    unique_events = unique_events.reset_index(drop=True)

    # Assert that all unique now
    assert len(unique_events) == len(events[COL_EVENT_DESCRIPTIVE_NAME].unique())

    return events



def get_individual_patient_data(events, molecular, constant):

    #: preprocess the data first, based on data_manager from the digital_twins app
    drop_cols = ['Unnamed: 0.1', 'Unnamed: 0']
    events = events.drop(columns=drop_cols, errors="ignore")
    molecular = molecular.drop(columns=drop_cols, errors="ignore")
    constant = constant.drop(columns=drop_cols, errors="ignore")
    events = preprocess_events(events)
    # Adjust biomarker stuff
    rename_dic = {
        COL_BIOMARKER_CATEGORY: COL_EVENT_CATEGORY,
        COL_BIOMARKER_EVENT: COL_EVENT_NAME,
        COL_BIOMARKER_VALUE: COL_EVENT_VALUE,
        COL_BIOMARKER_DESCRIPTIVE_NAME: COL_EVENT_DESCRIPTIVE_NAME
    }
    molecular = molecular.rename(columns=rename_dic)

    #: extract for each patientid in constant, the row
    all_patientids = constant["patientid"].drop_duplicates().to_list()

    ret_values = []

    for patientid in tqdm(all_patientids):
        
        # Extract the patient data
        patient_events = events[events["patientid"] == patientid].copy()
        patient_molecular = molecular[molecular["patientid"] == patientid].copy()
        patient_constant = constant[constant["patientid"] == patientid].copy()

        #: set source column
        patient_events["source"] = "events"
        patient_molecular["source"] = "genetic"

        #: merge and sort
        patient_data = pd.concat([patient_events, patient_molecular], axis=0, ignore_index=True)
        patient_data["date"] = pd.to_datetime(patient_data["date"])
        patient_data = patient_data.sort_values(by="date")

        # Remove any duplicates in case they get in events
        patient_data = patient_data.drop_duplicates()

        # Append to running list
        ret_values.append((patient_constant, patient_data))

    return ret_values

    
