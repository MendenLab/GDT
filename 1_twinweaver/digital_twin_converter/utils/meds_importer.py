import pandas as pd





def convert_meds_to_dtc(df_codes, df_data, df_split,
                        prefer_text_value_over_numeric = True,
                        no_value_default = "occurred",
                        event_category_mapping = None,
                        default_category = "generic_event"):
    """Converts raw medical data into a format compatible with the digital_twin_converter package.

    This function takes multiple DataFrames containing patient data, medical codes,
    and patient splits, and transforms them into three distinct DataFrames:
    one for static (constant) patient data, one for descriptions of that static
    data, and one for time-series event data.

    The function separates events with timestamps from those without. Events
    without timestamps are treated as static patient characteristics and are
    pivoted into a wide-format DataFrame (`converted_constant`). Events with
    timestamps are formatted into a long-format DataFrame (`converted_events`).

    Parameters
    ----------
    df_codes : pd.DataFrame
        DataFrame containing the mapping from medical codes to their descriptions.
        Must contain 'code' and 'description' columns.
    df_data : pd.DataFrame
        The primary DataFrame with event data for each patient. Must include
        'subject_id', 'code', 'time', 'text_value', and 'numeric_value' columns.
    df_split : pd.DataFrame
        DataFrame containing static patient information, such as train/test
        split assignments. Must include a 'subject_id' column.
    prefer_text_value_over_numeric : bool, optional
        If True, when an event has both a text and a numeric value, the text
        value is used. If False, the numeric value is prioritized.
        Defaults to True.
    no_value_default : str, optional
        The default value to assign to an event's 'event_value' if it has
        a timestamp but no associated text or numeric value. Defaults to "occurred".
    event_category_mapping : dict, optional
        A dictionary to map event codes ('event_name') to event categories.
        If None, no specific category mapping is applied. Defaults to None.
    default_category : str, optional
        The category to assign to events that are not found in the
        `event_category_mapping`. Defaults to "generic_event".

    Returns
    -------
    converted_constant : pd.DataFrame
        A DataFrame with one row per patient, containing static features. This
        includes data from `df_split` and pivoted data from `df_data` for
        events that had no timestamp.
    converted_constant_description : pd.DataFrame
        A DataFrame providing human-readable descriptions for each column in
        the `converted_constant` DataFrame.
    converted_events : pd.DataFrame
        A long-format DataFrame containing all time-stamped events, structured
        for time-series analysis. Includes columns like 'patientid', 'date',
        'event_name', 'event_value', and 'event_category'.

    Warns
    -----
    Warning
        Prints a warning to the console if duplicate rows are detected in the
        final `converted_events` DataFrame. Duplicates are checked based on
        the combination of "patientid", "date", "event_name", and "event_value".

    """


    #: Set all subject_id to strings
    df_data["subject_id"] = df_data["subject_id"].astype(str)
    df_split["subject_id"] = df_split["subject_id"].astype(str)

    #: assert all actually used codes in df_data are in df_codes and have a non-NA description
    all_used_codes = df_data["code"]
    assert all_used_codes.isin(df_codes["code"]).all(), "Not all used codes are present in df_codes"
    assert df_codes[df_codes["code"].isin(all_used_codes)]["description"].notna().all(), "Not all used codes have a description in df_codes"

    #: put all events in df_data with no time into constant
    no_time_events = df_data[df_data["time"].isna()].copy()
    if prefer_text_value_over_numeric:
        no_time_events["event_value"] = no_time_events["text_value"]
        no_time_events["event_value"] = no_time_events["event_value"].fillna(no_time_events["numeric_value"])
    else:
        no_time_events["event_value"] = no_time_events["numeric_value"]
        no_time_events["event_value"] = no_time_events["event_value"].fillna(no_time_events["text_value"])

    no_time_events = no_time_events[["subject_id", "code", "event_value"]].drop_duplicates()
    no_time_events = no_time_events.pivot(index="subject_id", columns="code", values="event_value")
    converted_constant = no_time_events.reset_index()

    #: put split data into constant
    converted_constant = converted_constant.merge(df_split, on="subject_id", how="left")
    converted_constant = converted_constant.rename(columns={"subject_id": "patientid"})

    #: generate corresponding constant_description_mapping file, which maps for every column in converted_constant the description of the code
    converted_constant_description = []
    for column in converted_constant.columns:
        if column in df_codes["code"].values:
            description = df_codes[df_codes["code"] == column]["description"].values[0]
            converted_constant_description.append({"variable": column, "comment": description})
        else:
            converted_constant_description.append({"variable": column, "comment": "No description available"})
    converted_constant_description = pd.DataFrame(converted_constant_description)

    #: create first general events file
    converted_events = df_data[df_data["time"].notna()].copy()
    converted_events = converted_events.rename(columns={"subject_id": "patientid", "time": "date", "code": "event_name"})
    converted_events["event_value"] = None  # Placeholder for event_value

    #: based on prefer_text_value_over_numeric, assign event_value
    if prefer_text_value_over_numeric:
        converted_events["event_value"] = converted_events["text_value"]
        converted_events["event_value"] = converted_events["event_value"].fillna(converted_events["numeric_value"])
    else:
        converted_events["event_value"] = converted_events["numeric_value"]
        converted_events["event_value"] = converted_events["event_value"].fillna(converted_events["text_value"])
    converted_events = converted_events.drop(columns=["text_value", "numeric_value"])

    #: fill in event_value with no_value_default if no value is present
    converted_events["event_value"] = converted_events["event_value"].fillna(no_value_default)

    #: apply event_category_mapping to df_data if provided
    if event_category_mapping is not None:
        converted_events["event_category"] = converted_events["event_name"].map(event_category_mapping)
    else:
        converted_events["event_category"] = pd.NA

    #: apply default_category to all events without a category
    converted_events["event_category"] = converted_events["event_category"].fillna(default_category)

    #: add in event_descriptive_name as a merge of event_name and df_codes
    converted_events = converted_events.merge(df_codes[["code", "description"]], left_on="event_name", right_on="code", how="left")
    converted_events = converted_events.rename(columns={"description": "event_descriptive_name"})
    converted_events = converted_events.drop(columns=["code"])

    #: add in empty meta_data column
    converted_events["meta_data"] = pd.NA

    #: issue warning if duplicates are present
    all_duplicate_rows = converted_events[converted_events.duplicated(subset=["patientid", "date", "event_name", "event_value"], keep=False)]
    if not all_duplicate_rows.empty:
        print("Warning: Duplicates found in converted events data. Please make sure to handle them appropriately!")
        print(all_duplicate_rows)

    #: retrun constant, constant_description, events
    return converted_constant, converted_constant_description, converted_events
