import pandas as pd
import re
from io import StringIO
from fuzzywuzzy import process
import os
import sys


def setup_imports_nb():
    try:
        # In case you run this cell from a .py file later via an IDE that defines __file__
        notebook_parent_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ is not defined in a standard Jupyter notebook cell, so use os.getcwd()
        # os.getcwd() usually points to the directory where the notebook file is located,
        # or where the Jupyter server was started.
        # Be sure this is the correct directory for your structure.
        notebook_parent_dir = os.getcwd()

    project_root_dir = os.path.abspath(os.path.join(notebook_parent_dir, "../2_forecasting_eval_utils"))
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)



def add_extra_prompt_to_instruction(raw_df, extra_prompt):
    """
    Add extra prompt to instruction column in the dataframe.
    """
    raw_df = raw_df.copy()
    raw_df["instruction"] = raw_df["instruction"].apply(lambda x: f"{x} {extra_prompt}")
    return raw_df




def process_raw_data_to_list(raw_df):
    ret = raw_df[["patientid", "instruction"]].values.tolist()
    return ret



def make_nr_of_copies(raw_df, nr_of_copies):
    """
    Make a specified number of copies of each row in the DataFrame.
    
    Args:
        raw_df (pandas.DataFrame): The input DataFrame to copy.
        nr_of_copies (int): The number of copies to create for each row.
        
    Returns:
        pandas.DataFrame: A new DataFrame with the specified number of copies.
    """
    adjusted = pd.concat([raw_df] * nr_of_copies, ignore_index=True)
    return adjusted




def parse_lab_data(text_data):
    """
    Parses multi-line text data containing lab results and converts it into a Pandas DataFrame.

    This function can handle several formats, including single-line entries and
    multi-line formats where a 'Week X:' header applies to subsequent lines.

    Args:
        text_data (str): A string containing the lab results.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'week', 'event_name', and 'event_value'.
                          Returns an empty DataFrame if no data could be parsed.
    """
    records = []
    lines = text_data.strip().split('\n')
    current_week = None

    # Pattern for lines that define the current week (e.g., "Week 1:")
    week_header_pattern = re.compile(r"^\s*Week\s*(\d+):?\s*$", re.IGNORECASE)

    # Pattern for indented lab data lines following a week header (e.g., "  calcium - 17861-6: 9.1")
    indented_lab_pattern = re.compile(r"^\s*(.+?)\s*-\s*([\d-]+)\s*:\s*([\d.]+)$")

    # MODIFIED: Pattern for lines that start with "Week X:" or "Week X,". Added a '$' to anchor the match to the end of the line.
    week_prefix_pattern = re.compile(
        r"^\s*Week\s*(\d+)\s*[:,]?\s*(.*?)\s*(?:-\s*([\d-]+))?\s*(?:is|:)?\s*([\d.]+)\s*$",
        re.IGNORECASE
    )

    # Pattern for lines like "5 weeks - ...", "1 week: ...", or "11 weeks, ..."
    weeks_keyword_pattern = re.compile(
        r"^\s*(\d+)\s+weeks?\s*[-:,]\s*(.*?)\s*-\s*([\d-]+)\s*[:,]?\s*([\d.]+)",
        re.IGNORECASE
    )

    # Pattern for lines like "3, erythrocytes - 26453-1, 4.95"
    week_first_comma_pattern = re.compile(
        r"^\s*(\d+)\s*,\s*(.*?)\s*-\s*([\d-]+)\s*,\s*([\d.]+)\s*$",
        re.IGNORECASE
    )

    # Pattern for lines like "2 basophils - 704-7: 5.0"
    week_first_pattern = re.compile(
        r"^\s*(\d+)\s+(?!weeks?)(.*?)\s*-\s*([\d-]+)\s*:\s*([\d.]+)",
        re.IGNORECASE
    )

    # Pattern for lines like "creatinine - 2160-0, week 4, 0.83"
    lab_first_pattern = re.compile(
        r"^\s*(.*?)\s*-\s*([\d-]+)\s*,\s*week\s*(\d+)\s*,\s*([\d.]+)\s*$",
        re.IGNORECASE
    )

    # Generic pattern for single-line lab data
    single_line_pattern = re.compile(
        r"^(.*?)\s*(?:-\s*([\d-]+))?\s*(?:for week|week|at week|is|:)\s*(\d+)\s*(?:is|:)?\s*([\d.]+)",
        re.IGNORECASE
    )

    # Patterns for lines to ignore
    ignore_patterns = [
        re.compile(p, re.IGNORECASE) for p in [
            r"^Task \d+:",
            r"^==== response: =====",
            r"^Forecasting the future values"
        ]
    ]

    for line in lines:
        line = line.strip()
        if not line or any(p.search(line) for p in ignore_patterns):
            continue

        # 1. Check for the "Week X:" header format
        week_header_match = week_header_pattern.match(line)
        if week_header_match:
            current_week = int(week_header_match.group(1))
            continue

        # 2. Check for the indented lab data format
        indented_lab_match = indented_lab_pattern.match(line)
        if indented_lab_match and current_week is not None:
            try:
                lab_name_part = indented_lab_match.group(1).strip()
                loinc = indented_lab_match.group(2).strip()
                value = float(indented_lab_match.group(3))
                lab_name = f"{lab_name_part} - {loinc}"
                records.append({'week': current_week, 'event_name': lab_name, 'event_value': value})
                continue
            except (ValueError, IndexError):
                print(f"Skipping line due to conversion or parsing error: {line}")
                continue

        # 3. Check for the "Week X, ..." or "Week X: ..." format
        match = week_prefix_pattern.match(line)
        if match:
            try:
                week = int(match.group(1))
                lab_name_full = match.group(2).strip()
                loinc = match.group(3)  # Optional
                value = float(match.group(4))
                lab_name = lab_name_full
                if loinc:
                    lab_name_full = lab_name_full.strip()
                    loinc = loinc.strip()
                    # Ensure LOINC is not duplicated
                    if not lab_name_full.endswith(loinc):
                        lab_name = f"{lab_name_full} - {loinc}"
                records.append({'week': week, 'event_name': lab_name, 'event_value': value})
                continue
            except (ValueError, IndexError):
                print(f"Skipping line due to conversion or parsing error: {line}")
                continue

        # 4. Check for the "11 weeks, ...", "1 week: ...", or "5 weeks - ..." format
        match = weeks_keyword_pattern.match(line)
        if match:
            try:
                week = int(match.group(1))
                lab_name_part = match.group(2).strip()
                loinc = match.group(3).strip()
                value = float(match.group(4))
                lab_name = f"{lab_name_part} - {loinc}"
                records.append({'week': week, 'event_name': lab_name, 'event_value': value})
                continue
            except (ValueError, IndexError):
                print(f"Skipping line due to conversion or parsing error: {line}")
                continue

        # 5. Check for "3, erythrocytes..." format
        match = week_first_comma_pattern.match(line)
        if match:
            try:
                week = int(match.group(1))
                lab_name_part = match.group(2).strip()
                loinc = match.group(3).strip()
                value = float(match.group(4))
                lab_name = f"{lab_name_part} - {loinc}"
                records.append({'week': week, 'event_name': lab_name, 'event_value': value})
                continue
            except (ValueError, IndexError):
                print(f"Skipping line due to conversion or parsing error: {line}")
                continue

        # 6. Check for the "2 basophils..." format (requires a value)
        match = week_first_pattern.match(line)
        if match:
            try:
                week = int(match.group(1))
                lab_name_part = match.group(2).strip()
                loinc = match.group(3).strip()
                value = float(match.group(4))
                lab_name = f"{lab_name_part} - {loinc}"
                records.append({'week': week, 'event_name': lab_name, 'event_value': value})
                continue
            except (ValueError, IndexError):
                print(f"Skipping line due to conversion or parsing error: {line}")
                continue

        # 7. Check for "creatinine - 2160-0, week 4..." format
        match = lab_first_pattern.match(line)
        if match:
            try:
                lab_name_part = match.group(1).strip()
                loinc = match.group(2).strip()
                week = int(match.group(3))
                value = float(match.group(4))
                lab_name = f"{lab_name_part} - {loinc}"
                records.append({'week': week, 'event_name': lab_name, 'event_value': value})
                continue
            except (ValueError, IndexError):
                print(f"Skipping line due to conversion or parsing error: {line}")
                continue

        # 8. Check for the most generic single-line format
        match = single_line_pattern.match(line)
        if match:
            try:
                lab_description = match.group(1).strip()
                loinc = match.group(2) # Optional
                week = int(match.group(3))
                value = float(match.group(4))
                lab_name = lab_description
                if loinc and not lab_description.endswith(loinc):
                    lab_name = f"{lab_description} - {loinc}"
                records.append({'week': week, 'event_name': lab_name, 'event_value': value})
                continue
            except (ValueError, IndexError):
                print(f"Skipping line due to conversion or parsing error: {line}")
                continue

        # If no pattern matches, print a warning
        print(f"Skipping line due to format mismatch: {line}")

    if not records:
        return pd.DataFrame(columns=['week', 'event_name', 'event_value'])

    return pd.DataFrame(records)





def match_to_closest_true_names(parsed_results: pd.DataFrame, empty_target_df: pd.DataFrame) -> pd.DataFrame:
    
    # Create a dictionary mapping each patientid to a list of their true event names
    per_split_target_names_df = empty_target_df[["patientid", "event_descriptive_name"]].copy().drop_duplicates()
    per_split_target_names_dict = {}
    for i, row in per_split_target_names_df.iterrows():
        patientid = row["patientid"]
        event_name = row["event_descriptive_name"]
        if patientid not in per_split_target_names_dict:
            per_split_target_names_dict[patientid] = []
        per_split_target_names_dict[patientid].append(event_name)

    # Standardize the event names from parsed results to lowercase for better matching
    parsed_results["event_name"] = parsed_results["event_name"].str.lower()

    # Map the event names to the closest names in the empty target names
    def map_event_name(event_name: str, patientid: str) -> str:
        """
        Finds the closest matching event name from the target list for a given patient
        using fuzzy string matching based on Levenshtein distance.
        """

        if patientid in per_split_target_names_dict:
            possible_names = per_split_target_names_dict[patientid]
            # Ensure the list of possible names is not empty
            if possible_names:
                #: Find the closest name, based on the minimum edit distance
                # It returns a tuple of the best match and its score, e.g., ('blood_pressure', 90)
                best_match, score = process.extractOne(event_name, possible_names)
                return best_match

        return event_name  # Return original if no match found

    # Apply the mapping function to create a new column with the closest event names
    parsed_results['closest_event_name'] = parsed_results.apply(
        lambda row: map_event_name(row['event_name'], row['patientid']), axis=1
    )
    parsed_results["event_name"] = parsed_results["closest_event_name"]
    parsed_results = parsed_results.drop(columns=['closest_event_name'])
    return parsed_results





def parse_all_results_into_df(raw_results):
    parsed_results = []
    for patientid, result, logprobs in raw_results:
        parsed_result = parse_lab_data(result)
        parsed_result['patientid'] = patientid
        if parsed_result.empty:
            continue
        parsed_results.append(parsed_result)
    full_df = pd.concat(parsed_results, ignore_index=True)

    # drop all entries with no name for event_name
    full_df = full_df[full_df['event_name'].notna() & (full_df['event_name'] != '')]

    return full_df





def process_empty_targets_from_raw_data(raw_initial_data):

    all_empty_targets = []
    for i, row in raw_initial_data.iterrows():
        target_as_str = row["empty_target_as_string"]
        # parse from json str, orient="records"
        target_as_df = pd.read_json(StringIO(target_as_str), orient="records",
                                    dtype={'event_name': str})
        all_empty_targets.append(target_as_df)
    all_empty_targets_df = pd.concat(all_empty_targets, ignore_index=True)
    return all_empty_targets_df





def average_results_by_week(parsed_results: pd.DataFrame) -> pd.DataFrame:

    # Group by patientid, week and the newly found closest_event_name,
    # and calculate the mean for each group. The .mean() will only apply to numeric columns.
    grouping_cols = ['patientid', 'date', 'event_name']
    averaged_results = parsed_results.groupby(grouping_cols).mean().reset_index()

    return averaged_results





def match_to_event_name_and_adjust_patientid(results, empty_target_df):
   
    #: match event descriptive name to event name
    event_name_to_event_descriptive_name = empty_target_df[["event_name", "event_descriptive_name"]].drop_duplicates()
    results = results.rename(columns={"event_name": "event_name_raw"})
    
    # Now merge on event_name_raw to event_descriptive_name
    results = results.merge(event_name_to_event_descriptive_name, 
                            left_on="event_name_raw", 
                            right_on="event_descriptive_name", 
                            how="left")

    #: make patientid of the format 0142A35B825C23ABB790_split_0_var_26453_1
    results["patientid"] = results["patientid"].astype(str) + "_var_" + results["event_name"].astype(str)

    results = results.drop(columns=["event_name_raw", "event_descriptive_name"])

    return results





def fill_in_missing_values_with_copy_forward(results, raw_data, empty_target_df):


    #: match results to empty_target_df
    empty_target_df = empty_target_df[["patientid", "event_descriptive_name", "date"]].copy()
    results_with_all_target_dates = empty_target_df.merge(results, 
                                                          left_on=["patientid", "event_descriptive_name", "date"], 
                                                          right_on=["patientid", "event_name", "date"],
                                                          how="left", suffixes=("_target", ""))
    results = results_with_all_target_dates.sort_values(by=["patientid", "date", "event_name"])

    
    #: load in from raw_data the last observed values
    all_last_observed = []
    for i, row in raw_data.iterrows():
        last_observed_as_str = row["last_observed_values"]
        # parse from json str, orient="records"
        lo_as_df = pd.read_json(StringIO(last_observed_as_str), orient="records",
                                    dtype={'event_name': str})
        all_last_observed.append(lo_as_df)
    all_last_observed_df = pd.concat(all_last_observed, ignore_index=True)
    all_last_observed_df = all_last_observed_df[['patientid', 'event_descriptive_name', 'event_value']]

    #: match last observd to results
    results = results.merge(all_last_observed_df, on=["patientid", "event_descriptive_name"],
                            how="left", suffixes=("", "_last_observed"))

    #: for any missing values, forward fill, grouped by patientid and event_name
    results['event_value'] = results.groupby(['patientid', 'event_descriptive_name'])['event_value'].ffill()

    #: for any still missing values, apply the last observed value
    results['event_value'] = results.apply(
        lambda row: row['event_value'] if pd.notna(row['event_value']) else row['event_value_last_observed'], axis=1)
    
    # Clean up
    results["event_name"] = results["event_descriptive_name"]
    results = results.drop(columns=['event_value_last_observed', 'event_descriptive_name'])
    assert results["event_value"].notna().all(), "There are still missing values in the results!"

    return results



def convert_to_dates_and_match_to_closest_target_date(results, empty_target_df, raw_data):

    # Ensure date columns are in datetime format
    empty_target_df['date'] = pd.to_datetime(empty_target_df['date'])
    raw_data['split_date_included_in_input'] = pd.to_datetime(raw_data['split_date_included_in_input'])

    #: make dict for target dates
    target_dates_df = empty_target_df[['patientid', 'event_descriptive_name', 'date']].drop_duplicates()
    unique_patientid_event_descriptive_name = target_dates_df[['patientid', 'event_descriptive_name']].drop_duplicates()
    target_dates_dict = {}
    for index, row in unique_patientid_event_descriptive_name.iterrows():
        patientid = row['patientid']
        event_descriptive_name = row['event_descriptive_name']
        corresponding_date_list = target_dates_df[
            (target_dates_df['patientid'] == patientid) &
            (target_dates_df['event_descriptive_name'] == event_descriptive_name)
        ]['date'].tolist()
        target_dates_dict[(patientid, event_descriptive_name)] = corresponding_date_list

    #: convert results to dates, based on split date and weeks
    split_date = raw_data[["patientid", "split_date_included_in_input"]].drop_duplicates()
    results = results.merge(split_date, on="patientid", how="left")
    
    # Cap week results to 156 (3 years) to avoid overflow issues
    if (results['week'] > 156).any():
        print("Warning: Some week values exceed 156. Capping them to 156 to avoid overflow.")
        results['week'] = results['week'].clip(upper=156)
    
    results['date'] = results['split_date_included_in_input'] + pd.to_timedelta(
        results['week'] * 7, unit='d')

    #: match to closest empty target date
    def find_closest_date(current_date, patientid, event_name):
        candidate_dates = target_dates_dict.get((patientid, event_name), [])

        if not candidate_dates or pd.isna(current_date):
            return pd.NaT
        # Find and return the date in the candidate list with the minimum absolute difference
        return min(candidate_dates, key=lambda d: abs(current_date - d))

    # For each row in the results, find the list of candidate dates and then find the closest one.
    results['closest_target_date'] = results.apply(
        lambda row: find_closest_date(
            row['date'], 
            row['patientid'], 
            row['event_name'],
        ),
        axis=1
    )
    results["date"] = results['closest_target_date']
    results = results.drop(columns=['split_date_included_in_input', 'week', 'closest_target_date'])
    assert results['date'].notna().all(), "Some dates are NaT after conversion!"
    return results







