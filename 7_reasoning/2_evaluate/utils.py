import pandas as pd
from tqdm import tqdm
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import re


# Set constants
START_OF_TEXT = ""  # Leave the start blank, so to be as close as possible to the original prompt
AFTER_PATIENT_PROMPT = ""  

START_PREDICTION_RANK_TOKEN = "<prediction>"
END_PREDICTION_RANK_TOKEN = "</prediction>"
START_THOUGHT_TAG = "<thinking>"
END_THOUGHT_TAG = "</thinking>"
EOT_TOKEN = "<|eot_id|>"


# Based on training set up
PREDICTION_ADD_SYSTEM_PROMPT = True
SYSTEM_PROMPT = ("As a specialist predictive model in personalized medicine,"
                " your task is to forecast the health trajectory of cancer "
                "patients by integrating genomic data, lifestyle factors, "
                "treatment history and anything else provided about the patient. "
                "Use the provided patient data, including genetic mutations, "
                "biomarker levels, and previous treatment responses, to predict"
                " all requested tasks. Deliver precise and clinically relevant "
                "predictions to enhance patient care and treatment planning.")



BEGINNING_OF_TEXT = (
    """
    You are an expert hematologist-oncologist. You will receive a complete 
    patient history and a specific task to predict the patient's neutrophil 
    trajectory.

    Your primary goal is to generate a step-by-step reasoning chain that 
    explains to your prediction. This rationale is more important than the 
    prediction itself.

    Structure your entire response using the following tags. Do not include 
    any text outside of these tags.

    [Place the final, formatted prediction here as specified in the task 
    description. Do not include an opening tag.]
    </prediction>

    <thinking>
    Inside this tag, you must follow this four-step reasoning process:

    1.  **Patient Summary:** Briefly summarize the patient's current oncological 
        and hematological status. Focus on the diagnosis, active treatments, 
        and the most recent relevant lab values.
    2.  **Key Predictive Factors:** Identify the **top 5 most influential factors** from the patient's record that will drive the neutrophil trajectory. 
        List each factor (e.g., specific drug, time since last treatment, 
        comorbidity, recent lab trend) and provide a concise justification 
        for its high importance.
    3.  **Mechanistic Analysis:** This is the most critical step. Synthesize 
        the 5 factors you identified. Provide a detailed, step-by-step 
        biological explanation of how these factors will interact to 
        influence the neutrophil count *over time*.
        * Describe the specific biological pathways involved (e.g., 
            myelosuppression from a specific drug class, hematopoietic 
            recovery kinetics, effects of G-CSF on bone marrow 
            precursors, inflammatory cytokine release).
        * Explain the expected *timing* of these effects (e.g., "The 
            patient is X days post-[Chemo], so we expect the nadir 
            around day Y," or "The recent G-CSF administration will 
            likely cause a transient leukocytosis followed by...").
    4.  **Confounding Factors:** Briefly mention 1-2 other factors (e.g., 
        potential infection, patient age, nutritional status) that could 
        complicate or alter your primary predicted trajectory.

    </thinking>

    <prognosis_summary>
    Based on your thinking and rationale, provide a 1-2 sentence summary of the expected 
    neutrophil trend (e.g., "Expect sharp decline into severe neutropenia," 
    or "Anticipate slow but steady recovery") and the primary clinical risk 
    (e.g., "High risk of febrile neutropenia," or "Risk of treatment delay").
    </prognosis_summary>
    """
)




def prepare_prompts_for_vllm(prompts_df, num_copies):

    new_prompts = prompts_df.copy()

    # Make copies 
    new_prompts = new_prompts.loc[new_prompts.index.repeat(num_copies)].reset_index(drop=True)

    return new_prompts


def generate_llm_prompt_selection(prompts_df, target_df,
                                  system_prompt_override=None):

    all_patientids = prompts_df['patientid'].unique().tolist()
    all_patientids.sort()
    all_prompts = []

    for patientid in tqdm(all_patientids):
        
        # Grab data
        patient_prompts = prompts_df[prompts_df['patientid'] == patientid]
        instruction_prompt = patient_prompts["input_prompt"].iloc[0]
        curr_target = target_df[target_df["patientid"] == patientid].iloc[0]["target_df"]
        curr_target_prompt = target_df[target_df["patientid"] == patientid].iloc[0]["target_prompt"]
        curr_target_list = curr_target["event_value"].tolist()
        
        if system_prompt_override is not None:
            system_prompt = system_prompt_override
        else:
            system_prompt = SYSTEM_PROMPT
        
        new_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction_prompt}
        ]

        # Save in format needed for TRL
        all_prompts.append({"patientid": patientid, 
                            "prompt": new_prompt, 
                            "target_prompt": curr_target_prompt,
                            "target_list_of_values" : curr_target_list})
        
    all_prompts_df = pd.DataFrame(all_prompts)

    return all_prompts_df




def get_target(target_df, train_prompts_df):

    # Get target DF and text
    patientids = target_df["patientid"].unique().tolist()
    patientids.sort()
    targets = []

    for patientid in patientids:
        
        # Get all predictions
        curr_target = target_df[target_df["patientid"] == patientid]
        curr_target = curr_target.sort_values("date")

        # Get target prompt
        target_prompt = train_prompts_df[train_prompts_df["patientid"] == patientid]["target_prompt"].iloc[0]

        #: add to df
        targets.append({"patientid": patientid,
                        "target_df": curr_target,
                        "target_prompt": target_prompt})        
    
    # Target
    targets_df = pd.DataFrame(targets)
    return targets_df




def get_dataframe_of_results_and_targets_for_llms(all_patient_data):

    target_list = []
    prompts_list = []

    for constant, patient_data, converted_data, in tqdm(all_patient_data):
        
        # Grab DFs
        target = converted_data["meta"]["target_meta_detailed"][0]["target_data_raw"]
        patientid = constant["patientid"].iloc[0]
        input_prompt = converted_data["instruction"]
        target_prompt = converted_data["answer"]
        last_observed_values = converted_data["meta"]["target_meta_detailed"][0]["last_observed_values"].to_dict("records")[0]

        target_list.append(target)
        prompts_list.append({"patientid": patientid,
                             "input_prompt": input_prompt,
                             "target_prompt": target_prompt,
                             "last_observed_values": last_observed_values})
                            
    # Create DFs
    target_df = pd.concat(target_list)
    target_df = target_df.drop_duplicates()
    prompts_df = pd.DataFrame(prompts_list)

    # Return
    return target_df, prompts_df




def post_process_responses(responses, target_df, pred_then_cot=False):
    """
    Processes multiple responses per patient by averaging their predictions.
    If no valid predictions are found for a patient, the overall mean is used.
    Using a hacky and dirty way to ensure reasoning edge cases are handled correctly.
    
    Parameters:
    - responses: List of tuples in the format (patientid, response, logprobs).
    - target_df: DataFrame containing the target information with a 'patientid' column.
    
    Returns:
    - selected_df: DataFrame with averaged predictions per patient.
    """
    
    ret_list = []
    
    # Group responses by patientid
    grouped_responses = defaultdict(list)
    for patientid, response, logprobs in responses:
        grouped_responses[patientid].append(response)
    
    # Calculate the overall mean of 'event_value' to use as a fallback
    overall_mean = target_df["event_value"].mean()
    
    # Iterate over each patient group
    for patientid, resp_list in tqdm(grouped_responses.items(), desc="Processing Patients"):

        event_values_accumulator = []
        curr_target = target_df[target_df["patientid"] == patientid]
        
        for response in resp_list:
            try:
                # Define the regex pattern to extract relevant values

                # First try extracting from <prediction></prediction> tags
                def extract_value(resp):
                    # Then extract neutrophils part
                    pattern = r'neutrophils\s*-\s*26499-4\s*is\s*([0-9.,]+)'
                    matches = re.findall(pattern, resp)
                    return matches
                
                if pred_then_cot:
                    pattern_in_prediction = r'(.*?)<\/prediction>'
                else:
                    pattern_in_prediction = r'<prediction>.*?<\/prediction>'
                matches_pred = re.findall(pattern_in_prediction, response, re.DOTALL) # Find the prediction section
                if matches_pred:
                    curr_response = matches_pred[0]
                    matches = extract_value(curr_response)
                    if len(matches) == 0:
                        # if no matches found, try extracting from rest of test
                        matches = extract_value(response)

                else:
                    curr_response = response
                    matches = extract_value(response)

                # Clean the extracted matches
                cleaned_matches = []
                for match in matches:
                    match = match.strip().rstrip('.')  # Remove trailing period if present
                    match = match.replace(",", "")      # Remove commas
                    if match:                           # Ensure the match is not empty
                        cleaned_matches.append(match)
                
                # Convert matches to numeric values
                new_values = pd.to_numeric(cleaned_matches, errors='coerce').tolist()
                
                # Handle length discrepancies by forward filling or truncating
                if len(new_values) < len(curr_target):
                    new_values.extend([np.nan] * (len(curr_target) - len(new_values)))
                elif len(new_values) > len(curr_target):
                    new_values = new_values[:len(curr_target)]
                
                # Create a Series for forward filling
                new_values_series = pd.Series(new_values)

                if new_values_series.isna().any():
                    print(f"Warning: NaN values found in predictions for patientid {patientid}. Attempting to forward fill.")

                # We actually now just skip series with nan values
                if new_values_series.isna().any():
                    new_values_series = new_values_series.ffill()
                
                # Ensure no NaNs remain after forward fill
                if new_values_series.isna().any():
                    new_values_series = new_values_series.fillna(method='bfill')
                
                # Final check to ensure no NaNs
                assert not new_values_series.isna().any(), "Still NaN values found after processing."
                
                # Append the cleaned values to the accumulator
                event_values_accumulator.append(new_values_series.tolist())
            
            except Exception as e:
                # Log the error and skip to the next response
                print("========================================= Full response =========================================")
                print(response)
                print("===")
                print(f"Error processing patientid {patientid}: {e} - Using mean for this response.")
                continue
        
        if event_values_accumulator:
            # Convert the list of lists to a NumPy array for averaging
            event_array = np.array(event_values_accumulator)
            
            # Calculate the mean across all valid responses
            mean_event_values = np.nanmean(event_array, axis=0)
        else:
            # If no valid responses, use the overall mean
            mean_event_values = [overall_mean] * len(curr_target)
            print(f"No valid predictions for patientid {patientid}. Using overall mean.")
        
        # Assign the averaged values to the 'event_value' column
        curr_target = curr_target.copy()
        curr_target["event_value"] = np.nan   # First override with nans
        assert curr_target["event_value"].isna().all(), "Failed to set event_value to NaN."

        # Then assign new values so that dates etc. are easily preserved
        curr_target["event_value"] = mean_event_values
        
        # Append the processed DataFrame to the return list
        ret_list.append(curr_target)
    
    # Concatenate all patient DataFrames into a single DataFrame
    selected_df = pd.concat(ret_list, ignore_index=True)
    
    return selected_df
        