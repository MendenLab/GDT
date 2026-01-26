import pandas as pd
from tqdm import tqdm
import sys
import os



# Set constants
START_OF_TEXT = ""  # Leave the start blank, so to be as close as possible to the original prompt
AFTER_PATIENT_PROMPT = ""  

START_PREDICTION_RANK_TOKEN = "<prediction>"
END_PREDICTION_RANK_TOKEN = "</prediction>"
START_THOUGHT_TAG = "<thinking>"
END_THOUGHT_TAG = "</thinking>"
EOT_TOKEN = "<|eot_id|>"


SYSTEM_PROMPT = ("As a specialist predictive model in personalized medicine,"
                " your task is to forecast the health trajectory of cancer "
                "patients by integrating genomic data, lifestyle factors, "
                "treatment history and anything else provided about the patient. "
                "Use the provided patient data, including genetic mutations, "
                "biomarker levels, and previous treatment responses, to predict"
                " all requested tasks. Deliver precise and clinically relevant "
                "predictions to enhance patient care and treatment planning.")


def generate_llm_prompt_selection(prompts_df, target_df,
                                  task_instruction,
                                  system_prompt_and_chat_template=False,
                                  system_prompt_override=None):

    all_patientids = prompts_df['patientid'].unique().tolist()
    all_prompts = []

    for patientid in tqdm(all_patientids):
        
        # Grab data
        patient_prompts = prompts_df[prompts_df['patientid'] == patientid]
        instruction_prompt = patient_prompts["input_prompt"].iloc[0]
        curr_target = target_df[target_df["patientid"] == patientid].iloc[0]["target_df"]
        curr_target_prompt = target_df[target_df["patientid"] == patientid].iloc[0]["target_prompt"]
        curr_target_list = curr_target["event_value"].tolist()
        
        if system_prompt_and_chat_template:
            curr_instruction = START_OF_TEXT + instruction_prompt + AFTER_PATIENT_PROMPT + task_instruction
            if system_prompt_override is not None:
                system_prompt = system_prompt_override
            else:
                system_prompt = SYSTEM_PROMPT
            new_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": curr_instruction}
            ]
        else:
            new_prompt = START_OF_TEXT + instruction_prompt + AFTER_PATIENT_PROMPT
            new_prompt += task_instruction

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

        