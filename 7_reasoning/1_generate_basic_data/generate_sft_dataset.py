import pandas as pd
import pickle
from tqdm import tqdm
from utils import (get_dataframe_of_results_and_targets_for_llms,
                   get_target, generate_llm_prompt_selection)
import wandb
from openai import OpenAI
from tqdm import tqdm
import time
import subprocess
import requests
import os
import sys

NUM_TO_SELECT = "all"   # Nr of SFT samples to generate
END_OF_TEXT_TOP_5 = (
    """
    You are an expert hematologist-oncologist. You will receive a complete 
    patient history and a specific task to predict the patient's neutrophil 
    trajectory.

    Your primary goal is to generate a step-by-step reasoning chain that 
    leads to your prediction. This rationale is more important than the 
    prediction itself.

    Structure your entire response using the following tags. Do not include 
    any text outside of these tags.

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
    
    <prediction>
    [Place the final, formatted prediction here as specified in the task 
    description.]
    </prediction>
    """
)


PORT = 7983
VLLM_BASE_URL = f"http://localhost:{PORT}/v1/"
MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"   # Large model
LOAD_PATH = "genie-dt-grpo-forecasting/0_data/converted_data/2025_03_18_converted_2385_train.pkl"
SAVE_PATH = "genie-dt-grpo-forecasting/0_data/sft_data/"
NUM_GPUS = 2
PYTHON_PATH = "miniforge3/envs/mamba_vllm_b200/bin/python"
CACHE_DIR="genie-dt-grpo-forecasting/0_data/model_cache"


def main(port=PORT, model_name=MODEL_NAME, load_path=LOAD_PATH, save_path=SAVE_PATH, 
         num_gpus=NUM_GPUS, python_path=PYTHON_PATH, cache_dir=CACHE_DIR):

    vllm_base_url = f"http://localhost:{port}/v1/"

    wandb.init(project="genie-dt-grpo-forecasting-paper", group="generate_data")
    wandb.run.name = "Generate SFT dataset"

    env = os.environ.copy()
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    vllm_command = [python_path, "-m", "vllm.entrypoints.openai.api_server", "--port", str(port),
                    "--model", model_name, "--enable-prefix-caching",
                    "--download-dir", cache_dir,
                    "--tensor-parallel-size", str(num_gpus)]
    
    print(f"Launching vLLM server with command: {' '.join(vllm_command)}")

    # Launch the subprocess and capture its stdout and stderr
    process = subprocess.Popen(
        vllm_command,
        env=env,
        text=True, # Decode output as text
        bufsize=1, # Line-buffered
        universal_newlines=True
    )

    # Start a thread to watch and print the subprocess output
    # Poll the server for up to a certain amount of time
    server_ready = False
    start_time = time.time()
    print("\nWaiting for the vLLM server to load...")
    while time.time() - start_time < 1800:
        try:
            # Check if the API endpoint is responding
            response = requests.get(vllm_base_url + "models") 
            if response.status_code == 200:
                print("✅ vLLM server is ready.")
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            # Server is not up yet, wait a bit
            time.sleep(5)

    if not server_ready:
        print("❌ vLLM server failed to start within given time. Terminating.")
        if process:
            process.terminate()
        sys.exit(1) # Exit the script with an error code


    print("Loading data...")
    with open(load_path, 'rb') as f:
        train_data = pickle.load(f)


    print("Preprocessing data...")
    train_target_df, train_prompts_df = get_dataframe_of_results_and_targets_for_llms(train_data)

    #: generate target prompt and df!!
    print("Generating targets...")
    train_targets = get_target(train_target_df, train_prompts_df)

    #: setup prompts and targets, each entry with columns "patientid", "input_prompt", "target_prompt"
    print("Setting up prompts and targets...")
    train_data_input_and_target = generate_llm_prompt_selection(train_prompts_df, train_targets, 
                                                                system_prompt_and_chat_template=True, 
                                                                task_instruction=END_OF_TEXT_TOP_5,
                                                                system_prompt_override=END_OF_TEXT_TOP_5)


    # Generate the respective prompts
    horizontal_rule = "\n---------------------------------------------------------\n"
    pre_prompt = horizontal_rule + "Here is the background data:\n"
    post_prompt = horizontal_rule + "Here is the correct prediction:\n"
    main_prompt = horizontal_rule + "Now your actual task is to provide a final correct thinking process and output for the tasks in the background data, reasoning as best you can and using the correct prediction as a reference point for the reasoning (though never mention it explicitly in the reasoning!) and the prediction in the <prediction> </prediction> tags."

    train_prompts_df['final_prompt'] = pre_prompt + train_prompts_df['input_prompt'] + END_OF_TEXT_TOP_5 + post_prompt + train_prompts_df['target_prompt'] + main_prompt
    train_prompts_df["task_instruction"] = END_OF_TEXT_TOP_5

    print(train_prompts_df['final_prompt'].iloc[0])


    # Merge on patientid
    final_prompts = train_prompts_df.merge(train_data_input_and_target[["patientid", "prompt", "target_list_of_values"]], on='patientid', how='left')


    # Shuffle and select first NUM_TO_SELECT samples
    if NUM_TO_SELECT != "all":
        final_prompts = final_prompts.sample(frac=1, random_state=9867).reset_index(drop=True)
        final_prompts = final_prompts.iloc[:NUM_TO_SELECT]
        print(f"Selected {NUM_TO_SELECT} samples for SFT dataset.")

    print(f"Final SFT dataset size: {final_prompts.shape[0]} samples.")



        
    # --- Client Initialization ---
    # Initialize the OpenAI client to point to your local vLLM server.
    # The API key is not used by a default vLLM setup but is required by the client.
    client = OpenAI(
        base_url=vllm_base_url,
        api_key="EMPTY" # vLLM doesn't require a key by default
    )

    def generate(text_prompt: str) -> str:
        """
        Generates a response from the vLLM server using the OpenAI client.
        """
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": text_prompt}
                ],
                extra_body={
                    "temperature" : 0.7,  # From HF best practices
                    "top_p" : 0.8,   # From HF best practices
                    "top_k" : 20,   # From HF best practices
                    "min_p" : 0,   # From HF best practices
                    "max_tokens" : 2000,    # Need to see if this is enough
                    "seed": 8762,
                }
            )
            # The response structure is different from the Google AI client
            full_response_text = response.choices[0].message.content
            return full_response_text.strip()

        except Exception as e:
            print(f"An error occurred: {e}")
            return f"Error: Could not generate response. Details: {e}"


    # First, test the generate function with a simple prompt
    print("--- Testing with a single prompt ---")
    r = generate("Hello! Tell me a fun fact about the Qwen model.")
    print(r)
    print("\n" + "="*50 + "\n")


    
    # Go through all samples and generate the predictions, then save as a new column
    print("--- Processing DataFrame prompts ---")
    for index, row in tqdm(final_prompts.iterrows(), total=final_prompts.shape[0]):
        text_prompt = row['final_prompt']
        final_response = generate(text_prompt)
        final_prompts.at[index, 'final_response'] = final_response
        

    
    print(len(final_prompts))

    # Save the final prompts with responses
    final_prompts.to_csv(save_path + "train_" + model_name.replace("/", "-") + "_num_samples_" + str(len(final_prompts)) + "_sft_prompts.csv", index=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SFT dataset using vLLM")
    parser.add_argument("--port", type=int, default=PORT, help="Port for the vLLM server")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Model name to use")
    parser.add_argument("--load_path", type=str, default=LOAD_PATH, help="Path to load the training data (pickle file)")
    parser.add_argument("--save_path", type=str, default=SAVE_PATH, help="Path to save the generated SFT dataset")
    parser.add_argument("--num_gpus", type=int, default=NUM_GPUS, help="Number of GPUs to use")
    parser.add_argument("--python_path", type=str, default=PYTHON_PATH, help="Path to Python executable")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR, help="Cache directory for model downloads")
    
    args = parser.parse_args()
    
    main(args.port, args.model_name, args.load_path, args.save_path, 
         args.num_gpus, args.python_path, args.cache_dir)


