import pandas as pd
import numpy as np
import wandb
import argparse
from datasets import Dataset
from trl import (
    GRPOTrainer,  
    GRPOConfig,  
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datetime
import os
import re
import ast


from utils import SYSTEM_PROMPT, START_THOUGHT_TAG, END_THOUGHT_TAG, START_PREDICTION, END_PREDICTION, START_PROGNOSIS, END_PROGNOSIS


DEBUG = False


# Use SFT model as base
MODEL_NAME = "0_data/6_sft_pred_then_cot/runs/2025-10-30_15-57-43/saves"


SAVE_FOLDER = "0_data/7_grpo_pred_then_cot/"
PATH_TO_TRAIN_DF = "0_data/sft_data/cot_after_target/train_cot_after_target_Qwen-Qwen3-Next-80B-A3B-Instruct_num_samples_2385.csv"
PATH_TO_VALIDATION_DF = "0_data/sft_data/cot_after_target/validation_cot_after_target_Qwen-Qwen3-Next-80B-A3B-Instruct_num_samples_235.csv"
EVAL_SAVE_STEPS = 500
EVAL_BATCH_SIZE = 4


#: setup reward functions
def _extract_df_of_values_of_neutrophils(response, curr_target_list):
    try:

        # Hacky, convert manually back to prediction DF
        pattern = r'neutrophils\s*-\s*26499-4\s*is\s*([0-9].+)'
        matches = re.findall(pattern, response)
        matches = [x[:-1] if x[-1] == "." else x for x in matches]  # Remove if not ending in a period
        matches = [x.replace(",", "") for x in matches]  # Remove commas

        # Make new DF, based on the target
        new_values = pd.to_numeric(matches, errors='coerce').tolist()
        # Forward fill if too short
        if len(new_values) < len(curr_target_list):
            new_values.extend([np.nan] * (len(curr_target_list) - len(new_values)))
        
        # Cut if too long
        if len(new_values) > len(curr_target_list):
            new_values = new_values[:len(curr_target_list)]
        
        # Forward fill list
        prediction = pd.Series(new_values).ffill().tolist()
        return prediction
    
    except:
        return None


def _calculate_mae(prediction, target, max_mae=20):
    
    try:
        prediction_values = np.asarray(prediction)
        target_values = np.asarray(target)

        # Check no nans
        assert np.isnan(prediction_values).sum() == 0
        assert np.isnan(target_values).sum() == 0

        # Calculate MAE
        mae = np.mean(np.abs(prediction_values - target_values))
        mae_clipped = np.clip(mae, 0, max_mae)
        return mae_clipped
    except:
        return max_mae


def _nr_of_correct_tags(completion):
    # Max nr of tags = 9
    
    
    # FYI: START_THOUGHT_TAG = "<thinking>", END_THOUGHT_TAG = "</thinking>"
    # FYI: START_PREDICTION_RANK_TOKEN = "<prediction>", END_PREDICTION_RANK_TOKEN = "</prediction>"
    correct_count = 0

    # Check that the completion contains a <thinking> tag
    if START_THOUGHT_TAG in completion:
        correct_count += 1

    # Check that the completion contains an </thinking> tag
    if END_THOUGHT_TAG in completion:
        correct_count += 1

    # Check that the completion contains a <prediction> tag
    if START_PREDICTION in completion:
        correct_count += 1

    # Check that the completion contains an </prediction> tag
    if END_PREDICTION in completion:
        correct_count += 1
    
    # First, remove empty tags, i.e. only whitespace between the tags, if e.g. model is repeating stuff
    completion = re.sub(r"<thinking>\s*</thinking>", "", completion)
    completion = re.sub(r"<prediction>\s*</prediction>", "", completion)
    
    # Check that the completion starts with a <thinking> tag
    if completion.strip().startswith(START_THOUGHT_TAG):
        correct_count += 1
    
    # Check that the thought tag is closed before the prediction tag
    thought_end_index = completion.find(END_THOUGHT_TAG)
    prediction_start_index = completion.find(START_PREDICTION)
    
    if thought_end_index != -1 and prediction_start_index != -1 and thought_end_index < prediction_start_index:
        correct_count += 1
    
    # Check that the completion contains a <prognosis_summary> tag
    if START_PROGNOSIS in completion:
        correct_count += 1
    
    # Check that the completion contains an </prognosis_summary> tag
    if END_PROGNOSIS in completion:
        correct_count += 1

    # Check that the completion ends with a </prediction> tag
    if completion.strip().endswith(END_PREDICTION):
        correct_count += 1
    
    return correct_count




def reward_function(completions, **kwargs):
    
    # Setup rewards
    rewards = [0] * len(completions)

    # For monitoring
    print_debug = np.random.rand(1)[0] < 0.10  # i.e. in 10% of cases print

    # Setup constants
    max_mae = 20    
    multiplier_for_mae = 30.0

    # Go through all completions
    for idx, raw_completion in enumerate(completions):
        
        # Get response
        completion = raw_completion[0]["content"]

        if print_debug:
            print("=====================================================") 
            print(f"Completion {idx}: {completion}")
            print("---------------")

        #: Keep reward 0, but measure to check how good it is
        nr_correct_tags = _nr_of_correct_tags(completion)
        
        #: make rewards for MAE
        curr_target = kwargs["target_list_of_values"][idx]
        total_reward_for_mae = 0
        mae = max_mae
        parsing_failed = 0
        prediction = _extract_df_of_values_of_neutrophils(completion, curr_target)
        if prediction is not None:
            mae = _calculate_mae(prediction, curr_target, max_mae)  # Clamped by max_mae
            # We want to flip it, since its better to add the rewards together (rather than subtract)
            mae_from_max = (max_mae - mae) / max_mae
            total_reward_for_mae = mae_from_max * multiplier_for_mae
        else:
            parsing_failed = 1
            mae = max_mae
            total_reward_for_mae = 0

        #: Sum it all up
        rewards[idx] = total_reward_for_mae        

        if print_debug:
            #: add in correct debug statements
            print(f"Target {idx}: {curr_target}")
            print(f"Prediction {idx}: {prediction}")
            print(f"Reward for completion {idx}: {rewards[idx]}")
            print("---------------")
            print(f"Reward for MAE: {total_reward_for_mae}")
            print(f"MAE: {mae}")
            print(f"Nr of correct tags: {nr_correct_tags}")
            print("=====================================================")

        # Save to wandb
        wandb.log({
            "reward_for_mae": total_reward_for_mae,
            "mae" : mae,
            "nr_correct_tags": nr_correct_tags,
            "total_reward": rewards[idx],
            "parsing_failed": parsing_failed,
        })

    # Save max and min rewards to wandb
    wandb.log({
        "max_reward_in_batch": max(rewards),
        "min_reward_in_batch": min(rewards),
    })
        
    # Return rewards
    return rewards


def main(num_train_epochs, learning_rate, weight_decay, lr_warm_up_ratio, batch_size, max_grad_norm, gradient_accumulation,
         num_generations, base_model, save_folder=SAVE_FOLDER, path_to_train_df=PATH_TO_TRAIN_DF, 
         path_to_validation_df=PATH_TO_VALIDATION_DF):


    #: setup basics
    split = "train"
    wandb.init(project="genie-dt-grpo-forecasting-paper", mode="offline" if DEBUG else "online", 
               group="genie-dt-grpo-pred-then-cot-norm-only-mae")


    #: set name and group correctly
    wandb.run.name = f"Genie DT GRPO - Pred then COT - with norm - only MAE reward - {num_train_epochs} epochs - {learning_rate} lr - {lr_warm_up_ratio} lr warmup ratio - {weight_decay} weight decay - {batch_size} batch size - {max_grad_norm} max grad norm - {gradient_accumulation} grad acc - {num_generations} generations"
    wandb.config.update({
        "split": split,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "model": "genie-dt-grpo-reward-both-format-mae",
        "weight_decay": weight_decay,
        "lr_warm_up_ratio": lr_warm_up_ratio,
        "batch_size": batch_size,
        "max_grad_norm": max_grad_norm,
        "gradient_accumulation": gradient_accumulation,
    }, allow_val_change=True)

    
    #: setup folders, based on datetime str
    curr_output_dir = save_folder + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/"
    os.makedirs(curr_output_dir, exist_ok=True)
    output_dir = curr_output_dir + "training_run/"
    model_save_folder = curr_output_dir + "saves/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_save_folder, exist_ok=True)


    #: load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto")
    
    #: load in training data
    train_df = pd.read_csv(path_to_train_df)
    validation_df = pd.read_csv(path_to_validation_df)
    print("Loaded in training and validation DFs")
    print(f"Training DF shape: {train_df.shape}, validation DF shape: {validation_df.shape}")
    

    # Parse the 'prompt' column from JSON strings to dictionaries if necessary
    def prep_df(df):

        df = df.copy()

        def add_input_in_format(row):
            input_str = row["input_prompt"]
            #: add system message and convert to dictionary
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": input_str}]
            return messages
        df['prompt'] = df.apply(add_input_in_format, axis=1)
        
        #: add "final_response" to the prompt as assistant message
        def add_final_response(row):
            # Add the final response as an assistant message
            messages = [{"role": "assistant", "content": row['post_processed_targets']}]
            return messages

        df['completion'] = df.apply(add_final_response, axis=1)
        

        # We need to keep the 'target_list_of_values' column for the reward function.
        # It's likely stored as a string, so we use ast.literal_eval to parse it.
        if 'target_list_of_values' in df.columns:
            # Check if it's actually a string before trying to parse
            if isinstance(df['target_list_of_values'].iloc[0], str):
                print("Parsing 'target_list_of_values' from string to list...")
                df['target_list_of_values'] = df['target_list_of_values'].apply(ast.literal_eval)
            
            # Keep the columns needed by GRPOTrainer and the reward function
            # See if we event need the column "completion"
            df = df[["prompt", "target_list_of_values"]]
        else:
            raise ValueError("Column 'target_list_of_values' not found in DataFrame. It is required for the reward function.")
        
        return df

    # Prepare the DataFrame
    train_df = prep_df(train_df)
    validation_df = prep_df(validation_df)

    # Print one example to check the format
    print("Example prompt data (system):")
    print(train_df.iloc[0]['prompt'][0]["content"])
    print("Example prompt data (user):")
    print(train_df.iloc[0]['prompt'][1]["content"])
    #print("Example reference completion (assistant):")
    #print(train_df.iloc[0]['completion'][0]['content'])
    print("Example target data (target_list_of_values):")
    print(train_df.iloc[0]['target_list_of_values'])


    #: setup Tokenizer
    print("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"  # Found in the tokenizer config - warning hacky, depends on LLM


    #: convert to dataset
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)

    final_eval_steps = round(EVAL_SAVE_STEPS / (batch_size * gradient_accumulation))


    # Define generation kwargs for vLLM
    generation_kwargs = {
        "top_p": 1.0,       # Setting to high value
        "temperature": 1.0, # Use 1.0 for sampling many different trajectories
    }

    train_params = GRPOConfig(  # Changed from SFTConfig
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,  # If this OOM then put into gradient accumulation steps
        gradient_accumulation_steps=gradient_accumulation,  # Effect batch size = per_device_train_batch_size * gradient_accumulation_steps *num_gpus
        gradient_checkpointing=False,
        eval_strategy="steps",
        eval_steps=final_eval_steps,
        save_strategy="steps",
        save_steps=final_eval_steps,                
        save_only_model=True,
        per_device_eval_batch_size=num_generations,
        optim="adamw_torch",
        logging_steps=1,
        learning_rate=learning_rate, 
        warmup_ratio=lr_warm_up_ratio,  
        bf16=True,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,        
        num_train_epochs=num_train_epochs,              
        save_total_limit=2,
        group_by_length=True,
        report_to="wandb",
        seed=42,
        load_best_model_at_end=True,        
        beta=0.0,                           
        loss_type="dapo",                   # Since improves over some GRPO issues     
        scale_rewards=True,                 
        log_completions=False,             
        num_generations=num_generations,    
        generation_batch_size=num_generations,      
        use_vllm=True,              
        vllm_mode="colocate",       
        generation_kwargs=generation_kwargs,
        epsilon_high=0.28,                  # From DAPO paper
        max_prompt_length=None,             # Allow any length
        max_completion_length=2048,         
    )

    #: setup trainer
    trainer = GRPOTrainer(
            model=model,
            args=train_params,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=tokenizer,
            reward_funcs=reward_function, # Pass the reward function
    )
    print("Trainer setup complete, starting training")


    #: run GRPO training
    trainer.train()

    # Final eval
    eval_results = trainer.evaluate()
    print(f"Evaluation results at the end of training: {eval_results}")
    wandb.log(eval_results)

    #: save model and tokenizer
    model.save_pretrained(model_save_folder)
    tokenizer.save_pretrained(model_save_folder)

    #: finish wandb
    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run GRPO and evaluation for Genie DT")

    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training (GRPO default 1e-6)")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for AdamW optimizer")
    parser.add_argument("--lr_warm_up_ratio", type=float, default=0.1, help="Learning rate warm-up ratio")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--max_grad_norm", type=float, default=0.2, help="Maximum gradient norm for clipping")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt during training")
    parser.add_argument("--base_model", type=str, default=MODEL_NAME, help="Base model to use")
    parser.add_argument("--save_folder", type=str, default=SAVE_FOLDER, help="Folder to save the model outputs")
    parser.add_argument("--path_to_train_df", type=str, default=PATH_TO_TRAIN_DF, help="Path to the training data CSV")
    parser.add_argument("--path_to_validation_df", type=str, default=PATH_TO_VALIDATION_DF, help="Path to the validation data CSV")
    args = parser.parse_args()

    main(args.num_train_epochs, args.learning_rate, args.weight_decay,
         args.lr_warm_up_ratio, args.batch_size, args.max_grad_norm, args.gradient_accumulation,
            args.num_generations, args.base_model, args.save_folder, args.path_to_train_df, args.path_to_validation_df)



