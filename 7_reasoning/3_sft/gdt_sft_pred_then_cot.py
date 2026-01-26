import pandas as pd
import numpy as np
import wandb
import argparse
from datasets import Dataset
from trl import (
    SFTTrainer,
    SFTConfig,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datetime
import os


from utils import SYSTEM_PROMPT



DEBUG = False


# Training parameters
# Update these paths to your local model path
MODEL_NAME = "chkpt/llama_3_1_8b_10x_280k_release_1"
TOKENIZER = "chkpt/llama_3_1_8b_10x_280k_release_1"
SAVE_FOLDER = "genie-dt-grpo-forecasting/0_data/6_sft_pred_then_cot/runs/"
PATH_TO_TRAIN_DF = "genie-dt-grpo-forecasting/0_data/sft_data/cot_after_target/train_cot_after_target_Qwen-Qwen3-Next-80B-A3B-Instruct_num_samples_2385.csv"
PATH_TO_VALIDATION_DF = "genie-dt-grpo-forecasting/0_data/sft_data/cot_after_target/validation_cot_after_target_Qwen-Qwen3-Next-80B-A3B-Instruct_num_samples_235.csv"

EVAL_SAVE_STEPS = 500
EVAL_BATCH_SIZE = 1






def main(num_train_epochs, learning_rate, weight_decay, lr_warm_up_ratio, batch_size, max_grad_norm, gradient_accumulation,
         cosine_scheduler=False, constant_scheduler=False, model_name=MODEL_NAME, tokenizer_path=TOKENIZER,
         save_folder=SAVE_FOLDER, path_to_train_df=PATH_TO_TRAIN_DF, path_to_validation_df=PATH_TO_VALIDATION_DF):


    #: setup basics
    split = "train"
    wandb.init(project="genie-dt-grpo-forecasting-paper", mode="offline" if DEBUG else "online", 
                group="genie-dt-sft-pred-then-cot")


    #: set name and group correctly
    run_name = f"Genie DT SFT - Pred then COT - {num_train_epochs} epochs - {learning_rate} lr - {lr_warm_up_ratio} lr warmup ratio - {weight_decay} weight decay - {batch_size} batch size - {max_grad_norm} max grad norm - {gradient_accumulation} grad acc"
    if cosine_scheduler:
        run_name += " - cosine scheduler"
    elif constant_scheduler:
        run_name += " - constant scheduler"
    wandb.run.name = run_name
    wandb.config.update({
        "split": split,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "model": "genie-dt-sft",
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
        df = df[["prompt", "completion"]]
        return df

    # Prepare the DataFrame
    train_df = prep_df(train_df)
    validation_df = prep_df(validation_df)

    # Print one example to check the format
    print("Example prompt data:")
    print(train_df.iloc[0]['prompt'][0]["content"])
    print("Example training data:")
    print(train_df.iloc[0]['prompt'][1]["content"])
    print("Example final response:")
    print(train_df.iloc[0]['completion'][0]['content'])


    #: setup SFT
    print("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"  # Found in the tokenizer config - warning hacky, depends on LLM
    # tokenizer.model_max_length = MAX_LEN  # Don't need it

    #: convert to dataset (using chat template from llama3.1 8b), 
    # make sure to use DataCollatorForCompletionOnlyLM, with the appropriate response template, so only the response is used for loss calculation
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)

    final_eval_steps = round(EVAL_SAVE_STEPS / (batch_size * gradient_accumulation))

    #: setup SFTconfig
    train_params = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,   # If this OOM then put into gradient accumulation steps
        gradient_accumulation_steps=gradient_accumulation,   # Effect batch size = per_device_train_batch_size * gradient_accumulation_steps *num_gpus
        gradient_checkpointing=False,
        eval_strategy="steps",
        eval_steps=final_eval_steps,
        save_strategy="steps",
        eval_on_start=True,
        save_steps=final_eval_steps,        
        per_device_eval_batch_size=EVAL_BATCH_SIZE,     
        optim="adamw_torch",
        logging_steps=1,
        learning_rate=learning_rate,   # 1e-5 is good for 7B
        warmup_ratio=lr_warm_up_ratio,   
        lr_scheduler_type="cosine" if cosine_scheduler else ("constant_with_warmup" if constant_scheduler else "linear"),
        bf16=True,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        num_train_epochs=num_train_epochs,
        save_total_limit=2,
        group_by_length=True,
        report_to="wandb",
        seed=42,
        save_only_model=True,   
        max_length=None,             # Including any length
        packing=False,           
        load_best_model_at_end=True,
        completion_only_loss=True,      # Important for instruction tuning
    )


    #: load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    

    #: setup trainer
    trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            args=train_params,
            eval_dataset=validation_dataset,
    )
    print("Trainer setup complete, starting training")


    #: run SFT
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

    parser = argparse.ArgumentParser(description="Run SFT and evaluation for Genie DT")

    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for AdamW optimizer")
    parser.add_argument("--lr_warm_up_ratio", type=float, default=0.1, help="Learning rate warm-up ratio")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--cosine_scheduler", action='store_true', help="Whether to use cosine learning rate scheduler")
    parser.add_argument("--constant_scheduler", action='store_true', help="Whether to use constant learning rate scheduler")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Path to the base model")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER, help="Path to the tokenizer")
    parser.add_argument("--save_folder", type=str, default=SAVE_FOLDER, help="Folder to save the model outputs")
    parser.add_argument("--path_to_train_df", type=str, default=PATH_TO_TRAIN_DF, help="Path to the training data CSV")
    parser.add_argument("--path_to_validation_df", type=str, default=PATH_TO_VALIDATION_DF, help="Path to the validation data CSV")
    
    args = parser.parse_args()

    main(args.num_train_epochs, args.learning_rate, args.weight_decay,
         args.lr_warm_up_ratio, args.batch_size, args.max_grad_norm, args.gradient_accumulation,
            args.cosine_scheduler, args.constant_scheduler, args.model_name, args.tokenizer_path,
            args.save_folder, args.path_to_train_df, args.path_to_validation_df)



