import pandas as pd
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


SYSTEM_PROMPT = ("As a specialist predictive model in personalized medicine,"
                " your task is to forecast the health trajectory of cancer "
                "patients by integrating genomic data, lifestyle factors, "
                "treatment history and anything else provided about the patient. "
                "Use the provided patient data, including genetic mutations, "
                "biomarker levels, and previous treatment responses, to predict"
                " all requested tasks. Deliver precise and clinically relevant "
                "predictions to enhance patient care and treatment planning.")


DEBUG = False

# Core parameters
EVALUATION_TIMELINES = [8, 26, 52, 104] 


# Training parameters
# Update these paths to your local model path
MODEL_NAME = "chkpt/llama_3_1_8b_10x_280k_release_1"
TOKENIZER = "chkpt/llama_3_1_8b_10x_280k_release_1"
MAX_LEN = 8192
SAVE_FOLDER = "genie-dt-cit-eval-events/0_data/genie_dt_correct_sft/runs/"
PATH_TO_TRAIN_DF = "genie-dt-cit-eval-events/0_data/train_text/text_table_cit_train_num_samples_per_lot_1.csv"
PATH_TO_VALIDATION_DF = "genie-dt-cit-eval-events/0_data/validation_text/text_table_cit_validation_num_samples_per_lot_1.csv"
EVAL_SAVE_STEPS = 500
VALIDATION_SUBSAMPLE_FRACTION = 0.3  # Use 30% of validation data for quicker evaluation during training
LR_WARM_UP_RATIO = 0.1






def main(num_train_epochs, learning_rate,
         weight_decay, lr_warm_up_ratio, batch_size, max_grad_norm,
         gradient_accumulation, model_name=MODEL_NAME, tokenizer_path=TOKENIZER,
         save_folder=SAVE_FOLDER, path_to_train_df=PATH_TO_TRAIN_DF, 
         path_to_validation_df=PATH_TO_VALIDATION_DF):


    #: setup basics
    indication = "cit"
    split = "train"
    wandb.init(project="genie-dt-cit-events-landmark", mode="offline" if DEBUG else "online", 
                group="genie-dt-sft-corrected")


    #: set name and group correctly
    wandb.run.name = f"Genie DT - Train - samplers per var - {indication} - {num_train_epochs} epochs - {learning_rate} lr - {lr_warm_up_ratio} lr warmup ratio - {VALIDATION_SUBSAMPLE_FRACTION} validation_subsample_fraction - {weight_decay} weight decay - {batch_size} batch size - {max_grad_norm} max grad norm - {gradient_accumulation} grad acc"
    wandb.config.update({
        "split": split,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "model": "genie-dt-sft",
        "indication": indication,
        "validation_subsample_fraction": VALIDATION_SUBSAMPLE_FRACTION,
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
    
    #: get correct weeks
    def subsample_df(df):
        df = df[df["week_to_predict"].isin(EVALUATION_TIMELINES)]
        return df
    
    train_df = subsample_df(train_df)
    validation_df = subsample_df(validation_df)

    # Then subsample validation
    validation_df = validation_df.sample(frac=VALIDATION_SUBSAMPLE_FRACTION, random_state=42)

    assert all([week in train_df["week_to_predict"].unique() for week in EVALUATION_TIMELINES]), "Not all evaluation timelines are present in the training data"
    assert all([week in validation_df["week_to_predict"].unique() for week in EVALUATION_TIMELINES]), "Not all evaluation timelines are present in the validation data"
    print(f"After sampling, training DF shape: {train_df.shape}, validation DF shape: {validation_df.shape}")


    print(f"Number of training samples after subsampling: {len(train_df)}")
    print(f"Number of validation samples after subsampling: {len(validation_df)}")

    # Parse the 'prompt' column from JSON strings to dictionaries if necessary
    def prep_df(df):

        df = df.copy()

        def add_input_in_format(row):
            input_str = row["instruction"]
            #: add system message and convert to dictionary
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": input_str}]
            return messages
        df['prompt'] = df.apply(add_input_in_format, axis=1)
        
        #: add "final_response" to the prompt as assistant message
        def add_final_response(row):
            # Add the final response as an assistant message
            messages = [{"role": "assistant", "content": row['answer']}]
            return messages
        df['completion'] = df.apply(add_final_response, axis=1)
        df = df[["prompt", "completion"]]
        return df

    # Prepare the DataFrame
    train_df = prep_df(train_df)
    validation_df = prep_df(validation_df)

    # Print one example to check the format
    print("Example training data:")
    print(train_df.iloc[0]['prompt'][1]["content"])
    print("Example final response:")
    print(train_df.iloc[0]['completion'][0]['content'])


    #: setup SFT
    print("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"  # Found in the tokenizer config - warning hacky, depends on LLM
    tokenizer.model_max_length = MAX_LEN

    #: convert to dataset (using chat template from llama3.1 8b), 
    # make sure to use DataCollatorForCompletionOnlyLM, with the appropriate response template, so only the response is used for loss calculation
    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)

    final_eval_steps = round(EVAL_SAVE_STEPS / (batch_size * gradient_accumulation))

    #: setup SFTconfig
    train_params = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,   
        gradient_accumulation_steps=gradient_accumulation,   
        gradient_checkpointing=False,
        eval_strategy="steps",
        eval_steps=final_eval_steps,
        save_strategy="steps",
        save_steps=final_eval_steps,             
        optim="adamw_torch",
        logging_steps=1,
        learning_rate=learning_rate, 
        warmup_ratio=LR_WARM_UP_RATIO,   
        bf16=True,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        num_train_epochs=num_train_epochs,
        save_total_limit=2,
        group_by_length=True,
        report_to="wandb",
        seed=42,
        max_length=MAX_LEN,            
        packing=False,           
        load_best_model_at_end=True,
        completion_only_loss=True,     
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
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Path to the base model")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER, help="Path to the tokenizer")
    parser.add_argument("--save_folder", type=str, default=SAVE_FOLDER, help="Folder to save the model outputs")
    parser.add_argument("--path_to_train_df", type=str, default=PATH_TO_TRAIN_DF, help="Path to the training data CSV")
    parser.add_argument("--path_to_validation_df", type=str, default=PATH_TO_VALIDATION_DF, help="Path to the validation data CSV")
    args = parser.parse_args()

    main(args.num_train_epochs, args.learning_rate,
         args.weight_decay, args.lr_warm_up_ratio, args.batch_size, args.max_grad_norm, args.gradient_accumulation,
         args.model_name, args.tokenizer_path, args.save_folder, args.path_to_train_df, args.path_to_validation_df)

