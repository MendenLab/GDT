from time import time, strftime
from transformers import AutoTokenizer
import wandb  # Ensure wandb is installed if you plan to use it



from base_loader import get_custom_dataset_base



def get_custom_dataset(dataset_config, tokenizer, split):
    return get_custom_dataset_base(dataset_config, tokenizer, split)



if __name__ == "__main__":

    # Initialize wandb if needed
    wandb.init(project="Genie-DT-Finetuning-V1", group="Data-Processing")
    wandb.run.name = "Data Processing from JSONL - " + strftime("%Y-%m-%d %H:%M:%S")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct',
        use_fast=True,
    )

    class DatasetConfig:
        splits_per_patient_therapy = 10

    dataset_config = DatasetConfig()

    # Process each dataset split
    for split in ['train', 'validation', 'test']:
        get_custom_dataset(dataset_config, tokenizer, split)  # Adjust num_proc based on your CPU cores

    wandb.finish()
