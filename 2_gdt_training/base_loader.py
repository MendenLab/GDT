
import os
from time import time
from datasets import load_dataset
from datasets import load_from_disk


# Note, due to the way llama-recipes was loaded, we needed to hard the values here

# Needed to fix tokenizer chat template for Llama 3 back when it had some issues
LLAMA3_CHAT_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>

'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>

' }}{% endif %}"""  # noqa: E501


LLAMA3_UTTERANCE_TEMPLATE = "{% set loop_messages = messages %}" \
                            "{% for message in loop_messages %}" \
                            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+" \
                            " message['content'] | trim + '<|eot_id|>' %}" \
                            "{{ content }}" \
                            "{% endfor %}"


# System prompt
FLATIRON_SYSTEM_PROMPT = ("As a specialist predictive model in personalized medicine,"
                          " your task is to forecast the health trajectory of cancer "
                          "patients by integrating genomic data, lifestyle factors, "
                          "treatment history and anything else provided about the patient. "
                          "Use the provided patient data, including genetic mutations, "
                          "biomarker levels, and previous treatment responses, to predict"
                          " all requested tasks. Deliver precise and clinically relevant "
                          "predictions to enhance patient care and treatment planning.")



# Configuration constants
MAX_NUM_FILES = 1_000_000_000

SAVE_FOLDER = "./fine_tuning_experiments/data_cache/v1"

MAX_LENGTH = 8000
CACHE_FOLDER = "./fine_tuning_experiments/data_cache/huggingface_cache/"
BATCH_SIZE = 1000
NUM_PROC = 16



# Override validation and test sets to use the 1 lot per patient splits for consistency
VALID_SET_OVERRIDE = "./flatiron_cgdb/instruction_local/combined/2024_11_14_processed_1_lots_per_patient"  
TEST_SET_OVERRIDE = "/flatiron_cgdb/instruction_local/combined/2024_11_14_processed_1_lots_per_patient"




def find_sublist_index(main_list, sublist):
    sublist_len = len(sublist)
    for i in range(len(main_list) - sublist_len + 1):
        if main_list[i:i + sublist_len] == sublist:
            return i
    print("Didn't find the sublist in the main list")
    return -1


def process_batch(batch, tokenizer, assistant_start, EOT_token, FLATIRON_SYSTEM_PROMPT):
    # Generate prompts for the entire batch
    prompts = [
        [
            {"role": "system", "content": FLATIRON_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": answer},
        ]
        for instruction, answer in zip(batch["instruction"], batch["answer"])
    ]

    # Apply chat template & tokenize the entire batch
    tokens_list = tokenizer.apply_chat_template(
        prompts, tokenize=True, chat_template=LLAMA3_UTTERANCE_TEMPLATE
    )

    # Prepare inputs and targets for the entire batch
    input_ids_list = [[tokenizer.bos_token_id] + tokens for tokens in tokens_list]

    # Limit inputs to MAX_LENGTH
    input_ids_list = [input_ids[:MAX_LENGTH] for input_ids in input_ids_list]

    # Prepare targets for the entire batch
    target_ids_list = [input_ids.copy() for input_ids in input_ids_list]

    # Find assistant start indices for the entire batch
    assistant_start_indices = [
        find_sublist_index(target_ids, assistant_start) for target_ids in target_ids_list
    ]

    processed_samples = {
        "input_ids": [],
        "labels": [],
        "attention_mask": []
    }

    for idx, (input_ids, target_ids, assistant_start_index) in enumerate(
        zip(input_ids_list, target_ids_list, assistant_start_indices)
    ):
        if assistant_start_index == -1:
            print(f"Assistant header not found in the tokens for index {idx}")
            # Skip examples where the assistant header is not found
            continue

        # Mask out everything before the assistant header
        if assistant_start_index > 0:
            labels = [-100] * assistant_start_index + target_ids[assistant_start_index:]
        else:
            # In case no assistant header is found, mask out everything
            print("Assistant header not found in the tokens")
            labels = [-100] * len(target_ids)

        # Perform checks on the first example
        if idx == 0 and len(labels) > 0:
            assert len(input_ids) <= MAX_LENGTH*1.05, "Input length exceeds MAX_LENGTH significantly"
            assert input_ids[0] == tokenizer.bos_token_id, "BOS token mismatch"
            assert len(input_ids) == len(labels), "Input and target length mismatch"
            detokenized_output = tokenizer.decode([l for l in labels if l != -100],
                                    skip_special_tokens=True)[len("assistant\n\n"):]
            if len(input_ids) <= MAX_LENGTH - 1:
                # We allow those which we cut off to not have EOT/check full answer
                # so that during inference it doesn't cut off to early
                # when trying out very long seqs
                assert input_ids[-1] == target_ids[-1] == EOT_token, "EOT token mismatch: " + str(input_ids[-1])
                assert detokenized_output == batch["answer"][0].strip(), "Answer mismatch" + detokenized_output + " vs " + batch["answer"][0].strip()
            else:
                assert detokenized_output in batch["answer"][0].strip(), "Answer mismatch" + detokenized_output + " vs " + batch["answer"][0].strip()
            assert all(
                inp == tgt
                for inp, tgt in zip(input_ids, labels) if tgt != -100
            ), "Input and target mismatch: " + str(input_ids) + " vs " + str(labels)
            
        # Prepare the sample
        processed_samples["input_ids"].append(input_ids)
        processed_samples["labels"].append(labels)
        processed_samples["attention_mask"].append([1] * len(input_ids))

    return processed_samples


def get_custom_dataset_base(dataset_config, tokenizer, split):
    """Process the dataset for a given split ('train', 'validation', 'test')."""

    print("Processing dataset:", split)
    assert tokenizer.chat_template == LLAMA3_CHAT_TEMPLATE, "This datapipe only works with LLAMA3 tokenizer."
    num_splits_per_patient = dataset_config.splits_per_patient_therapy
    
    # Dataset name and caching location
    dataset_name = f"fh_cgdb_instruction_v1_2024_11_28_{num_splits_per_patient}_splits_{split}_{MAX_NUM_FILES}_{MAX_LENGTH}"
    dataset_location = os.path.join(SAVE_FOLDER, dataset_name)
    if os.path.exists(dataset_location):
        print("Dataset already exists, loading from cache:", dataset_location)
        curr_dataset = load_from_disk(dataset_location, keep_in_memory=False)
        return curr_dataset

    # Directory containing the JSONL files
    if split == 'validation':
        data_dir = VALID_SET_OVERRIDE
    elif split == 'test':
        data_dir = TEST_SET_OVERRIDE
    else:
        data_dir = f"/flatiron_cgdb/instruction_local/combined/2024_11_14_processed_{num_splits_per_patient}_lots_per_patient"
    print("Data directory: ", data_dir)

    # Adjust split name if needed
    split_dir = os.path.join(data_dir, split)

    # Get list of JSONL files in the directory
    data_files = [os.path.join(split_dir, filename)
                  for filename in os.listdir(split_dir) if filename.endswith('.jsonl')]
    data_files = data_files[:MAX_NUM_FILES]

    if not data_files:
        print(f"No JSONL files found for split '{split}' in directory '{split_dir}'")
        return None

    # Load dataset
    dataset = load_dataset('json', data_files=data_files, split='train',
                           cache_dir=CACHE_FOLDER)

    # Precompute tokens
    assistant_start_tokens = ["<|start_header_id|>", "assistant", "<|end_header_id|>"]
    assistant_start = tokenizer.convert_tokens_to_ids(assistant_start_tokens)
    EOT_token = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    # Define the batch processing function with partial application
    def batch_process(batch):
        return process_batch(batch, tokenizer, assistant_start, EOT_token, FLATIRON_SYSTEM_PROMPT)

    # Apply the map function with batching and parallel processing
    start_time = time()
    processed_dataset = dataset.map(
        batch_process,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=dataset.column_names,  # Remove original columns if not needed
        desc=f"Processing {split} split",
    )
    print(f"Mapping completed in {time() - start_time:.2f} seconds.")

    # Remove examples where input_ids or labels are empty
    processed_dataset = processed_dataset.filter(
        lambda x: len(x['input_ids']) > 0 and len(x['labels']) > 0,
        desc="Filtering empty examples",
        num_proc=NUM_PROC,
    )

    # Save the processed dataset to disk
    processed_dataset.save_to_disk(dataset_location)
    print("Dataset saved to:", dataset_location)
    print("Number of samples in final dataset:", len(processed_dataset))
    
    return processed_dataset


