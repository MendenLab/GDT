import pandas as pd
import numpy as np
import argparse
import re


# Sometimes there might be errors in the reasoning chains so we replace the model's
# predicted target with the true target while keeping the reasoning intact.

OPENING_TAG = "<prediction>"
CLOSING_TAG = "</prediction>"


def main(load_path, save_path):

    df = pd.read_csv(load_path)

    def replace_prediction_with_true(row):
        reasoning_with_target = row["final_response"]
        # Ensure true_target is a string, in case it's numeric or other types
        true_target = str(row["target_prompt"])

        # Find the indices of the opening and closing tags
        start_index = reasoning_with_target.find(OPENING_TAG)
        end_index = reasoning_with_target.find(CLOSING_TAG)

        # Check if either tag is missing, or if they are in the wrong order
        if start_index == -1 or end_index == -1 or end_index < start_index:
            return np.nan  # Return NaN to mark this row for dropping

        # Construct the new string:
        # 1. Get the part before and including the opening tag
        part_before = reasoning_with_target[:start_index + len(OPENING_TAG)]
        
        # 2. Get the part from the closing tag to the end
        part_after = reasoning_with_target[end_index:]

        # 3. Concatenate them with the true target in the middle
        concat = part_before + true_target + part_after
        return concat
    
            
    df["post_processed_targets"] = df.apply(replace_prediction_with_true, axis=1)

    
    # Fulfill the "drop the sample" requirement by dropping rows where
    # the tags were missing (and thus post_processed_targets is NaN)
    df.dropna(subset=["post_processed_targets"], inplace=True)
    
    # Save the processed dataset
    df.to_csv(save_path, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SFT and evaluation for Genie DT")
    parser.add_argument("--load_path", type=str, default="genie-dt-grpo-forecasting/0_data/sft_data/train_Qwen-Qwen3-Next-80B-A3B-Instruct_num_samples_2385_sft_prompts.csv", help="Path to the test targets file for evaluation")
    parser.add_argument("--save_path", type=str, default="genie-dt-grpo-forecasting/0_data/sft_data/post_processed/post_processed_train_Qwen-Qwen3-Next-80B-A3B-Instruct_num_samples_2385_sft_prompts.csv", help="Path to save the post-processed dataset")
    args = parser.parse_args()

    main(args.load_path, args.save_path)