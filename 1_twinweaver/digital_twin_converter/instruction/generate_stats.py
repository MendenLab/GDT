"""
Generate stats which might be interesting for later analysis.
"""
import os
import glob
import json
import pandas as pd
import argparse



def main(path_to_data_folder, save_path):

    # : find all files with "_instruction_stats.json" ending
    pattern = os.path.join(path_to_data_folder, "*_instruction_stats.json")
    files = glob.glob(pattern)

    # : load the json
    all_data = []
    for file in files:
        with open(file) as f:
            data = json.load(f)
            all_data.append(data)

    # : get stats
    final_stats = {}

    # : total number of samples
    total_samples = sum([data["num_total_samples"] for data in all_data])

    # : sum of which tasks
    num_events = sum([data["num_tasks_events"] for data in all_data])
    num_forecasting = sum([data["num_tasks_forecasting"] for data in all_data])
    num_forecasting_qa = sum([data["num_tasks_forecasting_qa"] for data in all_data])
    num_forecasting_or_qa = sum([data["num_tasks_forecasting_OR_qa"] for data in all_data])

    # : nr of patients
    num_patients = sum([data["num_patients"] for data in all_data])
    num_patients_no_split = sum([data["num_patients_no_splits"] for data in all_data])

    # : overall distribution of tasks
    num_events_only = 0
    num_forecasting_only = 0
    num_forecasting_qa_only = 0
    num_events_forecasting = 0
    num_events_forecasting_qa = 0
    num_forecasting_forecasting_qa = 0
    num_all = 0
    for data in all_data:
        for curr_nr_forecasting, curr_nr_forecasting_qa, curr_nr_events in data["sample_task_distribtuion"]:
            if curr_nr_forecasting > 0 and curr_nr_forecasting_qa == 0 and curr_nr_events == 0:
                num_forecasting_only += 1
            elif curr_nr_forecasting == 0 and curr_nr_forecasting_qa > 0 and curr_nr_events == 0:
                num_forecasting_qa_only += 1
            elif curr_nr_forecasting == 0 and curr_nr_forecasting_qa == 0 and curr_nr_events > 0:
                num_events_only += 1
            elif curr_nr_forecasting > 0 and curr_nr_forecasting_qa == 0 and curr_nr_events > 0:
                num_events_forecasting += 1
            elif curr_nr_forecasting == 0 and curr_nr_forecasting_qa > 0 and curr_nr_events > 0:
                num_events_forecasting_qa += 1
            elif curr_nr_forecasting > 0 and curr_nr_forecasting_qa > 0 and curr_nr_events == 0:
                num_forecasting_forecasting_qa += 1
            elif curr_nr_forecasting > 0 and curr_nr_forecasting_qa > 0 and curr_nr_events > 0:
                num_all += 1

    # : sum of nr instruction tokens
    nr_tokens_instructions = sum([sum(map(int, data["num_tokens_input"])) for data in all_data])

    # Sum of task tokens
    nr_tokens_tasks = sum([sum(map(int, data["num_tokens_target"])) for data in all_data])

    # Sum of total number of tokens
    nr_tokens_total = sum([sum(map(int, data["num_tokens_total"])) for data in all_data])

    # : avg/min/max of tokens
    avg_tokens_instructions = nr_tokens_instructions / total_samples
    avg_tokens_tasks = nr_tokens_tasks / total_samples
    avg_tokens_total = nr_tokens_total / total_samples

    min_tokens_instructions = min(min([data["num_tokens_input"] for data in all_data]))
    min_tokens_tasks = min(min([data["num_tokens_target"] for data in all_data]))
    min_tokens_total = min(min([data["num_tokens_total"] for data in all_data]))

    max_tokens_instructions = max(max([data["num_tokens_input"] for data in all_data]))
    max_tokens_tasks = max(max([data["num_tokens_target"] for data in all_data]))
    max_tokens_total = max(max([data["num_tokens_total"] for data in all_data]))

    # Count how often total goes above 8192 tokens
    total_above_8192 = sum([1 for data in all_data if max(data["num_tokens_total"]) > 8192])
    total_above_7692 = sum([1 for data in all_data if max(data["num_tokens_total"]) > 7692])

    # : make dataframe with two columns: "Property" and "value"
    final_stats["total_samples"] = total_samples
    final_stats["num_events"] = num_events
    final_stats["num_forecasting"] = num_forecasting
    final_stats["num_forecasting_qa"] = num_forecasting_qa
    final_stats["num_forecasting_or_qa"] = num_forecasting_or_qa
    final_stats["num_patients"] = num_patients
    final_stats["num_patients_no_split"] = num_patients_no_split
    final_stats["task_split_num_forecasting_only"] = num_forecasting_only
    final_stats["task_split_num_forecasting_qa_only"] = num_forecasting_qa_only
    final_stats["task_split_num_events_only"] = num_events_only
    final_stats["task_split_num_events_forecasting"] = num_events_forecasting
    final_stats["task_split_num_events_forecasting_qa"] = num_events_forecasting_qa
    final_stats["task_split_num_forecasting_forecasting_qa"] = num_forecasting_forecasting_qa
    final_stats["task_split_num_all"] = num_all
    final_stats["nr_tokens_instructions"] = nr_tokens_instructions
    final_stats["nr_tokens_tasks"] = nr_tokens_tasks
    final_stats["nr_tokens_total"] = nr_tokens_total
    final_stats["avg_tokens_instructions"] = avg_tokens_instructions
    final_stats["avg_tokens_tasks"] = avg_tokens_tasks
    final_stats["avg_tokens_total"] = avg_tokens_total
    final_stats["min_tokens_instructions"] = min_tokens_instructions
    final_stats["min_tokens_tasks"] = min_tokens_tasks
    final_stats["min_tokens_total"] = min_tokens_total
    final_stats["max_tokens_instructions"] = max_tokens_instructions
    final_stats["max_tokens_tasks"] = max_tokens_tasks
    final_stats["max_tokens_total"] = max_tokens_total
    final_stats["total_samples_above_8192_tokens"] = total_above_8192
    final_stats["total_samples_above_7692_tokens"] = total_above_7692

    df = pd.DataFrame(list(final_stats.items()), columns=["Property", "Value"])
    df.to_csv(save_path, index=False)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Save statistics to a CSV file.')
    parser.add_argument('path_to_data_folder', type=str, nargs='?',
                        default='/flatiron_cgdb/instruction/combined/2024_08_08',
                        help='Path to the folder containing the instruction stats JSON files.')
    default_save = ('instruction_stats.csv')
    parser.add_argument('save_path', type=str, nargs='?',
                        default=default_save,
                        help='Path to save the aggregated statistics CSV file.')

    args = parser.parse_args()

    main(args.path_to_data_folder, args.save_path)
