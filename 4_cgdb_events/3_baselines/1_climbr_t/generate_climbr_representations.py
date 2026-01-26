import wandb
import os
import pandas as pd
import json
import femr.models.transformer
import pyarrow.csv
import datasets
import numpy as np
import shutil


DEBUG = False

all_indications = [
    'enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
    'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
    'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
    'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
    'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma'
]



def main(path_to_climbr_input_data_folder, save_folder, split):
    
    # Setup wandb
    wandb.init(project="genie-dt-cgdb-baselines-events", mode="offline" if DEBUG else "online")
    wandb.config.update({
        "path_to_climbr_input_data_folder": path_to_climbr_input_data_folder,
        "save_folder": save_folder,
        "split": split,
    })
    wandb.run.name = f"Generating representations for CLIMBR-T - {split}"


    # First delete all files in the path to save, if they exist
    if os.path.exists(save_folder):
        print(f"Previous files exist in {save_folder}. Deleting them.")
        shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)


    for indication in all_indications:

        print("=" * 50)
        print(f"Processing indication: {indication}")
        print("=" * 50)

        #: load in the input, the labels and the mapping
        path_to_input_patient_data = os.path.join(path_to_climbr_input_data_folder, indication + "_patients.parquet")
        path_to_labels = os.path.join(path_to_climbr_input_data_folder, indication + "_labels.csv")
        path_to_mapping = os.path.join(path_to_climbr_input_data_folder, indication + "_gdt_patientid_to_climbr_t_patientid_map.json")

        with open(path_to_mapping, 'r') as f:
            mapping = json.load(f)

        # Load some labels (need to handle datetime explicitly)
        date_columns = ['prediction_time']
        column_types = {col: pyarrow.timestamp('s') for col in date_columns}
        convert_options = pyarrow.csv.ConvertOptions(column_types=column_types)
        labels = pyarrow.csv.read_csv(path_to_labels, convert_options=convert_options).to_pylist()

        # Load our data
        dataset = datasets.Dataset.from_parquet(path_to_input_patient_data)

        #: get representations for the input data
        features = femr.models.transformer.compute_features(dataset,'StanfordShahLab/clmbr-t-base', 
                                                            labels, num_proc=4, tokens_per_batch=128)

        #: save the representations, and mapping to save folder
        save_path_rep = os.path.join(save_folder, f"{indication}_representations.npz")
        save_path_mapping = os.path.join(save_folder, f"{indication}_gdt_patientid_to_climbr_t_patientid_map.json")
        
        np.savez_compressed(save_path_rep, **features)
        
        with open(save_path_mapping, 'w') as f:
            json.dump(mapping, f)

        print(f"Saved representations to {save_path_rep}")
        print(f"Saved mapping to {save_path_mapping}")

    
    # Finalize wandb run
    wandb.finish()





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate data for forecasting")
    # Add defaults
    parser.add_argument("--path_to_climbr_input_data_folder", type=str, default="genie-dt-cgdb-eval-events/0_data/climbr_t/train/",
                        help="Path to the folder with CLIMBR-T input data")
    parser.add_argument("--save_folder", type=str, default="genie-dt-cgdb-eval-events/0_data/climbr_t/representations/small_tests",
                        help="Path to the folder where to save the representations")
    parser.add_argument("--split", type=str, default="train",
                        help="Split to process (train, val, test)")
    args = parser.parse_args()
    main(args.path_to_climbr_input_data_folder, args.save_folder, args.split)










