import wandb
import os
import pandas as pd
import json
import numpy as np
import argparse
import sys
def setup_imports_nb():
    try:
        # In case you run this cell from a .py file later via an IDE that defines __file__
        notebook_parent_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ is not defined in a standard Jupyter notebook cell, so use os.getcwd()
        # os.getcwd() usually points to the directory where the notebook file is located,
        # or where the Jupyter server was started.
        # Be sure this is the correct directory for your structure.
        notebook_parent_dir = os.getcwd()

    project_root_dir = os.path.abspath(os.path.join(notebook_parent_dir, "../../2_eval_tools/"))
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
setup_imports_nb()
from utils_landmark_eval import LandmarkEventsEval



DEBUG = False

all_indications = ["cit"]
all_variables = ["death", "progression"]   # All variables in CIT
all_timelines = [8, 26, 52, 104]





def evaluate(majority_class, load_folder_splits_test):
    
    #: evaluate model on each indication
    for indication in all_indications:
        
        # Setup wandb
        wandb.init(project="genie-dt-cit-events-landmark-probability", mode="offline" if DEBUG else "online",
                group="majority_class")
        wandb.config.update({
            "indication": indication,
            "load_folder_splits_test": load_folder_splits_test,
        })
        wandb.run.name = f"Majority Class Eval Test - Indication: {indication}"

        print(f"Evaluating models for indication: {indication}")

        #: grab the test labels, by loading all DFs and concat
        split = "test"
        test_labels = pd.read_csv(os.path.join(load_folder_splits_test, f"targets_{split}_{indication}.csv"))
        test_labels["occurred"] = pd.NA
        test_labels["censored"] = pd.NA
        test_labels["true_censoring"] = pd.NA
        test_labels["true_time"] = pd.NA
    
        # Make sure no accidental leakage of true occurence or censoring
        assert test_labels["occurred"].isna().all(), "Test labels should not have any occurred values yet."
        assert test_labels["censored"].isna().all(), "Test labels should not have any censored values yet."
        assert test_labels["true_censoring"].isna().all(), "Test labels should not have any true_censoring values yet."
        assert test_labels["true_time"].isna().all(), "Test labels should not have any ture_time values yet."

        #: go over every variable and timeline
        all_predictions = []
        all_curr_variables = [x for x in all_variables if x in test_labels["sampled_category"].unique()]

        for variable in all_curr_variables:
            for timeline in all_timelines:

                print(f"Evaluating model for variable: {variable}, timeline: {timeline}")

                #: load in correct input data and arrange
                labels_for_variable_timeline = test_labels[
                    (test_labels["sampled_category"] == variable) &
                    (test_labels["week_to_predict"] == timeline)
                ]
                
                #: process into correct format - same format as labels
                predictions = labels_for_variable_timeline.copy().reset_index(drop=True)
                predictions = predictions.drop(columns=["occurred", "censored", "true_censoring", "true_time"], errors='ignore')
                predictions["occurred"] = majority_class[variable][timeline]  # Set to majority class
                predictions["censored"] = pd.NA   # Setting to NA for now
                predictions["probability_occurrence"] = predictions["occurred"].astype(int)
                predictions["probability_no_occurrence"] = 1 - predictions["probability_occurrence"]
                all_predictions.append(predictions)

        #: post process predictions and concatenate into one large DF
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)

        #: evaluate using eval tools
        evaluator = LandmarkEventsEval(indication=indication, 
                                       data_loading_path=load_folder_splits_test, 
                                       split="test")
        res = evaluator.evaluate(all_predictions_df)

        # Print for quick eval
        print(res["death"][52])

        # Wrap up wandb
        wandb.finish()



def main(load_folder_splits_train, load_folder_splits_test):


    # Setup wandb
    wandb.init(project="genie-dt-cit-events-landmark-probability", mode="offline" if DEBUG else "online",
               group="majority_class")
    wandb.config.update({
        "load_folder_splits_train": load_folder_splits_train,
        "load_folder_splits_test": load_folder_splits_test,
    })
    wandb.run.name = f"Getting stats for Majority class"

    all_train_labels = []

    #: go across all indications
    for indication in all_indications:
        print(f"Processing indication: {indication}")
       
        #: load in the labels, by loading all DFs and concat
        split = "train"
        train_labels = pd.read_csv(os.path.join(load_folder_splits_train, f"targets_{split}_{indication}.csv"))

        #: add to concatenation sets
        all_train_labels.append(train_labels)
    
    # Concat
    all_train_labels = pd.concat(all_train_labels, ignore_index=True)

    # For each variable and time, get majority class for occurence
    all_majority_classes = {}
    all_curr_variables = [x for x in all_variables if x in all_train_labels["sampled_category"].unique()]

    for variable in all_curr_variables:
        all_majority_classes[variable] = {}

        for timeline in all_timelines:

            print(f"Training model for variable: {variable}, timeline: {timeline}")

            #: get subset of all none-censored data for variable and timeline (very naive, biased estimator)
            subset = all_train_labels[
                (all_train_labels["sampled_category"] == variable) & 
                (all_train_labels["week_to_predict"] == timeline) &
                (all_train_labels["censored"] == False)
            ].copy()

            #: get majority class for variable and timeline
            majority_occurence = subset["occurred"].mode()[0]
            
            #: save
            all_majority_classes[variable][timeline] = majority_occurence

    # Wrap up wandb
    wandb.finish()

    #: evaluate
    evaluate(all_majority_classes, load_folder_splits_test)




if __name__ == "__main__":

    # with defaults genie-dt-cgdb-eval-events/0_data/climbr_t/representations/train/
    parser = argparse.ArgumentParser(description="Models")
    parser.add_argument("--load_folder_splits_train", type=str, default="genie-dt-cit-eval-events/0_data/train/")
    parser.add_argument("--load_folder_splits_test", type=str, default="genie-dt-cit-eval-events/0_data/test/")
    
    args = parser.parse_args()
    load_folder_splits_train = args.load_folder_splits_train
    load_folder_splits_test = args.load_folder_splits_test
    print(f"Loading splits from {load_folder_splits_train} and {load_folder_splits_test}")

    main(load_folder_splits_train, load_folder_splits_test)




    