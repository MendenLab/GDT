import wandb
import os
import pandas as pd
import numpy as np
import argparse
import sys
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import OneHotEncoder
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
WANDB_GROUP = "survival_forest"

all_indications = ["cit"]
all_variables = ["death", "progression"]   # All variables in CIT
all_timelines = [8, 26, 52, 104]

# Note that in CIT the true_time is in days, whilst our timeline is in weeks



def evaluate(all_models, load_folder_splits_test):
    
    #: evaluate model on each indication
    for indication in all_indications:
        
        # Setup wandb
        wandb.init(project="genie-dt-cit-events-landmark-probability", mode="offline" if DEBUG else "online",
                group=WANDB_GROUP)
        wandb.config.update({
            "indication": indication,
            "load_folder_splits_test": load_folder_splits_test,
        })
        wandb.run.name = f"SurvivalForest - Test - Indication: {indication}"

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

        #: load in input data for indication
        input_df = pd.read_csv(os.path.join(load_folder_splits_test, f"ml_input_{split}_{indication}.csv"))
        input_df = input_df.drop_duplicates()

        #: go over every variable and timeline
        all_predictions = []
        all_curr_variables = [x for x in all_variables if x in test_labels["sampled_category"].unique()]

        for variable in all_curr_variables:

            #: grab correct data
            curr_input_df = input_df.copy()
            curr_input_df = curr_input_df.drop_duplicates()

            #: setup predictions
            model = all_models[variable]["model"]
            encoder = all_models[variable]["encoder"]
            numeric_cols = all_models[variable]["numeric_cols"]
            categorical_cols = all_models[variable]["categorical_cols"]

            # Prepare the input data for the model
            X_test_numeric = curr_input_df[numeric_cols].values
            if encoder is not None:
                # If there are categorical columns, encode them using the fitted encoder.
                X_test_categorical_encoded = encoder.transform(curr_input_df[categorical_cols])
                # Combine the numeric and one-hot encoded categorical features horizontally.
                X_test = np.hstack([X_test_numeric, X_test_categorical_encoded])
            else:
                # If no categorical columns exist, the feature set is just the numeric data.
                X_test = X_test_numeric
            
            
            # 1. Get a PERSONALIZED survival function for EACH patient in X_test.
            #    The i-th function corresponds to the i-th patient.
            survival_functions = model.predict_survival_function(X_test)
            
            # Create a dictionary mapping each patient's ID to their personal survival function
            patient_id_to_sf_map = dict(zip(curr_input_df["generic_patientid"], survival_functions))

            for timeline in all_timelines:
                print(f"Evaluating model for variable: {variable}, timeline: {timeline}")

                # 2. For the current timeline, calculate the event probability FOR EACH PATIENT.
                patient_ids = curr_input_df["generic_patientid"].values
                prob_occurrence_per_patient = []
                prob_survival_per_patient = []

                for patient_id in patient_ids:
                    sf = patient_id_to_sf_map[patient_id] # Get the specific patient's survival function
                    
                    # Find survival probability for this patient at the specific timeline
                    time_points = sf.x
                    probabilities = sf.y
                    
                    time_index = np.searchsorted(time_points, timeline * 7, side='right') - 1
                    
                    # If timeline is before the first time point, survival prob is 1.0
                    if time_index < 0:
                        prob_survival = 1.0
                    else:
                        prob_survival = probabilities[time_index]

                    # Probability of event is 1 - survival probability
                    prob_occurrence_per_patient.append(1.0 - prob_survival)
                    prob_survival_per_patient.append(prob_survival)

                # 3. Categorize the prediction FOR EACH PATIENT based on the 0.5 probability threshold.
                #    This results in a boolean array, one value per patient.
                predictions_categorized = np.array(prob_occurrence_per_patient) >= 0.5

                #: assert identical generic_patientid between test_labels and curr_input_df
                assert set(test_labels["generic_patientid"].unique()) == set(curr_input_df["generic_patientid"].unique()), \
                    "Generic patient IDs in test labels and input data must match."

                #: process into correct format - same format as labels
                #: match via generic_patientid to test_labels
                predictions_formatted = test_labels[
                    (test_labels["sampled_category"] == variable) &
                    (test_labels["week_to_predict"] == timeline) &
                    (test_labels["generic_patientid"].isin(curr_input_df["generic_patientid"]))
                ].copy()

                #: order by same patientid
                predictions_formatted = predictions_formatted.set_index("generic_patientid").reindex(
                    curr_input_df["generic_patientid"].values
                ).reset_index()

                #: assign and save
                predictions_formatted["probability_occurrence"] = prob_occurrence_per_patient
                predictions_formatted["probability_no_occurrence"] = prob_survival_per_patient
                predictions_formatted["occurred"] = predictions_categorized
                predictions_formatted["censored"] = pd.NA  # No censoring information in
                all_predictions.append(predictions_formatted)

        #: post process predictions and concatenate into one large DF
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        all_predictions_df = all_predictions_df.sort_values(by=["patientid", "sampled_category", "week_to_predict"]).reset_index(drop=True)

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
               group=WANDB_GROUP)
    wandb.config.update({
        "load_folder_splits_test": load_folder_splits_test,
        "load_folder_splits_train": load_folder_splits_train,
        "all_indications": all_indications,
        "all_variables": all_variables,
        "all_timelines": all_timelines
    })
    wandb.run.name = f"Training survival forest"

    all_inputs = []
    all_targets = []

    for indication in all_indications:
        
        #: load in the labels, by loading all DFs and concat
        split = "train"
        target_df = pd.read_csv(os.path.join(load_folder_splits_train, f"targets_{split}_{indication}.csv"))
        input_df = pd.read_csv(os.path.join(load_folder_splits_train, f"ml_input_{split}_{indication}.csv"))

        all_inputs.append(input_df)
        all_targets.append(target_df)

    #: concatenate them together
    all_inputs_df = pd.concat(all_inputs, ignore_index=True)
    all_targets_df = pd.concat(all_targets, ignore_index=True)

    #: go over each variable
    all_curr_variables = [x for x in all_variables if x in all_targets_df["sampled_category"].unique()]
    all_models = {}

    for variable in all_curr_variables:

        print(f"Training model for variable: {variable}")

        #: extract correct data
        curr_input_df = all_inputs_df.copy()   # Since one row per generic_patientid
        curr_input_df = curr_input_df.drop_duplicates()

        curr_target_df = all_targets_df[all_targets_df["sampled_category"] == variable].copy()
        curr_target_df = curr_target_df[["generic_patientid", "sampled_category", "true_censoring", "true_time"]]
        curr_target_df = curr_target_df.drop_duplicates()

        # Make some checks
        assert len(curr_input_df) == len(curr_target_df), "Input and target dataframes must have the same length."
        assert curr_input_df["generic_patientid"].tolist() == curr_target_df["generic_patientid"].tolist(), "Patient IDs must match in input and target dataframes."

        # Make same order
        patientid_order = curr_input_df["generic_patientid"].values
        curr_input_df = curr_input_df.set_index("generic_patientid").reindex(patientid_order).reset_index()
        curr_target_df = curr_target_df.set_index("generic_patientid").reindex(patientid_order).reset_index()
        
        # Assert that the same order of patient IDs is maintained
        assert all(curr_input_df["generic_patientid"] == curr_target_df["generic_patientid"]), \
            "Patient IDs must match in input and target dataframes after reindexing."

        #: formulate input and output (censored) data
        # Separate column types based on their data type.
        # The 'generic_patientid' column is explicitly excluded from feature processing.
        numeric_cols = curr_input_df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = curr_input_df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Ensure patient ID is not treated as a feature.
        if 'generic_patientid' in curr_input_df.columns:
            # Use list comprehension to create a new list without the id
            categorical_cols = [col for col in categorical_cols if col != 'generic_patientid']


        X_train_numeric = curr_input_df[numeric_cols].values
        encoder = None # Initialize encoder to None. It will be created if categorical columns exist.

        # If there are categorical columns, initialize and fit the OneHotEncoder.
        if categorical_cols:
            # handle_unknown='ignore' will prevent errors if the test set has categories
            # not seen during training.
            # sparse_output=False ensures the output is a dense numpy array.
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            # Fit the encoder on the training data and transform it.
            X_train_categorical_encoded = encoder.fit_transform(curr_input_df[categorical_cols])

            # Combine the numeric and one-hot encoded categorical features horizontally.
            X_train = np.hstack([X_train_numeric, X_train_categorical_encoded])
        else:
            # If no categorical columns exist, the feature set is just the numeric data.
            X_train = X_train_numeric

        # Prepare the target variable for the survival model
        # Need to flip the censoring since sksurv expects False for censored and True for occurred.
        curr_target_df["true_censoring"] = ~curr_target_df["true_censoring"].astype(bool)
        y_train = curr_target_df[["true_censoring", "true_time"]].to_numpy()
        y_train = np.array([(bool(censor), time) for censor, time in y_train], dtype=[('censored', '?'), ('time', '<f8')])
                
        #: train survival forest
        model = RandomSurvivalForest(
            n_jobs=-1, random_state=9782,
        ).fit(X_train, y_train)

        #: save
        all_models[variable] = {
            "model": model,
            "encoder": encoder,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols
        }
        print(f"Model for variable {variable} trained and saved.")


    wandb.finish()

    #: eval
    evaluate(all_models, load_folder_splits_test)
    


if __name__ == "__main__":
    # with defaults
        # with defaults genie-dt-cgdb-eval-events/0_data/climbr_t/representations/train/
    parser = argparse.ArgumentParser(description="Models")
    parser.add_argument("--load_folder_splits_train", type=str, default="genie-dt-cit-eval-events/0_data/train/")
    parser.add_argument("--load_folder_splits_test", type=str, default="genie-dt-cit-eval-events/0_data/test/")
    
    args = parser.parse_args()
    load_folder_splits_train = args.load_folder_splits_train
    load_folder_splits_test = args.load_folder_splits_test
    print(f"Loading splits from {load_folder_splits_train} and {load_folder_splits_test}")

    main(load_folder_splits_train, load_folder_splits_test)







