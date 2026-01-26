import wandb
import os
import pandas as pd
import json
import numpy as np
import argparse
import sys
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA




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
WANDB_GROUP = "clmbr-t-coxph" # Updated wandb group

all_indications = ["cit"]
all_variables = ["death", "progression"]   # All variables in CIT
all_timelines = [8, 26, 52, 104]


# Note that in CIT the true_time is in days, whilst our timeline is in weeks


def evaluate(all_trained_models, load_folder_train_reps, load_folder_test_reps,
             load_folder_splits_train, load_folder_splits_test):
    
    #: evaluate model on each indication
    for indication in all_indications:
        
        # Setup wandb
        wandb.init(project="genie-dt-cit-events-landmark-probability", mode="offline" if DEBUG else "online",
                   group=WANDB_GROUP)
        wandb.config.update({
            "indication": indication,
            "load_folder_train_reps": load_folder_train_reps,
            "load_folder_test_reps": load_folder_test_reps,
            "load_folder_splits_train": load_folder_splits_train,
            "load_folder_splits_test": load_folder_splits_test,
        })
        wandb.run.name = f"CLMBR-T CoxPH Eval Test - Indication: {indication}"

        print(f"Evaluating models for indication: {indication}")

        #: load in all the test representations, assign them the correct original patientids
        path_to_reps = os.path.join(load_folder_test_reps, f"{indication}_representations.csv")
        precomputed_test_representations = pd.read_csv(path_to_reps)
        
        print(f"Loaded {len(precomputed_test_representations)} representations for {indication}")

        #: grab the test labels, by loading all DFs and concat, so that we have the correct patientids
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
            
            # 1. Get the single trained CoxPH model for the current variable
            model = all_trained_models[variable]["model"]
            preprocessor = all_trained_models[variable]["preprocessor"]

            # 2. Prepare the input data (X_test) for all unique patients for this variable
            curr_test_labels = test_labels[test_labels["sampled_category"] == variable]
            unique_patients_df = curr_test_labels[["generic_patientid"]].drop_duplicates()

            patients_matched = pd.merge(unique_patients_df, precomputed_test_representations, on="generic_patientid", how="inner")
            cols_features = [col for col in precomputed_test_representations.columns if col.startswith("feature_")]
            X_test_raw = patients_matched[cols_features]
            
            # Preprocess
            X_test = preprocessor.transform(X_test_raw)

            # 3. Get a PERSONALIZED survival function for EACH patient in X_test.
            survival_functions = model.predict_survival_function(X_test)
            patient_id_to_sf_map = dict(zip(patients_matched["generic_patientid"], survival_functions))

            for timeline in all_timelines:
                print(f"Evaluating model for variable: {variable}, timeline: {timeline}")

                # 4. For the current timeline, calculate the event probability FOR EACH PATIENT.
                # Filter the labels to get the specific patient cohort for this timeline
                predictions_formatted = test_labels[
                    (test_labels["sampled_category"] == variable) &
                    (test_labels["week_to_predict"] == timeline)
                ].copy()

                prob_occurrence_per_patient = []
                prob_survival_per_patient = []

                for patient_id in predictions_formatted["generic_patientid"]:
                    sf = patient_id_to_sf_map[patient_id] # Get the specific patient's survival function
                    
                    # Find survival probability for this patient at the specific timeline
                    time_points = sf.x
                    probabilities = sf.y
                    
                    time_in_days = timeline * 7
                    time_index = np.searchsorted(time_points, time_in_days, side='right') - 1
                    
                    # If timeline is before the first time point, survival prob is 1.0
                    if time_index < 0:
                        prob_survival = 1.0
                    else:
                        prob_survival = probabilities[time_index]

                    # Probability of event is 1 - survival probability
                    prob_occurrence_per_patient.append(1.0 - prob_survival)
                    prob_survival_per_patient.append(prob_survival)

                # 5. Categorize the prediction FOR EACH PATIENT based on the 0.5 probability threshold.
                predictions_categorized = np.array(prob_occurrence_per_patient) >= 0.5
                
                # Assign predictions and format for evaluation
                predictions_formatted["occurred"] = predictions_categorized
                predictions_formatted["censored"] = pd.NA
                predictions_formatted["probability_occurrence"] = prob_occurrence_per_patient
                predictions_formatted["probability_no_occurrence"] = prob_survival_per_patient
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




def main(load_folder_train_reps, load_folder_test_reps,
         load_folder_splits_train, load_folder_splits_test):
    
    # Setup wandb
    wandb.init(project="genie-dt-cit-events-landmark-probability", mode="offline" if DEBUG else "online",
               group=WANDB_GROUP)
    wandb.config.update({
        "load_folder_train_reps": load_folder_train_reps,
        "load_folder_test_reps": load_folder_test_reps,
        "load_folder_splits_train": load_folder_splits_train,
        "load_folder_splits_test": load_folder_splits_test,
        "all_indications": all_indications,
    })
    wandb.run.name = f"Training CoxPH heads for CLMBR-T"
    
    ############################################ TRAINING HEADS ############################################

    all_train_representations = []
    all_train_labels = []

    #: go across all indications
    for indication in all_indications:
        print(f"Processing indication: {indication}")
        
        #: load in all the precomputed representations
        path_to_reps = os.path.join(load_folder_train_reps, f"{indication}_representations.csv")
        precomputed_train_reps = pd.read_csv(path_to_reps)

        #: load in the labels, by loading all DFs and concat
        # These targets are in days
        split = "train"
        train_labels = pd.read_csv(os.path.join(load_folder_splits_train, f"targets_{split}_{indication}.csv"))

        #: add to concatenation sets
        all_train_representations.append(precomputed_train_reps)
        all_train_labels.append(train_labels)

        print(f"Loaded {len(precomputed_train_reps)} representations for {indication}")

    #: concatenate into full dataset
    all_train_representations = pd.concat(all_train_representations, ignore_index=True)
    all_train_labels = pd.concat(all_train_labels, ignore_index=True)
    print(f"Total representations: {len(all_train_representations)}")

    #: Train one CoxPH model per variable
    all_trained_models = {}
    all_curr_variables = [x for x in all_variables if x in all_train_labels["sampled_category"].unique()]

    for variable in all_curr_variables:
        print(f"Training model for variable: {variable}")

        #: Get unique survival outcomes for each patient for this variable
        labels_for_variable = all_train_labels[
            (all_train_labels["sampled_category"] == variable)
        ]
        labels_for_variable = labels_for_variable[["generic_patientid", "true_censoring", "true_time"]].drop_duplicates()
        
        #: Match labels to representations to align X and y
        labels_matched = pd.merge(labels_for_variable, all_train_representations, on="generic_patientid", how="inner")
        cols_features = [col for col in all_train_representations.columns if col.startswith("feature_")]
        X_train_raw = labels_matched[cols_features]

        # Prepare the target variable for the survival model
        # sksurv expects True for an event (not censored) and False for censored.
        # Note that the time is in days here
        labels_matched["event_observed"] = ~labels_matched["true_censoring"].astype(bool)
        y_train = labels_matched[["event_observed", "true_time"]].to_numpy()
        y_train = np.array(
            [(bool(event), time) for event, time in y_train], 
            dtype=[('event', '?'), ('time', '<f8')]
        )
        
        #1. Standardize and dimred the features
        # having convergence issues without PCA
        preprocessor = Pipeline([
            ('scaler', StandardScaler()),
        ])

        # CHANGED: Fit and transform the raw data using the entire pipeline
        X_train = preprocessor.fit_transform(X_train_raw)
        print(f"Original dimensions: {X_train_raw.shape[1]}, Reduced dimensions: {X_train.shape[1]}")
                
        # Define the parameter grid to search
        param_grid = {
            'alphas': [[alpha] for alpha in np.logspace(-2, 0, 5)], 
            'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.99, 1.0]
        }

        # Create the Coxnet model instance
        model = CoxnetSurvivalAnalysis(fit_baseline_model=True, max_iter=200000)

        # Set up GridSearchCV
        cv = GridSearchCV(
            model,
            param_grid,
            scoring=lambda estimator, X, y: concordance_index_censored(y["event"], y["time"], estimator.predict(X))[0],
            cv=5,
            verbose=3,
            n_jobs=-1,
            refit=True
        ).fit(X_train, y_train)

        print(f"Best parameters found: {cv.best_params_}")

        # The 'best_estimator_' IS the final model, already retrained on the full X_train and y_train.
        best_model = cv.best_estimator_

        # Now, assign this fully trained model to your dictionary
        all_trained_models[variable] = {
            "model": best_model,
            "preprocessor": preprocessor,
        }

        print(f"Trained model for variable: {variable} with {len(X_train)} samples")

    print(f"Trained models for all variables: {list(all_trained_models.keys())}")
    wandb.finish()

    ############################################ EVALUATION ############################################
    print("Starting evaluation...")
    evaluate(all_trained_models, load_folder_train_reps, load_folder_test_reps,
             load_folder_splits_train, load_folder_splits_test)
    print("Finished evaluation.")

    




if __name__ == "__main__":

    # with defaults genie-dt-cgdb-eval-events/0_data/climbr_t/representations/train/
    parser = argparse.ArgumentParser(description="Train CoxPH heads and evaluate for CLIMBR-T")
    parser.add_argument("--load_folder_train_reps", type=str, default="genie-dt-cit-eval-events/0_data/clmbr_t/representations/train/")
    parser.add_argument("--load_folder_test_reps", type=str, default="genie-dt-cit-eval-events/0_data/clmbr_t/representations/test/")
    parser.add_argument("--load_folder_splits_train", type=str, default="genie-dt-cit-eval-events/0_data/train/")
    parser.add_argument("--load_folder_splits_test", type=str, default="genie-dt-cit-eval-events/0_data/test/")
    
    args = parser.parse_args()
    load_folder_train_reps = args.load_folder_train_reps
    load_folder_test_reps = args.load_folder_test_reps  
    load_folder_splits_train = args.load_folder_splits_train
    load_folder_splits_test = args.load_folder_splits_test
    print(f"Loading representations from {load_folder_train_reps} and {load_folder_test_reps}")
    print(f"Loading splits from {load_folder_splits_train} and {load_folder_splits_test}")

    main(load_folder_train_reps, load_folder_test_reps,
         load_folder_splits_train, load_folder_splits_test)
    
