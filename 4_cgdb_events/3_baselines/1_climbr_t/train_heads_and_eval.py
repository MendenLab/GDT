import wandb
import os
import pandas as pd
import json
import numpy as np
import argparse
import glob
import sys

# SKSURV and SKLEARN imports for CoxPH model
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def setup_imports_nb():
    try:
        notebook_parent_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        notebook_parent_dir = os.getcwd()

    project_root_dir = os.path.abspath(os.path.join(notebook_parent_dir, "../2_eval_tools/"))
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
setup_imports_nb()
from utils_events_eval import EventsEval


DEBUG = False
WANDB_GROUP = "clmbr-t-coxph"

all_indications = [
    'enhanced_multiplemyeloma', 'enhanced_rcc', 'enhanced_breast', 'enhanced_crc',
    'enhanced_ovarian', 'enhanced_endometrial', 'enhanced_pantumor', 'enhanced_metprostate',
    'enhanced_advurothelial', 'enhanced_cll', 'enhanced_sclc', 'enhanced_headneck',
    'enhanced_pancreatic', 'enhanced_dlbcl', 'enhanced_hcc', 'enhanced_acutemyeloidleukemia',
    'enhanced_melanoma', 'enhanced_nsclc', 'enhanced_gastricesoph', 'enhanced_follicularlymphoma'
]

all_variables = ["death", "progression", "lot", "metastasis"]

all_timelines = [8, 26, 52, 104]


def evaluate(all_trained_models, load_folder_test_reps, load_folder_splits_test):

    #: evaluate model on each indication
    for indication in all_indications:
        # Setup wandb
        wandb.init(project="genie-dt-cgdb-baselines-events-probabilities", mode="offline" if DEBUG else "online",
                   group=WANDB_GROUP)
        wandb.config.update({
            "indication": indication,
            "load_folder_test_reps": load_folder_test_reps,
            "load_folder_splits_test": load_folder_splits_test,
        })
        wandb.run.name = f"CoxPH Eval Test - Indication: {indication}"

        print(f"Evaluating models for indication: {indication}")

        #: load in all the test representations and map to original patientids
        path_to_reps = os.path.join(load_folder_test_reps, f"{indication}_representations.npz")
        path_original_patientids = os.path.join(load_folder_test_reps, f"{indication}_gdt_patientid_to_climbr_t_patientid_map.json")
        precomputed_test_reps = np.load(path_to_reps)
        with open(path_original_patientids, 'r') as f:
            original_patientid_mapping = json.load(f)
        reverse_patientid_mapping = {v: k for k, v in original_patientid_mapping.items()}
        precomputed_test_original_patientids = np.asarray([reverse_patientid_mapping[k] for k in precomputed_test_reps['patient_ids']])

        #: grab the test labels
        folder_of_labels = os.path.join(load_folder_splits_test, f"test_{indication}")
        all_targets = glob.glob(os.path.join(folder_of_labels, "*_targets.csv"))
        all_targets_dfs = [pd.read_csv(f) for f in all_targets]
        test_labels = pd.concat(all_targets_dfs, ignore_index=True)
        # Clear columns that will be filled with predictions
        for col in ["occurred", "censored", "true_censoring", "true_time"]:
            if col in test_labels.columns:
                test_labels[col] = pd.NA

        # Make sure no accidental leakage of true occurence or censoring
        assert test_labels["occurred"].isna().all(), "Test labels should not have any occurred values yet."
        assert test_labels["censored"].isna().all(), "Test labels should not have any censored values yet."
        assert test_labels["true_censoring"].isna().all(), "Test labels should not have any true_censoring values yet."
        assert test_labels["true_time"].isna().all(), "Test labels should not have any ture_time values yet."

        #: link representations to patientids
        precomputed_test_representations = precomputed_test_reps["features"]
        test_patientids = pd.DataFrame({
            "generic_patientid": precomputed_test_original_patientids,
            "index_of_repr": np.arange(len(precomputed_test_representations))
        })
        print(f"Loaded {len(precomputed_test_representations)} representations for {indication}")

        #: go over every variable and timeline
        all_predictions = []
        all_curr_variables = [x for x in all_variables if x in test_labels["sampled_category"].unique()]

        for variable in all_curr_variables:
            print(f"Predicting for variable: {variable}")
            
            model = all_trained_models[variable]["model"]
            preprocessor = all_trained_models[variable]["preprocessor"]

            # Get unique patient representations for this variable to avoid redundant predictions
            unique_patients_df = test_labels[test_labels["sampled_category"] == variable][["generic_patientid"]].drop_duplicates()
            labels_matched_unique = pd.merge(unique_patients_df, test_patientids, on="generic_patientid", how="left")

            X_test_raw = precomputed_test_representations[labels_matched_unique["index_of_repr"].values]
            X_test = preprocessor.transform(X_test_raw)
            
            # Predict personalized survival function for each patient
            survival_functions = model.predict_survival_function(X_test)
            patient_id_to_sf_map = dict(zip(labels_matched_unique["generic_patientid"], survival_functions))

            for timeline in all_timelines:
                print(f"  Evaluating timeline: {timeline}")

                labels_for_timeline = test_labels[
                    (test_labels["sampled_category"] == variable) &
                    (test_labels["week_to_predict"] == timeline)
                ].copy()

                prob_occurrence_per_patient = []
                prob_survival_per_patient = []

                # Calculate event probability for each patient at the given timeline
                for patient_id in labels_for_timeline["generic_patientid"]:
                    sf = patient_id_to_sf_map[patient_id]
                    time_points, probabilities = sf.x, sf.y
                    
                    time_index = np.searchsorted(time_points, timeline, side='right') - 1
                    
                    prob_survival = 1.0 if time_index < 0 else probabilities[time_index]
                    prob_occurrence_per_patient.append(1.0 - prob_survival)
                    prob_survival_per_patient.append(prob_survival)

                # Convert probabilities to binary predictions
                y_pred = np.array(prob_occurrence_per_patient) >= 0.5
                
                # Format predictions
                predictions = labels_for_timeline.reset_index(drop=True)
                predictions["occurred"] = y_pred
                predictions["censored"] = pd.NA
                predictions["probability_occurrence"] = prob_occurrence_per_patient
                predictions["probability_no_occurrence"] = prob_survival_per_patient
                all_predictions.append(predictions)

        #: post process predictions and evaluate
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        all_predictions_df = all_predictions_df.sort_values(by=["patientid", "sampled_category", "week_to_predict"]).reset_index(drop=True)

        evaluator = EventsEval(indication=indication, data_loading_path=load_folder_splits_test, split="test")
        res = evaluator.evaluate(all_predictions_df)

        print(res["death"][52])
        wandb.finish()



def main(load_folder_train_reps, load_folder_test_reps,
         load_folder_splits_train, load_folder_splits_test):
    
    wandb.init(project="genie-dt-cgdb-baselines-events-probabilities", mode="offline" if DEBUG else "online",
               group=WANDB_GROUP)
    wandb.run.name = f"Training CoxPH heads for CLMBR-T"

    ############################################ TRAINING HEADS ############################################
    all_train_representations, all_train_patientids, all_train_labels = [], [], []

    for indication in all_indications:
        print(f"Loading training data for indication: {indication}")
        path_to_reps = os.path.join(load_folder_train_reps, f"{indication}_representations.npz")
        path_original_patientids = os.path.join(load_folder_train_reps, f"{indication}_gdt_patientid_to_climbr_t_patientid_map.json")
        
        precomputed_train_reps = np.load(path_to_reps)
        with open(path_original_patientids, 'r') as f:
            original_patientid_mapping = json.load(f)
        reverse_patientid_mapping = {v: k for k, v in original_patientid_mapping.items()}
        
        all_train_representations.append(precomputed_train_reps["features"])
        all_train_patientids.append(np.asarray([reverse_patientid_mapping[k] for k in precomputed_train_reps['patient_ids']]))
        
        folder_of_labels = os.path.join(load_folder_splits_train, f"train_{indication}")
        all_targets = glob.glob(os.path.join(folder_of_labels, "*_targets.csv"))
        train_labels = pd.concat([pd.read_csv(f) for f in all_targets], ignore_index=True)
        all_train_labels.append(train_labels)

    #: concatenate into full dataset
    all_train_representations = np.concatenate(all_train_representations, axis=0)
    all_train_patientids_flat = np.concatenate(all_train_patientids, axis=0)
    all_train_patientids = pd.DataFrame({
        "generic_patientid": all_train_patientids_flat,
        "index_of_repr": np.arange(len(all_train_patientids_flat))
    })
    all_train_labels = pd.concat(all_train_labels, ignore_index=True)
    print(f"Total representations for training: {len(all_train_representations)}")

    all_trained_models = {}
    all_curr_variables = [x for x in all_variables if x in all_train_labels["sampled_category"].unique()]

    for variable in all_curr_variables:
        print(f"Training CoxPH model for variable: {variable}")

        # Get unique time-to-event labels for this variable
        labels_for_variable = all_train_labels[all_train_labels["sampled_category"] == variable].copy()

        # Drop patients with duplicate generic_patientid across indications, keeping the first
        labels_for_variable = labels_for_variable.drop_duplicates(subset=['generic_patientid'], keep='first')

        # Match labels to representations
        labels_matched = pd.merge(labels_for_variable, all_train_patientids, on="generic_patientid", how="left")
        
        X_train_raw = all_train_representations[labels_matched["index_of_repr"].values]
        
        # sksurv expects boolean event indicator (True=event, False=censored)
        event_indicator = ~labels_matched["true_censoring"].astype(bool)
        time_to_event = labels_matched["true_time"].values
        y_train = np.array(list(zip(event_indicator, time_to_event)), dtype=[('event', '?'), ('time', '<f8')])
        
        # Preprocessing pipeline
        preprocessor = Pipeline([('scaler', StandardScaler())])

        X_train = preprocessor.fit_transform(X_train_raw)
        print(f"Training with {X_train.shape[0]} samples. Feature dimensions: {X_train.shape[1]}")
        
        # Define the parameter grid for GridSearchCV
        param_grid = {
            'alphas': [[alpha] for alpha in np.logspace(-2, 2, 10)],
            'l1_ratio': [0.001, 0.1, 0.5, 0.7, 0.9, 0.99, 1.0]
        }

        model = CoxnetSurvivalAnalysis(fit_baseline_model=True, max_iter=200000)
        
        cv = GridSearchCV(
            model,
            param_grid,
            scoring=lambda estimator, X, y: concordance_index_censored(y["event"], y["time"], estimator.predict(X))[0],
            cv=5,
            verbose=3,
            n_jobs=-1,
            refit=True
        ).fit(X_train, y_train)

        print(f"Best parameters found for {variable}: {cv.best_params_}")
        best_model = cv.best_estimator_

        all_trained_models[variable] = {
            "model": best_model,
            "preprocessor": preprocessor,
        }

    print(f"Trained models for all variables: {all_trained_models.keys()}")
    wandb.finish()

    ############################################ EVALUATION ############################################
    print("\nStarting evaluation...")
    evaluate(all_trained_models, load_folder_test_reps, load_folder_splits_test)
    print("Finished evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate CoxPH heads for CLIMBR-T")
    parser.add_argument("--load_folder_train_reps", type=str, default="genie-dt-cgdb-eval-events/0_data/climbr_t/representations/train/")
    parser.add_argument("--load_folder_test_reps", type=str, default="genie-dt-cgdb-eval-events/0_data/climbr_t/representations/test/")
    parser.add_argument("--load_folder_splits_train", type=str, default="genie-dt-cgdb-eval-events/0_data/splits_only_train/")
    parser.add_argument("--load_folder_splits_test", type=str, default="genie-dt-cgdb-eval-events/0_data/splits_only/")
    
    args = parser.parse_args()
    
    print(f"Loading representations from {args.load_folder_train_reps} and {args.load_folder_test_reps}")
    print(f"Loading splits from {args.load_folder_splits_train} and {args.load_folder_splits_test}")

    main(args.load_folder_train_reps, args.load_folder_test_reps,
         args.load_folder_splits_train, args.load_folder_splits_test)
