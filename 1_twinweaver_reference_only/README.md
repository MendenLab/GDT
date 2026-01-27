# TwinWeaver for GDT


> **_NOTE:_  This package is an early version of TwinWeaver, and is provided for reference. All development should be made with the TwinWeaver package.**

The core of the Genie Digital Twin project is the package `digital_twin_converter` to transform clinical data into LLM-ingestible text.
The library can be used for any clinical patient data, assuming its in the correct format.

Based on the Flatiron/CGDB dataset, we build up 2 datasets for training Genie, as well as then further fine-tuning to make Genie Digital Twin.




## Installation of `digital_twin_converter` package (early version of `TwinWeaver`)

Run the following commands from this folder to install the core package:

```
cd 1_twinweaver
pip install -e .
```




# CGDB specific data generation details

The following section contains details on the datasets generated based on CGDB (Flatiron & FoundationOne Medicine) dataset of ~93k pan-cancer patients. 

You can find the main scripts for data generation in [`dataset_generation/instruction/slurm_instruction_dataset.sh`](dataset_generation/instruction/slurm_instruction_dataset.sh).


### Input Data Format

The general input data is based on two dataframes per patients: static and events.
Static is one row with many columns representing various demographic data.
Events is a long format dataframe with the key columns being `date`, `event_category`, `event_name`, `event_value` and `event_descriptive_name`.


## Dataset Generation for Instruction

This project creates one instruction datasets based on three tasks to enable digital twins.

### Tasks

1. Future forecasting with numeric values up to 90 days into the future (aka `forecasting`)
2. Future forecasting with categories, up to 90 days into the future (aka `forecasting qa`)
3. Predicting whether an important event will occur & be censored (aka `events`)

We then combine them, with up to 1 forecasting, 1 forecasting qa and potentially many events task (default 4).
At least one task is enabled per sample.
Currently, we make 10 patient sample per line of therapy per patient, averaging around 30 samples per patient.
The rough distribution is that half the samples have only events, and half have one of the forecasting tasks.


### Task Details & Assumptions


#### 0️⃣ Common base parameters

- Splitting with sometime randomly during first 3 months (=13 weeks) after (and including) start of LoT
    - Randomly selected on week level
- Use same setup as in pretraining
    - Including timing
- Put in as much as possible for context - length of max prediction
    - Remove rest, except for last observed value of variable to predict ,last genetic variables observed & last line of therapy (summarized row style)


#### 1️⃣ Task 1: Forecasting - Shared setup

- All variables, but samples inveresly with how complex the variable is
    - = how much the variables change
    - Sampling: `~ ld(NRMSE(copy_forward) * nr_of_occurences)`
        - `copy_forward` is if we use the last observed value for the current prediction
        - So that common variables are prioritized (since often rare variables fluctuate a lot)
        - Picked since from empirical observations it prioritized high variance variables without completely masking out the other variables
- Next 13 weeks
- Picking 3 (if possible) variables per patients
- Predict max 3 months ahead
- Only select from those variables which have at least one observed value on the input side within the last 90 days
- For filtering we apply 3-sigma filtering with clipping (i.e. not removing anything)

#### 1️⃣.1️⃣ Task 1.1 - numeric

- Predict numbers directly

#### 1️⃣.2️⃣ Task 1.2 - QA

- Predict category from ranges → use 5 categories which have equal sizes
- Should hopefully lead to less bias and more challenging task



### 2️⃣ Task 2: Time to event prediction - events

- Manually selected vars
    - Death
    - Metastases
    - Progression
        - Death is a progression event as well
    - Time to next treatment
- Actually, do not do diagnoses
    - Hard to include without biasing heavily or making useless predictions
    - Very dirty in any case in Flatiron
- Provide time + var in input
    - Censor + occurence in output
    - This way we can sample time points to find censoring/occurence
    - Occurence is causally dependent on censoring
        - e.g. if censored then did not occur
    - Can then condition on different censor events if we want to simulate more exactly


### Processing overview

The general flow is (for all three tasks):
1. `jsonl_converter_instruction.py` calls `data_manager` to retrieve the patient data
2. `jsonl_converter_instruction.py` calls `data_splitter_*` to split the patient trajectory into input/output
2. `jsonl_converter_instruction.py` calls `converter_manual_template_*` to convert proecedurally the structured data into text
3. `jsonl_converter_instruction.py` then saves the data
4. `split_and_save.py` then loads the data, does final data processing of splits and then uploads to S3 in usable (~100MB) chunks.

Both 'forecasting' and 'forecasting qa' are identical execept for the conversion done in `converter_manual_forecasting.py` and `converter_manual_forecasting_qa.py` respectively.

The conversion templates are based on our previous experience, [with the results being shown in our pre-print](https://www.medrxiv.org/content/10.1101/2024.07.05.24309957v1).


### Output Data Format


Each line is one JSON file, with the following entries:
```
{
    "instruction": input for patient sample,
    "answer": target answer for patient sample (can have multiple tasks per query),
    "meta": {
        "patientid": patientid, also consistent with CGDB database,
        "indication": indication where the patient data orginates from,
        "split": which train/val/test split,
        "forecasting_type":
        "constant": demographic pandas dataframe in json format and "split" orientation,
            to be loaded in using pd.read_json(dic["meta"]["constant"], orient = 'split'),
        "history_data": events & genetic pandas dataframe in json format and "split" orientation,
            to be loaded in using pd.read_json(dic["meta"]["events"], orient = 'split'),
        "split_date_included_in_input" : date of the trajectort split into input
            and output, to be included in the input,
        "target_meta": list of meta data, in same order as appearing in answer,
        "num_tokens_input": nr instruction tokens,
        "num_tokens_target": nr target tokens
        "num_tokens_total": total nr tokens
    }
}
```




