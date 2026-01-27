import pandas as pd


class Config:
    """
    Centralized configuration repository for data processing, prompt generation, and constants.

    This class consolidates various configuration settings essential for the data processing
    pipeline. It defines standardized column names, specific values for event categories (like
    'line of therapy', 'death'), data source identifiers, file paths, table names, and text
    templates (prompts) used for different language model tasks such as text conversion,
    forecasting (value prediction, time-to-event), quality assurance (QA) via binning, and
    setting up multi-task prompts. Default values are provided but can be overridden to
    adapt to specific datasets, model requirements, or experimental setups.

    Attributes
    ----------
    date_col : str
        Standardized column name for date information across datasets. Default: "date".
    patient_id_col : str
        Standardized column name for unique patient identifiers. Default: "patientid".
    event_category_col : str
        Standardized column name for the category of a recorded event (e.g., 'lab', 'diagnosis'). Default: "event_category".
    event_name_col : str
        Standardized column name for the specific name of an event within its category (e.g., 'Glucose level', 'Type 2 Diabetes'). Default: "event_name".
    event_descriptive_name_col : str
        Standardized column name for a more human-readable or descriptive name of the event. Default: "event_descriptive_name".
    event_value_col : str
        Standardized column name for the value associated with an event (e.g., a lab result, 'present'). Default: "event_value".
    source_col : str
        Standardized column name indicating the origin or type of the data record. Default: "source".
    meta_data_col : str
        Standardized column name for storing additional metadata related to an event. Default: "meta_data".
    lot_date_col : str
        Column name specifically used for dates related to line of therapy (LoT) events. Default: "lot_date".
    lot_name_col : str
        Column name for the name or identifier of the line of therapy (e.g., "First Line"). Default: "line_name".
        If set to None, it will use just last LoT event name.
    biomarker_category_col : str
        Column name for the category of a biomarker (e.g., 'Gene Mutation', 'Protein Expression'). Default: "biomarker_category".
    biomarker_event_col : str
        Column name for the specific name of the biomarker event (e.g., 'EGFR Exon 19 Deletion'). Default: "biomarker_event".
    biomarker_value_col : str
        Column name for the value or status of the biomarker (e.g., 'Positive', 'Detected'). Default: "biomarker_value".
    biomarker_descriptive_name_col : str
        Column name for a descriptive name of the biomarker. Default: "biomarker_descriptive_name".
    event_category_lot : str
        Specific string value used in `event_category_col` to identify 'line of therapy' events. Default: "lot".
    event_category_death : str
        Specific string value used in `event_category_col` to identify 'death' events. Default: "death".
    event_category_labs : str
        Specific string value used in `event_category_col` to identify 'lab result' events. Default: "lab".
    event_value_lot_start : str
        Specific string value used in `event_value_col` to denote the start of a line of therapy. Default: "LoT start".
        If set to None, it will use just last LoT event name.
    skip_future_lot_filtering : bool
        Flag indicating whether to skip filtering out future line of therapy events. Default: False.
        Useful in case you accidentially overlap LoTs which are actually the same, use with caution.
    lot_concatenate_descriptive_and_value : bool
        Flag indicating whether to concatenate the descriptive name and value for line of therapy events. Default: False.
    lot_concatenate_string : str
        String used to concatenate the descriptive name and value for line of therapy events when `lot_concatenate_descriptive_and_value` is True. Default: " - ".
    source_genetic : str
        Specific string value used in `source_col` to identify data originating from genetic testing. Default: "genetic".
    genetic_skip_text_value : str
        A specific event value (often for genetic data) that might be skipped during text generation to avoid redundancy if its presence is implied elsewhere. Default: "present".
    genetic_tag_opening : str
        Opening tag used to demarcate genetic information within generated text. Default: "<genetic>".
    genetic_tag_closing : str
        Closing tag used to demarcate genetic information within generated text. Default: "</genetic>".
    base_data_path : str
        The root directory path where input data files are located. Default: "/flatiron_cgdb/2024_07_05/". *Note: This default path is system-specific and likely needs overriding.*
    event_table_name : str
        The base name (without extension) for the primary file or table containing event data. Default: "events".
    train_split_name : str
        Identifier for the training dataset split (e.g., used in file naming or data loading). Default: "train".
    bins_split_name : str
        Identifier for a data split used for binning tasks, often related to QA. Default: "5_equal_sized_bins".
    preamble_text : str
        Introductory text inserted at the beginning of the textual representation of a patient's record. Default: Explains structure and LOINC codes.
    constant_text : str
        Text used to introduce the section containing static demographic data in the textual patient record. Default: "\\n\\nStarting with demographic data:\\n".
    first_day_text : str
        Text used to introduce the events that occurred on the patient's very first recorded visit day. Default: "\\nOn the first visit, the patient experienced the following: \\n".
    event_day_preamble : str
        Text inserted before the description of events for visits subsequent to the first one. Default: "\\n".
    event_day_text : str
        Template text used to introduce events on subsequent visit days, indicating the time elapsed since the previous visit. Default: " weeks later, the patient visited and experienced the following: \\n".
    post_event_text : str
        Text appended after listing all events for a specific visit day. Default: ".\\n".
    forecasting_fval_prompt_start : str
        Initial text for prompts instructing a language model to predict future numerical values of specified variables over time. Default: Instructs prediction per cumulative week.
    forecasting_prompt_var_time : str
        Text segment used within forecasting prompts to specify the time frame (e.g., future weeks) for prediction. Default: " the future weeks ".
    forecasting_prompt_summarized_start : str
        Initial text for prompts that include a summary of the last known values of variables being forecasted. Default: "\\nThe last values of the variables in the input data are:\\n".
    forecasting_firstday_override : str
        Alternative introductory text for forecasting prompts, possibly used when only a subset of initial data is presented, hinting at omissions. Default: Mentions included events, potential omissions.
    forecasting_prompt_summarized_genetic : str
        Text used to introduce a summary section listing the last observed genetic event statuses within a forecasting prompt. Default: "\\n\\n\\n\\nHere we repeat the last observed values of each genetic event in the input data:\\n".
    forecasting_prompt_summarized_lot : str
        Text used to introduce a summary section describing the most recent line of therapy within a forecasting prompt. Default: "\\nThe most recent line of therapy:\\n".
    forecasting_tte_prompt_start : str
        Initial text for prompts instructing a language model to predict time-to-event (TTE) outcomes, specifically focusing on whether an event is censored. Default: Asks for censoring prediction.
    forecasting_tte_prompt_mid : str
        Middle text segment for TTE prompts, specifying the prediction horizon (in weeks) and asking about event occurrence status. Default: Specifies weeks and asks about occurrence.
    forecasting_tte_prompt_end : str
        Concluding text for TTE prompts, detailing the required output format for the prediction (censoring and occurrence). Default: Specifies format like "'Here is the prediction: the event (<name>) was [not] censored and [did not occur]/[occured].'".
    target_prompt_start : str
        Template string used to begin constructing the target (ground truth) output string for TTE tasks, includes placeholder for event name. Default: "\\nHere is the prediction: the event ({event_name}) was ".
    target_prompt_censor_true : str
        Text segment used in the TTE target output to indicate that the event *was* censored within the observation period. Default: "censored ".
    target_prompt_censor_false : str
        Text segment used in the TTE target output to indicate that the event *was not* censored. Default: "not censored ".
    target_prompt_before_occur : str
        Conjunction used in the TTE target output between the censoring status and the occurrence status. Default: "and ".
    target_prompt_occur : str
        Text segment used in the TTE target output to indicate that the event *did* occur. Default: "occurred.".
    target_prompt_not_occur : str
        Text segment used in the TTE target output to indicate that the event *did not* occur. Default: "did not occur.".
    qa_prompt_start : str
        Initial text for prompts instructing a model to perform a Quality Assurance (QA) task, specifically predicting value bins for future variable values. Default: Asks for bin prediction per week.
    qa_bins_start : str
        Text used within QA prompts to introduce the list of possible bins the model should choose from. Default: "\\tThe possible bins are: ".
    task_prompt_start : str
        Introductory text for multi-task prompts, explaining that multiple tasks follow and instructing the model on the required response format (e.g., prefixing each answer with 'Task X:'). Default: Explains multi-task format.
    task_prompt_each_task : str
        Template string used to introduce each individual task within a multi-task prompt, includes placeholder for task number. Default: "Task {task_nr} is ".
    task_prompt_end : str
        Concluding text for the overall multi-task prompt setup. Default: "" (empty string).
    task_prompt_forecasting : str
        Identifier text appended to `task_prompt_each_task` to specify a forecasting sub-task. Default: "forecasting:".
    task_prompt_forecasting_qa : str
        Identifier text appended to `task_prompt_each_task` to specify a forecasting QA (binning) sub-task. Default: "forecasting QA:".
    task_prompt_events : str
        Identifier text appended to `task_prompt_each_task` to specify a time-to-event prediction sub-task. Default: "time to event prediction:".
    task_prompt_custom : str
        Identifier text appended to `task_prompt_each_task` to specify a custom-defined sub-task. Default: " a custom task:".
    task_target_start : str
        Template string used to begin the target (ground truth) output corresponding to a specific task number in a multi-task setting. Default: "Task {task_nr} is ".
    task_target_end : str
        Concluding text for the target output of a specific task within a multi-task response. Default: "" (empty string).
    decimal_precision : int
        Number of decimal places to use when rounding numerical values (e.g., lab results) during text conversion. Default: 2.
    event_category_preamble_mapping_override : dict | None
        Optional dictionary to override the introductory text used before listing events of a specific category on a given day. Structure: `{<event_category>: <preamble_string>}`. Default: None.
    event_category_and_name_replace_override : dict | None
        Optional nested dictionary to define specific replacements for event descriptions based on category and name. Allows replacing the entire event string and defining a value for reverse mapping.
        Structure: `{<event_category>: {<event_name>: {"full_replacement_string": <str>, "reverse_string_value": <str>}}}`. Default: None.
    always_keep_first_visit : bool
        Flag indicating whether the events from the very first visit should always be included in the patient history, regardless of token budget constraints. Default: True.
    seed : int
        Seed value for random number generators to ensure reproducibility in processes like data splitting or sampling. Default: 768921.
    nr_tokens_budget_padding : int
        Number of tokens reserved as a buffer when calculating token budgets, ensuring outputs don't exceed limits. May need adjustment based on model/task. Default: 200.
    tokenizer_to_use : str
        Identifier string for the tokenizer model to be used for counting tokens (e.g., for budget calculations). Should correspond to a model available in the environment (e.g., from Hugging Face). Default: 'meta-llama/Meta-Llama-3-8B'.
    wandb_project : str
        Name of the Weights & Biases project to log experiments and results to. Default: "cgdb-pancancer-conversion".
    constant_use_cgdb_preprocessing : bool
        Flag indicating whether to use specific preprocessing steps defined for the CGDB dataset when handling constant (demographic) data. Default: True.
    constant_columns_to_use : list[str]
        List of column names from the constant (demographic) data source to be included in the processing and text conversion. *Note: Age might be handled separately.* Default: ["race", "gender", "ethnicity", "indication"].
    override_with_custom_dataset : bool
        Flag to enable the use of custom dataset paths instead of the default `base_data_path` derived paths. If True, relevant `override_datasets_*` attributes must be set. Default: False.
    override_datasets_events : str | None
        Path to the custom events dataset file/table, used if `override_with_custom_dataset` is True. Default: None.
    override_datasets_constant : str | None
        Path to the custom constant (demographic) dataset file/table, used if `override_with_custom_dataset` is True. Default: None.
    override_datasets_molecular : str | None
        Path to the custom molecular (e.g., genetic) dataset file/table, used if `override_with_custom_dataset` is True. Can be None if not applicable. Default: None.
    override_constant_description : str | None
        Path to a custom file describing the constant data columns, used if `override_with_custom_dataset` is True. Default: None.
    override_statistics : str | None
        Path to a custom file containing pre-computed statistics (e.g., for normalization or binning), used if `override_with_custom_dataset` is True. Can be None if not applicable. Default: None.
    """

    def __init__(self):

        # --- Import data parameters ---
        self.date_cutoff = pd.to_datetime("2024-03-31")  # Default cutoff date used for CGDB; used e.g. to determine censoring cutoff

        # --- Core Column Names ---
        self.date_col: str = "date"
        self.patient_id_col: str = "patientid"
        self.event_category_col: str = "event_category"
        self.event_name_col: str = "event_name"
        self.event_descriptive_name_col: str = "event_descriptive_name"
        self.event_value_col: str = "event_value"
        self.source_col : str = "source"
        self.meta_data_col : str = "meta_data"

        # --- Specific Category/Type Column Names ---
        self.lot_date_col: str = "lot_date"
        self.lot_name_col: str = "line_name"
        self.event_value_lot_start: str = "LoT start"
        self.skip_future_lot_filtering : bool = False  # Whether to skip filtering future LoT events, by default False.
        self.lot_concatenate_descriptive_and_value : bool = False  # If true, concatenate descriptive name and value for LoT events, by default False (only event_vale.)
        self.lot_concatenate_string : str = " - "  # String used to concatenate descriptive name and value for LoT events, by default " - ".

        # --- Biomarker Column Names ---
        self.biomarker_category_col: str = "biomarker_category"
        self.biomarker_event_col: str = "biomarker_event"
        self.biomarker_value_col: str = "biomarker_value"
        self.biomarker_descriptive_name_col: str = "biomarker_descriptive_name"

        # --- Specific Event Categories / Values / Sources ---
        self.event_category_lot: str = "lot"
        self.event_category_death: str = "death"
        self.event_category_labs: str = "lab"
        self.source_genetic: str = "genetic"
        self.genetic_skip_text_value: str = "present"
        self.genetic_tag_opening : str = "<genetic>"
        self.genetic_tag_closing : str = "</genetic>"

        # --- Data Paths, Tables, and Splits ---
        # Note: Default path is specific, override as needed.
        self.base_data_path: str = "/flatiron_cgdb/2024_07_05/"
        self.event_table_name: str = "events"
        self.train_split_name: str = "train"
        self.bins_split_name: str = "5_equal_sized_bins"

        # --- Text Conversion Prompts ---
        self.preamble_text: str = (
            "The following is a patient, starting with the demographic data, "
            "following visit by visit everything that the patient experienced. "
            "All lab codes refer to LOINC codes."
        )
        self.constant_text: str = "\n\nStarting with demographic data:\n"
        self.first_day_text: str = "\nOn the first visit, the patient experienced the following: \n"
        self.event_day_preamble: str = "\n"
        self.event_day_text: str = " weeks later, the patient visited and experienced the following: \n"
        self.post_event_text: str = ".\n"
        self.genetic_empty_text : str = "No genetic data available."

        # --- Forecasting Prompts (General & Summarization) ---
        self.forecasting_fval_prompt_start: str = (
            "\nYour task is to predict the future values of the following variables "
            "for each cumulative week starting from the last visit:\n"
        )
        self.forecasting_prompt_var_time: str = " the future weeks "
        self.forecasting_prompt_summarized_start: str = "\nThe last values of the variables in the input data are:\n"
        self.forecasting_firstday_override: str = (
            "\nThe following events are included in the input data, though "
            "potentially there are more which were omitted. Starting with: \n"
        )
        self.forecasting_prompt_summarized_genetic: str = (
            "\n\n\n\nHere we repeat the last observed values of each "
            "genetic event in the input data:\n"
        )
        self.forecasting_prompt_summarized_lot: str = ("\nThe most recent line of therapy:\n")

        # --- Forecasting Prompts (Time-to-Event Specific) ---
        self.forecasting_tte_prompt_start: str = (
            "\nYour task is to predict whether the following event was censored "
        )
        self.forecasting_tte_prompt_mid: str = (
            " weeks from the last clinical visit and whether the event occured or not: "
        )
        self.forecasting_tte_prompt_end: str = (
            ".\nPlease provide your prediction in the following format: "
            "'Here is the prediction: the event (<name of event>) was [not] censored "
            "and [did not occur]/[occured].'"
        )

        # --- Target Output Prompts (Time-to-Event) ---
        self.target_prompt_start: str = "\nHere is the prediction: the event ({event_name}) was "
        self.target_prompt_censor_true: str = "censored "
        self.target_prompt_censor_false: str = "not censored "
        self.target_prompt_before_occur: str = "and "
        self.target_prompt_occur: str = "occurred."
        self.target_prompt_not_occur: str = "did not occur."

        # --- QA Prompts (Binning) ---
        self.qa_prompt_start: str = (
            "\nYour task is to predict the appropriate bins for the future values of "
            "the following variables for each cumulative week starting from the date of the last visit:"
        )
        self.qa_bins_start: str = "\tThe possible bins are: "

        # --- Multi-Task Prompts ---
        self.task_prompt_start: str = (
            "\nYou will now have multiple tasks to complete. Please answer for each "
            "task in the same order as they are presented. Before every response state the "
            "task nr, e.g. 'Task 2:'.\n\n"
        )
        self.task_prompt_each_task: str = ("Task {task_nr} is ")
        self.task_prompt_end: str = ("")
        self.task_prompt_forecasting: str = "forecasting:"
        self.task_prompt_forecasting_qa: str = "forecasting QA:"
        self.task_prompt_events: str = "time to event prediction:"
        self.task_prompt_custom: str = " a custom task:"
        self.task_target_start: str = ("Task {task_nr} is ")
        self.task_target_end: str = ("")


        # --- Overrides -----
        self.decimal_precision = 2
        self.event_category_preamble_mapping_override = None
        # Override for the event category preamble mapping (default is None).
        # Structure is {<event_category>: <preamble_string>}

        self.event_category_and_name_replace_override = None
        # dict, optional
        #    Override for the event category and name replace mapping (default is None).
        #    Structure is {<event_category>: {
        #        <event_name>: {
        #            "full_replacement_string": <full_replacement_string>,
        #            "reverse_string_value": <reverse_string_value>
        #            }
        #        }
        #    }

        self.always_keep_first_visit : bool = True  # Whether to always keep the first visit in the patient history, by default True.

        # Seeds
        self.seed = 768921  # I like both of these numbers

        # Token budgets
        self.nr_tokens_budget_padding : int = 200  # Might need to be set to 500 for pretrain

        # Tokenizers for counting
        self.tokenizer_to_use : str = 'meta-llama/Meta-Llama-3-8B'

        # --- Logging and WandB ---
        self.wandb_project : str = "cgdb-pancancer-conversion"


        # --- Processing of constant ---
        self.constant_use_cgdb_preprocessing : bool = True
        self.constant_columns_to_use : list = ["race", "gender", "ethnicity", "indication"]   # Which columns to use from the constant data - note, in CGDB age is added in code
        self.constant_birthdate_column : str = None

        # --- Allow overriding of which datasets to load ----
        self.override_with_custom_dataset : bool = False   # Whether to override the datasets with custom ones - Note, need to set for all of them then
        self.override_datasets_events : str = None
        self.override_datasets_constant : str = None
        self.override_datasets_molecular : str = None   # Can be left none
        self.override_constant_description : str = None
        self.override_statistics : str = None # Can be left none
