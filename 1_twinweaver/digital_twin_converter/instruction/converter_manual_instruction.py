import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datetime import datetime
from collections import Counter

from digital_twin_converter.common.data_manager import SingleIndicationDataManager
from digital_twin_converter.common.converter_manual_base import ConverterManualBase
from digital_twin_converter.instruction.converter_manual_forecasting import (
    ConverterManualForecasting
)
from digital_twin_converter.instruction.converter_manual_events import (
    ConverterManualEvents
)
from digital_twin_converter.instruction.converter_manual_forecasting_qa import (
    ConverterManualForecastingQA
)
from digital_twin_converter.common.config import Config



class ConverterManualInstruction(ConverterManualBase):
    """
    Orchestrates the conversion of patient data into multi-task instruction-following
    prompts and target strings, suitable for training language models.

    This class acts as a high-level controller, utilizing specialized sub-converters
    (`ConverterManualForecasting`, `ConverterManualEvents`, `ConverterManualForecastingQA`)
    to generate prompts and targets for different task types (e.g., value forecasting,
    event prediction, question answering about forecasting). It then combines these
    individual task components into a single, structured multi-task instruction prompt
    and a corresponding multi-task target answer string, using templates defined in
    the provided `Config` object. It also handles the reverse conversion of model
    outputs back into structured data and provides methods for aggregation and comparison.
    """

    def __init__(self,
                constant_description : pd.DataFrame,
                nr_tokens_budget_total : int,
                config : Config,) -> None:
        """
        Initializes the ConverterManualInstruction class.

        Sets up the main converter by initializing base class settings, storing essential
        data descriptions and token budgets, configuring the tokenizer, loading multi-task
        prompt templates from the config, and initializing instances of the specialized
        sub-converters required for different task types.

        Parameters
        ----------
        constant_description : pd.DataFrame
            A DataFrame containing descriptions for constant patient features. Passed down
            to sub-converters that might use it.
        nr_tokens_budget_total : int
            The total token budget available for the generated context + instruction prompt.
            Used for dynamically selecting history length.
        config : Config
            A configuration object supplying tokenizer details, prompt templates (for tasks
            and context generation), padding values, and other settings.
        """

        super().__init__(config)

        # Set vars
        self.constant_description = constant_description
        self.nr_tokens_budget_total = nr_tokens_budget_total

        # Set seed
        np.random.seed(self.seed)

        # Set forecasting prompt texts using overrides or config defaults
        self.forecasting_prompt_start = config.forecasting_fval_prompt_start
        self.forecasting_prompt_var_time = config.forecasting_prompt_var_time
        self.forecasting_prompt_summarized_start = config.forecasting_prompt_summarized_start
        self.forecasting_prompt_summarized_genetic = config.forecasting_prompt_summarized_genetic
        self.forecasting_prompt_summarized_lot = config.forecasting_prompt_summarized_lot

        # Initialize tokenizer and related settings
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_to_use)
        self.nr_tokens_budget_padding = self.config.nr_tokens_budget_padding
        self.always_keep_first_visit = self.config.always_keep_first_visit

        # Set multi-task prompt texts using overrides or config defaults
        self.task_prompt_start = config.task_prompt_start
        self.task_prompt_each_task = config.task_prompt_each_task
        self.task_prompt_end = config.task_prompt_end
        self.task_target_start = config.task_target_start
        self.task_target_end = config.task_target_end


        # Initialize converters - can this be done more elegantly?
        self.converter_forecasting = ConverterManualForecasting(
            constant_description,
            nr_tokens_budget_total,
            config=config,
        )

        self.converter_events = ConverterManualEvents(
            config,
            constant_description,
            nr_tokens_budget_total,
        )

        self.converter_forecasting_qa = ConverterManualForecastingQA(
            constant_description,
            nr_tokens_budget_total,
            config=config,
        )



    def _generate_prompt(self, all_tasks : list) -> tuple:
        """
        Constructs the multi-task instruction prompt string from individual task prompts.

        Takes a list of generated tasks (where each task is a tuple containing its specific
        prompt, target, metadata, and type identifier) and formats them into a single
        instruction prompt using wrapper templates defined in the config (e.g., numbering
        tasks, adding introductory/concluding text).

        Parameters
        ----------
        all_tasks : list[tuple[str, str, dict, str]]
            A list where each element represents a single task. The tuple structure is
            expected to be: (task_prompt_string, task_target_string, task_metadata_dict, task_type_identifier_string).

        Returns
        -------
        str
            The combined multi-task instruction prompt string ready to be prepended
            with context (patient history, etc.).
        """

        ret_str = self.task_prompt_start

        for idx in range(len(all_tasks)):

            task = all_tasks[idx]

            ret_str += self.task_prompt_each_task.format(task_nr = idx + 1)
            ret_str += task[3]
            ret_str += task[0]
            ret_str += "\n\n"

        ret_str += self.task_prompt_end

        return ret_str

    def _generate_target_string(self, all_tasks : list) -> tuple:
        """
        Constructs the multi-task target answer string and aggregates metadata.

        Takes a list of generated tasks and combines their individual target strings
        into a single, formatted multi-task answer string using wrapper templates
        (e.g., numbering answers corresponding to tasks). It also collects the
        metadata dictionaries from each individual task into a list.

        Parameters
        ----------
        all_tasks : list[tuple[str, str, dict, str]]
            A list where each element represents a single task. The tuple structure is
            expected to be: (task_prompt_string, task_target_string, task_metadata_dict, task_type_identifier_string).

        Returns
        -------
        tuple
            - str: The combined multi-task target answer string.
            - list[dict]: A list containing the metadata dictionaries from each
                          individual task included in the target string.
        """

        ret_str = ""
        ret_meta = []

        for idx in range(len(all_tasks)):

            task = all_tasks[idx]
            task_type = task[3]
            task_text = task[1]

            ret_str += self.task_target_start.format(task_nr = idx + 1)
            ret_str += task_type
            ret_str += task_text
            ret_str += self.task_target_end
            ret_str += "\n"


            curr_meta = task[2]
            curr_meta["task_type"] = task_type
            ret_meta.append(task[2])

        return ret_str, ret_meta


    def get_nr_tokens(self, curr_string):
        """
        Calculates the number of tokens in a given string using the initialized tokenizer.

        Parameters
        ----------
        curr_string : str
            The string to tokenize.

        Returns
        -------
        int
            The number of tokens generated by the tokenizer for the input string.
        """
        return len(self.tokenizer(curr_string)["input_ids"])


    def forward_conversion(self,
                           forecasting_splits : list,
                           event_splits : list,
                           variable_stats : pd.DataFrame,
                           override_mode_to_select_forecasting : str = None) -> dict:
        """
        Generates a multi-task instruction prompt and target answer from patient data splits.

        This method orchestrates the creation of a training example. It randomly selects
        scenarios (splits) for event prediction and forecasting tasks from the provided lists.
        It then uses the specialized sub-converters to generate prompt/target pairs for these
        tasks (including potentially forecasting QA tasks). These individual tasks are shuffled
        and combined into a multi-task instruction prompt and a multi-task target answer using
        helper methods. Patient history (constant features, time-series events, summarized info)
        is formatted and prepended to the instruction prompt, respecting the total token budget.

        Parameters
        ----------
        forecasting_splits : list[dict]
            List of dictionaries, each representing a potential forecasting task scenario
            (containing patient history up to a split date and future target values).
        event_splits : list[dict]
            List of dictionaries, each representing a potential event prediction task scenario
            (containing patient history up to a split date and event outcome/censoring info).
        variable_stats : pd.DataFrame
            DataFrame containing statistics about variables, used by the forecasting QA converter.
        override_mode_to_select_forecasting : str, optional
            If provided, forces the selection mode for forecasting tasks ('forecasting',
            'forecasting_qa', or 'both'). If None, the mode is chosen randomly. Defaults to None.

        Returns
        -------
        dict
            A dictionary containing the final formatted data:
            - 'instruction': The complete input string for the model (context + multi-task prompt).
            - 'answer': The complete target string for the model (multi-task answer).
            - 'meta': A dictionary holding metadata including patient ID, structured constant and
                      history data used, split date, combined metadata from sub-converters, and
                      a list of detailed metadata for each individual task generated ('target_meta_detailed').

        Raises
        ------
        AssertionError
            If input splits have inconsistent Line of Therapy (LOT) dates or split dates,
            or if no tasks are generated.
        """


        #: make assertions that data has same split and lot date
        all_lot_dates_events = [x["lot_date"] for x in event_splits]
        all_lot_dates_forecasting = [x["lot_date"] for x in forecasting_splits]
        assert len(set(all_lot_dates_events) | set(all_lot_dates_forecasting)) == 1
        all_split_dates_events = [x["split_date_included_in_input"] for x in event_splits]
        all_split_dates_forecasting = [x["split_date_included_in_input"] for x in forecasting_splits]
        assert len(set(all_split_dates_events) | set(all_split_dates_forecasting)) == 1
        # assert that the split date is after or equal to lot date (checking 0, since all same)
        assert len(event_splits) == 0 or all_split_dates_events[0] >= all_lot_dates_events[0]


        #: randomly select the number of events and which splits to use and generate events
        min_nr_events = 1 if len(forecasting_splits) == 0 else 0
        nr_events_to_use = np.random.randint(min_nr_events, len(event_splits) + 1)
        event_splits_to_use = np.random.choice(event_splits, nr_events_to_use, replace=False)

        event_converted = []
        for event_split in event_splits_to_use:
            prompt, target, target_meta = self.converter_events.forward_conversion(event_split)
            event_converted.append((prompt, target, target_meta, self.config.task_prompt_events))

        #: then randomly select the forecasting splits and generate forecasting
        forecasting_tasks = []
        forecasting_qa_tasks = []

        if len(forecasting_splits) > 0:

            if override_mode_to_select_forecasting is None:
                #: randomly select whether to do forecasting, forecasting QA or both
                mode_to_select = np.random.choice(["forecasting", "forecasting_qa", "both"], 1, replace=False)[0]
            else:
                mode_to_select = override_mode_to_select_forecasting

            # Randomly select the forecasting split
            if mode_to_select == "forecasting" or mode_to_select == "both":
                forecasting_split_to_use = np.random.choice(forecasting_splits, 1, replace=False)[0]
                ret_forecasting = self.converter_forecasting.forward_conversion(forecasting_split_to_use)
                prompt_forecasting, target_forecasting, target_meta_forecasting = ret_forecasting
                forecasting_tasks.append((prompt_forecasting, target_forecasting, target_meta_forecasting,
                                          self.config.task_prompt_forecasting))

            #: then randomly select the QA splits and generate QA
            if mode_to_select == "forecasting_qa" or mode_to_select == "both":
                forecasting_qa_split_to_use = np.random.choice(forecasting_splits, 1, replace=False)[0]
                ret_forecasting_qa = self.converter_forecasting_qa.forward_conversion(forecasting_qa_split_to_use,
                                                                                      variable_stats=variable_stats)
                prompt_forecasting_qa, target_forecasting_qa, target_meta_forecasting_qa = ret_forecasting_qa
                forecasting_qa_tasks.append((prompt_forecasting_qa, target_forecasting_qa, target_meta_forecasting_qa,
                                             self.config.task_prompt_forecasting_qa))

        #: determine random order of all tasks
        all_tasks = forecasting_tasks + forecasting_qa_tasks + event_converted
        all_tasks = np.random.permutation(all_tasks)
        assert len(all_tasks) > 0, "No tasks generated!"

        #: generate overall prompt
        prompt = self._generate_prompt(all_tasks)

        #: generate overall target
        target_str, target_meta = self._generate_target_string(all_tasks)

        #: aggregate all meta needed and add into target_meta for the summarized row
        combined_meta = {
            "dates_per_variable": {},
            "variable_name_mapping": {},
        }
        for curr_meta in target_meta:
            if "dates_per_variable" in curr_meta:
                combined_meta["dates_per_variable"].update(curr_meta["dates_per_variable"])
            if "variable_name_mapping" in curr_meta:
                combined_meta["variable_name_mapping"].update(curr_meta["variable_name_mapping"])

        # Since the history and constant are the same, we can just use the first
        data_to_use = forecasting_splits[0] if len(forecasting_splits) > 0 else event_splits[0]
        events_until_split = data_to_use["events_until_split"]
        constant = data_to_use["constant_data"]
        patientid = data_to_use["constant_data"]["patientid"]
        split_date_included_in_input = data_to_use["split_date_included_in_input"]

        #: generate constant string
        constant, constant_description = self._preprocess_constant_date(events_until_split,
                                                                        constant,
                                                                        self.constant_description)
        constant_string = self._get_constant_string(constant, constant_description)

        #: generate summarized row string
        summarized_row_string = self._generate_summarized_row_string(events_until_split,
                                                                     combined_meta)

        #: get current budget
        budget_total = self.nr_tokens_budget_total
        budget_total -= self.get_nr_tokens(prompt)
        budget_total -= self.get_nr_tokens(constant_string)
        budget_total -= self.get_nr_tokens(summarized_row_string)
        budget_total -= self.get_nr_tokens(target_str)
        budget_total -= self.nr_tokens_budget_padding

        #: select events within token budget
        patient_history = self._get_all_most_recent_events_within_budget(events_until_split,
                                                                         budget_total)

        #: generate history string
        patient_history_processed = patient_history.copy()
        patient_history_processed = self._preprocess_events(patient_history_processed)
        history_str = self._get_event_string(patient_history,
                                             use_accumulative_dates = False)

        #: generate final string
        input_string = self.config.preamble_text + constant_string + history_str + summarized_row_string + prompt

        #: generate return & meta (incl. structured data)
        ret = {
            "instruction": input_string,
            "answer": target_str,
            "meta" : {
                "patientid": patientid,
                "constant_data": constant,
                "history_data": patient_history,
                "split_date_included_in_input" : split_date_included_in_input,
                "combined_meta": combined_meta,
                "target_meta_detailed": target_meta,
            }
        }
        return ret



    def forward_conversion_inference(self,
                                     forecasting_split : dict = None,
                                     forecasting_future_weeks_per_variable : dict = None,
                                     event_split : dict = None,
                                     custom_tasks : list = None) -> dict:
        """
        Generates a multi-task instruction prompt suitable for inference.

        Constructs the input prompt for the model based on provided task scenarios
        (forecasting, event prediction, and/or custom text tasks). It does not generate
        a target 'answer' string. Patient history context (constant, time-series, summarized)
        is generated and prepended, respecting token limits.

        Parameters
        ----------
        forecasting_split : dict, optional
            Data for a forecasting task scenario. Defaults to None.
        forecasting_future_weeks_per_variable : dict, optional
            Specifies future time points (in weeks) for forecasting specific variables.
            Used by the forecasting sub-converter during inference. Defaults to None.
        event_split : dict, optional
            Data for an event prediction task scenario. Defaults to None.
        custom_tasks : list[str], optional
            A list of strings, each representing a custom task/question to include
            in the prompt. Defaults to None.

        Returns
        -------
        dict
            A dictionary containing:
            - 'instruction': The complete input string for the model (context + multi-task prompt).
            - 'answer': None (as this is for inference).
            - 'meta': Metadata including patient ID, structured data used for context,
                      split date, and metadata associated with the generated tasks.

        Raises
        ------
        AssertionError
            If both `forecasting_split` and `event_split` are None, or if their
            split dates do not match when both are provided.
        """

        #: make assertions that data has same split date
        if forecasting_split is not None and event_split is not None:
            assert forecasting_split["split_date_included_in_input"] == event_split["split_date_included_in_input"], (
                "Split dates do not match!"
            )
        assert forecasting_split is not None or event_split is not None, "No data provided!"

        #: convert events, if needed
        event_prompt_str = event_meta = None
        if event_split is not None:
            event_prompt_str, event_meta = self.converter_events.forward_conversion_inference(event_split)

        #: convert forecasting, if needed
        forecasting_prompt_str = forecasting_meta = None
        if forecasting_split is not None:
            f_ret = self.converter_forecasting.forward_conversion_inference(forecasting_split,
                                                                            future_weeks_per_variable=forecasting_future_weeks_per_variable)
            forecasting_prompt_str, forecasting_meta = f_ret

        #: generate meta and aggregate
        all_tasks = []
        target_meta = []
        if event_prompt_str is not None:
            all_tasks.append((event_prompt_str, None, event_meta, self.config.task_prompt_events))
            target_meta.append(event_meta)
        if forecasting_prompt_str is not None:
            all_tasks.append((forecasting_prompt_str, None, forecasting_meta, self.config.task_prompt_forecasting))
            target_meta.append(forecasting_meta)

        # Add in custom tasks
        if custom_tasks is not None:
            for custom_task in custom_tasks:
                all_tasks.append((custom_task, None, None, self.config.task_prompt_custom + "\n"))
                target_meta.append({})

        # Generate prompt
        prompt = self._generate_prompt(all_tasks)

        #: generate constant string
        data_to_use = forecasting_split if forecasting_split is not None else event_split
        events_until_split = data_to_use["events_until_split"]
        constant = data_to_use["constant_data"]
        patientid = data_to_use["constant_data"]["patientid"]
        split_date_included_in_input = data_to_use["split_date_included_in_input"]
        constant, constant_description = self._preprocess_constant_date(events_until_split,
                                                                        constant,
                                                                        self.constant_description)
        constant_string = self._get_constant_string(constant, constant_description)

        #: generate summarized row string
        combined_meta = {
            "dates_per_variable": {},
            "variable_name_mapping": {},
        }
        for curr_meta in target_meta:
            if "dates_per_variable" in curr_meta:
                combined_meta["dates_per_variable"].update(curr_meta["dates_per_variable"])
            if "variable_name_mapping" in curr_meta:
                combined_meta["variable_name_mapping"].update(curr_meta["variable_name_mapping"])
        summarized_row_string = self._generate_summarized_row_string(events_until_split,
                                                                     combined_meta)

        #: get current budget
        budget_total = self.nr_tokens_budget_total
        budget_total -= self.get_nr_tokens(prompt)
        budget_total -= self.get_nr_tokens(constant_string)
        budget_total -= self.get_nr_tokens(summarized_row_string)
        budget_total -= self.nr_tokens_budget_padding

        #: select events within token budget
        patient_history = self._get_all_most_recent_events_within_budget(events_until_split,
                                                                         budget_total)

        #: generate history string
        patient_history_processed = patient_history.copy()
        patient_history_processed = self._preprocess_events(patient_history_processed)
        history_str = self._get_event_string(patient_history,
                                             use_accumulative_dates = False)

        #: generate final string
        input_string = self.config.preamble_text + constant_string + history_str + summarized_row_string + prompt

        #: generate return & meta (incl. structured data)
        ret = {
            "instruction": input_string,
            "answer": None,
            "meta" : {
                "patientid": patientid,
                "constant_data": constant,
                "history_data": patient_history,
                "split_date_included_in_input" : split_date_included_in_input,
                "combined_meta": combined_meta,
                "target_meta_detailed": target_meta,
            }
        }
        return ret


    def generate_target_manual(self, reverse_converted, split_date, events_until_split):
        """
        Reconstructs a multi-task target string from previously reverse-converted structured data.

        This method takes a list of dictionaries (where each dictionary represents the
        parsed result of a single task from a model's output) and generates the corresponding
        multi-task target string. It calls the appropriate sub-converter's manual target
        generation method for each task type.

        Parameters
        ----------
        reverse_converted : list[dict]
            A list of dictionaries, each resulting from the `reverse_conversion` method,
            containing keys 'task_type' and 'result' (the structured data).
        split_date : datetime
            The reference split date, needed for reconstructing forecasting targets.
        events_until_split : pd.DataFrame
            Patient's historical event data up to the split date, needed for context
            by some sub-converters' target generation (e.g., forecasting).

        Returns
        -------
        tuple
            - str: The reconstructed multi-task target string.
            - list[dict]: A list containing the metadata generated during the reconstruction
                          of each individual task's target string.
            - list[dict]: The input `reverse_converted` list, potentially updated with
                          'original_text' (the reconstructed target snippet for each task)
                          and 'target_meta'.
        """

        #: go through all reverse converted and generate only the target string
        individual_targets = []

        for idx, reverse_converted_dic in enumerate(reverse_converted):

            if reverse_converted_dic["task_type"] == self.config.task_prompt_events:

                occurred = reverse_converted_dic["result"]["occurred"].iloc[0]
                censored = reverse_converted_dic["result"]["censoring"].iloc[0]
                target_name = reverse_converted_dic["result"]["target_name"].iloc[0]

                ret_prompt, ret_meta = self.converter_events.generate_target_manual(target_name=target_name,
                                                                   event_occurred=occurred,
                                                                   event_censored=censored)
                individual_targets.append((None, ret_prompt, {}, self.config.task_prompt_events))
                reverse_converted[idx]["original_text"] = ret_prompt

            if reverse_converted_dic["task_type"] == self.config.task_prompt_forecasting:

                ret_prompt, ret_meta = self.converter_forecasting.generate_target_manual(
                        split_date_included_in_input=split_date,
                        events_until_split=events_until_split,
                        target_events_after_split=reverse_converted_dic["result"]
                    )
                individual_targets.append((None, ret_prompt, {}, self.config.task_prompt_forecasting))
                reverse_converted[idx]["original_text"] = ret_prompt
                reverse_converted[idx]["target_meta"] = ret_meta

            if reverse_converted_dic["task_type"] == self.config.task_prompt_custom:
                individual_targets.append((None, reverse_converted_dic["result"], {}, self.config.task_prompt_custom))
                reverse_converted[idx]["original_text"] = reverse_converted_dic["result"]

        #: then aggegrate all target strings with own
        target_str, target_meta = self._generate_target_string(individual_targets)

        # Return
        return target_str, target_meta, reverse_converted


    def aggregate_multiple_responses(self, all_responses : list[list[dict]]) -> dict:
        """
        Aggregates multiple model responses for the same multi-task prompt.

        Takes a list of responses, where each response is itself a list of parsed results
        (one dictionary per task from the `reverse_conversion` method). It groups the results
        for each task position across all responses and then calls the appropriate
        sub-converter's aggregation method to determine the most likely result for that task.

        Parameters
        ----------
        all_responses : list[list[dict]]
            A list where each inner list corresponds to one full model response (parsed by
            `reverse_conversion`) to the multi-task prompt. Each dictionary in the inner list
            should contain 'task_type' and 'result'.

        Returns
        -------
        list[dict]
            A list of dictionaries, one for each task position in the original prompt. Each
            dictionary contains:
            - 'task_type': The type of the task at this position.
            - 'result': The aggregated result (e.g., the most common structured data).
            - 'original_text': None (as this aggregates structured results, not text).
            - 'aggregated_meta': Metadata about the aggregation process (e.g., distribution
                               of responses for this task), provided by the sub-converter.

        Raises
        ------
        ValueError
            If `all_responses` is empty or if the inner lists have inconsistent lengths.
        """

        #: first restructure the format to get appropriate responses together
        ret_order = {
            idx : [] for idx in range(len(all_responses[0]))
        }

        for response_list in all_responses:
            # This is one response, which is a list since we may have multiple tasks per response
            for idx, response in enumerate(response_list):
                ret_order[idx].append(response)

        #: for each, aggregate by correctly calling the correct converter
        ret_aggregated = []

        for idx in range(max(ret_order.keys()) + 1):

            task_type = ret_order[idx][0]["task_type"]
            results = [x["result"] for x in ret_order[idx]]

            if task_type == self.config.task_prompt_events:
                ret, meta = self.converter_events.aggregate_multiple_responses(results)
            if task_type == self.config.task_prompt_forecasting:
                ret, meta = self.converter_forecasting.aggregate_multiple_responses(results)
            if task_type == self.config.task_prompt_forecasting_qa:
                ret, meta = self.converter_forecasting_qa.aggregate_multiple_responses(results)
            if task_type == self.config.task_prompt_custom:

                # Aggregate by taking the most common answer
                all_responses_count = Counter([x["original_text"] for x in results])
                most_common_response = all_responses_count.most_common(1)[0][0]
                ret = most_common_response
                # Get all responses and their counts as DF
                all_responses_count = pd.DataFrame(all_responses_count.items(), columns=["response", "count"])
                all_responses_count = all_responses_count.sort_values("count", ascending=False).reset_index(drop=True)
                meta = {
                    "all_custom_responses": all_responses_count,
                }

            # Set as appropriate dic
            ret_dic = {
                "task_type": task_type,
                "result": ret,
                "original_text": None,
                "aggregated_meta" : meta,
            }
            ret_aggregated.append(ret_dic)

        #: return
        return ret_aggregated





    def reverse_conversion(self,
                           target_string : str,
                           data_manager : SingleIndicationDataManager,
                           split_date : datetime,
                           patientid : str = None,
                           inference_override : bool = False) -> list[dict]:
        """
        Parses a multi-task model output string back into a list of structured results.

        Splits the input string based on task markers (e.g., "Task 1:", "Task 2:").
        For each segment, it identifies the task type using configured prompt identifiers
        and calls the corresponding sub-converter's `reverse_conversion` method to parse
        the text into structured data (e.g., a DataFrame).

        Parameters
        ----------
        target_string : str
            The complete multi-task output string generated by the language model.
        data_manager : SingleIndicationDataManager
            The data manager instance, needed by some sub-converters for context
            (e.g., unique event mappings).
        split_date : datetime
            The reference split date, required by some sub-converters (e.g., forecasting)
            to interpret time-related outputs correctly.
        patientid : str, optional
            If provided, this patient ID is added to the resulting structured data
            for each task. Defaults to None.
        inference_override : bool, optional
            If True, allows processing even if a task type cannot be strictly identified
            (e.g., by falling back to heuristics). Primarily for inference robustness
            where model output might be slightly malformed. Defaults to False.

        Returns
        -------
        list[dict]
            A list of dictionaries, one for each task identified and parsed in the
            `target_string`. Each dictionary contains:
            - 'task_type': The identified type of the task.
            - 'original_text': The raw text segment corresponding to this task's answer.
            - 'result': The structured data parsed from 'original_text' by the appropriate
                        sub-converter (e.g., a pandas DataFrame).

        Raises
        ------
        ValueError
            If `inference_override` is False and the type of a task segment cannot be
            determined using the configured prompt identifiers.
        """

        #: split up by task
        all_tasks = target_string.split("Task")
        ret_list = []

        for curr_task in all_tasks:

            # Basic preprocessing
            curr_task = curr_task.strip()
            if len(curr_task) == 0:
                continue

            #: determine which task it is
            task_type = None
            standard_extraction = True

            if self.config.task_prompt_forecasting in curr_task:
                task_type = self.config.task_prompt_forecasting
            elif self.config.task_prompt_forecasting_qa in curr_task:
                task_type = self.config.task_prompt_forecasting_qa
            elif self.config.task_prompt_events in curr_task:
                task_type = self.config.task_prompt_events
            elif "custom" in curr_task:
                task_type = self.config.task_prompt_custom
            else:

                #: Try determining by whether the task contains a "censored"
                if "censored" in curr_task:
                    task_type = self.config.task_prompt_events
                    standard_extraction = False
                elif inference_override is False:
                    raise ValueError("Could not determine task type")

            #: extract the relevant parts
            if standard_extraction:
                curr_task_text = curr_task.split(task_type)[1]
            else:
                curr_task_text = ":".join(curr_task.split(":")[1:])

                # Often we have "[""]" which need to be removed
                curr_task_text = curr_task_text.replace("[", "")
                curr_task_text = curr_task_text.replace("]", "")

            #: pass to correct reverse converter
            if task_type == self.config.task_prompt_forecasting:
                ret = self.converter_forecasting.reverse_conversion(curr_task_text,
                                                                    data_manager.unique_events,
                                                                    split_date)
            elif task_type == self.config.task_prompt_forecasting_qa:
                ret = self.converter_forecasting_qa.reverse_conversion(curr_task_text,
                                                                    data_manager.unique_events,
                                                                    split_date)
            elif task_type == self.config.task_prompt_events:
                ret = self.converter_events.reverse_conversion(curr_task_text)

            elif task_type == self.config.task_prompt_custom:
                ret = {"original_text": curr_task_text}

            if patientid is not None:
                ret["patientid"] = patientid

            ret_dic = {
                "task_type": task_type,
                "original_text" : curr_task_text,
                "result": ret,
            }

            #: make list of results
            ret_list.append(ret_dic)

        #: return list of results
        return ret_list


    def get_difference_in_event_dataframes(self, list_of_original_conversions_meta : list[dict],
                                           list_of_reversed_conversions : list[dict]) -> list[pd.DataFrame]:
        """
        Compares original structured data with reverse-converted data for each task in a multi-task scenario.

        Iterates through the tasks, matching the original metadata (which contains the
        ground truth structured data, e.g., in 'target_data_processed') with the result
        of the reverse conversion for that task. It then calls the appropriate sub-converter's
        difference calculation method.

        Parameters
        ----------
        list_of_original_conversions_meta : list[dict]
            The list of metadata dictionaries generated during the *forward* conversion,
            where each dictionary corresponds to a task and should contain the original
            structured target data (e.g., under 'target_data_processed'). Usually obtained
            from the 'target_meta_detailed' key in the forward conversion output meta.
        list_of_reversed_conversions : list[dict]
            The list of dictionaries resulting from the `reverse_conversion` method, where
            each dictionary contains the parsed structured data ('result') for a task.

        Returns
        -------
        list[pd.DataFrame]
            A list of pandas DataFrames. Each DataFrame highlights the differences found
            for the corresponding task between the original data and the reverse-converted data.
            An empty DataFrame indicates no differences were found for that task.

        Raises
        ------
        AssertionError
            If the number of original metadata items does not match the number of reversed
            conversion results.
        ValueError
             If a task type mismatch is detected or if expected data keys are missing.
        """

        # Check and setup data
        #list_of_original_conversions_meta = original_conversion["meta"]["target_meta_detailed"]

        assert len(list_of_original_conversions_meta) == len(list_of_reversed_conversions)
        ret_diffs = []

        #: go through each task and compare using the correct converter
        for idx in range(len(list_of_original_conversions_meta)):

            original = list_of_original_conversions_meta[idx]
            reversed = list_of_reversed_conversions[idx]

            if reversed["task_type"] == self.config.task_prompt_forecasting:
                ret_diff = self.converter_forecasting.get_difference_in_event_dataframes(reversed["result"],
                                                                                        original["target_data_processed"])
            elif original["task_type"] == self.config.task_prompt_forecasting_qa:
                ret_diff = self.converter_forecasting_qa.get_difference_in_event_dataframes(reversed["result"],
                                                                                        original["target_data_processed"])
            elif original["task_type"] == self.config.task_prompt_events:
                ret_diff = self.converter_events.get_difference_in_event_dataframes(reversed["result"],
                                                                                    original["target_data_processed"])

            ret_diffs.append(ret_diff)

        #: return list of differences
        return ret_diffs
