import pandas as pd
import copy

from digital_twin_converter.instruction.converter_manual_forecasting import (
    ConverterManualForecasting
)
from digital_twin_converter.common.converter_manual_base import round_and_strip


class ConverterManualForecastingQA(ConverterManualForecasting):
    """
    A class for converting patient data into a format suitable for manual forecasting
    with a question-answering approach.

    This class inherits from ConverterManualForecasting and implements additional methods
    to preprocess target data into
    categorical bins and generate prompts for the forecasting model.
    """

    def _preprocess_target_into_qa(self,
                                     patient_split: dict,
                                     variable_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the target data into categories suitable for question-answering format.

        Parameters
        ----------
        patient_split : dict
            A dictionary containing patient data split into events and target variables.
        variable_stats : pd.DataFrame
            A DataFrame containing statistics of the variables, used to determine the binning strategy.

        Returns
        -------
        pd.DataFrame
            The updated patient_split dictionary with target events converted into categorical bins.

        """

        #: deep copy dict
        curr_patient_split = copy.deepcopy(patient_split)

        #: loop through all target variables
        targets = curr_patient_split["target_events_after_split"]
        all_variables = targets[self.config.event_name_col].unique().tolist()
        targets[self.config.event_value_col] = targets[self.config.event_value_col].astype("string")
        curr_patient_split["category_splits"] = {}

        for var in all_variables:

            #: change every variable in target to be a category based on variable stats in event_value
            #: use e.g. "a" (= "Bin (0,2]") as new value
            splits = variable_stats.loc[variable_stats[self.config.event_name_col] == var, self.config.bins_split_name].values[0]

            # Replace both edges for splits with -inf/inf
            splits[0] = -float("inf")
            splits[-1] = float("inf")
            bin_ranges = [f"Bin ({round_and_strip(splits[i], self.decimal_precision)}, " +
                          f"{round_and_strip(splits[i+1], self.decimal_precision)}]"
                          for i in range(len(splits) - 1)]

            # Change last bin closing to be exclusive for inf
            bin_ranges[-1] = bin_ranges[-1].replace("]", ")")

            #: make bins into A, B, C, etc.
            bin_names = ["A", "B", "C", "D", "E", "F"]
            bin_mapping = {bin_names[i]: bin_ranges[i] for i in range(len(bin_ranges))}
            all_bins = list(bin_mapping.keys())

            # Apply binning
            curr_targets = targets[targets[self.config.event_name_col] == var].loc[:, self.config.event_value_col].copy()
            target_selection = targets[self.config.event_name_col] == var

            new_targets = pd.cut(
                curr_targets.astype(float),
                bins=splits,
                labels=all_bins,
                include_lowest=True
            )
            new_targets = new_targets.astype("string")
            targets.loc[target_selection, self.config.event_value_col] = new_targets

            #: add in curr_patient_split["category_splits"]
            curr_patient_split["category_splits"][var] = bin_mapping

        #: save & return
        curr_patient_split["target_events_after_split"] = targets
        return curr_patient_split

    def _generate_prompt(self,
                         target_meta: dict) -> str:
        """
        Generates a prompt string for the forecasting model based on the target
        metadata and variable statistics.

        Parameters
        ----------
        target_meta : dict
            A dictionary containing metadata about the target variables.

        Returns
        -------
        str
            The generated prompt string for the forecasting model.

        """

        # start ret
        ret_prompt = ""

        #: make sure to round weeks from future_weeks_per_variable to predict to 2 decimals
        future_weeks_per_variable = target_meta["future_weeks_per_variable"]
        future_weeks_per_variable = {k: [round_and_strip(v2, self.decimal_precision) for v2 in v] for
                                     k, v in future_weeks_per_variable.items()}

        #: add base prompt
        ret_prompt += self.config.qa_prompt_start

        #: sort alphabetically by variable name
        future_weeks_per_variable = dict(sorted(future_weeks_per_variable.items(), key=lambda item: item[0]))

        #: create prompt for which variables to predict
        for variable, weeks in future_weeks_per_variable.items():

            #: need event_descriptive_name
            variable_desc_name = target_meta["variable_name_mapping"][variable]

            # Create prompt
            ret_prompt += "\n\t" + variable_desc_name
            ret_prompt += self.config.forecasting_prompt_var_time
            ret_prompt += str(weeks).replace('[', '').replace(']', '').replace("'", "")

            #: add in string for which varaibles have which category values
            processed_bins = str(target_meta["category_splits"][variable]).lower()
            processed_bins = processed_bins.replace('{', '').replace('}', '').replace("'", "")

            ret_prompt += "\n" + self.config.qa_bins_start
            ret_prompt += processed_bins

        #: return
        return ret_prompt

    def forward_conversion(self,
                           patient_split: dict,
                           variable_stats: pd.DataFrame) -> dict:
        """
        Converts the patient data into a format suitable for the forecasting model,
        including preprocessing and prompt generation.

        Parameters
        ----------
        patient_split : dict
            A dictionary containing patient data split into events and target variables.
        variable_stats : pd.DataFrame
            A DataFrame containing statistics of the variables, used to determine the binning strategy.

        Returns
        -------
        dict
            A dictionary containing the input string for the model, the target string,
            constant data, history data, and metadata.

        """

        #: preprocess target data into categories
        raw_target = patient_split["target_events_after_split"].copy()
        patient_split = self._preprocess_target_into_qa(patient_split, variable_stats)

        #: generate target string
        target_str, target_meta = self._generate_target_string(patient_split)

        #: make sure that target_meta contains the new targets in categories
        target_meta["category_splits"] = patient_split["category_splits"]
        target_meta["target_data_raw"] = raw_target

        #: generate prompt (including when to generate what)
        prompt_str = self._generate_prompt(target_meta)

        return prompt_str, target_str, target_meta
