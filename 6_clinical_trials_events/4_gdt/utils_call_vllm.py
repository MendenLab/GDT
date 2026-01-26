from openai import (AsyncOpenAI, OpenAIError, AuthenticationError,
                    RateLimitError, APIConnectionError)
import asyncio
import pandas as pd
import os
import sys
import time
import random


# Based on training set up
PREDICTION_ADD_SYSTEM_PROMPT = True
SYSTEM_PROMPT = ("As a specialist predictive model in personalized medicine,"
                " your task is to forecast the health trajectory of cancer "
                "patients by integrating genomic data, lifestyle factors, "
                "treatment history and anything else provided about the patient. "
                "Use the provided patient data, including genetic mutations, "
                "biomarker levels, and previous treatment responses, to predict"
                " all requested tasks. Deliver precise and clinically relevant "
                "predictions to enhance patient care and treatment planning.")



BEGINNING_OF_TEXT = "<|being_of_text|>"  # This we do not add since it is added automatically by the tokenizer on vllm
BEGINNING_OF_TEXT_PROMPT = "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
AFTER_SYSTEM_PROMPT = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
BEGINNING_OF_PREDICTION = "<|eot_id|><|start_header_id|>system<|end_header_id|>\n\n"
EVENT_PROMPT = "Task 1 is time to event prediction:\nHere is the prediction: the event ({event}) was "


LABEL_OCCURRED = "occurred"
LABEL_NOT_OCCURRED = "not_occurred"
LABEL_CENSORED = "censored"

PROMPT_OCCURRED = (LABEL_OCCURRED, "not censored and occurred.")
PROMPT_NOT_OCCURRED = (LABEL_NOT_OCCURRED, "not censored and did not occur.")
PROMPT_CENSORED = (LABEL_CENSORED, "censored and did not occur.")


async def _call_llm_for_prediction_async(client, model_to_use, patientid, curr_instruction, 
                                         semaphore, tokenizer):

    #: generate correct input prompt
    curr_instruction_processed = curr_instruction.strip()   # Since the ending \n is removed usually by chat template
    curr_event = curr_instruction.split("whether the event occured or not: ")[1].split("Please provide your prediction")[0].strip().replace(".", "")
    prompt_start = EVENT_PROMPT.format(event=curr_event)
    instruction_setup = BEGINNING_OF_TEXT_PROMPT + SYSTEM_PROMPT + AFTER_SYSTEM_PROMPT + curr_instruction_processed + BEGINNING_OF_PREDICTION + prompt_start
    
    # Get correct slicing index
    prompt_tokens = tokenizer.encode(instruction_setup, add_special_tokens=False)
    slicing_index = len(prompt_tokens)

    async with semaphore:
        try:
            
            # Setup return object
            return_dic = {
                "patientid": patientid
            }

            for case, curr_prompt in [PROMPT_OCCURRED, PROMPT_NOT_OCCURRED, PROMPT_CENSORED]:

                #: run through all 3 prompts and get logprobs
                combined_prompt = instruction_setup + curr_prompt

                response = await client.completions.create(
                    # IMPORTANT: Use the model name your fine-tune is based on, e.g., "davinci-002"
                    # or your custom fine-tune model ID. gpt-3.5-turbo-instruct is a modern option.
                    model=model_to_use,
                    prompt=combined_prompt,
                    max_tokens=0,   # We want to score, not generate
                    logprobs=1,     # Get the log probability of the top token
                    echo=True       # The key parameter!
                )

                # Get logprobs
                logprobs = response.choices[0].logprobs.token_logprobs

                #: extract correct logprobs as list
                completion_logprobs = logprobs[slicing_index:]

                # Save
                return_dic[case] = completion_logprobs

                #: do some assertions that the sliced versions are correct
                input_sliced = "".join(response.choices[0].logprobs.tokens[:slicing_index])
                output_sliced = "".join(response.choices[0].logprobs.tokens[slicing_index:])
                assert "was" not in output_sliced, f"Output sliced not correct (was): {output_sliced}"
                assert prompt_start.strip() in input_sliced, f"Input sliced not correct: {input_sliced}"
                assert curr_prompt in output_sliced, f"Output sliced not correct: {output_sliced}"
                

            #: return logprobs of all 3 as dictionary
            return return_dic

        except AuthenticationError as e:
            print(f"Authentication error: {e}")
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
        except APIConnectionError as e:
            print(f"Network error: {e}")
            raise e
        except OpenAIError as e:
            print(f"An OpenAI error occurred: {e}")

        return None




async def run_across_all_patients_to_get_probs_for_three_states(list_of_instructions_and_patientids, 
                                   tokenizer,
                                   prediction_url="http://0.0.0.0:9068/v1/",
                                   prediction_model="chkpt/llama_3_1_8b_10x_280k_release_1",  # Update this path
                                   max_concurrent_requests = 40,
                                   ):

    #: setup client
    client_prediction = AsyncOpenAI(
        base_url=prediction_url,
        api_key="EMPTY",  # vLLM local server key
        timeout=10 * 60,  # 10 minutes timeout
    )

    #: make semaphore to not overload the system
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    #: setup task list
    tasks = []
    for idx, (patientid, instruction) in enumerate(list_of_instructions_and_patientids):
        
        # Call task
        task = _call_llm_for_prediction_async(client_prediction, prediction_model, patientid,
                                            instruction, semaphore, tokenizer)
        tasks.append(task)
    
    #: async call
    responses = await asyncio.gather(*tasks)

    #: gather return list appropriately
    return responses


