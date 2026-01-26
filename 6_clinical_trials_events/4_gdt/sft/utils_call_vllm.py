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


TIME_OUTPUTS = False  # Adding debug



async def _call_llm_for_prediction_async(client, model_to_use, patientid, curr_instruction, max_tokens, seed, semaphore, temperature, top_p):

    messages = [
                {"role": "user", "content": curr_instruction}
        ]

    if PREDICTION_ADD_SYSTEM_PROMPT:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    async with semaphore:
        try:
            
            start_time = time.time()

            completion = await client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                logprobs=1,
                extra_body={
                        "max_tokens": max_tokens,
                        "seed": seed,
                        "temperature": temperature,
                        "top_p": top_p,
                }
            )
            response = completion.choices[0].message.content
            logprobs = completion.choices[0].logprobs.content

            if random.random() < 0.01:
                print("==== Response: =====")
                print(response)

            if TIME_OUTPUTS:
                end_time = time.time()
                elapsed_time = end_time - start_time
                return patientid, response, logprobs, elapsed_time
            else:
                return patientid, response, logprobs
        except AuthenticationError as e:
            print(f"Authentication error: {e}")
        except RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
        except APIConnectionError as e:
            print(f"Network error: {e}")
            raise e
        except OpenAIError as e:
            print(f"An OpenAI error occurred: {e}")

        return ""




async def run_across_all_patients(list_of_instructions_and_patientids, 
                                   temperature, top_p,
                                   prediction_url="http://0.0.0.0:9068/v1/",
                                   prediction_model="chkpt/llama_3_1_8b_10x_280k_release_1",  # Update this path
                                   max_concurrent_requests = 40,
                                   max_tokens = 8192,
                                   seed = 42):

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
        
        # Gather seed
        if isinstance(seed, list):
            curr_seed = seed[idx]
        else:
            curr_seed = seed

        # Call task
        task = _call_llm_for_prediction_async(client_prediction, prediction_model, patientid,
                                            instruction, max_tokens, curr_seed, 
                                            semaphore, temperature, top_p)
        tasks.append(task)
    
    #: async call
    responses = await asyncio.gather(*tasks)

    #: gather return list appropriately
    return responses


