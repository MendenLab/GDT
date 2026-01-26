from openai import (AsyncOpenAI, OpenAIError, AuthenticationError,
                    RateLimitError, APIConnectionError)
import asyncio
import pandas as pd
import os
import sys
import random






async def _call_llm_for_prediction_async(client, model_to_use, patientid, curr_instruction, max_tokens, seed, semaphore, temperature, top_p):

    # curr_instruction is already a list of messages

    async with semaphore:
        try:
            completion = await client.chat.completions.create(
                model=model_to_use,
                messages=curr_instruction,
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
                                   max_concurrent_requests = 100,
                                   max_tokens = 8192,
                                   seed = 42):

    #: setup client
    client_prediction = AsyncOpenAI(
        base_url=prediction_url,
        api_key="EMPTY",  # vLLM local server key
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


