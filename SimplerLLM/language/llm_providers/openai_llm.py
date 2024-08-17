# add streaming
from openai import AsyncOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
import os
import time
from .llm_response_models import LLMFullResponse,LLMEmbeddingsResponse

# Load environment variables
load_dotenv()

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))


def generate_response(
    model_name,
    messages=None,
    temperature=0.7,
    max_tokens=300,
    top_p=1.0,
    full_response=False,
    api_key = None,
    base_url = None,
):
    start_time = time.time() if full_response else None
    openai_client = OpenAI(api_key=api_key,base_url=base_url)
    
    for attempt in range(MAX_RETRIES):
        try:
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            generated_text = completion.choices[0].message.content

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=completion,
                )
            return generated_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts"
                if full_response:
                    end_time = time.time()
                    process_time = end_time - start_time
                    error_msg += f" and {process_time} seconds"
                error_msg += f" due to: {e}"
                print(error_msg)
                return None

async def generate_response_async(
    model_name,
    messages=None,
    temperature=0.7,
    max_tokens=300,
    top_p=1.0,
    full_response=False,
    api_key = None,
    base_url = None,
):
    start_time = time.time() if full_response else None
    async_openai_client = AsyncOpenAI(api_key=api_key,base_url=base_url)
   
    for attempt in range(MAX_RETRIES):
        try:
            completion = await async_openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            generated_text = completion.choices[0].message.content

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMFullResponse(
                    generated_text=generated_text,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=completion,
                )
            return generated_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts"
                if full_response:
                    end_time = time.time()
                    process_time = end_time - start_time
                    error_msg += f" and {process_time} seconds"
                error_msg += f" due to: {e}"
                print(error_msg)
                return None

def generate_embeddings(
    model_name,
    user_input=None,
    full_response = False,
    api_key = None,
    base_url = None,
):
    
    if not user_input:
        raise ValueError("user_input must be provided.")
    
    start_time = time.time() if full_response else None

    openai_client = OpenAI(api_key=api_key,base_url=base_url)

    for attempt in range(MAX_RETRIES):
        try:
            
            response = openai_client.embeddings.create(
                model= model_name,
                input=user_input
            )
            generate_embeddings = response.data

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMEmbeddingsResponse(
                    generated_embedding=generate_embeddings,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=response,
                )
            return generate_embeddings

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts"
                if full_response:
                    end_time = time.time()
                    process_time = end_time - start_time
                    error_msg += f" and {process_time} seconds"
                error_msg += f" due to: {e}"
                print(error_msg)
                return None

async def generate_embeddings_async(
    model_name,
    user_input=None,
    full_response = False,
    api_key = None,
    base_url = None,
):
    async_openai_client = AsyncOpenAI(api_key=api_key,base_url=base_url)
    if not user_input:
        raise ValueError("user_input must be provided.")
    
    start_time = time.time() if full_response else None
    for attempt in range(MAX_RETRIES):
        try:
            result = await async_openai_client.embeddings.create(
                model=model_name,
                input=user_input,
            )
            generate_embeddings = result.data

            if full_response:
                end_time = time.time()
                process_time = end_time - start_time
                return LLMEmbeddingsResponse(
                    generated_embedding=generate_embeddings,
                    model=model_name,
                    process_time=process_time,
                    llm_provider_response=result,
                )
            return generate_embeddings

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
            else:
                error_msg = f"Failed after {MAX_RETRIES} attempts"
                if full_response:
                    end_time = time.time()
                    process_time = end_time - start_time
                    error_msg += f" and {process_time} seconds"
                error_msg += f" due to: {e}"
                print(error_msg)
                return None
