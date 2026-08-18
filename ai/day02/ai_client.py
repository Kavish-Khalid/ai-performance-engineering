import time
from openai import OpenAI, APIError
from pydantic import BaseModel

client = OpenAI()


class AIResult(BaseModel):
    output: object
    model: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    estimated_cost: float


def ask_llm(prompt):
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response


def ask_llm_structured(prompt, response_model):
    start_time = time.perf_counter()

    try:
        response = client.responses.parse(
            model="gpt-5-mini",
            input=prompt,
            text_format=response_model
        )
    except APIError as e:
        print(f"AI API error: {e}")
        raise

    end_time = time.perf_counter()

    latency = end_time - start_time

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
    total_tokens = response.usage.total_tokens

    input_cost = (input_tokens / 1_000_000) * 0.25
    output_cost = (output_tokens / 1_000_000) * 2.00
    total_cost = input_cost + output_cost

    return AIResult(
        output=response.output_parsed,
        model=response.model,
        latency_seconds=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        estimated_cost=total_cost
    )
