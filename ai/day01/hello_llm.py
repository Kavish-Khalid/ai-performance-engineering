from openai import OpenAI

client = OpenAI()

performance_data = """
A banking application was tested with 5,000 concurrent users.
Average response time increased from 2 seconds to 7 seconds.
Application CPU is 45%.
JVM heap utilization is 82%.
Database CPU is 91%.
Database connection pool utilization is 95%.
"""

prompt = f"""
You are a senior performance engineer.
Analyze the following performance test data.
{performance_data}

Identify:
1. Observations supported directly by the data.
2. Potential bottlenecks.
3. Hypotheses that require validation.
4. Additional metrics that should be collected.

Do not invent metrics or facts that are not provided.
"""

response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)

input_cost = (response.usage.input_tokens / 1_000_000) * 0.25
output_cost = (response.usage.output_tokens / 1_000_000) * 2.00

total_cost = input_cost + output_cost

print("=" * 50)
print("AI PERFORMANCE METRICS")
print("=" * 50)

print(f"{'Model':20} {response.model}")
print(f"{'Input tokens':20} {response.usage.input_tokens}")
print(f"{'Output tokens':20} {response.usage.output_tokens}")
print(f"{'Reasoning tokens':20} {response.usage.output_tokens_details.reasoning_tokens}")
print(f"{'Total tokens':20} {response.usage.total_tokens}")
print(f"{'Estimated cost':20} ${total_cost:.6f}")

print("=" * 50)
print("AI RESPONSE")
print("=" * 50)

print(response.output_text)