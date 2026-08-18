from ai_client import ask_llm_structured
from pydantic import BaseModel


class PerformanceAnalysis(BaseModel):
    observations: list[str]
    bottlenecks: list[str]
    hypotheses: list[str]
    additional_metrics: list[str]


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

Analyze the following performance test data:

{performance_data}

Identify:
1. Observations supported directly by the data.
2. Potential bottlenecks.
3. Hypotheses that require validation.
4. Additional metrics that should be collected.

Do not invent metrics or facts that are not provided.
"""


result = ask_llm_structured(prompt, PerformanceAnalysis)
analysis = result.output

print("=" * 50)
print("AI PERFORMANCE METRICS")
print("=" * 50)

print(f"Model:            {result.model}")
print(f"Latency:          {result.latency_seconds:.2f} seconds")
print(f"Input tokens:     {result.input_tokens}")
print(f"Output tokens:    {result.output_tokens}")
print(f"Reasoning tokens: {result.reasoning_tokens}")
print(f"Total tokens:     {result.total_tokens}")
print(f"Estimated cost:   ${result.estimated_cost:.6f}")

print("=" * 50)

print("OBSERVATIONS")
for observation in analysis.observations:
    print(f"- {observation}")

print("\nPOTENTIAL BOTTLENECKS")
for bottleneck in analysis.bottlenecks:
    print(f"- {bottleneck}")

print("\nHYPOTHESES")
for hypothesis in analysis.hypotheses:
    print(f"- {hypothesis}")

print("\nADDITIONAL METRICS")
for metric in analysis.additional_metrics:
    print(f"- {metric}")
