from pydantic_ai import RunContext, Tool

from ollama2a.agent_executor import OllamaAgentExecutor

# Define a simple tool function for the agent
async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
    return f"{ctx.deps} {x} {y}"

ae = OllamaAgentExecutor(
    ollama_host="localhost",
    ollama_port=11434,
    ollama_model="qwen3:0.6b",
    system_prompt="You are a helpful assistant.",
    description="An agent that uses the Ollama API to execute tasks.",
    tools=[Tool(my_tool)],
)
app = ae.app

# To run the FastAPI app, use the command:
# uvicorn main:app --host 0.0.0.0 --port 8000
