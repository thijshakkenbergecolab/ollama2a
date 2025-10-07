# ollama2a

A Python library for running Ollama agents with automated server management.

## Features

- 🚀 **Automated Ollama Server Management**: Automatically starts and manages Ollama servers
- 🤖 **Pydantic AI Integration**: Seamless integration with pydantic-ai for agent creation
- 🔧 **Configurable**: Easy configuration of host, port, models, and tools
- 🧪 **Well Tested**: Comprehensive test suite with high coverage
- 📦 **Production Ready**: Robust error handling and resource management
- 🌊 **Streaming Support**: A2A-compliant SSE streaming for real-time responses

## Installation

```bash
pip install ollama2a
```

### Development Installation

```bash
pip install ollama2a[dev]
```

## Quick Start

### Basic Usage

```python
from ollama2a.agent_executor import OllamaAgentExecutor

# Create an agent executor with default settings
executor = OllamaAgentExecutor(
    ollama_model="qwen3:0.6b",
    system_prompt="You are a helpful assistant."
)

# The server starts automatically and the agent is ready to use
result = executor.agent.run_sync(user_prompt="What is the capital of France?")
print(result)
```

### With Custom Tools

```python
from pydantic_ai import Tool, RunContext
from ollama2a.agent_executor import OllamaAgentExecutor

# Define a custom tool
async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
    return f"Result: {x + y}"

# Create executor with custom tools
executor = OllamaAgentExecutor(
    ollama_model="qwen3:0.6b",
    system_prompt="You are a math assistant.",
    tools=[Tool(my_tool)]
)
```

### FastAPI Integration

```python
from pydantic_ai import Tool
from ollama2a.agent_executor import OllamaAgentExecutor
def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
    return f"Result: {x + y}"

# Create the agent executor
executor = OllamaAgentExecutor(
    ollama_host="localhost",
    ollama_port=11434,
    ollama_model="qwen3:0.6b",
    system_prompt="You are a helpful assistant.",
    tools=[Tool(my_tool)]
)

# Get the FastAPI app
app = executor.app

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Streaming API

The library now includes A2A-compliant SSE streaming support for real-time responses:

```python
from ollama2a.agent_executor import OllamaAgentExecutor

# Create executor with streaming support
executor = OllamaAgentExecutor(
    ollama_model="qwen3:0.6b",
    system_prompt="You are a helpful assistant."
)

# The streaming endpoint is automatically available at /stream
app = executor.app

# Example client usage with httpx
import httpx
import json

async def stream_chat():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/stream",
            json={
                "prompt": "Tell me a story",
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=60.0
        )

        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "text" in data:
                    print(data["text"], end="", flush=True)
```

#### Streaming Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | Required | The prompt to stream responses for |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0-2.0) |
| `max_tokens` | `int` | `None` | Maximum tokens to generate |
| `top_p` | `float` | `None` | Top-p sampling parameter (0.0-1.0) |
| `frequency_penalty` | `float` | `None` | Frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | `float` | `None` | Presence penalty (-2.0 to 2.0) |
| `timeout` | `float` | `60.0` | Stream timeout in seconds |

#### SSE Event Types

The streaming endpoint emits the following A2A-compliant SSE events:

- `start`: Initial event with model info and parameters
- `content`: Streaming text chunks with incremental content
- `complete`: Final event with full response and statistics
- `timeout`: Emitted if the stream exceeds the timeout
- `error`: Emitted on errors with error details
- `end`: Final event marking stream completion

## Configuration

### OllamaAgentExecutor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ollama_host` | `str` | `"localhost"` | Ollama server host |
| `ollama_port` | `int` | `11434` | Ollama server port |
| `ollama_model` | `str` | `"qwen3:0.6b"` | Model to use |
| `system_prompt` | `str` | `"You are a helpful assistant."` | System prompt for the agent |
| `description` | `str` | `"An agent that uses the Ollama API to execute tasks."` | Agent description |
| `tools` | `List[Tool]` | `[]` | Custom tools for the agent |
| `a2a_port` | `int` | `8000` | Port for the A2A server |

## Server Management

The `HybridOllamaManager` automatically handles:

- ✅ **Server startup**: Starts Ollama server if not running
- ✅ **Model downloading**: Downloads models if not available locally
- ✅ **Health checks**: Monitors server health
- ✅ **Graceful shutdown**: Properly terminates processes
- ✅ **Error handling**: Robust error handling and retries

### Manual Server Management

```python
from ollama2a.ollama_manager import HybridOllamaManager

manager = HybridOllamaManager(host="localhost", port=11434)
manager.ensure_server_running()

# Use the manager
response = manager.run_model("qwen3:0.6b", "Hello world!")
print(response)

# Cleanup when done
manager.cleanup()
```

## Requirements

- Python 3.9+
- Ollama installed on your system

## Installation of Ollama

Follow the [official Ollama installation guide](https://ollama.ai/download) for your operating system.

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/ollama2a.git
cd ollama2a

# Install in development mode
pip install -e .[dev]
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ollama2a --cov-report=html

# Run specific test file
pytest tests/test_ollama_manager.py -v
```

### Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy ollama2a/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for the amazing local LLM runtime
- [Pydantic AI](https://ai.pydantic.dev/) for the agent framework
- [FastA2A](https://github.com/tomasonjo/fasta2a) for the API framework

## Changelog

### 1.1.0 (2025-10-07)

- Added A2A-compliant SSE streaming support
- New `/stream` endpoint for real-time responses
- Configurable streaming parameters (temperature, max_tokens, etc.)
- Timeout handling for streaming requests
- Comprehensive test coverage for streaming features

### 1.0.0 (2025-09-12)

- Initial release
- Basic Ollama server management
- Pydantic AI integration
- Starlette app generation
- Comprehensive test suite