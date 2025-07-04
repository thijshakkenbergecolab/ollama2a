from logging import info
from fasta2a import FastA2A
from pydantic_ai import Agent, Tool
from ollama import pull, list as ollama_list
from openai import AsyncOpenAI
from dataclasses import dataclass, field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from typing import List

from ollama2a.ollama_manager import HybridOllamaManager


@dataclass
class OllamaAgentExecutor:
    """A class to execute agents using the Ollama API."""

    # Define the Ollama host and port
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_model: str = "qwen3:0.6b"
    system_prompt: str = "You are a helpful assistant."
    description: str = "An agent that uses the Ollama API to execute tasks."
    tools: List[Tool] = field(default_factory=list)

    a2a_port: int = 8000
    manager: HybridOllamaManager() = field(init=False)
    client: AsyncOpenAI = field(init=False)
    llm_provider: OpenAIProvider = field(init=False)
    model: OpenAIModel = field(init=False)
    agent: Agent = field(init=False)
    app: FastA2A = field(init=False)

    def pull_if_model_not_exists(self):
        """Check if the specified model exists on the Ollama server."""
        models = ollama_list()
        exists = False
        for model in models.models:
            if model.model == self.ollama_model:
                exists = True
                break
        if not exists:
            info(f"Model {self.ollama_model} not found. Pulling from Ollama server...")
            pull(self.ollama_model)
            info(f"Model {self.ollama_model} pulled successfully.")

    def __post_init__(self):
        # Check if the model exists, if not, pull it
        self.pull_if_model_not_exists()
        # # Create an Ollama manager
        self.manager = HybridOllamaManager(host=self.ollama_host, port=self.ollama_port)
        # Ensure the Ollama server is running
        self.manager.ensure_server_running()
        # Create an AsyncOpenAI client until someone fixes https://github.com/pydantic/pydantic-ai/issues/112
        self.client = AsyncOpenAI(
            base_url=f"{self.manager.base_url}/v1",
            api_key="we-have-to-set-this-to-something",  # noqa: S105
        )
        self.llm_provider = OpenAIProvider(openai_client=self.client)
        self.model = OpenAIModel(
            self.ollama_model,
            provider=self.llm_provider,  # noqa: type-arg
        )
        # Create an Agent with the model and instructions
        self.agent = Agent(
            self.model, instructions=self.system_prompt, tools=self.tools
        )
        # Create the A2A server application
        self.app = self.agent.to_a2a(
            url=f"http://{self.ollama_host}:{self.a2a_port}",
            description=self.description,
        )


if __name__ == "__main__":
    # Initialize the OllamaAgentExecutor
    executor = OllamaAgentExecutor()
    print(f"Initialized OllamaAgentExecutor with model: {executor.ollama_model}")
    print(f"Client: {executor.client}")
    print(f"LLM Provider: {executor.llm_provider}")
    print(f"Model: {executor.model}")
    print(f"Agent: {executor.agent}")
    print(f"App: {executor.app}")
    print(executor.agent.run_sync(user_prompt="What is the capital of France?"))
