from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from typing import List, AsyncIterator, Dict, Any
from fastapi.responses import StreamingResponse
from ollama import pull, list as ollama_list
from dataclasses import dataclass, field
from pydantic_ai import Agent, Tool
from openai import AsyncOpenAI
from datetime import datetime
from fasta2a import FastA2A
from fastapi import Request
from logging import info
from json import dumps

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
    manager: HybridOllamaManager = field(init=False)
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
        # Create the A2A server application with streaming support
        self.app = self.agent.to_a2a(
            url=f"http://{self.ollama_host}:{self.a2a_port}",
            description=self.description,
        )

        # Add streaming endpoint to the existing app
        self._add_streaming_endpoint()

    def _add_streaming_endpoint(self):
        """Add A2A-compliant SSE streaming endpoint to the existing FastA2A app."""

        @self.app.post("/stream")
        async def stream_endpoint(request: Request):
            """A2A-compliant SSE streaming endpoint."""
            data = await request.json()
            user_prompt = data.get("prompt", "")

            return StreamingResponse(
                self._generate_sse_stream(user_prompt),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

    async def _generate_sse_stream(self, prompt: str) -> AsyncIterator[str]:
        """Generate A2A-compliant SSE events for streaming responses."""

        # Send initial event
        yield self._format_sse_event("start", {
            "timestamp": datetime.utcnow().isoformat(),
            "model": self.ollama_model,
            "prompt": prompt
        })

        try:
            # Create streaming completion
            stream = await self.client.chat.completions.create(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                temperature=0.7,
            )

            full_response = ""
            chunk_count = 0

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    chunk_count += 1

                    # Send content event
                    yield self._format_sse_event("content", {
                        "text": content,
                        "chunk_id": chunk_count,
                        "timestamp": datetime.utcnow().isoformat()
                    })

            # Send completion event
            yield self._format_sse_event("complete", {
                "full_response": full_response,
                "total_chunks": chunk_count,
                "timestamp": datetime.utcnow().isoformat()
            })

        except Exception as e:
            # Send error event
            yield self._format_sse_event("error", {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

        finally:
            # Send end event
            yield self._format_sse_event("end", {
                "timestamp": datetime.utcnow().isoformat()
            })

    def _format_sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Format data as SSE event according to A2A protocol."""
        event = f"event: {event_type}\n"
        event += f"data: {dumps(data)}\n\n"
        return event

