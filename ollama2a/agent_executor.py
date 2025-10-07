from typing import List, AsyncIterator, Dict, Any, Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai.models.openai import OpenAIModel
from asyncio import TimeoutError as ATimeoutError
from fastapi.responses import StreamingResponse
from ollama import pull, list as ollama_list
from fastapi import Request, HTTPException
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pydantic_ai import Agent, Tool
from openai import AsyncOpenAI
from fasta2a import FastA2A
from typing import Literal
from logging import info
from json import dumps

from ollama2a.ollama_manager import HybridOllamaManager


class StreamRequest(BaseModel):
    """Validation model for stream endpoint requests."""
    prompt: str = Field(..., min_length=1, description="The prompt to stream responses for")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Presence penalty")
    timeout: Optional[float] = Field(default=60.0, gt=0, description="Stream timeout in seconds")


def format_sse_event(
    event_type: Literal["start", "content", "complete", "timeout", "error", "end"],
    data: Dict[str, Any],
) -> str:
    """Format data as SSE event according to A2A protocol."""
    event = f"event: {event_type}\n"
    event += f"data: {dumps(data)}\n\n"
    return event


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
        from starlette.routing import Route

        async def stream_endpoint(request: Request):
            """A2A-compliant SSE streaming endpoint with request validation."""
            try:
                # Parse and validate request data
                data = await request.json()
                stream_request = StreamRequest(**data)
            except ValidationError as e:
                # Return validation errors as HTTP 422
                raise HTTPException(status_code=422, detail=e.errors())
            except Exception as e:
                # Handle JSON parsing errors
                raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

            return StreamingResponse(
                self._generate_sse_stream(stream_request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        # Add the route using FastA2A's add_route method
        route = Route("/stream", endpoint=stream_endpoint, methods=["POST"])
        self.app.routes.append(route)

    async def _generate_sse_stream(self, request: StreamRequest) -> AsyncIterator[str]:
        """Generate A2A-compliant SSE events for streaming responses with timeout handling."""
        import asyncio

        # Send initial event
        yield format_sse_event("start", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.ollama_model,
            "prompt": request.prompt,
            "timeout": request.timeout
        })

        try:
            # Build completion parameters with configurable values
            completion_params = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": request.prompt},
                ],
                "stream": True,
                "temperature": request.temperature,
            }

            # Add optional parameters if provided
            if request.max_tokens is not None:
                completion_params["max_tokens"] = request.max_tokens
            if request.top_p is not None:
                completion_params["top_p"] = request.top_p
            if request.frequency_penalty is not None:
                completion_params["frequency_penalty"] = request.frequency_penalty
            if request.presence_penalty is not None:
                completion_params["presence_penalty"] = request.presence_penalty

            # Create streaming completion with timeout
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(**completion_params),
                timeout=request.timeout
            )

            full_response = ""
            chunk_count = 0
            start_time = asyncio.get_event_loop().time()

            async for chunk in stream:
                # Check if we've exceeded the timeout during streaming
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > request.timeout:
                    raise ATimeoutError(f"Stream timeout after {elapsed:.2f} seconds")

                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    chunk_count += 1

                    # Send content event
                    yield format_sse_event("content", {
                        "text": content,
                        "chunk_id": chunk_count,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

            # Send completion event
            yield format_sse_event("complete", {
                "full_response": full_response,
                "total_chunks": chunk_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        except ATimeoutError as e:
            # Send timeout event
            yield format_sse_event("timeout", {
                "error": str(e) if str(e) else f"Stream timeout after {request.timeout} seconds",
                "timeout_seconds": request.timeout,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            # Send error event
            yield format_sse_event("error", {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        finally:
            # Send end event
            yield format_sse_event("end", {
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
