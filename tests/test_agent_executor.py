from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pytest import raises, main
import pytest
from pydantic import ValidationError

from ollama2a.agent_executor import OllamaAgentExecutor, StreamRequest, format_sse_event


class TestOllamaAgentExecutor:
    """Test suite for OllamaAgentExecutor"""

    @patch("ollama2a.agent_executor.HybridOllamaManager")
    @patch("ollama2a.agent_executor.AsyncOpenAI")
    @patch("ollama2a.agent_executor.OpenAIProvider")
    @patch("ollama2a.agent_executor.OpenAIModel")
    @patch("ollama2a.agent_executor.Agent")
    @patch("ollama2a.agent_executor.FastA2A")
    @patch("ollama2a.agent_executor.list")
    @patch("ollama2a.agent_executor.pull")
    @patch("ollama2a.agent_executor.info")
    def test_init_default_values(
        self,
        mock_info,
        mock_pull,
        mock_list,
        mock_fasta2a,
        mock_agent,
        mock_openai_model,
        mock_openai_provider,
        mock_async_openai,
        mock_manager_class,
    ):
        """Test executor initialization with default values"""
        # Mock the list response to show model exists
        mock_models_response = Mock()
        mock_model = Mock()
        mock_model.model = "qwen3:0.6b"
        mock_models_response.models = [mock_model]
        mock_list.return_value = mock_models_response

        # Mock manager instance
        mock_manager_instance = Mock()
        mock_manager_instance.base_url = "http://localhost:11434"
        mock_manager_class.return_value = mock_manager_instance

        # Mock agent instance with to_a2a method
        mock_agent_instance = Mock()
        mock_agent_instance.to_a2a.return_value = Mock()
        mock_agent.return_value = mock_agent_instance

        executor = OllamaAgentExecutor()

        assert executor.ollama_host == "localhost"
        assert executor.ollama_port == 11434
        assert executor.ollama_model == "qwen3:0.6b"
        assert executor.system_prompt == "You are a helpful assistant."
        assert executor.a2a_port == 8000
        assert executor.tools == []

        # Verify post_init setup was called
        mock_manager_class.assert_called_once_with(host="localhost", port=11434)
        mock_manager_instance.ensure_server_running.assert_called_once()

    @patch("ollama2a.agent_executor.HybridOllamaManager")
    @patch("ollama2a.agent_executor.AsyncOpenAI")
    @patch("ollama2a.agent_executor.OpenAIProvider")
    @patch("ollama2a.agent_executor.OpenAIModel")
    @patch("ollama2a.agent_executor.Agent")
    @patch("ollama2a.agent_executor.FastA2A")
    @patch("ollama2a.agent_executor.list")
    @patch("ollama2a.agent_executor.pull")
    @patch("ollama2a.agent_executor.info")
    def test_init_custom_values(
        self,
        mock_info,
        mock_pull,
        mock_list,
        mock_fasta2a,
        mock_agent,
        mock_openai_model,
        mock_openai_provider,
        mock_async_openai,
        mock_manager_class,
    ):
        """Test executor initialization with custom values"""
        # Mock the list response to show model exists
        mock_models_response = Mock()
        mock_model = Mock()
        mock_model.model = "custom-model:1.0"
        mock_models_response.models = [mock_model]
        mock_list.return_value = mock_models_response

        # Mock manager instance
        mock_manager_instance = Mock()
        mock_manager_instance.base_url = "http://custom-host:9999"
        mock_manager_class.return_value = mock_manager_instance

        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.to_a2a.return_value = Mock()
        mock_agent.return_value = mock_agent_instance

        executor = OllamaAgentExecutor(
            ollama_host="custom-host",
            ollama_port=9999,
            ollama_model="custom-model:1.0",
            system_prompt="Custom prompt",
            a2a_port=9000,
        )

        assert executor.ollama_host == "custom-host"
        assert executor.ollama_port == 9999
        assert executor.ollama_model == "custom-model:1.0"
        assert executor.system_prompt == "Custom prompt"
        assert executor.a2a_port == 9000

        mock_manager_class.assert_called_once_with(host="custom-host", port=9999)

    @patch("ollama2a.agent_executor.ollama_list")
    @patch("ollama2a.agent_executor.pull")
    @patch("ollama2a.agent_executor.info")
    def test_pull_if_model_not_exists_model_exists(
        self, mock_info, mock_pull, mock_list
    ):
        """Test pull_if_model_not_exists when model already exists"""
        # Create executor without triggering post_init
        executor = OllamaAgentExecutor.__new__(OllamaAgentExecutor)
        executor.ollama_model = "qwen3:0.6b"

        # Mock the list response to show model exists
        mock_models_response = Mock()
        mock_model = Mock()
        mock_model.model = "qwen3:0.6b"
        mock_models_response.models = [mock_model]
        mock_list.return_value = mock_models_response

        executor.pull_if_model_not_exists()

        mock_list.assert_called_once()
        mock_pull.assert_not_called()
        mock_info.assert_not_called()

    @patch("ollama2a.agent_executor.ollama_list")
    @patch("ollama2a.agent_executor.pull")
    @patch("ollama2a.agent_executor.info")
    def test_pull_if_model_not_exists_model_missing(
        self, mock_info, mock_pull, mock_list
    ):
        """Test pull_if_model_not_exists when model doesn't exist"""
        # Create executor without triggering post_init
        executor = OllamaAgentExecutor.__new__(OllamaAgentExecutor)
        executor.ollama_model = "missing-model:1.0"

        # Mock the list response to show model doesn't exist
        mock_models_response = Mock()
        mock_other_model = Mock()
        mock_other_model.model = "qwen3:0.6b"
        mock_models_response.models = [mock_other_model]
        mock_list.return_value = mock_models_response

        executor.pull_if_model_not_exists()

        mock_list.assert_called_once()
        mock_pull.assert_called_once_with("missing-model:1.0")
        assert mock_info.call_count == 2
        mock_info.assert_any_call(
            "Model missing-model:1.0 not found. Pulling from Ollama server..."
        )
        mock_info.assert_any_call("Model missing-model:1.0 pulled successfully.")

    @patch("ollama2a.agent_executor.ollama_list")
    @patch("ollama2a.agent_executor.pull")
    @patch("ollama2a.agent_executor.info")
    def test_pull_if_model_not_exists_empty_list(self, mock_info, mock_pull, mock_list):
        """Test pull_if_model_not_exists when no models exist"""
        # Create executor without triggering post_init
        executor = OllamaAgentExecutor.__new__(OllamaAgentExecutor)
        executor.ollama_model = "any-model:1.0"

        # Mock empty list response
        mock_models_response = Mock()
        mock_models_response.models = []
        mock_list.return_value = mock_models_response

        executor.pull_if_model_not_exists()

        mock_list.assert_called_once()
        mock_pull.assert_called_once_with("any-model:1.0")

    @patch("ollama2a.agent_executor.HybridOllamaManager")
    @patch("ollama2a.agent_executor.AsyncOpenAI")
    @patch("ollama2a.agent_executor.OpenAIProvider")
    @patch("ollama2a.agent_executor.OpenAIModel")
    @patch("ollama2a.agent_executor.Agent")
    @patch("ollama2a.agent_executor.FastA2A")
    @patch("ollama2a.agent_executor.list")
    @patch("ollama2a.agent_executor.pull")
    @patch("ollama2a.agent_executor.info")
    def test_post_init_setup_sequence(
        self,
        mock_info,
        mock_pull,
        mock_list,
        mock_fasta2a,
        mock_agent,
        mock_openai_model,
        mock_openai_provider,
        mock_async_openai,
        mock_manager_class,
    ):
        """Test the post_init setup sequence"""
        # Mock the list response
        mock_models_response = Mock()
        mock_model = Mock()
        mock_model.model = "qwen3:0.6b"
        mock_models_response.models = [mock_model]
        mock_list.return_value = mock_models_response

        # Mock manager instance
        mock_manager_instance = Mock()
        mock_manager_instance.base_url = "http://localhost:11434"
        mock_manager_class.return_value = mock_manager_instance

        # Mock OpenAI components
        mock_client_instance = Mock()
        mock_async_openai.return_value = mock_client_instance

        mock_provider_instance = Mock()
        mock_openai_provider.return_value = mock_provider_instance

        mock_model_instance = Mock()
        mock_openai_model.return_value = mock_model_instance

        # Mock agent instance
        mock_agent_instance = Mock()
        mock_app_instance = Mock()
        mock_agent_instance.to_a2a.return_value = mock_app_instance
        mock_agent.return_value = mock_agent_instance

        executor = OllamaAgentExecutor()

        # Verify the setup sequence
        mock_manager_class.assert_called_once_with(host="localhost", port=11434)
        mock_manager_instance.ensure_server_running.assert_called_once()

        mock_async_openai.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="we-have-to-set-this-to-something",
        )

        mock_openai_provider.assert_called_once_with(openai_client=mock_client_instance)
        mock_openai_model.assert_called_once_with(
            "qwen3:0.6b", provider=mock_provider_instance
        )

        mock_agent.assert_called_once_with(
            mock_model_instance, instructions="You are a helpful assistant.", tools=[]
        )

        mock_agent_instance.to_a2a.assert_called_once_with(
            url="http://localhost:8000",
            description="An agent that uses the Ollama API to execute tasks.",
        )

        # Verify the instances are assigned
        assert executor.manager == mock_manager_instance
        assert executor.client == mock_client_instance
        assert executor.llm_provider == mock_provider_instance
        assert executor.model == mock_model_instance
        assert executor.agent == mock_agent_instance
        assert executor.app == mock_app_instance

    @patch("ollama2a.agent_executor.HybridOllamaManager")
    @patch("ollama2a.agent_executor.list")
    @patch("ollama2a.agent_executor.pull")
    def test_manager_server_startup_failure(
        self, mock_pull, mock_list, mock_manager_class
    ):
        """Test behavior when server startup fails"""
        # Mock the list response
        mock_models_response = Mock()
        mock_model = Mock()
        mock_model.model = "qwen3:0.6b"
        mock_models_response.models = [mock_model]
        mock_list.return_value = mock_models_response

        # Mock manager that fails to start server
        mock_manager_instance = Mock()
        mock_manager_instance.ensure_server_running.side_effect = RuntimeError(
            "Server failed"
        )
        mock_manager_class.return_value = mock_manager_instance

        with raises(RuntimeError, match="Server failed"):
            OllamaAgentExecutor()

    @patch("ollama2a.agent_executor.list")
    @patch("ollama2a.agent_executor.pull")
    def test_model_pull_failure(self, mock_pull, mock_list):
        """Test behavior when model pull fails"""
        # Create executor without triggering post_init
        executor = OllamaAgentExecutor.__new__(OllamaAgentExecutor)
        executor.ollama_model = "missing-model:1.0"

        # Mock empty list response
        mock_models_response = Mock()
        mock_models_response.models = []
        mock_list.return_value = mock_models_response

        # Mock pull failure
        mock_pull.side_effect = Exception("Pull failed")

        with raises(Exception, match="Pull failed"):
            executor.pull_if_model_not_exists()

    @patch("ollama2a.agent_executor.HybridOllamaManager")
    @patch("ollama2a.agent_executor.AsyncOpenAI")
    @patch("ollama2a.agent_executor.OpenAIProvider")
    @patch("ollama2a.agent_executor.OpenAIModel")
    @patch("ollama2a.agent_executor.Agent")
    @patch("ollama2a.agent_executor.FastA2A")
    @patch("ollama2a.agent_executor.list")
    @patch("ollama2a.agent_executor.pull")
    @patch("ollama2a.agent_executor.info")
    def test_with_custom_tools(
        self,
        mock_info,
        mock_pull,
        mock_list,
        mock_fasta2a,
        mock_agent,
        mock_openai_model,
        mock_openai_provider,
        mock_async_openai,
        mock_manager_class,
    ):
        """Test executor initialization with custom tools"""
        # Mock the list response
        mock_models_response = Mock()
        mock_model = Mock()
        mock_model.model = "qwen3:0.6b"
        mock_models_response.models = [mock_model]
        mock_list.return_value = mock_models_response

        # Mock manager and other components
        mock_manager_instance = Mock()
        mock_manager_instance.base_url = "http://localhost:11434"
        mock_manager_class.return_value = mock_manager_instance

        mock_agent_instance = Mock()
        mock_agent_instance.to_a2a.return_value = Mock()
        mock_agent.return_value = mock_agent_instance

        # Create custom tools
        mock_tool1 = Mock()
        mock_tool2 = Mock()
        custom_tools = [mock_tool1, mock_tool2]

        executor = OllamaAgentExecutor(tools=custom_tools)

        assert executor.tools == custom_tools

        # Verify agent was created with custom tools
        mock_agent.assert_called_once_with(
            mock_openai_model.return_value,
            instructions="You are a helpful assistant.",
            tools=custom_tools,
        )

    @patch("ollama2a.agent_executor.ollama_list")
    def test_list_models_failure(self, mock_list):
        """Test behavior when listing models fails"""
        # Create executor without triggering post_init
        executor = OllamaAgentExecutor.__new__(OllamaAgentExecutor)
        executor.ollama_model = "any-model:1.0"

        # Mock list failure
        mock_list.side_effect = Exception("Failed to list models")

        with raises(Exception, match="Failed to list models"):
            executor.pull_if_model_not_exists()


class TestOllamaAgentExecutorIntegration:
    """Integration tests for OllamaAgentExecutor"""

    @patch("ollama2a.agent_executor.HybridOllamaManager")
    @patch("ollama2a.agent_executor.AsyncOpenAI")
    @patch("ollama2a.agent_executor.OpenAIProvider")
    @patch("ollama2a.agent_executor.OpenAIModel")
    @patch("ollama2a.agent_executor.Agent")
    @patch("ollama2a.agent_executor.FastA2A")
    @patch("ollama2a.agent_executor.list")
    @patch("ollama2a.agent_executor.pull")
    @patch("ollama2a.agent_executor.info")
    def test_full_initialization_workflow(
        self,
        mock_info,
        mock_pull,
        mock_list,
        mock_fasta2a,
        mock_agent,
        mock_openai_model,
        mock_openai_provider,
        mock_async_openai,
        mock_manager_class,
    ):
        """Test complete initialization workflow with model pulling"""
        # Mock the list response to show model doesn't exist initially
        mock_models_response = Mock()
        mock_models_response.models = []
        mock_list.return_value = mock_models_response

        # Mock all the components
        mock_manager_instance = Mock()
        mock_manager_instance.base_url = "http://test-host:8080"
        mock_manager_class.return_value = mock_manager_instance

        mock_client_instance = Mock()
        mock_async_openai.return_value = mock_client_instance

        mock_provider_instance = Mock()
        mock_openai_provider.return_value = mock_provider_instance

        mock_model_instance = Mock()
        mock_openai_model.return_value = mock_model_instance

        mock_agent_instance = Mock()
        mock_app_instance = Mock()
        mock_agent_instance.to_a2a.return_value = mock_app_instance
        mock_agent.return_value = mock_agent_instance

        # Create executor with custom settings
        executor = OllamaAgentExecutor(
            ollama_host="test-host",
            ollama_port=8080,
            ollama_model="test-model:latest",
            system_prompt="Test prompt",
            description="Test description",
        )

        # Verify model was pulled because it didn't exist
        mock_pull.assert_called_once_with("test-model:latest")
        mock_info.assert_any_call(
            "Model test-model:latest not found. Pulling from Ollama server..."
        )
        mock_info.assert_any_call("Model test-model:latest pulled successfully.")

        # Verify complete setup chain
        mock_manager_class.assert_called_once_with(host="test-host", port=8080)
        mock_manager_instance.ensure_server_running.assert_called_once()

        mock_async_openai.assert_called_once_with(
            base_url="http://test-host:8080/v1",
            api_key="we-have-to-set-this-to-something",
        )

        mock_openai_provider.assert_called_once_with(openai_client=mock_client_instance)
        mock_openai_model.assert_called_once_with(
            "test-model:latest", provider=mock_provider_instance
        )
        mock_agent.assert_called_once_with(
            mock_model_instance, instructions="Test prompt", tools=[]
        )
        mock_agent_instance.to_a2a.assert_called_once_with(
            url="http://test-host:8000", description="Test description"
        )

        # Verify all components are properly assigned
        assert executor.manager == mock_manager_instance
        assert executor.client == mock_client_instance
        assert executor.llm_provider == mock_provider_instance
        assert executor.model == mock_model_instance
        assert executor.agent == mock_agent_instance
        assert executor.app == mock_app_instance


class TestStreamRequest:
    """Test suite for StreamRequest model"""

    def test_valid_stream_request_minimal(self):
        """Test StreamRequest with minimal valid data"""
        request = StreamRequest(prompt="Test prompt")
        assert request.prompt == "Test prompt"
        assert request.temperature == 0.7
        assert request.max_tokens is None
        assert request.top_p is None
        assert request.frequency_penalty is None
        assert request.presence_penalty is None
        assert request.timeout == 60.0

    def test_valid_stream_request_full(self):
        """Test StreamRequest with all parameters"""
        request = StreamRequest(
            prompt="Test prompt",
            temperature=1.5,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=-0.5,
            timeout=30.0
        )
        assert request.prompt == "Test prompt"
        assert request.temperature == 1.5
        assert request.max_tokens == 100
        assert request.top_p == 0.9
        assert request.frequency_penalty == 0.5
        assert request.presence_penalty == -0.5
        assert request.timeout == 30.0

    def test_invalid_empty_prompt(self):
        """Test StreamRequest validation with empty prompt"""
        with raises(ValidationError) as exc_info:
            StreamRequest(prompt="")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("prompt",) for e in errors)

    def test_invalid_temperature_range(self):
        """Test StreamRequest validation with invalid temperature"""
        with raises(ValidationError):
            StreamRequest(prompt="Test", temperature=-0.1)

        with raises(ValidationError):
            StreamRequest(prompt="Test", temperature=2.1)

    def test_invalid_max_tokens(self):
        """Test StreamRequest validation with invalid max_tokens"""
        with raises(ValidationError):
            StreamRequest(prompt="Test", max_tokens=0)

    def test_invalid_top_p_range(self):
        """Test StreamRequest validation with invalid top_p"""
        with raises(ValidationError):
            StreamRequest(prompt="Test", top_p=-0.1)

        with raises(ValidationError):
            StreamRequest(prompt="Test", top_p=1.1)

    def test_invalid_penalty_ranges(self):
        """Test StreamRequest validation with invalid penalties"""
        with raises(ValidationError):
            StreamRequest(prompt="Test", frequency_penalty=-2.1)

        with raises(ValidationError):
            StreamRequest(prompt="Test", frequency_penalty=2.1)

        with raises(ValidationError):
            StreamRequest(prompt="Test", presence_penalty=-2.1)

        with raises(ValidationError):
            StreamRequest(prompt="Test", presence_penalty=2.1)

    def test_invalid_timeout(self):
        """Test StreamRequest validation with invalid timeout"""
        with raises(ValidationError):
            StreamRequest(prompt="Test", timeout=0)

        with raises(ValidationError):
            StreamRequest(prompt="Test", timeout=-1)


class TestStreamingMethods:
    """Test suite for streaming methods in OllamaAgentExecutor"""

    @pytest.mark.asyncio
    @patch("ollama2a.agent_executor.HybridOllamaManager")
    @patch("ollama2a.agent_executor.AsyncOpenAI")
    @patch("ollama2a.agent_executor.OpenAIProvider")
    @patch("ollama2a.agent_executor.OpenAIModel")
    @patch("ollama2a.agent_executor.Agent")
    @patch("ollama2a.agent_executor.FastA2A")
    @patch("ollama2a.agent_executor.list")
    @patch("ollama2a.agent_executor.pull")
    async def test_stream_endpoint_added(
        self,
        mock_pull,
        mock_list,
        mock_fasta2a,
        mock_agent,
        mock_openai_model,
        mock_openai_provider,
        mock_async_openai,
        mock_manager_class,
    ):
        """Test that stream endpoint is added to the app"""
        # Mock the list response
        mock_models_response = Mock()
        mock_model = Mock()
        mock_model.model = "qwen3:0.6b"
        mock_models_response.models = [mock_model]
        mock_list.return_value = mock_models_response

        # Mock manager and other components
        mock_manager_instance = Mock()
        mock_manager_instance.base_url = "http://localhost:11434"
        mock_manager_class.return_value = mock_manager_instance

        mock_agent_instance = Mock()
        mock_app_instance = Mock()
        mock_app_instance.routes = []
        mock_agent_instance.to_a2a.return_value = mock_app_instance
        mock_agent.return_value = mock_agent_instance

        executor = OllamaAgentExecutor()

        # Check that stream endpoint was added
        assert len(mock_app_instance.routes) == 1
        route = mock_app_instance.routes[0]
        assert route.path == "/stream"
        assert "POST" in route.methods

    @pytest.mark.asyncio
    async def test_generate_sse_stream_success(self):
        """Test successful SSE stream generation"""
        import asyncio

        # Create executor without triggering post_init
        executor = OllamaAgentExecutor.__new__(OllamaAgentExecutor)
        executor.ollama_model = "test-model"
        executor.system_prompt = "Test prompt"

        # Mock client
        mock_client = AsyncMock()
        executor.client = mock_client

        # Mock streaming response
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.content = "Hello"

        async def mock_stream_generator():
            yield mock_chunk

        mock_stream = mock_stream_generator()

        # Mock the create method to return our stream
        async def mock_create(**kwargs):
            return mock_stream

        mock_client.chat.completions.create = mock_create

        # Create request
        request = StreamRequest(prompt="Test", timeout=60)

        # Generate stream
        events = []
        async for event in executor._generate_sse_stream(request):
            events.append(event)

        # Verify events
        assert len(events) >= 3  # start, content, complete/end
        assert "event: start" in events[0]
        assert "event: content" in events[1]
        assert '"text": "Hello"' in events[1]

    @pytest.mark.asyncio
    async def test_generate_sse_stream_timeout(self):
        """Test SSE stream generation with timeout"""
        import asyncio
        from asyncio import TimeoutError as ATimeoutError

        # Create executor without triggering post_init
        executor = OllamaAgentExecutor.__new__(OllamaAgentExecutor)
        executor.ollama_model = "test-model"
        executor.system_prompt = "Test prompt"

        # Mock client that times out
        mock_client = AsyncMock()
        executor.client = mock_client

        # Mock the create method to raise TimeoutError
        mock_client.chat.completions.create.side_effect = ATimeoutError("Stream timeout")

        # Create request with short timeout
        request = StreamRequest(prompt="Test", timeout=0.1)

        # Generate stream
        events = []
        async for event in executor._generate_sse_stream(request):
            events.append(event)

        # Verify timeout event was sent
        assert any("event: timeout" in e for e in events)
        assert any("Stream timeout" in e or "0.1 seconds" in e for e in events)
        assert any("event: end" in e for e in events)

    @pytest.mark.asyncio
    async def test_generate_sse_stream_error(self):
        """Test SSE stream generation with error"""
        # Create executor without triggering post_init
        executor = OllamaAgentExecutor.__new__(OllamaAgentExecutor)
        executor.ollama_model = "test-model"
        executor.system_prompt = "Test prompt"

        # Mock client that raises error
        mock_client = AsyncMock()
        executor.client = mock_client

        async def error_create(**kwargs):
            raise Exception("API Error")

        mock_client.chat.completions.create = error_create

        # Create request
        request = StreamRequest(prompt="Test")

        # Generate stream
        events = []
        async for event in executor._generate_sse_stream(request):
            events.append(event)

        # Verify error event was sent
        assert any("event: error" in e for e in events)
        assert any("API Error" in e for e in events)
        assert any("event: end" in e for e in events)


class TestFormatSSEEvent:
    """Test suite for format_sse_event function"""

    def test_format_start_event(self):
        """Test formatting start event"""
        result = format_sse_event("start", {"timestamp": "2024-01-01T00:00:00Z"})
        assert result == 'event: start\ndata: {"timestamp": "2024-01-01T00:00:00Z"}\n\n'

    def test_format_content_event(self):
        """Test formatting content event"""
        result = format_sse_event("content", {"text": "Hello", "chunk_id": 1})
        assert result == 'event: content\ndata: {"text": "Hello", "chunk_id": 1}\n\n'

    def test_format_complete_event(self):
        """Test formatting complete event"""
        result = format_sse_event("complete", {"full_response": "Done", "total_chunks": 5})
        assert result == 'event: complete\ndata: {"full_response": "Done", "total_chunks": 5}\n\n'

    def test_format_error_event(self):
        """Test formatting error event"""
        result = format_sse_event("error", {"error": "Something went wrong"})
        assert result == 'event: error\ndata: {"error": "Something went wrong"}\n\n'

    def test_format_timeout_event(self):
        """Test formatting timeout event"""
        result = format_sse_event("timeout", {"error": "Timeout", "timeout_seconds": 60})
        assert result == 'event: timeout\ndata: {"error": "Timeout", "timeout_seconds": 60}\n\n'

    def test_format_end_event(self):
        """Test formatting end event"""
        result = format_sse_event("end", {"timestamp": "2024-01-01T00:01:00Z"})
        assert result == 'event: end\ndata: {"timestamp": "2024-01-01T00:01:00Z"}\n\n'


if __name__ == "__main__":
    main([__file__])
