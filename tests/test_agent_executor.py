from unittest.mock import Mock, patch, MagicMock
from pytest import raises, main

from ollama2a.agent_executor import OllamaAgentExecutor


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


if __name__ == "__main__":
    main([__file__])
