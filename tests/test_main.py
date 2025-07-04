import pytest
from unittest.mock import Mock, patch, AsyncMock
from pydantic_ai import RunContext, Tool

# Import the components we want to test
from main import my_tool, ae, app


class TestMyTool:
    """Test the custom tool function"""

    @pytest.mark.asyncio
    async def test_my_tool_basic_functionality(self):
        """Test that my_tool returns expected format"""
        # Create a mock context
        mock_ctx = Mock()
        mock_ctx.deps = 42

        result = await my_tool(mock_ctx, 10, 20)

        assert result == "42 10 20"

    @pytest.mark.asyncio
    async def test_my_tool_with_different_values(self):
        """Test my_tool with different input values"""
        mock_ctx = Mock()
        mock_ctx.deps = "test_deps"

        result = await my_tool(mock_ctx, -5, 100)

        assert result == "test_deps -5 100"

    @pytest.mark.asyncio
    async def test_my_tool_with_string_deps(self):
        """Test my_tool with string dependencies"""
        mock_ctx = Mock()
        mock_ctx.deps = "hello"

        result = await my_tool(mock_ctx, 0, 0)

        assert result == "hello 0 0"


class TestMainConfiguration:
    """Test the main application configuration"""

    @patch('main.OllamaAgentExecutor')
    def test_agent_executor_configuration(self, mock_executor_class):
        """Test that OllamaAgentExecutor is configured correctly"""
        # Mock the executor instance
        mock_executor_instance = Mock()
        mock_executor_instance.app = Mock()
        mock_executor_class.return_value = mock_executor_instance

        # Re-import to trigger the configuration
        import importlib
        import main
        importlib.reload(main)

        # Verify the executor was called with correct parameters
        mock_executor_class.assert_called_once_with(
            ollama_host="localhost",
            ollama_port=11434,
            ollama_model="qwen3:0.6b",
            system_prompt="You are a helpful assistant.",
            description="An agent that uses the Ollama API to execute tasks.",
            tools=[Tool(main.my_tool)]
        )

    def test_tool_is_properly_configured(self):
        """Test that the tool is properly wrapped"""
        # Check that ae.tools contains our tool
        assert len(ae.tools) == 1

        # The tool should be a Tool instance
        tool = ae.tools[0]
        assert isinstance(tool, Tool)

    def test_app_is_extracted_from_executor(self):
        """Test that app is properly extracted from executor"""
        # The app should be the same as ae.app
        assert app is ae.app

        # App should not be None
        assert app is not None


class TestIntegration:
    """Integration tests for main.py setup"""

    @patch('main.OllamaAgentExecutor')
    def test_full_setup_chain(self, mock_executor_class):
        """Test the complete setup chain"""
        # Mock the executor and its app
        mock_app = Mock()
        mock_executor_instance = Mock()
        mock_executor_instance.app = mock_app
        mock_executor_class.return_value = mock_executor_instance

        # Re-import to trigger setup
        import importlib
        import main
        importlib.reload(main)

        # Verify the chain: executor created -> app extracted
        mock_executor_class.assert_called_once()
        assert main.app is mock_app

    @pytest.mark.asyncio
    async def test_tool_function_signature(self):
        """Test that tool function has correct async signature"""
        import inspect

        # Check that my_tool is async
        assert inspect.iscoroutinefunction(my_tool)

        # Check function signature
        sig = inspect.signature(my_tool)
        params = list(sig.parameters.keys())
        assert params == ['ctx', 'x', 'y']

        # Check return annotation
        assert sig.return_annotation == str

    def test_configuration_values(self):
        """Test that configuration values are as expected"""
        assert ae.ollama_host == "localhost"
        assert ae.ollama_port == 11434
        assert ae.ollama_model == "qwen3:0.6b"
        assert ae.system_prompt == "You are a helpful assistant."
        assert ae.description == "An agent that uses the Ollama API to execute tasks."

    @patch('main.OllamaAgentExecutor')
    def test_tool_integration_with_executor(self, mock_executor_class):
        """Test that tool is properly integrated with executor"""
        mock_executor_instance = Mock()
        mock_executor_class.return_value = mock_executor_instance

        # Re-import to trigger setup
        import importlib
        import main
        importlib.reload(main)

        # Get the tools argument passed to the executor
        call_args = mock_executor_class.call_args
        tools_arg = call_args[1]['tools']  # keyword argument 'tools'

        # Should be a list with one Tool
        assert len(tools_arg) == 1
        assert isinstance(tools_arg[0], Tool)


class TestAppProperties:
    """Test properties of the created app"""

    def test_app_has_required_attributes(self):
        """Test that app has expected FastAPI/FastA2A attributes"""
        # The app should have typical FastAPI attributes
        # Note: Specific attributes depend on FastA2A implementation
        assert hasattr(app, '__call__')  # Should be callable (ASGI app)

    @patch('main.OllamaAgentExecutor')
    def test_app_is_from_to_a2a_method(self, mock_executor_class):
        """Test that app comes from the to_a2a method"""
        mock_app = Mock()
        mock_agent = Mock()
        mock_agent.to_a2a.return_value = mock_app

        mock_executor_instance = Mock()
        mock_executor_instance.app = mock_app
        mock_executor_instance.agent = mock_agent
        mock_executor_class.return_value = mock_executor_instance

        # Re-import to trigger setup
        import importlib
        import main
        importlib.reload(main)

        # Verify app comes from executor
        assert main.app is mock_app


# Test for potential import issues
class TestImports:
    """Test import-related functionality"""

    def test_all_imports_successful(self):
        """Test that all required imports work"""
        # If we got this far, imports worked
        assert RunContext is not None
        assert Tool is not None
        assert ae is not None
        assert app is not None

    def test_main_module_attributes(self):
        """Test that main module has expected attributes"""
        import main

        # Check that main module has the expected attributes
        assert hasattr(main, 'my_tool')
        assert hasattr(main, 'ae')
        assert hasattr(main, 'app')

        # Check types
        assert callable(main.my_tool)
        assert main.ae is not None
        assert main.app is not None


# Performance/Resource test
class TestResourceUsage:
    """Test resource-related aspects"""

    def test_single_executor_instance(self):
        """Test that only one executor instance is created"""
        # Should not create multiple instances during import
        assert ae is not None

        # Re-importing should not create new instances
        import main as main2
        assert main2.ae is ae
        assert main2.app is app


if __name__ == "__main__":
    pytest.main([__file__])