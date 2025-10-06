from unittest.mock import Mock, patch, AsyncMock
from pydantic_ai import RunContext, Tool
from pytest import main as pytest_main, mark
import importlib
import sys


class TestMyTool:
    """Test the custom tool function"""

    @mark.asyncio
    async def test_my_tool_basic_functionality(self):
        """Test that my_tool returns expected format"""
        # Import inside test to avoid module loading issues
        from ollama2a.main import my_tool

        # Create a mock context
        mock_ctx = Mock()
        mock_ctx.deps = 42

        result = await my_tool(mock_ctx, 10, 20)

        assert result == "42 10 20"

    @mark.asyncio
    async def test_my_tool_with_different_values(self):
        """Test my_tool with different input values"""
        from ollama2a.main import my_tool

        mock_ctx = Mock()
        mock_ctx.deps = "test_deps"

        result = await my_tool(mock_ctx, -5, 100)

        assert result == "test_deps -5 100"

    @mark.asyncio
    async def test_my_tool_with_string_deps(self):
        """Test my_tool with string dependencies"""
        from ollama2a.main import my_tool

        mock_ctx = Mock()
        mock_ctx.deps = "hello"

        result = await my_tool(mock_ctx, 0, 0)

        assert result == "hello 0 0"


class TestMainConfiguration:
    """Test the main application configuration"""

    def test_tool_is_properly_configured(self):
        """Test that the tool is properly wrapped"""
        from ollama2a.main import ae

        # Check that ae.tools contains our tool
        assert len(ae.tools) == 1

        # The tool should be a Tool instance
        tool = ae.tools[0]
        assert isinstance(tool, Tool)

    def test_app_is_extracted_from_executor(self):
        """Test that app is properly extracted from executor"""
        from ollama2a.main import ae, app

        # The app should be the same as ae.app
        assert app is ae.app

        # App should not be None
        assert app is not None

    def test_configuration_values(self):
        """Test that configuration values are as expected"""
        from ollama2a.main import ae

        assert ae.ollama_host == "localhost"
        assert ae.ollama_port == 11434
        assert ae.ollama_model == "qwen3:0.6b"
        assert ae.system_prompt == "You are a helpful assistant."
        assert ae.description == "An agent that uses the Ollama API to execute tasks."


class TestIntegration:
    """Integration tests for main.py setup"""

    @mark.asyncio
    async def test_tool_function_signature(self):
        """Test that tool function has correct async signature"""
        from ollama2a.main import my_tool
        import inspect

        # Check that my_tool is async
        assert inspect.iscoroutinefunction(my_tool)

        # Check function signature
        sig = inspect.signature(my_tool)
        params = list(sig.parameters.keys())
        assert params == ["ctx", "x", "y"]

        # Check return annotation
        assert sig.return_annotation == str


class TestAppProperties:
    """Test properties of the created app"""

    def test_app_has_required_attributes(self):
        """Test that app has expected FastAPI/FastA2A attributes"""
        from ollama2a.main import app

        # The app should have typical FastAPI attributes
        assert hasattr(app, "__call__")  # Should be callable (ASGI app)


class TestImports:
    """Test import-related functionality"""

    def test_all_imports_successful(self):
        """Test that all required imports work"""
        from ollama2a.main import my_tool, ae, app

        # If we got this far, imports worked
        assert RunContext is not None
        assert Tool is not None
        assert ae is not None
        assert app is not None

    def test_main_module_attributes(self):
        """Test that main module has expected attributes"""
        import ollama2a.main as main

        # Check that main module has the expected attributes
        assert hasattr(main, "my_tool")
        assert hasattr(main, "ae")
        assert hasattr(main, "app")

        # Check types
        assert callable(main.my_tool)
        assert main.ae is not None
        assert main.app is not None


class TestModuleBehavior:
    """Test module-level behavior"""

    def test_module_creates_single_instances(self):
        """Test that module creates consistent instances"""
        # Import multiple times should give same instances
        from ollama2a.main import ae as ae1, app as app1
        from ollama2a.main import ae as ae2, app as app2

        assert ae1 is ae2
        assert app1 is app2

    def test_executor_configuration_integration(self):
        """Test the executor is configured correctly"""
        from ollama2a.main import ae
        from pydantic_ai import Tool

        # Verify configuration
        assert ae.ollama_host == "localhost"
        assert ae.ollama_port == 11434
        assert ae.ollama_model == "qwen3:0.6b"

        # Verify tool integration
        assert len(ae.tools) == 1
        assert isinstance(ae.tools[0], Tool)


# Test class for mock-based testing (if needed for edge cases)
class TestWithMocking:
    """Tests that require mocking for specific scenarios"""

    @patch("ollama2a.agent_executor.HybridOllamaManager")
    @patch("ollama2a.agent_executor.AsyncOpenAI")
    @patch("ollama2a.agent_executor.OpenAIProvider")
    @patch("ollama2a.agent_executor.OpenAIModel")
    @patch("ollama2a.agent_executor.Agent")
    @patch("ollama2a.agent_executor.list")
    def test_mocked_execution_path(
        self,
        mock_list,
        mock_agent,
        mock_model,
        mock_provider,
        mock_openai,
        mock_manager,
    ):
        """Test execution with all external dependencies mocked"""
        # Mock the list response to show model exists
        mock_models_response = Mock()
        mock_model_obj = Mock()
        mock_model_obj.model = "qwen3:0.6b"
        mock_models_response.models = [mock_model_obj]
        mock_list.return_value = mock_models_response

        # Mock manager instance
        mock_manager_instance = Mock()
        mock_manager_instance.base_url = "http://localhost:11434"
        mock_manager.return_value = mock_manager_instance

        # Mock agent instance
        mock_agent_instance = Mock()
        mock_agent_instance.to_a2a.return_value = Mock()
        mock_agent.return_value = mock_agent_instance

        # Import fresh module (this will use mocks)
        if "ollama2a.main" in sys.modules:
            del sys.modules["ollama2a.main"]

        import ollama2a.main as main

        # Verify the mocked components were used
        mock_manager.assert_called_once()
        assert main.ae is not None
        assert main.app is not None


if __name__ == "__main__":
    pytest_main([__file__])
