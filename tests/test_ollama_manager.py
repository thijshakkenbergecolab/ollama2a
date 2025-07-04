from os import environ
from unittest.mock import Mock, patch, MagicMock
from subprocess import Popen, TimeoutExpired, PIPE
from pytest import raises, main
from httpx import Response

from ollama2a.ollama_manager import HybridOllamaManager


class TestHybridOllamaManager:
    """Test suite for HybridOllamaManager"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.manager = HybridOllamaManager()

    def test_init_default_values(self):
        """Test manager initialization with default values"""
        manager = HybridOllamaManager()
        assert manager.host == "localhost"
        assert manager.port == 11434
        assert manager.server_process is None
        assert manager.client is None

    def test_init_custom_values(self):
        """Test manager initialization with custom values"""
        manager = HybridOllamaManager(host="0.0.0.0", port=8080)
        assert manager.host == "0.0.0.0"
        assert manager.port == 8080

    def test_base_url_property(self):
        """Test base_url property construction"""
        manager = HybridOllamaManager(host="192.168.1.100", port=9999)
        assert manager.base_url == "http://192.168.1.100:9999"

    @patch("ollama2a.ollama_manager.get")
    def test_is_server_running_success(self, mock_get):
        """Test server running check when server is available"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = self.manager._is_server_running()

        assert result is True
        mock_get.assert_called_once_with(
            f"{self.manager.base_url}/api/version", timeout=5
        )

    @patch("ollama2a.ollama_manager.get")
    @patch("ollama2a.ollama_manager.warning")
    def test_is_server_running_failure(self, mock_warning, mock_get):
        """Test server running check when server is not available"""
        # Mock exception
        mock_get.side_effect = Exception("Connection failed")

        result = self.manager._is_server_running()

        assert result is False
        mock_warning.assert_called_once()

    @patch("ollama2a.ollama_manager.get")
    def test_is_server_running_wrong_status(self, mock_get):
        """Test server running check with non-200 status code"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = self.manager._is_server_running()

        assert result is False

    @patch("ollama2a.ollama_manager.Client")
    @patch.object(HybridOllamaManager, "_start_server")
    @patch.object(HybridOllamaManager, "_is_server_running")
    def test_ensure_server_running_server_not_running(
        self, mock_is_running, mock_start, mock_client
    ):
        """Test ensure_server_running when server is not running"""
        mock_is_running.return_value = False
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        self.manager.ensure_server_running()

        mock_start.assert_called_once()
        mock_client.assert_called_once()
        assert self.manager.client == mock_client_instance

    @patch("ollama2a.ollama_manager.Client")
    @patch.object(HybridOllamaManager, "_start_server")
    @patch.object(HybridOllamaManager, "_is_server_running")
    def test_ensure_server_running_server_already_running(
        self, mock_is_running, mock_start, mock_client
    ):
        """Test ensure_server_running when server is already running"""
        mock_is_running.return_value = True
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        self.manager.ensure_server_running()

        mock_start.assert_not_called()
        mock_client.assert_called_once()

    @patch("ollama2a.ollama_manager.register")
    @patch("ollama2a.ollama_manager.sleep")
    @patch("ollama2a.ollama_manager.Popen")
    @patch.object(HybridOllamaManager, "_is_server_running")
    def test_start_server_success(
        self, mock_is_running, mock_popen, mock_sleep, mock_register
    ):
        """Test successful server startup"""
        # Mock server becomes available on first check
        mock_is_running.return_value = True
        mock_process = Mock()
        mock_popen.return_value = mock_process

        with patch.dict(environ, {}, clear=False):
            self.manager._start_server()

        # Check that Popen was called with correct arguments
        mock_popen.assert_called_once()
        call_args, call_kwargs = mock_popen.call_args

        # Verify command
        assert call_args[0] == ["ollama", "serve"]

        # Verify PIPE constants
        assert call_kwargs["stdout"] == PIPE
        assert call_kwargs["stderr"] == PIPE

        # Verify env was passed and contains OLLAMA_HOST
        assert "env" in call_kwargs
        assert "OLLAMA_HOST" in call_kwargs["env"]
        assert call_kwargs["env"]["OLLAMA_HOST"] == "http://localhost:11434"

        assert self.manager.server_process == mock_process
        mock_register.assert_called_once_with(self.manager.cleanup)

    @patch("ollama2a.ollama_manager.sleep")
    @patch("ollama2a.ollama_manager.Popen")
    @patch.object(HybridOllamaManager, "_is_server_running")
    def test_start_server_timeout(self, mock_is_running, mock_popen, mock_sleep):
        """Test server startup timeout"""
        # Server never becomes available
        mock_is_running.return_value = False
        mock_process = Mock()
        mock_popen.return_value = mock_process

        with raises(RuntimeError, match="Server failed to start"):
            self.manager._start_server()

        # Should have tried 30 times
        assert mock_is_running.call_count == 30

    @patch.object(HybridOllamaManager, "ensure_server_running")
    def test_run_model_success(self, mock_ensure):
        """Test successful model execution"""
        # Mock client and response
        mock_client = Mock()
        mock_response = {"response": "Quantum computing explanation..."}
        mock_client.generate.return_value = mock_response
        self.manager.client = mock_client

        result = self.manager.run_model("qwen3:0.6b", "Explain quantum computing")

        assert result == "Quantum computing explanation..."
        mock_ensure.assert_called_once()
        mock_client.generate.assert_called_once_with(
            model="qwen3:0.6b", prompt="Explain quantum computing"
        )

    @patch.object(HybridOllamaManager, "ensure_server_running")
    def test_run_model_failure(self, mock_ensure):
        """Test model execution failure"""
        # Mock client that raises exception
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("Model not found")
        self.manager.client = mock_client

        with raises(RuntimeError, match="Model execution failed"):
            self.manager.run_model("invalid-model", "test prompt")

    @patch.object(HybridOllamaManager, "ensure_server_running")
    def test_chat_success(self, mock_ensure):
        """Test successful chat execution"""
        # Mock client and response
        mock_client = Mock()
        mock_response = {"message": {"content": "Hello! How can I help you?"}}
        mock_client.chat.return_value = mock_response
        self.manager.client = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        result = self.manager.chat("qwen3:0.6b", messages)

        assert result == "Hello! How can I help you?"
        mock_ensure.assert_called_once()
        mock_client.chat.assert_called_once_with(model="qwen3:0.6b", messages=messages)

    @patch.object(HybridOllamaManager, "ensure_server_running")
    def test_chat_failure(self, mock_ensure):
        """Test chat execution failure"""
        # Mock client that raises exception
        mock_client = Mock()
        mock_client.chat.side_effect = Exception("Chat failed")
        self.manager.client = mock_client

        with raises(RuntimeError, match="Chat failed"):
            self.manager.chat("model", [])

    def test_cleanup_process_running(self):
        """Test cleanup when process is running"""
        # Mock running process
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.wait.return_value = 0
        self.manager.server_process = mock_process

        self.manager.cleanup()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)

    def test_cleanup_process_timeout(self):
        """Test cleanup when process doesn't respond to terminate"""
        # Mock process that doesn't terminate gracefully
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = TimeoutExpired(cmd="ollama", timeout=5)
        self.manager.server_process = mock_process

        self.manager.cleanup()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_cleanup_no_process(self):
        """Test cleanup when no process exists"""
        self.manager.server_process = None

        # Should not raise any exceptions
        self.manager.cleanup()

    def test_cleanup_process_already_stopped(self):
        """Test cleanup when process already stopped"""
        mock_process = Mock()
        mock_process.poll.return_value = 0  # Process already stopped
        self.manager.server_process = mock_process

        self.manager.cleanup()

        # Should not call terminate or kill
        mock_process.terminate.assert_not_called()
        mock_process.kill.assert_not_called()

    @patch("ollama2a.ollama_manager.environ")
    @patch("ollama2a.ollama_manager.Popen")
    @patch.object(HybridOllamaManager, "_is_server_running")
    def test_start_server_environment_variables(
        self, mock_is_running, mock_popen, mock_environ
    ):
        """Test that environment variables are set correctly"""
        mock_is_running.return_value = True
        mock_environ.copy.return_value = {"PATH": "/usr/bin"}

        manager = HybridOllamaManager(host="0.0.0.0", port=8080)
        manager._start_server()

        # Verify environment was updated
        expected_env = {"PATH": "/usr/bin", "OLLAMA_HOST": "http://0.0.0.0:8080"}
        call_args = mock_popen.call_args
        assert call_args[1]["env"] == expected_env


class TestIntegration:
    """Integration-style tests that test multiple components together"""

    @patch("ollama2a.ollama_manager.get")
    @patch("ollama2a.ollama_manager.Client")
    @patch("ollama2a.ollama_manager.Popen")
    def test_full_workflow_server_not_running(self, mock_popen, mock_client, mock_get):
        """Test full workflow when server needs to be started"""
        # Setup mocks
        mock_get.side_effect = [
            Exception("Connection refused"),  # First check - server not running
            Mock(status_code=200),  # Second check - server is running
        ]

        mock_process = Mock()
        mock_popen.return_value = mock_process

        mock_client_instance = Mock()
        mock_client_instance.generate.return_value = {"response": "Test response"}
        mock_client.return_value = mock_client_instance

        # Execute
        manager = HybridOllamaManager()
        with patch("ollama2a.ollama_manager.sleep"):  # Speed up the test
            result = manager.run_model("test-model", "test prompt")

        # Verify
        assert result == "Test response"
        mock_popen.assert_called_once()
        mock_client_instance.generate.assert_called_once()


if __name__ == "__main__":
    main([__file__])
