from os.path import dirname, abspath
from sys import path
from unittest.mock import Mock
from pytest import fixture


# Add parent directory to path for imports
path.insert(0, dirname(dirname(abspath(__file__))))


@fixture
def mock_ollama_client():
    """Fixture providing a mock ollama client"""
    client = Mock()
    client.generate.return_value = {"response": "Mock response"}
    client.chat.return_value = {"message": {"content": "Mock chat response"}}
    return client


@fixture
def mock_subprocess():
    """Fixture providing a mock subprocess"""
    process = Mock()
    process.poll.return_value = None  # Process is running
    process.wait.return_value = 0
    return process


@fixture
def sample_chat_messages():
    """Fixture providing sample chat messages"""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with Python?"},
    ]
