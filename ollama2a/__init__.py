"""
ollama2a - A Python library for running Ollama agents with automated server management.
"""

__version__ = "0.1.0"
__author__ = "Thijs Hakkenberg"
__email__ = "thijs.hakkenberg@ecolab.com"

from .ollama_manager import HybridOllamaManager
from .agent_executor import OllamaAgentExecutor

__all__ = [
    "HybridOllamaManager",
    "OllamaAgentExecutor",
    "__version__",
]
