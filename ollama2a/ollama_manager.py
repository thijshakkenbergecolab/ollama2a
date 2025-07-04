from subprocess import Popen, PIPE, TimeoutExpired
from dataclasses import dataclass
from atexit import register
from logging import warning
from ollama import Client
from os import environ
from time import sleep
from httpx import get


@dataclass
class HybridOllamaManager:
    """Hybrid Ollama manager to handle server lifecycle and model execution"""

    server_process: Popen = None
    client: Client = None
    host: str = "localhost"
    port: int = 11434

    @property
    def base_url(self) -> str:
        """Construct base URL for Ollama API"""
        return f"http://{self.host}:{self.port}"

    def ensure_server_running(self):
        """Ensure ollama server is running, start if needed"""
        if not self._is_server_running():
            self._start_server()

        # Initialize client
        self.client = Client()

    def _start_server(self):
        """Start ollama server process"""
        # define environment variables for base URL
        env = environ.copy()
        env["OLLAMA_HOST"] = self.base_url

        self.server_process = Popen(
            ["ollama", "serve"], stdout=PIPE, stderr=PIPE, env=env
        )

        # Wait for server to be ready
        for _ in range(30):  # 30 second timeout
            if self._is_server_running():
                break
            sleep(1)
        else:
            raise RuntimeError("Server failed to start")

        # Register cleanup
        register(self.cleanup)

    def run_model(self, model, prompt):
        """Run model using official library"""
        self.ensure_server_running()

        try:
            response = self.client.generate(model=model, prompt=prompt)
            return response["response"]
        except Exception as e:
            raise RuntimeError(f"Model execution failed: {e}")

    def chat(self, model, messages):
        """Chat using official library"""
        self.ensure_server_running()

        try:
            response = self.client.chat(model=model, messages=messages)
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Chat failed: {e}")

    def cleanup(self):
        """Clean up server process"""
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except TimeoutExpired:
                self.server_process.kill()

    def _is_server_running(self):
        try:
            response = get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except Exception as e:
            warning(f"Server check failed: {e}")
            return False


if __name__ == "__main__":
    # Usage
    manager = HybridOllamaManager()
    out = manager.run_model("qwen3:0.6b", "Explain quantum computing")
    print(out)
