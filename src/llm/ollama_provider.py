
import requests

from src.config import settings
from src.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    """LLM provider using a locally hosted Ollama model."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.ollama_model
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.timeout = 180

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0
    ) -> str:
        if not prompt.strip():
            raise ValueError("The prompt cannot be empty.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "")

            if not generated_text:
                raise RuntimeError(
                    "Ollama returned an empty response."
                )

            return generated_text

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Check that the Ollama service is running."
            ) from exc

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Ollama API error: {response.text}"
            ) from exc
