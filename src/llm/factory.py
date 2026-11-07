from src.config import settings
from src.llm.api_provider import OpenAIProvider
from src.llm.ollama_provider import OllamaProvider


def get_llm():
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return OpenAIProvider()
    if provider == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
