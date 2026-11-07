
from types import SimpleNamespace

import pytest
import requests

from src.config import settings
from src.llm.factory import get_llm
from src.llm.ollama_provider import OllamaProvider


def test_ollama_provider_initialization():
    provider = OllamaProvider()

    assert provider.model == "qwen2.5:7b"
    assert provider.base_url == "http://localhost:11434"


def test_ollama_generate_success(monkeypatch):
    def mock_post(url, json, timeout):
        assert url == "http://localhost:11434/api/generate"
        assert json["model"] == "qwen2.5:7b"
        assert json["prompt"] == "Explain RAG"
        assert json["stream"] is False
        assert json["options"]["temperature"] == 0.2
        assert timeout == 180

        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": "RAG combines retrieval and generation."}
        )

    monkeypatch.setattr(
        "src.llm.ollama_provider.requests.post",
        mock_post
    )

    provider = OllamaProvider()

    result = provider.generate(
        "Explain RAG",
        temperature=0.2
    )

    assert result == "RAG combines retrieval and generation."


def test_ollama_rejects_empty_prompt():
    provider = OllamaProvider()

    with pytest.raises(ValueError, match="prompt cannot be empty"):
        provider.generate("   ")


def test_ollama_connection_error(monkeypatch):
    def mock_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Connection refused")

    monkeypatch.setattr(
        "src.llm.ollama_provider.requests.post",
        mock_post
    )

    provider = OllamaProvider()

    with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
        provider.generate("Test connection")


def test_factory_selects_ollama(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "ollama")

    provider = get_llm()

    assert isinstance(provider, OllamaProvider)


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "unknown")

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_llm()
