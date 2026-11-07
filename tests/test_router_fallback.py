import pytest
from src.llm.base import LLMProvider
from src.query_engine.query_router import QueryRouter


class FailingLLM(LLMProvider):
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # Simule un LLM qui renvoie du texte non structuré
        return "Désolé, je ne sais pas répondre au format JSON."


def test_router_fallback_sql():
    router = QueryRouter(FailingLLM())
    decision = router.route("Quel est le montant total des transactions ?")
    assert decision.intent == "SQL_QUERY"


def test_router_fallback_general():
    router = QueryRouter(FailingLLM())
    decision = router.route("Quelles sont les colonnes disponibles ?")
    assert decision.intent == "GENERAL_QUESTION"


def test_router_fallback_default_hybrid():
    router = QueryRouter(FailingLLM())
    decision = router.route("Recherche des achats divers")
    assert decision.intent == "HYBRID_SEARCH"
