import json

from src.llm.base import LLMProvider
from src.query_engine.schemas import RouteDecision


ROUTER_PROMPT = """You are a router that classifies a user's question into exactly one of 4 categories:

1. GENERAL_QUESTION:
   The user asks about table schema, available columns, or dataset structure.
   Examples: "Quelles sont les colonnes disponibles ?", "Quel est le schéma des données ?"

2. SQL_QUERY:
   The user asks for exact aggregations, counts, averages, or rankings over the whole table or standard columns (merchant, category, date, amount), WITHOUT searching for a fuzzy textual topic.
   Examples:
   - "Combien de transactions au total ?" -> SQL_QUERY
   - "Quel est le montant moyen des transactions ?" -> SQL_QUERY
   - "Donne-moi le top 5 des commerçants par dépenses" -> SQL_QUERY

3. VECTOR_SEARCH:
   The user wants to find, list, or discover transactions matching a specific topic, keyword, or description, WITHOUT asking for a total, sum, or calculation.
   Examples:
   - "Trouve les transactions qui parlent d'achats de billets d'avion ou de vacances" -> VECTOR_SEARCH
   - "Cherche les dépenses liées aux abonnements de streaming ou musique" -> VECTOR_SEARCH

4. HYBRID_SEARCH:
   The user asks for a total, sum, or calculation on a specific fuzzy topic, keyword, or theme that requires searching within textual descriptions.
   Examples:
   - "Quel est le total dépensé pour les achats de voyages ?" -> HYBRID_SEARCH
   - "Montant cumulé des transactions relatives aux restaurants et cafés" -> HYBRID_SEARCH

STRICT RULES:
- If asking to find/search transactions on a topic WITHOUT calculating a sum -> VECTOR_SEARCH.
- If asking for a total/sum on a fuzzy topic (like voyages, restaurants, cadeaux) -> HYBRID_SEARCH.
- If asking for a global count/average/ranking without a fuzzy topic -> SQL_QUERY.
- Output ONLY valid JSON. No markdown code blocks.

Format:
{{
  "intent": "SQL_QUERY" | "VECTOR_SEARCH" | "HYBRID_SEARCH" | "GENERAL_QUESTION",
  "rationale": "one sentence reason",
  "search_query": "extracted keywords or null"
}}

Question:
{question}
"""


class QueryRouter:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def _fallback_route(self, question: str) -> RouteDecision:
        """Fallback heuristique si le LLM ne renvoie pas un JSON exploitable."""
        q_lower = question.lower()

        # Questions générales sur le schéma
        general_keywords = ["schéma", "colonne", "colonnes", "structure", "tables", "variables"]
        if any(kw in q_lower for kw in general_keywords):
            return RouteDecision(
                intent="GENERAL_QUESTION",
                rationale="Fallback heuristique : question sur le schéma ou les colonnes."
            )

        # Mots-clés thématiques flous
        fuzzy_themes = ["voyage", "vacance", "restaurant", "café", "abonnement", "streaming", "cadeau", "billet"]
        has_fuzzy_theme = any(ft in q_lower for ft in fuzzy_themes)

        # Mots d'agrégation
        calc_keywords = ["combien", "total", "somme", "moyenne", "montant cumulé", "dépensé"]
        has_calc = any(ck in q_lower for ck in calc_keywords)

        if has_fuzzy_theme and has_calc:
            return RouteDecision(
                intent="HYBRID_SEARCH",
                rationale="Fallback heuristique : calcul sur un thème textuel flou.",
                search_query=question
            )

        if has_fuzzy_theme or any(w in q_lower for w in ["trouve", "cherche", "recherche", "quelles sont les"]):
            return RouteDecision(
                intent="VECTOR_SEARCH",
                rationale="Fallback heuristique : recherche sémantique de descriptions.",
                search_query=question
            )

        # Calculs et statistiques purs
        if has_calc or any(rk in q_lower for rk in ["top", "maximum", "minimum", "classement"]):
            return RouteDecision(
                intent="SQL_QUERY",
                rationale="Fallback heuristique : calcul exact ou ranking SQL."
            )

        return RouteDecision(
            intent="HYBRID_SEARCH",
            rationale="Fallback heuristique par défaut.",
            search_query=question
        )

    def route(self, question: str) -> RouteDecision:

        if not question or not question.strip():
            raise ValueError("The question cannot be empty.")

        try:
            raw_response = self.llm.generate(
                ROUTER_PROMPT.format(question=question),
                temperature=0.0,
            ).strip()

            start = raw_response.find("{")
            end = raw_response.rfind("}")

            if start != -1 and end != -1 and end > start:
                data = json.loads(raw_response[start:end + 1])
                return RouteDecision.model_validate(data)

        except Exception:
            pass

        return self._fallback_route(question)
