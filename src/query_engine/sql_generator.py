import re
from src.llm.base import LLMProvider
from src.query_engine.sql_executor import validate_readonly_sql


SQL_PROMPT = """You are an expert DuckDB SQL query generator for a banking transactions database.

Table name: transactions
Available columns:
{schema}

STRICT RULES:
1. Generate exactly ONE read-only SQL query starting with SELECT.
2. Use only the `transactions` table.
3. For counts, use `COUNT(*)` with an alias (e.g. `SELECT COUNT(*) AS total_transactions FROM transactions`).
4. For averages, sums, min, max, use standard SQL aggregate functions with aliases (e.g. `AVG(amount) AS avg_amount`).
5. For rankings (top N), use `GROUP BY ... ORDER BY ... DESC LIMIT N`.
6. Return ONLY the raw SQL query. Do not wrap in markdown fences. Do not write explanations.
7. Only if the question is completely impossible to translate or asks for data outside the table schema, return:
CLARIFICATION_REQUIRED

Examples:
- "Combien y a-t-il de transactions au total ?"
SELECT COUNT(*) AS total_transactions FROM transactions

- "Quel est le montant moyen des transactions ?"
SELECT AVG(amount) AS average_amount FROM transactions

- "Donne-moi le top 5 des commerçants par dépenses"
SELECT merchant, SUM(amount) AS total_spent FROM transactions GROUP BY merchant ORDER BY total_spent DESC LIMIT 5

- "Total des dépenses par catégorie"
SELECT category, SUM(amount) AS total_amount FROM transactions GROUP BY category ORDER BY total_amount DESC

Question:
{question}

SQL:
"""


class SQLGenerator:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def _fallback_sql(self, question: str) -> str | None:
        """Génère du SQL déterministe pour les questions analytiques triviales."""
        q = question.lower().strip()
        if any(p in q for p in ["combien de transactions", "nombre de transactions", "total de transactions"]):
            return "SELECT COUNT(*) AS total_transactions FROM transactions"
        if any(p in q for p in ["montant moyen", "moyenne des transactions", "panier moyen"]):
            return "SELECT AVG(amount) AS average_amount FROM transactions"
        return None

    def generate(
        self,
        question: str,
        schema: list[dict],
    ) -> str:
        if not question or not question.strip():
            raise ValueError("The question cannot be empty.")

        if not schema:
            raise ValueError("The database schema is empty.")

        # 1. Tenter d'abord la règle déterministe simple
        fallback = self._fallback_sql(question)

        schema_text = ", ".join(
            f"{column.get('column_name', column.get('name'))} ({column.get('column_type', column.get('type'))})"
            for column in schema
        )

        prompt = SQL_PROMPT.format(
            schema=schema_text,
            question=question,
        )

        raw_response = self.llm.generate(
            prompt,
            temperature=0.0,
        ).strip()

        if not raw_response:
            if fallback:
                return fallback
            raise ValueError("The SQL generator returned an empty response.")

        # Nettoyage des balises Markdown éventuelles
        cleaned = re.sub(r"```(?:sql|SQL)?", "", raw_response).replace("```", "").strip()

        # Si le modèle a mis du texte explicatif avant ou après, extraire le bloc SELECT
        match = re.search(r"(SELECT\s+[\s\S]+?)(?:;|\n\n|$)", cleaned, re.IGNORECASE)
        if match:
            candidate_sql = match.group(1).strip()
            try:
                return validate_readonly_sql(candidate_sql)
            except Exception:
                pass

        # Vérification explicite de demande de clarification
        if cleaned.upper().startswith("CLARIFICATION_REQUIRED"):
            if fallback:
                return fallback
            return "CLARIFICATION_REQUIRED"

        # Dernier essai avec le SQL nettoyé ou le fallback
        try:
            return validate_readonly_sql(cleaned)
        except Exception:
            if fallback:
                return fallback
            return "CLARIFICATION_REQUIRED"
