from src.llm.base import LLMProvider
from src.query_engine.sql_generator import SQLGenerator


class MockLLM(LLMProvider):
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self.response


def test_sql_generator_counts():
    generator = SQLGenerator(MockLLM("SELECT COUNT(*) AS total FROM transactions"))
    schema = [{"column_name": "transaction_id", "column_type": "BIGINT"}]
    sql = generator.generate("Combien de transactions au total ?", schema)
    assert "COUNT(*)" in sql.upper()
    assert "TRANSACTIONS" in sql.upper()


def test_sql_generator_extracts_from_chatter():
    chatter = "Voici la requête SQL demandée pour votre analyse :\n```sql\nSELECT AVG(amount) AS avg_amount FROM transactions\n```\nJ'espère que cela vous aide !"
    generator = SQLGenerator(MockLLM(chatter))
    schema = [{"column_name": "amount", "column_type": "DOUBLE"}]
    sql = generator.generate("Quel est le montant moyen ?", schema)
    assert "AVG(AMOUNT)" in sql.upper()


def test_sql_generator_fallback():
    generator = SQLGenerator(MockLLM("CLARIFICATION_REQUIRED"))
    schema = [{"column_name": "transaction_id", "column_type": "BIGINT"}]
    sql = generator.generate("Combien de transactions au total ?", schema)
    assert sql == "SELECT COUNT(*) AS total_transactions FROM transactions"
