import pytest
from src.query_engine.sql_executor import validate_readonly_sql


def test_valid_select_query():
    sql = "SELECT category, SUM(amount) AS total FROM transactions GROUP BY category"
    validated = validate_readonly_sql(sql)
    assert "SELECT" in validated.upper()
    assert "TRANSACTIONS" in validated.upper()


def test_forbid_drop_or_delete():
    with pytest.raises(ValueError, match="Only SELECT queries are allowed"):
        validate_readonly_sql("DROP TABLE transactions")

    with pytest.raises(ValueError, match="Only SELECT queries are allowed"):
        validate_readonly_sql("DELETE FROM transactions WHERE amount > 100")


def test_forbid_forbidden_functions():
    with pytest.raises(ValueError, match="Function not allowed"):
        validate_readonly_sql("SELECT * FROM read_csv('secret.csv')")

    with pytest.raises(ValueError, match="Function not allowed"):
        validate_readonly_sql("SELECT http_get('http://malicious.site') FROM transactions")


def test_forbid_unauthorized_tables():
    with pytest.raises(ValueError, match="Table not allowed"):
        validate_readonly_sql("SELECT * FROM users")


def test_forbid_ctes():
    with pytest.raises(ValueError, match="Common table expressions"):
        validate_readonly_sql("WITH sub AS (SELECT * FROM transactions) SELECT * FROM sub")
