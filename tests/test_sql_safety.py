import pytest
from src.query_engine.sql_executor import validate_readonly_sql


def test_select_is_accepted():
    assert "SELECT" in validate_readonly_sql(
        "SELECT SUM(amount) FROM transactions"
    )


@pytest.mark.parametrize("sql", [
    "DELETE FROM transactions",
    "DROP TABLE transactions",
    "SELECT * FROM transactions; DROP TABLE transactions",
    "SELECT * FROM other_table",
])
def test_unsafe_sql_is_rejected(sql):
    with pytest.raises(ValueError):
        validate_readonly_sql(sql)
