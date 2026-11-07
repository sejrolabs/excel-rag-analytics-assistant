import duckdb
import sqlglot
from sqlglot import exp

from src.config import settings


ALLOWED_TABLES = {"transactions"}

FORBIDDEN_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "read_blob",
    "glob",
    "query",
    "query_table",
    "parquet_scan",
    "csv_scan",
    "sqlite_scan",
    "postgres_scan",
    "mysql_scan",
    "http_get",
    "http_post",
    "load_extension",
}


def validate_readonly_sql(sql: str) -> str:
    """
    Validate that a SQL query is a single read-only SELECT
    referencing only the transactions table.
    """
    cleaned = sql.strip()

    if not cleaned:
        raise ValueError("Empty SQL query.")

    try:
        statements = sqlglot.parse(cleaned, read="duckdb")
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"Invalid SQL syntax: {exc}") from exc

    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Exactly one SQL statement is required.")

    tree = statements[0]

    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT queries are allowed.")

    # CTEs are deliberately excluded to keep table validation simple.
    if tree.args.get("with_") is not None:
        raise ValueError("Common table expressions (WITH) are not allowed.")

    for node in tree.walk():
        if isinstance(node, exp.Func):
            function_name = node.sql_name().lower()

            if function_name in FORBIDDEN_FUNCTIONS:
                raise ValueError(
                    f"Function not allowed: {function_name}"
                )

        # Also inspect anonymous functions such as read_csv_auto(...).
        if isinstance(node, exp.Anonymous):
            function_name = node.name.lower()

            if function_name in FORBIDDEN_FUNCTIONS:
                raise ValueError(
                    f"Function not allowed: {function_name}"
                )

    tables = list(tree.find_all(exp.Table))

    if not tables:
        raise ValueError(
            "The query must read from the transactions table."
        )

    for table in tables:
        if table.name.lower() not in ALLOWED_TABLES:
            raise ValueError(
                f"Table not allowed: {table.name}"
            )

        # Prevent references such as database.transactions or schema.transactions.
        if table.db or table.catalog:
            raise ValueError(
                "Qualified table names are not allowed."
            )

    return tree.sql(dialect="duckdb")


class SQLExecutor:
    def __init__(self, database_path: str | None = None):
        self.database_path = database_path or settings.duckdb_path
        self.last_result_truncated = False

    def execute(self, sql: str) -> list[dict]:
        """
        Execute a validated read-only SQL query.

        Fetch at most max_sql_rows + 1 rows to detect truncation
        without loading the entire result into memory.
        """
        safe_sql = validate_readonly_sql(sql)

        max_rows = settings.max_sql_rows

        if max_rows < 1:
            raise ValueError("MAX_SQL_ROWS must be greater than zero.")

        connection = duckdb.connect(
            self.database_path,
            read_only=True,
        )

        try:
            connection.execute("SET memory_limit='1GB'")
            connection.execute("SET threads=2")

            cursor = connection.execute(safe_sql)

            # Fetch one extra row to determine whether results were truncated.
            rows = cursor.fetchmany(max_rows + 1)

            self.last_result_truncated = len(rows) > max_rows

            if self.last_result_truncated:
                rows = rows[:max_rows]

            columns = [description[0] for description in cursor.description]

            return [
                dict(zip(columns, row))
                for row in rows
            ]

        finally:
            connection.close()