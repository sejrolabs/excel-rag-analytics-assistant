from pathlib import Path
import re

import duckdb
import pandas as pd
import polars as pl


class DuckDBStore:
    """Manage the transactions table in a DuckDB database."""

    def __init__(self, database_path: str):
        self.database_path = str(database_path)

        Path(self.database_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _validate_table_name(table_name: str) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            raise ValueError("Invalid table name.")

    def replace_transactions(
        self,
        df: pl.DataFrame,
        table_name: str = "transactions",
    ) -> None:
        """Replace a table with the content of a Polars DataFrame."""

        self._validate_table_name(table_name)

        if not isinstance(df, pl.DataFrame):
            raise TypeError("df must be a Polars DataFrame.")

        if df.is_empty():
            raise ValueError("Cannot create a transactions table from empty data.")

        arrow_table = df.to_arrow()

        with duckdb.connect(self.database_path) as connection:
            connection.register("incoming_df", arrow_table)

            try:
                connection.execute(
                    f'''
                    CREATE OR REPLACE TABLE "{table_name}"
                    AS SELECT * FROM incoming_df
                    '''
                )
            finally:
                connection.unregister("incoming_df")

    def query(self, sql: str) -> list[dict]:
        """Execute a read-only query and return records as dictionaries."""

        if not Path(self.database_path).exists():
            raise FileNotFoundError(
                f"DuckDB database does not exist: {self.database_path}"
            )

        with duckdb.connect(
            self.database_path,
            read_only=True,
        ) as connection:
            result = connection.execute(sql).fetchdf()
            return result.to_dict(orient="records")

    def describe(
        self,
        table_name: str = "transactions",
    ) -> list[dict]:
        """Return the schema of a table."""

        self._validate_table_name(table_name)

        if not Path(self.database_path).exists():
            raise FileNotFoundError(
                f"DuckDB database does not exist: {self.database_path}"
            )

        with duckdb.connect(
            self.database_path,
            read_only=True,
        ) as connection:
            return connection.execute(
                f'DESCRIBE "{table_name}"'
            ).fetchdf().to_dict(orient="records")

    def count_transactions(
        self,
        table_name: str = "transactions",
    ) -> int:
        """Return the number of rows in a table."""

        self._validate_table_name(table_name)

        if not Path(self.database_path).exists():
            raise FileNotFoundError(
                f"DuckDB database does not exist: {self.database_path}"
            )

        with duckdb.connect(
            self.database_path,
            read_only=True,
        ) as connection:
            result = connection.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()

            return int(result[0])