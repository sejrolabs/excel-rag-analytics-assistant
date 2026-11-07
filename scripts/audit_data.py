
from src.storage.duckdb_store import DuckDBStore

db = DuckDBStore("data/processed/transactions.duckdb")

queries = {
    "DESCRIPTIONS DISTINCTES": """
        SELECT COUNT(DISTINCT description) AS descriptions_uniques
        FROM transactions
    """,
    "REPARTITION DES CATEGORIES": """
        SELECT category, COUNT(*) AS n
        FROM transactions
        GROUP BY category
        ORDER BY n DESC
    """,
    "EXEMPLES DE DESCRIPTIONS": """
        SELECT category, description, COUNT(*) AS n
        FROM transactions
        GROUP BY category, description
        ORDER BY category, n DESC
        LIMIT 30
    """,
    "STATISTIQUES DES MONTANTS": """
        SELECT
            MIN(amount) AS minimum,
            MEDIAN(amount) AS mediane,
            AVG(amount) AS moyenne,
            MAX(amount) AS maximum
        FROM transactions
    """
}

for title, sql in queries.items():
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for row in db.query(sql):
        print(row)