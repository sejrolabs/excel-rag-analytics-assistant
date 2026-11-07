
from src.storage.duckdb_store import DuckDBStore
from src.processing.text_enricher import enrich_transaction_text

db = DuckDBStore("data/processed/transactions.duckdb")

df = db.query("""
    SELECT *
    FROM transactions
    LIMIT 5
""")

# Conversion des résultats en DataFrame Polars
import polars as pl

df = pl.DataFrame(df)

enriched_df = enrich_transaction_text(df)

for row in enriched_df.select(
    ["transaction_id", "description", "transaction_text"]
).iter_rows(named=True):
    print("\n" + "=" * 80)
    print("TRANSACTION :", row["transaction_id"])
    print("DESCRIPTION :", row["description"])
    print("TEXTE ENRICHI :", row["transaction_text"])
