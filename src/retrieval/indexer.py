from datetime import date, datetime
import math
from typing import Callable, Any
import polars as pl

from src.config import settings
from src.storage.duckdb_store import DuckDBStore
from src.processing.text_enricher import enrich_transaction_text
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore


def make_json_safe(value: Any) -> Any:
    """Convertit les valeurs Python en valeurs compatibles JSON."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None:
        return None
    return value


def index_transactions(
    database_path: str | None = None,
    batch_size: int = 128,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """
    Extrait les transactions de DuckDB, enrichit leur description textuelle,
    génère les embeddings et les indexe dans Qdrant.

    Args:
        database_path: Chemin vers la base DuckDB (défaut: settings.duckdb_path)
        batch_size: Taille des lots pour l'inférence d'embedding et l'upsert
        progress_callback: Fonction de callback optionnelle appelée avec (indexed_count, total_count)

    Returns:
        int: Nombre total de transactions indexées
    """
    db_path = database_path or settings.duckdb_path
    db = DuckDBStore(db_path)

    rows = db.query("""
        SELECT
            rowid AS vector_id,
            *
        FROM transactions
        ORDER BY rowid
    """)

    if not rows:
        raise ValueError("Aucune transaction trouvée dans la base DuckDB pour l'indexation.")

    df = pl.DataFrame(rows)
    df = enrich_transaction_text(df)

    embedding_service = EmbeddingService()
    vector_store = VectorStore(vector_size=384)

    total = df.height
    indexed = 0

    try:
        for start in range(0, total, batch_size):
            batch = df.slice(start, batch_size)
            texts = batch["transaction_text"].to_list()
            vectors = embedding_service.embed_documents(texts)

            ids = [int(v) for v in batch["vector_id"].to_list()]
            payloads = []

            for row in batch.to_dicts():
                payload = {
                    key: make_json_safe(value)
                    for key, value in row.items()
                    if key != "vector_id"
                }
                payloads.append(payload)

            vector_store.upsert(
                ids=ids,
                vectors=vectors,
                payloads=payloads,
            )

            indexed += len(ids)
            if progress_callback:
                progress_callback(indexed, total)

        final_count = vector_store.count()
        return final_count
    finally:
        vector_store.close()
