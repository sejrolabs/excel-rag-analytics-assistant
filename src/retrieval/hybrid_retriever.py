from src.retrieval.bm25_search import BM25Search
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore


class HybridRetriever:
    """
    Hybrid retrieval combining dense vector search and BM25.

    Results are merged using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        bm25: BM25Search,
        dense_weight: float = 0.6,
        rrf_constant: int = 60,
    ):
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense_weight must be between 0 and 1.")

        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be greater than zero.")

        self.embeddings = embeddings
        self.vector_store = vector_store
        self.bm25 = bm25
        self.dense_weight = dense_weight
        self.lexical_weight = 1.0 - dense_weight
        self.rrf_constant = rrf_constant

    def search(
        self,
        query: str,
        top_k: int = 8,
    ) -> list[dict]:

        if not query or not query.strip():
            raise ValueError("The search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Dense retrieval.
        query_vector = self.embeddings.embed_query(query)

        dense_results = self.vector_store.search(
            query_vector,
            limit=top_k,
        )

        # Lexical retrieval.
        lexical_results = self.bm25.search(
            query,
            top_k=top_k,
        )

        scores: dict[str, float] = {}
        documents: dict[str, dict] = {}

        # Merge dense results.
        for rank, item in enumerate(dense_results, start=1):

            payload = item.get("payload") or {}

            transaction_id = payload.get(
                "transaction_id",
                item.get("id"),
            )

            if transaction_id is None:
                continue

            key = str(transaction_id)

            scores[key] = scores.get(key, 0.0) + (
                self.dense_weight / (self.rrf_constant + rank)
            )

            documents[key] = {
                **documents.get(key, {}),
                **payload,
            }

        # Merge lexical results.
        for rank, item in enumerate(lexical_results, start=1):

            transaction_id = item.get(
                "transaction_id",
                item.get("id"),
            )

            if transaction_id is None:
                continue

            key = str(transaction_id)

            scores[key] = scores.get(key, 0.0) + (
                self.lexical_weight / (self.rrf_constant + rank)
            )

            # Preserve existing fields and add lexical metadata.
            documents[key] = {
                **item,
                **documents.get(key, {}),
            }

        ordered_ids = sorted(
            scores,
            key=lambda key: scores[key],
            reverse=True,
        )[:top_k]

        return [
            {
                **documents[key],
                "score": scores[key],
            }
            for key in ordered_ids
        ]