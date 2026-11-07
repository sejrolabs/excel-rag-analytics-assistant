from qdrant_client import QdrantClient, models

from src.config import settings


class VectorStore:
    def __init__(
        self,
        path: str | None = None,
        collection: str | None = None,
        vector_size: int = 384,
    ):
        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero.")

        self.client = QdrantClient(
            path=path or settings.qdrant_path
        )

        self.collection = collection or settings.qdrant_collection
        self.vector_size = vector_size

        existing_collections = {
            collection_info.name: collection_info
            for collection_info in self.client.get_collections().collections
        }

        if self.collection not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def count(self) -> int:
        """Return the number of indexed vectors."""
        result = self.client.count(
            collection_name=self.collection,
            exact=True,
        )
        return result.count

    def upsert(
        self,
        ids: list[int],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        """Insert or update vectors and their associated payloads."""

        if not (
            len(ids) == len(vectors) == len(payloads)
        ):
            raise ValueError(
                "ids, vectors and payloads must have the same length."
            )

        if not ids:
            return

        points = []

        for point_id, vector, payload in zip(ids, vectors, payloads):

            if len(vector) != self.vector_size:
                raise ValueError(
                    f"Invalid vector dimension for point {point_id}: "
                    f"expected {self.vector_size}, got {len(vector)}."
                )

            if payload is None:
                raise ValueError(
                    f"Payload cannot be None for point {point_id}."
                )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=True,
        )

    def search(
        self,
        vector: list[float],
        limit: int = 8,
    ) -> list[dict]:
        """Search the nearest vectors."""

        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        if len(vector) != self.vector_size:
            raise ValueError(
                f"Invalid query vector dimension: "
                f"expected {self.vector_size}, got {len(vector)}."
            )

        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        ).points

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload or {},
            }
            for hit in hits
        ]

    def close(self) -> None:
        """Close the local Qdrant client."""
        self.client.close()