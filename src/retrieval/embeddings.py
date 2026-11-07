from sentence_transformers import SentenceTransformer
from src.config import settings


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        self.model = SentenceTransformer(model_name or settings.embedding_model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = [f"passage: {text}" for text in texts]
        vectors = self.model.encode(prepared, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode([f"query: {query}"], normalize_embeddings=True)
        return vector[0].tolist()
