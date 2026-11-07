
from src.config import settings
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore


def main():
    print("\n[1/3] Initializing embedding service...")
    embedding_service = EmbeddingService()

    print("\n[2/3] Connecting to Qdrant...")
    vector_store = VectorStore()
    
    try:
        count = vector_store.count()
        print(f"Indexed vectors: {count:,}")

        if count == 0:
            raise RuntimeError("Qdrant collection is empty.")

        queries = [
            "Paiement dans un restaurant",
            "Retrait d'argent au distributeur",
            "Achat de billets de train",
            "Transaction suspecte de montant élevé",
        ]

        print("\n[3/3] Testing semantic search...")

        for query in queries:
            print("\n" + "=" * 70)
            print(f"QUERY: {query}")
            print("=" * 70)

            query_vector = embedding_service.embed_query(query)
            results = vector_store.search(
                vector=query_vector,
                limit=5,
            )

            for i, result in enumerate(results, start=1):
                payload = result["payload"]

                print(f"\nResult {i} | Score: {result['score']:.4f}")
                print(f"Transaction ID: {payload.get('transaction_id')}")
                print(f"Description: {payload.get('description')}")
                print(f"Merchant: {payload.get('merchant')}")
                print(f"Category: {payload.get('category')}")
                print(f"Amount: {payload.get('amount')} EUR")

    finally:
        vector_store.close()


if __name__ == "__main__":
    main()
