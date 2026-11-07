
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import VectorStore


TEST_CASES = [
    {
        "query": "Paiement dans un restaurant",
        "expected_categories": ["Restaurants"],
    },
    {
        "query": "Achat de billets de train",
        "expected_merchants": ["SNCF"],
    },
    {
        "query": "Achat de produits alimentaires",
        "expected_categories": ["Groceries"],
    },
    {
        "query": "Dépenses liées aux loisirs",
        "expected_categories": ["Entertainment"],
    },
    {
        "query": "Paiement chez Amazon",
        "expected_merchants": ["Amazon"],
    },
]


def evaluate():
    embedding_service = EmbeddingService()
    vector_store = VectorStore(vector_size=384)

    k = 5
    total_precision = 0
    total_relevant = 0
    total_retrieved = 0

    print("=" * 70)
    print("EVALUATION DE LA RECHERCHE VECTORIELLE")
    print("=" * 70)

    for test in TEST_CASES:
        query = test["query"]
        vector = embedding_service.embed_query(query)

        results = vector_store.search(vector, limit=k)

        expected_categories = set(
            test.get("expected_categories", [])
        )
        expected_merchants = set(
            test.get("expected_merchants", [])
        )

        relevant = 0

        for result in results:
            payload = result["payload"]

            category_match = (
                payload.get("category") in expected_categories
                if expected_categories else False
            )

            merchant_match = (
                payload.get("merchant") in expected_merchants
                if expected_merchants else False
            )

            if category_match or merchant_match:
                relevant += 1

        precision = relevant / k

        total_relevant += relevant
        total_retrieved += k
        total_precision += precision

        print("\n" + "-" * 70)
        print(f"Query : {query}")
        print(f"Résultats pertinents : {relevant}/{k}")
        print(f"Precision@{k} : {precision:.2%}")

        for result in results:
            payload = result["payload"]

            print(
                f"  - {payload.get('merchant')} | "
                f"{payload.get('category')} | "
                f"Score : {result['score']:.4f}"
            )

    print("\n" + "=" * 70)
    print(f"Precision@{k} moyenne : "
          f"{total_precision / len(TEST_CASES):.2%}")

    print(f"Pertinence globale : "
          f"{total_relevant}/{total_retrieved}")

    vector_store.close()


if __name__ == "__main__":
    evaluate()
