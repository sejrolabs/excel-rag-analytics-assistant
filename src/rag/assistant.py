from src.config import settings

from src.llm.factory import get_llm

from src.query_engine.query_router import QueryRouter
from src.query_engine.sql_generator import SQLGenerator
from src.query_engine.sql_executor import SQLExecutor

from src.rag.generator import AnswerGenerator

from src.storage.duckdb_store import DuckDBStore


class RAGAssistant:
    """
    Main assistant orchestrating:
    - Question routing
    - SQL generation and execution
    - Dense vector retrieval
    - Hybrid dense + BM25 retrieval
    - Answer generation
    """

    def __init__(self):

        self.llm = get_llm()

        self.router = QueryRouter(self.llm)
        self.sql_generator = SQLGenerator(self.llm)
        self.sql_executor = SQLExecutor()

        self.answer_generator = AnswerGenerator(self.llm)

        self.store = DuckDBStore(settings.duckdb_path)

        # Retrieval components are initialized lazily.
        self._retriever = None
        self._vector_store = None
        self._embeddings = None


    def _get_retriever(self):
        """
        Initialize the dense and lexical retrieval components.

        BM25 and Qdrant use the same enriched transaction text.
        """

        if self._retriever is not None:
            return self._retriever

        import polars as pl

        from src.processing.text_enricher import enrich_transaction_text
        from src.retrieval.bm25_search import BM25Search
        from src.retrieval.embeddings import EmbeddingService
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.hybrid_retriever import HybridRetriever

        # 1. Load the complete transaction dataset.
        rows = self.store.query("SELECT * FROM transactions")

        if not rows:
            raise ValueError(
                "No transactions are available for retrieval."
            )

        # 2. Build the same enriched text used for Qdrant indexing.
        df = pl.DataFrame(rows)
        enriched_df = enrich_transaction_text(df)

        documents = enriched_df.to_dicts()

        # 3. Initialize embedding service and Qdrant.
        self._embeddings = EmbeddingService()

        self._vector_store = VectorStore(
            vector_size=384,
        )

        indexed_count = self._vector_store.count()

        if indexed_count == 0:
            raise RuntimeError(
                "The Qdrant collection is empty. "
                "Run scripts.index_text before retrieval."
            )

        # 4. Build BM25 index using enriched transaction texts.
        bm25 = BM25Search()

        bm25.fit([
            {
                **row,
                "text": row["transaction_text"],
            }
            for row in documents
        ])

        # 5. Initialize hybrid retriever.
        self._retriever = HybridRetriever(
            embeddings=self._embeddings,
            vector_store=self._vector_store,
            bm25=bm25,
            dense_weight=0.6,
            rrf_constant=60,
        )

        print(
            f"Hybrid retriever initialized: "
            f"{len(documents):,} BM25 documents, "
            f"{indexed_count:,} Qdrant vectors."
        )

        return self._retriever


    def ask(self, question: str) -> dict:
        """
        Process a user question and return a structured response.
        """

        if not question or not question.strip():
            raise ValueError("The question cannot be empty.")

        # ---------------------------------------------------------
        # STEP 1 — ROUTING
        # ---------------------------------------------------------

        route = self.router.route(question)

        sql = None
        sources = []
        evidence = None

        # ---------------------------------------------------------
        # STEP 2 — EXACT SQL QUERY
        # ---------------------------------------------------------

        if route.intent == "SQL_QUERY":

            schema = self.store.describe()

            try:
                sql = self.sql_generator.generate(
                    question=question,
                    schema=schema,
                )

                if sql == "CLARIFICATION_REQUIRED":
                    return {
                        "answer": (
                            "Je ne peux pas traduire cette question "
                            "en une requête SQL fiable. "
                            "Peux-tu préciser les filtres, la période "
                            "ou la mesure recherchée ?"
                        ),
                        "route": route.intent,
                        "sources": [],
                        "sql": None,
                    }

                evidence = self.sql_executor.execute(sql)

                sources = [
                    {
                        "type": "duckdb",
                        "sql": sql,
                        "rows": evidence,
                        "truncated": (
                            self.sql_executor.last_result_truncated
                        ),
                    }
                ]

            except ValueError as exc:
                return {
                    "answer": (
                        "La requête SQL n'a pas pu être validée : "
                        f"{exc}"
                    ),
                    "route": route.intent,
                    "sources": [
                        {
                            "type": "sql_validation_error",
                            "message": str(exc),
                        }
                    ],
                    "sql": sql,
                }

        # ---------------------------------------------------------
        # STEP 3 — GENERAL DATASET QUESTION
        # ---------------------------------------------------------

        elif route.intent == "GENERAL_QUESTION":

            evidence = self.store.describe()

            sources = [
                {
                    "type": "database_schema",
                    "content": evidence,
                }
            ]

        # ---------------------------------------------------------
        # STEP 4 — DENSE VECTOR SEARCH
        # ---------------------------------------------------------

        elif route.intent == "VECTOR_SEARCH":

            self._get_retriever()

            query = route.search_query or question

            query_vector = self._embeddings.embed_query(query)

            evidence = self._vector_store.search(
                vector=query_vector,
                limit=8,
            )

            sources = [
                {
                    "type": "dense_vector_search",
                    "retrieved_count": len(evidence),
                    "records": evidence,
                    "limitation": (
                        "Vector search returns the most relevant "
                        "candidates. It does not guarantee exhaustive "
                        "coverage of the transaction population."
                    ),
                }
            ]

        # ---------------------------------------------------------
        # STEP 5 — HYBRID SEARCH: DENSE + BM25
        # ---------------------------------------------------------

        elif route.intent == "HYBRID_SEARCH":

            retriever = self._get_retriever()

            query = route.search_query or question

            evidence = retriever.search(
                query=query,
                top_k=8,
            )

            sources = [
                {
                    "type": "hybrid_retrieval",
                    "retrieved_count": len(evidence),
                    "records": evidence,
                    "limitation": (
                        "Hybrid retrieval combines dense vector search "
                        "and BM25 lexical search. The returned records "
                        "are relevant candidates, not an exhaustive "
                        "selection of the entire transaction population. "
                        "Aggregates over these candidates must not be "
                        "presented as exact population totals."
                    ),
                }
            ]

        else:
            raise ValueError(
                f"Unsupported route intent: {route.intent}"
            )

        # ---------------------------------------------------------
        # STEP 6 — ANSWER GENERATION
        # ---------------------------------------------------------

        answer = self.answer_generator.generate(
            question=question,
            route=route.intent,
            evidence=evidence,
        )

        return {
            "answer": answer,
            "route": route.intent,
            "sources": sources,
            "sql": sql,
        }