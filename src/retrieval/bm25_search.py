import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Search:
    def __init__(self):
        self.documents: list[dict] = []
        self.index = None

    def fit(self, documents: list[dict], text_key: str = "text") -> None:
        self.documents = documents
        corpus = [tokenize(str(doc.get(text_key, ""))) for doc in documents]
        self.index = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        if self.index is None:
            return []
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"score": float(score), **self.documents[i]} for i, score in ranked if score > 0]
