import json
from pathlib import Path
from src.evaluation.metrics import exact_match


def load_benchmark(path: str = "data/sample/benchmark.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_exact(predictions: list, references: list) -> dict:
    if len(predictions) != len(references):
        raise ValueError("Prediction/reference length mismatch.")
    scores = [exact_match(p, r) for p, r in zip(predictions, references)]
    return {"n": len(scores), "exact_match": sum(scores) / len(scores) if scores else 0.0}
