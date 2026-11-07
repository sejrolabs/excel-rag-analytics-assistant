def exact_match(prediction, reference) -> float:
    return float(prediction == reference)


def mean_absolute_error(predictions: list[float], references: list[float]) -> float:
    if len(predictions) != len(references) or not references:
        raise ValueError("Predictions and references must have the same non-zero length.")
    return sum(abs(p - r) for p, r in zip(predictions, references)) / len(references)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    return sum(item in relevant_ids for item in retrieved_ids[:k]) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)
