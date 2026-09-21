from __future__ import annotations


def exact_match(expected: str, predicted: str) -> float:
    return 1.0 if expected.strip().lower() == predicted.strip().lower() else 0.0


def context_recall(relevant_contexts: list[str], retrieved_contexts: list[str]) -> float:
    if not relevant_contexts:
        return 1.0

    relevant = set(relevant_contexts)
    retrieved = set(retrieved_contexts)
    hits = len(relevant & retrieved)
    return hits / len(relevant)


def precision_at_k(relevant_contexts: list[str], retrieved_contexts: list[str], k: int) -> float:
    """Fração dos k primeiros contextos recuperados que são relevantes."""
    if k <= 0:
        return 0.0
    top_k = retrieved_contexts[:k]
    if not top_k:
        return 0.0
    relevant = set(relevant_contexts)
    hits = sum(1 for ctx in top_k if ctx in relevant)
    return hits / len(top_k)


def f1_score(relevant_contexts: list[str], retrieved_contexts: list[str]) -> float:
    """F1 combinando precisão e recall sobre o conjunto de contextos."""
    if not relevant_contexts and not retrieved_contexts:
        return 1.0
    precision = precision_at_k(relevant_contexts, retrieved_contexts, k=len(retrieved_contexts))
    recall = context_recall(relevant_contexts, retrieved_contexts)
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2 * precision * recall / denom


def mean_reciprocal_rank(relevant_contexts: list[str], retrieved_contexts: list[str]) -> float:
    """MRR: inverso da posição (1-indexada) do primeiro contexto relevante encontrado."""
    relevant = set(relevant_contexts)
    for rank, ctx in enumerate(retrieved_contexts, start=1):
        if ctx in relevant:
            return 1.0 / rank
    return 0.0
