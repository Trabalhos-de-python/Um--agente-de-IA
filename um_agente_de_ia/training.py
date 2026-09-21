from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TrainingExample:
    question: str
    expected_answer: str


def split_dataset(
    dataset: Sequence[TrainingExample],
    train_ratio: float = 0.8,
    shuffle: bool = False,
    random_seed: int | None = None,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Divide o dataset em treino e teste.

    Args:
        dataset: Sequência de exemplos de treinamento.
        train_ratio: Proporção para treino (0 a 1).
        shuffle: Se True, embaralha antes de dividir.
        random_seed: Semente para reprodutibilidade do embaralhamento.
    """
    if not 0 <= train_ratio <= 1:
        raise ValueError("train_ratio precisa estar entre 0 e 1 (inclusive)")

    items = list(dataset)
    if shuffle:
        rng = random.Random(random_seed)
        rng.shuffle(items)

    cut = int(len(items) * train_ratio)
    return items[:cut], items[cut:]


# Mapa simples de sinônimos em português para augmentação de dados
_SYNONYM_MAP: dict[str, list[str]] = {
    "problema": ["issue", "erro", "falha"],
    "projeto": ["initiative", "trabalho"],
    "status": ["estado", "situação"],
    "crítico": ["grave", "urgente", "severo"],
    "resolvido": ["corrigido", "solucionado", "fechado"],
}


def augment_dataset(
    dataset: Sequence[TrainingExample],
    synonym_map: dict[str, list[str]] | None = None,
    random_seed: int | None = None,
) -> list[TrainingExample]:
    """Gera exemplos adicionais substituindo palavras por sinônimos.

    Para cada exemplo, cria uma variação trocando a primeira palavra do
    ``synonym_map`` encontrada na pergunta por um sinônimo aleatório.
    Exemplos sem correspondência são incluídos sem alteração.
    """
    mapping = synonym_map if synonym_map is not None else _SYNONYM_MAP
    rng = random.Random(random_seed)
    augmented: list[TrainingExample] = list(dataset)

    for example in dataset:
        question_lower = example.question.lower()
        for word, synonyms in mapping.items():
            if word in question_lower:
                replacement = rng.choice(synonyms)
                new_question = re.sub(
                    re.escape(word), replacement, example.question, flags=re.IGNORECASE, count=1
                )
                augmented.append(TrainingExample(
                    question=new_question,
                    expected_answer=example.expected_answer,
                ))
                break

    return augmented
