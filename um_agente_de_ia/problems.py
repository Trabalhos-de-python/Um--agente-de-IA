from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, replace
from typing import Iterable

from .agent import Document, EchoReadyModel, RAGAgent, ReadyModel, SimpleRetriever
from .exceptions import DuplicateError, NotFoundError, ValidationError


VALID_PROBLEM_STATUSES = {"aberto", "investigando", "resolvido", "arquivado"}
VALID_PROBLEM_SEVERITIES = {"baixa", "media", "alta", "critica"}


@dataclass(frozen=True)
class Problem:
    id: str
    title: str
    description: str
    severity: str = "media"
    status: str = "aberto"

    def validate(self) -> None:
        if not self.id.strip():
            raise ValidationError("id do problema não pode ser vazio")
        if not self.title.strip():
            raise ValidationError("título do problema não pode ser vazio")
        if not self.description.strip():
            raise ValidationError("descrição do problema não pode ser vazia")
        if self.severity not in VALID_PROBLEM_SEVERITIES:
            raise ValidationError(
                f"severidade inválida: '{self.severity}'. Valores aceitos: {sorted(VALID_PROBLEM_SEVERITIES)}"
            )
        if self.status not in VALID_PROBLEM_STATUSES:
            raise ValidationError(
                f"status inválido: '{self.status}'. Valores aceitos: {sorted(VALID_PROBLEM_STATUSES)}"
            )


class ProblemManager:
    def __init__(self, problems: Iterable[Problem] | None = None) -> None:
        self._problems: dict[str, Problem] = {}
        for problem in problems or []:
            self.add_problem(problem)

    def add_problem(self, problem: Problem) -> None:
        problem.validate()
        if problem.id in self._problems:
            raise DuplicateError("problema", problem.id)
        self._problems[problem.id] = problem

    def get_problem(self, problem_id: str) -> Problem:
        if problem_id not in self._problems:
            raise NotFoundError("problema", problem_id)
        return self._problems[problem_id]

    def list_problems(
        self, status: str | None = None, severity: str | None = None
    ) -> list[Problem]:
        if status is not None and status not in VALID_PROBLEM_STATUSES:
            raise ValidationError(
                f"status inválido: '{status}'. Valores aceitos: {sorted(VALID_PROBLEM_STATUSES)}"
            )
        if severity is not None and severity not in VALID_PROBLEM_SEVERITIES:
            raise ValidationError(
                f"severidade inválida: '{severity}'. Valores aceitos: {sorted(VALID_PROBLEM_SEVERITIES)}"
            )

        return [
            problem
            for problem in self._problems.values()
            if (status is None or problem.status == status)
            and (severity is None or problem.severity == severity)
        ]

    def update_status(self, problem_id: str, status: str) -> Problem:
        if status not in VALID_PROBLEM_STATUSES:
            raise ValidationError(
                f"status inválido: '{status}'. Valores aceitos: {sorted(VALID_PROBLEM_STATUSES)}"
            )
        problem = self.get_problem(problem_id)
        updated = replace(problem, status=status)
        self._problems[problem_id] = updated
        return updated

    def update_severity(self, problem_id: str, severity: str) -> Problem:
        if severity not in VALID_PROBLEM_SEVERITIES:
            raise ValidationError(
                f"severidade inválida: '{severity}'. Valores aceitos: {sorted(VALID_PROBLEM_SEVERITIES)}"
            )
        problem = self.get_problem(problem_id)
        updated = replace(problem, severity=severity)
        self._problems[problem_id] = updated
        return updated

    def remove_problem(self, problem_id: str) -> None:
        if problem_id not in self._problems:
            raise NotFoundError("problema", problem_id)
        del self._problems[problem_id]

    def search(self, term: str) -> list[Problem]:
        normalized = term.strip().lower()
        if not normalized:
            return self.list_problems()
        return [
            problem
            for problem in self._problems.values()
            if normalized in problem.title.lower()
            or normalized in problem.description.lower()
        ]

    def _to_documents(self) -> list[Document]:
        return [
            Document(
                problem.id,
                (
                    f"Problema: {problem.title}. Severidade: {problem.severity}. "
                    f"Status: {problem.status}. Descrição: {problem.description}"
                ),
            )
            for problem in self._problems.values()
        ]

    def build_agent(self, model: ReadyModel | None = None) -> RAGAgent:
        ready_model = model if model is not None else EchoReadyModel()
        retriever = SimpleRetriever(self._to_documents())
        return RAGAgent(model=ready_model, retriever=retriever)

    def ask(self, question: str, model: ReadyModel | None = None) -> str:
        return self.build_agent(model=model).ask(question)

    def save_json(self, filepath: str) -> None:
        """Salva todos os problemas em um arquivo JSON local."""
        data = [
            {
                "id": p.id,
                "title": p.title,
                "description": p.description,
                "severity": p.severity,
                "status": p.status,
            }
            for p in self._problems.values()
        ]
        pathlib.Path(filepath).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load_json(cls, filepath: str) -> "ProblemManager":
        """Carrega problemas de um arquivo JSON local e retorna um novo ProblemManager."""
        raw = pathlib.Path(filepath).read_text(encoding="utf-8")
        data: list[dict] = json.loads(raw)
        manager = cls()
        for entry in data:
            problem = Problem(
                id=entry["id"],
                title=entry["title"],
                description=entry["description"],
                severity=entry.get("severity", "media"),
                status=entry.get("status", "aberto"),
            )
            manager.add_problem(problem)
        return manager
