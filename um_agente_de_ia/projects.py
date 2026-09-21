from __future__ import annotations

import csv
import io
import json
import pathlib
from dataclasses import asdict, dataclass, replace
from typing import Iterable

from .agent import Document, EchoReadyModel, RAGAgent, ReadyModel, SimpleRetriever
from .exceptions import DuplicateError, NotFoundError, ValidationError

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]


VALID_STATUSES = {"planejado", "em_andamento", "pausado", "concluido", "cancelado"}


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    done: bool = False

    def validate(self) -> None:
        if not self.id.strip():
            raise ValidationError("id da tarefa não pode ser vazio")
        if not self.title.strip():
            raise ValidationError("título da tarefa não pode ser vazio")


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    description: str
    status: str = "planejado"
    tasks: tuple[Task, ...] = ()

    def validate(self) -> None:
        if not self.id.strip():
            raise ValidationError("id do projeto não pode ser vazio")
        if not self.name.strip():
            raise ValidationError("nome do projeto não pode ser vazio")
        if not self.description.strip():
            raise ValidationError("descrição do projeto não pode ser vazia")
        if self.status not in VALID_STATUSES:
            raise ValidationError(
                f"status inválido: '{self.status}'. Valores aceitos: {sorted(VALID_STATUSES)}"
            )
        for task in self.tasks:
            task.validate()


class ProjectManager:
    def __init__(self, projects: Iterable[Project] | None = None) -> None:
        self._projects: dict[str, Project] = {}
        for project in projects or []:
            self.add_project(project)

    def add_project(self, project: Project) -> None:
        project.validate()
        if project.id in self._projects:
            raise DuplicateError("projeto", project.id)
        self._projects[project.id] = project

    def get_project(self, project_id: str) -> Project:
        if project_id not in self._projects:
            raise NotFoundError("projeto", project_id)
        return self._projects[project_id]

    def list_projects(self, status: str | None = None) -> list[Project]:
        if status is None:
            return list(self._projects.values())
        if status not in VALID_STATUSES:
            raise ValidationError(
                f"status inválido: '{status}'. Valores aceitos: {sorted(VALID_STATUSES)}"
            )
        return [project for project in self._projects.values() if project.status == status]

    def update_status(self, project_id: str, status: str) -> Project:
        if status not in VALID_STATUSES:
            raise ValidationError(
                f"status inválido: '{status}'. Valores aceitos: {sorted(VALID_STATUSES)}"
            )
        project = self.get_project(project_id)
        updated = replace(project, status=status)
        self._projects[project_id] = updated
        return updated

    def remove_project(self, project_id: str) -> None:
        if project_id not in self._projects:
            raise NotFoundError("projeto", project_id)
        del self._projects[project_id]

    def add_task(self, project_id: str, task: Task) -> Project:
        task.validate()
        project = self.get_project(project_id)
        updated = replace(project, tasks=project.tasks + (task,))
        self._projects[project_id] = updated
        return updated

    def list_tasks(self, project_id: str) -> list[Task]:
        return list(self.get_project(project_id).tasks)

    def search(self, term: str) -> list[Project]:
        normalized = term.strip().lower()
        if not normalized:
            return self.list_projects()
        return [
            project
            for project in self._projects.values()
            if normalized in project.name.lower() or normalized in project.description.lower()
        ]

    def _to_documents(self) -> list[Document]:
        def format_tasks(project: Project) -> str:
            if not project.tasks:
                return "Sem tarefas."
            items = [
                f"{task.id}: {task.title} ({'concluída' if task.done else 'pendente'})"
                for task in project.tasks
            ]
            return "Tarefas: " + "; ".join(items)

        return [
            Document(
                project.id,
                (
                    f"Projeto: {project.name}. Status: {project.status}. "
                    f"Descrição: {project.description}. {format_tasks(project)}"
                ),
            )
            for project in self._projects.values()
        ]

    def export_csv(self, filepath: str) -> None:
        fieldnames = ["id", "name", "description", "status"]
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for project in self._projects.values():
                writer.writerow({
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "status": project.status,
                })

    def save_json(self, filepath: str) -> None:
        """Salva todos os projetos (com tarefas) em um arquivo JSON local."""
        data: list[dict] = []
        for project in self._projects.values():
            entry: dict = {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "status": project.status,
                "tasks": [
                    {"id": t.id, "title": t.title, "done": t.done}
                    for t in project.tasks
                ],
            }
            data.append(entry)
        pathlib.Path(filepath).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load_json(cls, filepath: str) -> "ProjectManager":
        """Carrega projetos de um arquivo JSON local e retorna um novo ProjectManager."""
        raw = pathlib.Path(filepath).read_text(encoding="utf-8")
        data: list[dict] = json.loads(raw)
        manager = cls()
        for entry in data:
            tasks = tuple(
                Task(id=t["id"], title=t["title"], done=t.get("done", False))
                for t in entry.get("tasks", [])
            )
            project = Project(
                id=entry["id"],
                name=entry["name"],
                description=entry["description"],
                status=entry.get("status", "planejado"),
                tasks=tasks,
            )
            manager.add_project(project)
        return manager

    def export_csv_s3(self, bucket: str, key: str, region_name: str | None = None) -> None:
        """Exporta projetos como CSV para um bucket do Amazon S3."""
        if boto3 is None:
            raise ImportError(
                "boto3 é necessário para export_csv_s3. Instale com: pip install boto3"
            )
        fieldnames = ["id", "name", "description", "status"]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for project in self._projects.values():
            writer.writerow({
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "status": project.status,
            })
        s3 = boto3.client("s3", region_name=region_name)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )

    def import_csv_s3(self, bucket: str, key: str, region_name: str | None = None) -> None:
        """Importa projetos de um CSV armazenado em um bucket do Amazon S3.

        Projetos cujo ``id`` já existe no gerenciador são ignorados.
        """
        if boto3 is None:
            raise ImportError(
                "boto3 é necessário para import_csv_s3. Instale com: pip install boto3"
            )
        s3 = boto3.client("s3", region_name=region_name)
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            project = Project(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                status=row.get("status", "planejado"),
            )
            if project.id not in self._projects:
                self.add_project(project)

    def build_agent(self, model: ReadyModel | None = None) -> RAGAgent:
        ready_model = model if model is not None else EchoReadyModel()
        retriever = SimpleRetriever(self._to_documents())
        return RAGAgent(model=ready_model, retriever=retriever)

    def ask(self, question: str, model: ReadyModel | None = None) -> str:
        return self.build_agent(model=model).ask(question)
