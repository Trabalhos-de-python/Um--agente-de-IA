from __future__ import annotations

import argparse

from src.agent import Document, EchoReadyModel, RAGAgent, SimpleRetriever
from src.projects import Project, ProjectManager


def build_agent() -> RAGAgent:
    docs = [
        Document("1", "Pipeline CI valida lint, testes e segurança."),
        Document("2", "RAG combina recuperação de contexto com modelo pronto."),
        Document("3", "Segredos devem ficar em variáveis de ambiente."),
    ]
    return RAGAgent(model=EchoReadyModel(), retriever=SimpleRetriever(docs))


def build_project_manager() -> ProjectManager:
    return ProjectManager([
        Project("p1", "Site institucional", "Atualizar landing page e analytics", "em_andamento"),
        Project("p2", "App mobile", "Planejar backlog do próximo release", "planejado"),
    ])


def _separator(title: str = "") -> str:
    line = "─" * 60
    return f"\n{line}\n{title}\n{line}" if title else f"\n{line}"


def main(question: str) -> None:
    agent = build_agent()
    manager = build_project_manager()

    print(_separator("Agente RAG"))
    print(agent.ask(question))

    print(_separator("Gerenciador de Projetos"))
    print(manager.ask("Qual projeto está em andamento?"))

    print(_separator())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agente de IA com RAG")
    parser.add_argument(
        "question",
        nargs="?",
        default="O que é RAG?",
        help="Pergunta para o agente (padrão: 'O que é RAG?')",
    )
    args = parser.parse_args()
    main(args.question)
