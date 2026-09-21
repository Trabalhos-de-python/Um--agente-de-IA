from __future__ import annotations


class AgentError(Exception):
    """Classe base para erros do sistema de agente."""


class ValidationError(AgentError, ValueError):
    """Dados de entrada inválidos."""


class NotFoundError(AgentError, KeyError):
    """Recurso não encontrado."""

    def __init__(self, resource: str, resource_id: str) -> None:
        self.resource = resource
        self.resource_id = resource_id
        super().__init__(f"{resource} não encontrado: {resource_id}")

    def __str__(self) -> str:
        return f"{self.resource} não encontrado: {self.resource_id}"


class DuplicateError(AgentError, ValueError):
    """Recurso duplicado."""

    def __init__(self, resource: str, resource_id: str) -> None:
        self.resource = resource
        self.resource_id = resource_id
        super().__init__(f"{resource} já existe: {resource_id}")


class PromptSecurityError(AgentError, ValueError):
    """Prompt rejeitado pelas regras de segurança."""
