from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Iterable, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Stopwords comuns em português para melhorar a pontuação do retriever
# ---------------------------------------------------------------------------
_PT_STOPWORDS: frozenset[str] = frozenset({
    "a", "ao", "aos", "as", "com", "da", "das", "de", "do", "dos",
    "e", "em", "é", "na", "nas", "no", "nos", "o", "os", "ou",
    "para", "pela", "pelas", "pelo", "pelos", "por", "que", "se",
    "um", "uma", "uns", "umas",
})


@dataclass(frozen=True)
class Document:
    id: str
    content: str


@dataclass
class ConversationTurn:
    question: str
    answer: str


class ReadyModel(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class EchoReadyModel:
    """Implementação de exemplo para um modelo pronto.

    Troque por integrações reais (OpenAI, Azure OpenAI, Ollama, etc).
    """

    def generate(self, prompt: str) -> str:
        return f"[Resposta do modelo]\n{prompt}"


class AzureOpenAIModel:
    """Adapter ReadyModel para Azure OpenAI Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment_name: str,
        api_version: str = "2024-02-01",
        timeout: float = 30.0,
    ) -> None:
        normalized_key = api_key.strip()
        normalized_endpoint = endpoint.strip().rstrip("/")
        normalized_deployment = deployment_name.strip()
        if not normalized_key:
            raise ValueError("api_key do Azure OpenAI não pode ser vazia")
        if not normalized_endpoint.startswith(("http://", "https://")):
            raise ValueError(
                f"endpoint do Azure OpenAI precisa ser uma URL HTTP/HTTPS válida, recebido: {normalized_endpoint}"
            )
        if not normalized_deployment:
            raise ValueError("deployment_name do Azure OpenAI não pode ser vazio")
        self._api_key = normalized_key
        self._timeout = timeout
        self._url = (
            f"{normalized_endpoint}/openai/deployments/{normalized_deployment}"
            f"/chat/completions?api-version={api_version}"
        )

    def generate(self, prompt: str) -> str:
        payload = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "api-key": f"******",
        }
        request = Request(self._url, data=payload, headers=headers, method="POST")

        try:
            with urlopen(request, timeout=self._timeout) as response:
                charset = response.headers.get_content_charset("utf-8")
                if not isinstance(charset, str):
                    charset = "utf-8"
                response_body = response.read().decode(charset)
        except URLError as exc:
            raise RuntimeError(f"falha ao chamar Azure OpenAI em {self._url}: {exc}") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return response_body

        if isinstance(parsed, dict):
            choices = parsed.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content
            return json.dumps(parsed, ensure_ascii=False)

        return response_body


class N8NWebhookModel:
    """Adapter ReadyModel que envia prompts para um workflow n8n via webhook HTTP."""

    def __init__(self, webhook_url: str, timeout: float = 10.0, token: str | None = None) -> None:
        normalized_url = webhook_url.strip()
        if not normalized_url.startswith(("http://", "https://")):
            raise ValueError(
                f"webhook_url do n8n precisa ser uma URL HTTP/HTTPS válida, recebido: {normalized_url}"
            )
        self._webhook_url = normalized_url
        self._timeout = timeout
        self._token = token

    def generate(self, prompt: str) -> str:
        payload = json.dumps({"prompt": prompt}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        request = Request(self._webhook_url, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self._timeout) as response:
                charset = response.headers.get_content_charset("utf-8")
                if not isinstance(charset, str):
                    charset = "utf-8"
                response_body = response.read().decode(charset)
        except URLError as exc:
            raise RuntimeError(f"falha ao chamar webhook do n8n em {self._webhook_url}: {exc}") from exc

        if not response_body.strip():
            return ""

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return response_body

        if isinstance(parsed, dict):
            for key in ("response", "answer", "text", "output"):
                value = parsed.get(key)
                if isinstance(value, str):
                    return value
            return json.dumps(parsed, ensure_ascii=False)

        return response_body


class OpenAIModel:
    """Adapter ReadyModel para OpenAI Chat Completions API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout: float = 30.0) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("api_key da OpenAI não pode ser vazia")
        self._api_key = normalized_key
        self._model = model
        self._timeout = timeout
        self._url = "https://api.openai.com/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        request = Request(self._url, data=payload, headers=headers, method="POST")

        try:
            with urlopen(request, timeout=self._timeout) as response:
                charset = response.headers.get_content_charset("utf-8")
                response_body = response.read().decode(charset)
        except URLError as exc:
            raise RuntimeError(f"falha ao chamar OpenAI em {self._url}: {exc}") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return response_body

        if isinstance(parsed, dict):
            choices = parsed.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content
            return json.dumps(parsed, ensure_ascii=False)

        return response_body


class BedrockModel:
    """Adapter ReadyModel para Amazon Bedrock Runtime (modelos Anthropic Claude).

    Requer credenciais AWS configuradas via variáveis de ambiente, perfil ou IAM Role:
    - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN se usar credenciais temporárias)
    - Ou configure via ``aws configure`` / perfil de instância EC2 / IAM Role.
    """

    DEFAULT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        region_name: str = "us-east-1",
        max_tokens: int = 1024,
    ) -> None:
        if boto3 is None:
            raise ImportError(
                "boto3 é necessário para BedrockModel. Instale com: pip install boto3"
            )
        self._client = boto3.client("bedrock-runtime", region_name=region_name)
        self._model_id = model_id
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
        })
        try:
            response = self._client.invoke_model(
                modelId=self._model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
        except Exception as exc:
            raise RuntimeError(
                f"falha ao chamar Amazon Bedrock (model={self._model_id}): {exc}"
            ) from exc

        response_body = json.loads(response["body"].read())

        if isinstance(response_body, dict):
            content = response_body.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    text = first.get("text")
                    if isinstance(text, str):
                        return text
            return json.dumps(response_body, ensure_ascii=False)

        return str(response_body)


class SimpleRetriever:
    """Retriever baseado em TF-IDF simples usando apenas a stdlib."""

    def __init__(self, documents: Iterable[Document], default_k: int = 3) -> None:
        self._documents = list(documents)
        self.default_k = default_k
        self._idf: dict[str, float] = {}
        self._doc_tfs: list[dict[str, float]] = []
        self._build_index()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in cleaned.split() if t and t not in _PT_STOPWORDS]

    def _build_index(self) -> None:
        n = len(self._documents)
        if n == 0:
            return

        self._doc_tfs = []
        df: dict[str, int] = {}

        for doc in self._documents:
            tokens = self._tokenize(doc.content)
            tf: dict[str, float] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            total = max(len(tokens), 1)
            self._doc_tfs.append({k: v / total for k, v in tf.items()})
            for token in tf:
                df[token] = df.get(token, 0) + 1

        self._idf = {
            token: math.log((n + 1) / (count + 1)) + 1
            for token, count in df.items()
        }

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        top_k = k if k is not None else self.default_k
        query_tokens = self._tokenize(query)
        if not query_tokens or not self._documents:
            return []

        scores: list[tuple[float, Document]] = []
        for doc, tf in zip(self._documents, self._doc_tfs):
            score = sum(
                tf.get(token, 0.0) * self._idf.get(token, 0.0)
                for token in query_tokens
            )
            if score > 0:
                scores.append((score, doc))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]


class RAGAgent:
    """Agente RAG com suporte a histórico de conversa."""

    SYSTEM_PROMPT = (
        "Você é um agente de IA para suporte de projetos. "
        "Responda com base no contexto recuperado e cite o ID do documento de origem entre colchetes, "
        "por exemplo [doc-id], quando aplicável. "
        "Seja objetivo e preciso."
    )

    def __init__(
        self,
        model: ReadyModel,
        retriever: SimpleRetriever,
        max_history: int = 5,
    ) -> None:
        self.model = model
        self.retriever = retriever
        self.max_history = max_history
        self._history: list[ConversationTurn] = []

    @property
    def conversation_history(self) -> list[ConversationTurn]:
        return list(self._history)

    def reset(self) -> None:
        """Limpa o histórico de conversa."""
        self._history.clear()

    def ask(self, question: str) -> str:
        context_docs = self.retriever.retrieve(question)
        context = "\n\n".join(f"[{doc.id}] {doc.content}" for doc in context_docs)

        history_text = ""
        if self._history:
            recent = self._history[-self.max_history:]
            history_lines = []
            for turn in recent:
                history_lines.append(f"Usuário: {turn.question}")
                history_lines.append(f"Agente: {turn.answer}")
            history_text = "\n".join(history_lines) + "\n\n"

        user_prompt = (
            f"{history_text}"
            f"Pergunta: {question}\n"
            f"Contexto:\n{context if context else '(sem contexto encontrado)'}\n"
        )

        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"
        answer = self.model.generate(full_prompt)

        self._history.append(ConversationTurn(question=question, answer=answer))
        return answer
