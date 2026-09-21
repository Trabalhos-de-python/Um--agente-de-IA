from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
import re
import time
from typing import Iterable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .exceptions import PromptSecurityError, QdrantRequestError

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords comuns em português para melhorar a pontuação do retriever
# ---------------------------------------------------------------------------
_PT_STOPWORDS: frozenset[str] = frozenset({
    "a", "ao", "aos", "as", "com", "da", "das", "de", "do", "dos",
    "e", "em", "é", "na", "nas", "no", "nos", "o", "os", "ou",
    "para", "pela", "pelas", "pelo", "pelos", "por", "que", "se",
    "um", "uma", "uns", "umas",
})


def _decode_response_body(response: object, default_charset: str = "utf-8") -> str:
    headers = getattr(response, "headers", None)
    charset = default_charset
    if headers is not None and hasattr(headers, "get_content_charset"):
        detected = headers.get_content_charset(default_charset)
        if isinstance(detected, str):
            charset = detected
    return response.read().decode(charset)


@dataclass(frozen=True)
class Document:
    id: str
    content: str


@dataclass
class ConversationTurn:
    question: str
    answer: str


@dataclass(frozen=True)
class PromptSecurityResult:
    sanitized_text: str
    redacted_text: str
    allowed: bool
    blocked_reason: str | None
    triggered_rules: tuple[str, ...]
    truncated: bool


@dataclass(frozen=True)
class ObservabilityEvent:
    question_preview: str
    retrieved_document_ids: tuple[str, ...]
    blocked: bool
    duration_ms: float
    triggered_rules: tuple[str, ...]


class ReadyModel(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        ...

    def add_documents(self, documents: Iterable[Document]) -> int:
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
            "api-key": self._api_key,
        }
        request = Request(self._url, data=payload, headers=headers, method="POST")

        try:
            with urlopen(request, timeout=self._timeout) as response:
                response_body = _decode_response_body(response)
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
        normalized_token = token.strip() if token is not None else None
        self._webhook_url = normalized_url
        self._timeout = timeout
        self._token = normalized_token or None

    def generate(self, prompt: str) -> str:
        payload = json.dumps({"prompt": prompt}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = "Bearer " + self._token

        request = Request(self._webhook_url, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self._timeout) as response:
                response_body = _decode_response_body(response)
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
            "Authorization": "Bearer " + self._api_key,
        }
        request = Request(self._url, data=payload, headers=headers, method="POST")

        try:
            with urlopen(request, timeout=self._timeout) as response:
                response_body = _decode_response_body(response)
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


class PromptGuard:
    """Aplica sanitização, redaction e bloqueio básico de prompt injection."""

    _INJECTION_RULES: dict[str, re.Pattern[str]] = {
        "ignore_previous_instructions": re.compile(
            r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE
        ),
        "reveal_system_prompt": re.compile(
            r"(reveal|show|print).{0,20}(system|developer)\s+prompt", re.IGNORECASE
        ),
        "tool_override_attempt": re.compile(
            r"(tool|policy|developer)\s+instructions", re.IGNORECASE
        ),
        "jailbreak_attempt": re.compile(
            r"(act as|pretend to be).{0,20}(system|developer|root)", re.IGNORECASE
        ),
    }
    _SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bsk-[A-Za-z0-9]{12,}\b"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    )

    def __init__(self, max_chars: int = 2_000) -> None:
        self.max_chars = max_chars

    def inspect(self, text: str) -> PromptSecurityResult:
        sanitized = self._sanitize(text)
        truncated = len(sanitized) > self.max_chars
        if truncated:
            sanitized = sanitized[: self.max_chars].rstrip()

        triggered_rules = tuple(
            name for name, pattern in self._INJECTION_RULES.items() if pattern.search(sanitized)
        )
        allowed = not triggered_rules
        return PromptSecurityResult(
            sanitized_text=sanitized,
            redacted_text=self.redact(sanitized),
            allowed=allowed,
            blocked_reason="prompt_rejected_for_security" if not allowed else None,
            triggered_rules=triggered_rules,
            truncated=truncated,
        )

    def redact(self, text: str) -> str:
        redacted = text
        for pattern in self._SECRET_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted

    @staticmethod
    def _sanitize(text: str) -> str:
        without_controls = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
        compact_whitespace = re.sub(r"[ \t]+", " ", without_controls)
        compact_newlines = re.sub(r"\n{3,}", "\n\n", compact_whitespace)
        return compact_newlines.strip()


class AgentObservability:
    """Métricas em memória para API e execução local."""

    def __init__(self, max_events: int = 25) -> None:
        self.max_events = max_events
        self.total_requests = 0
        self.blocked_requests = 0
        self.failed_requests = 0
        self.documents_indexed = 0
        self.retrieval_hits = 0
        self.total_duration_ms = 0.0
        self._events: list[ObservabilityEvent] = []

    def record_documents_indexed(self, count: int) -> None:
        self.documents_indexed += count

    def record_query(
        self,
        *,
        question_preview: str,
        retrieved_document_ids: Iterable[str],
        blocked: bool,
        duration_ms: float,
        triggered_rules: Iterable[str] = (),
        failed: bool = False,
    ) -> None:
        doc_ids = tuple(retrieved_document_ids)
        rules = tuple(triggered_rules)
        self.total_requests += 1
        self.total_duration_ms += duration_ms
        self.retrieval_hits += len(doc_ids)
        if blocked:
            self.blocked_requests += 1
        if failed:
            self.failed_requests += 1
        event = ObservabilityEvent(
            question_preview=question_preview,
            retrieved_document_ids=doc_ids,
            blocked=blocked,
            duration_ms=round(duration_ms, 2),
            triggered_rules=rules,
        )
        self._events.insert(0, event)
        del self._events[self.max_events:]

    def snapshot(self) -> dict[str, object]:
        avg_duration = self.total_duration_ms / self.total_requests if self.total_requests else 0.0
        return {
            "requests_total": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "failed_requests": self.failed_requests,
            "documents_indexed": self.documents_indexed,
            "retrieval_hits": self.retrieval_hits,
            "avg_duration_ms": round(avg_duration, 2),
            "recent_events": [
                {
                    "question_preview": event.question_preview,
                    "retrieved_document_ids": list(event.retrieved_document_ids),
                    "blocked": event.blocked,
                    "duration_ms": event.duration_ms,
                    "triggered_rules": list(event.triggered_rules),
                }
                for event in self._events
            ],
        }


class HashingVectorizer:
    """Gera embeddings determinísticos para integração com bancos vetoriais."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions precisa ser maior que zero")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = SimpleRetriever._tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            signal = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0)
            vector[index] += signal * weight

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]


class QdrantVectorStore:
    """Integração HTTP simples com Qdrant."""

    def __init__(
        self,
        base_url: str,
        collection_name: str,
        *,
        vector_size: int = 64,
        timeout: float = 10.0,
        api_key: str | None = None,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url.startswith(("http://", "https://")):
            raise ValueError("base_url do Qdrant precisa ser HTTP/HTTPS")
        normalized_collection = collection_name.strip()
        if not normalized_collection:
            raise ValueError("collection_name do Qdrant não pode ser vazio")
        self._base_url = normalized_url
        self._collection_name = normalized_collection
        self._collection_path = quote(normalized_collection, safe="")
        self._vector_size = vector_size
        self._timeout = timeout
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._collection_ready = False

    def ensure_collection(self) -> None:
        if self._collection_ready:
            return
        try:
            self._request("GET", f"/collections/{self._collection_path}", None)
            self._collection_ready = True
            return
        except QdrantRequestError as exc:
            if exc.status_code != 404:
                raise
        self._request(
            "PUT",
            f"/collections/{self._collection_path}",
            {
                "vectors": {
                    "size": self._vector_size,
                    "distance": "Cosine",
                }
            },
        )
        self._collection_ready = True

    def upsert_documents(self, documents: Iterable[Document], embedder: HashingVectorizer) -> None:
        items = list(documents)
        if not items:
            return
        self.ensure_collection()
        points = [
            {
                "id": doc.id,
                "vector": embedder.embed(doc.content),
                "payload": {"content": doc.content},
            }
            for doc in items
        ]
        self._request(
            "PUT",
            f"/collections/{self._collection_path}/points?wait=true",
            {"points": points},
        )

    def search(
        self,
        query: str,
        embedder: HashingVectorizer,
        limit: int,
    ) -> list[Document]:
        self.ensure_collection()
        payload = {
            "vector": embedder.embed(query),
            "limit": limit,
            "with_payload": True,
        }
        response = self._request(
            "POST",
            f"/collections/{self._collection_path}/points/search",
            payload,
        )
        if not isinstance(response, dict):
            return []
        items = response.get("result")
        if not isinstance(items, list):
            return []
        documents: list[Document] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            payload_item = item.get("payload")
            if not isinstance(payload_item, dict):
                continue
            content = payload_item.get("content") or payload_item.get("text")
            identifier = item.get("id")
            if isinstance(content, str) and isinstance(identifier, (str, int)):
                documents.append(Document(str(identifier), content))
        return documents

    def _request(self, method: str, path: str, payload: dict[str, object] | None) -> object:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["api-key"] = self._api_key
        request = Request(f"{self._base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self._timeout) as response:
                response_body = _decode_response_body(response)
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise QdrantRequestError(
                f"falha na integração com Qdrant ({method} {path}): {exc.code} {response_body}",
                status_code=exc.code,
            ) from exc
        except URLError as exc:
            raise QdrantRequestError(
                f"falha na integração com Qdrant ({method} {path}): {exc}"
            ) from exc

        if not response_body.strip():
            return {}
        try:
            return json.loads(response_body)
        except json.JSONDecodeError:
            return response_body


class SimpleRetriever:
    """Retriever baseado em TF-IDF simples usando apenas a stdlib."""

    def __init__(self, documents: Iterable[Document], default_k: int = 3) -> None:
        self._documents_by_id: dict[str, Document] = {}
        self._documents: list[Document] = []
        self.default_k = default_k
        self._idf: dict[str, float] = {}
        self._doc_tfs: list[dict[str, float]] = []
        self.add_documents(documents)

    @property
    def document_count(self) -> int:
        return len(self._documents_by_id)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in cleaned.split() if t and t not in _PT_STOPWORDS]

    def add_documents(self, documents: Iterable[Document]) -> int:
        previous_count = len(self._documents_by_id)
        for document in documents:
            self._documents_by_id[document.id] = document
        self._documents = list(self._documents_by_id.values())
        self._build_index()
        return len(self._documents_by_id) - previous_count

    def _build_index(self) -> None:
        n = len(self._documents)
        if n == 0:
            self._doc_tfs = []
            self._idf = {}
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


class VectorRetriever:
    """Retriever com banco vetorial e fallback local."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        *,
        embedder: HashingVectorizer | None = None,
        fallback_documents: Iterable[Document] = (),
        default_k: int = 3,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder or HashingVectorizer()
        self.default_k = default_k
        self._fallback = SimpleRetriever(fallback_documents, default_k=default_k)

    @property
    def document_count(self) -> int:
        return self._fallback.document_count

    def add_documents(self, documents: Iterable[Document]) -> int:
        items = list(documents)
        if not items:
            return 0
        self.vector_store.upsert_documents(items, self.embedder)
        return self._fallback.add_documents(items)

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        top_k = k if k is not None else self.default_k
        try:
            documents = self.vector_store.search(query, self.embedder, top_k)
            if documents:
                return documents
        except RuntimeError as exc:
            LOGGER.warning("falha ao consultar Qdrant; usando fallback local: %s", exc)
        return self._fallback.retrieve(query, top_k)


class RAGAgent:
    """Agente RAG com suporte a histórico, observabilidade e segurança de prompts."""

    SYSTEM_PROMPT = (
        "Você é um agente de IA para suporte de projetos. "
        "Responda com base no contexto recuperado e cite o ID do documento de origem entre colchetes, "
        "por exemplo [doc-id], quando aplicável. "
        "Seja objetivo e preciso."
    )

    def __init__(
        self,
        model: ReadyModel,
        retriever: Retriever,
        max_history: int = 5,
        *,
        prompt_guard: PromptGuard | None = None,
        observability: AgentObservability | None = None,
    ) -> None:
        self.model = model
        self.retriever = retriever
        self.max_history = max_history
        self.prompt_guard = prompt_guard or PromptGuard()
        self.observability = observability or AgentObservability()
        self._history: list[ConversationTurn] = []
        self._last_prompt_security = PromptSecurityResult(
            sanitized_text="",
            redacted_text="",
            allowed=True,
            blocked_reason=None,
            triggered_rules=(),
            truncated=False,
        )

    @property
    def conversation_history(self) -> list[ConversationTurn]:
        return list(self._history)

    @property
    def last_prompt_security(self) -> PromptSecurityResult:
        return self._last_prompt_security

    def reset(self) -> None:
        """Limpa o histórico de conversa."""
        self._history.clear()

    def ask(self, question: str) -> str:
        started = time.perf_counter()
        security_result = self.prompt_guard.inspect(question)
        self._last_prompt_security = security_result

        if not security_result.allowed:
            duration_ms = (time.perf_counter() - started) * 1000
            self.observability.record_query(
                question_preview=security_result.redacted_text,
                retrieved_document_ids=(),
                blocked=True,
                duration_ms=duration_ms,
                triggered_rules=security_result.triggered_rules,
            )
            raise PromptSecurityError("Prompt bloqueado por política de segurança.")

        context_docs = self.retriever.retrieve(security_result.sanitized_text)
        context = "\n\n".join(f"[{doc.id}] {doc.content}" for doc in context_docs)

        history_text = ""
        if self._history and self.max_history > 0:
            recent = self._history[-self.max_history:]
            history_lines = []
            for turn in recent:
                history_lines.append(f"Usuário: {turn.question}")
                history_lines.append(f"Agente: {turn.answer}")
            history_text = "\n".join(history_lines) + "\n\n"

        user_prompt = (
            f"{history_text}"
            f"Pergunta: {security_result.sanitized_text}\n"
            f"Contexto:\n{context if context else '(sem contexto encontrado)'}\n"
        )

        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"
        try:
            answer = self.model.generate(full_prompt)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            self.observability.record_query(
                question_preview=security_result.redacted_text,
                retrieved_document_ids=(doc.id for doc in context_docs),
                blocked=False,
                duration_ms=duration_ms,
                triggered_rules=security_result.triggered_rules,
                failed=True,
            )
            raise

        self._history.append(ConversationTurn(question=security_result.sanitized_text, answer=answer))
        duration_ms = (time.perf_counter() - started) * 1000
        self.observability.record_query(
            question_preview=security_result.redacted_text,
            retrieved_document_ids=(doc.id for doc in context_docs),
            blocked=False,
            duration_ms=duration_ms,
            triggered_rules=security_result.triggered_rules,
        )
        return answer
