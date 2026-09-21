from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .agent import (
    AgentObservability,
    Document,
    EchoReadyModel,
    PromptGuard,
    QdrantVectorStore,
    RAGAgent,
    Retriever,
    SimpleRetriever,
    VectorRetriever,
)
from .exceptions import PromptSecurityError

LOGGER = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class DocumentPayload(BaseModel):
    id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class DocumentsRequest(BaseModel):
    documents: list[DocumentPayload] = Field(min_length=1)


def _seed_documents() -> list[Document]:
    return [
        Document("1", "Pipeline CI valida lint, testes e segurança."),
        Document("2", "RAG combina recuperação de contexto com modelo pronto."),
        Document("3", "Segredos devem ficar em variáveis de ambiente."),
    ]


def _build_retriever() -> Retriever:
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    if not qdrant_url:
        return SimpleRetriever(_seed_documents())

    retriever = VectorRetriever(
        QdrantVectorStore(
            qdrant_url,
            os.getenv("QDRANT_COLLECTION", "um-agente-de-ia"),
            api_key=os.getenv("QDRANT_API_KEY"),
        ),
        fallback_documents=_seed_documents(),
    )
    return retriever


def create_app(
    *,
    agent: RAGAgent | None = None,
    retriever: Retriever | None = None,
    observability: AgentObservability | None = None,
    prompt_guard: PromptGuard | None = None,
) -> FastAPI:
    observability = observability or AgentObservability()
    prompt_guard = prompt_guard or PromptGuard()
    retriever = retriever or _build_retriever()
    agent = agent or RAGAgent(
        model=EchoReadyModel(),
        retriever=retriever,
        observability=observability,
        prompt_guard=prompt_guard,
    )

    app = FastAPI(
        title="Um Agente de IA API",
        version="0.1.0",
        description="API REST para ingestão de documentos, perguntas RAG e métricas operacionais.",
    )
    app.state.agent = agent
    app.state.retriever = retriever
    app.state.observability = observability

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                "request_failed method=%s path=%s request_id=%s",
                request.method,
                request.url.path,
                request_id,
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        LOGGER.info(
            "request_completed method=%s path=%s status=%s request_id=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            request_id,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "vector_backend": "qdrant" if isinstance(retriever, VectorRetriever) else "simple",
            "local_documents_indexed": getattr(retriever, "document_count", observability.snapshot()["documents_indexed"]),
        }

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        return observability.snapshot()

    @app.post("/documents")
    def index_documents(payload: DocumentsRequest) -> dict[str, object]:
        documents = [Document(item.id, item.content) for item in payload.documents]
        try:
            retriever.add_documents(documents)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        observability.record_documents_indexed(len(documents))
        return {
            "indexed_documents": len(documents),
            "vector_backend": "qdrant" if isinstance(retriever, VectorRetriever) else "simple",
        }

    @app.post("/ask")
    def ask_question(payload: AskRequest) -> dict[str, object]:
        try:
            answer = agent.ask(payload.question)
        except PromptSecurityError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": str(exc),
                    "triggered_rules": list(agent.last_prompt_security.triggered_rules),
                },
            ) from exc

        return {
            "answer": answer,
            "conversation_length": len(agent.conversation_history),
            "triggered_rules": list(agent.last_prompt_security.triggered_rules),
            "prompt_truncated": agent.last_prompt_security.truncated,
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("um_agente_de_ia.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
