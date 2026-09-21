import io
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from um_agente_de_ia.agent import (
    AzureOpenAIModel,
    Document,
    EchoReadyModel,
    N8NWebhookModel,
    OpenAIModel,
    BedrockModel,
    RAGAgent,
    SimpleRetriever,
)


class AgentTests(unittest.TestCase):
    def test_n8n_webhook_model_success_response(self):
        model = N8NWebhookModel("https://example.com/webhook", token="abc123")
        with patch("um_agente_de_ia.agent.urlopen") as mock_urlopen:
            response = mock_urlopen.return_value.__enter__.return_value
            response.read.return_value = b'{"response":"ok n8n"}'
            response.headers.get_content_charset.return_value = "utf-8"

            answer = model.generate("Teste n8n")

            self.assertEqual(answer, "ok n8n")
            request = mock_urlopen.call_args.args[0]
            self.assertEqual(request.full_url, "https://example.com/webhook")
            self.assertEqual(request.get_method(), "POST")
            self.assertIn(b'"prompt": "Teste n8n"', request.data)
            self.assertEqual(request.headers["Authorization"], "Bearer " + "abc123")

    def test_n8n_webhook_model_wraps_transport_errors(self):
        model = N8NWebhookModel("https://example.com/webhook")
        with patch("um_agente_de_ia.agent.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(RuntimeError) as ctx:
                model.generate("Teste")
            self.assertIn("https://example.com/webhook", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, URLError)

    def test_openai_model_success_response(self):
        model = OpenAIModel("sk-test")
        with patch("um_agente_de_ia.agent.urlopen") as mock_urlopen:
            response = mock_urlopen.return_value.__enter__.return_value
            response.read.return_value = b'{"choices":[{"message":{"content":"resposta openai"}}]}'
            response.headers.get_content_charset.return_value = "utf-8"

            answer = model.generate("Teste OpenAI")

            self.assertEqual(answer, "resposta openai")
            request = mock_urlopen.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.openai.com/v1/chat/completions")
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.headers["Authorization"], "Bearer " + "sk-test")
            self.assertIn(b'"model": "gpt-4o-mini"', request.data)
            self.assertIn(b'"content": "Teste OpenAI"', request.data)

    def test_openai_model_wraps_transport_errors(self):
        model = OpenAIModel("sk-test")
        with patch("um_agente_de_ia.agent.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(RuntimeError) as ctx:
                model.generate("Teste")
            self.assertIn("https://api.openai.com/v1/chat/completions", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, URLError)

    def test_retriever_returns_relevant_document(self):
        retriever = SimpleRetriever([
            Document("1", "RAG usa recuperação de contexto."),
            Document("2", "Segredos ficam em variáveis de ambiente."),
        ])
        docs = retriever.retrieve("Como funciona RAG?", k=1)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].id, "1")

    def test_retriever_default_k(self):
        docs = [Document(str(i), f"documento único sobre assunto {i}") for i in range(5)]
        retriever = SimpleRetriever(docs, default_k=2)
        results = retriever.retrieve("documento assunto")
        self.assertLessEqual(len(results), 2)

    def test_retriever_empty_query_returns_empty(self):
        retriever = SimpleRetriever([Document("1", "conteúdo")])
        self.assertEqual(retriever.retrieve(""), [])
        self.assertEqual(retriever.retrieve("   "), [])

    def test_retriever_tfidf_ranks_by_relevance(self):
        retriever = SimpleRetriever([
            Document("rare", "xyzzy mágico"),
            Document("common", "projeto projeto projeto"),
        ])
        results = retriever.retrieve("xyzzy mágico", k=2)
        self.assertEqual(results[0].id, "rare")

    def test_rag_agent_builds_answer(self):
        retriever = SimpleRetriever([Document("1", "RAG reduz alucinação.")])
        agent = RAGAgent(model=EchoReadyModel(), retriever=retriever)
        answer = agent.ask("Como reduzir alucinação?")
        self.assertIn("Contexto", answer)
        self.assertIn("RAG reduz alucinação", answer)

    def test_rag_agent_stores_history(self):
        retriever = SimpleRetriever([Document("1", "RAG reduz alucinação.")])
        agent = RAGAgent(model=EchoReadyModel(), retriever=retriever)
        agent.ask("Primeira pergunta")
        agent.ask("Segunda pergunta")
        history = agent.conversation_history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].question, "Primeira pergunta")
        self.assertEqual(history[1].question, "Segunda pergunta")

    def test_rag_agent_reset_clears_history(self):
        retriever = SimpleRetriever([Document("1", "RAG reduz alucinação.")])
        agent = RAGAgent(model=EchoReadyModel(), retriever=retriever)
        agent.ask("Uma pergunta")
        agent.reset()
        self.assertEqual(agent.conversation_history, [])

    def test_rag_agent_history_included_in_prompt(self):
        retriever = SimpleRetriever([Document("1", "RAG reduz alucinação.")])
        prompts: list[str] = []

        class CapturingModel:
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return "ok"

        agent = RAGAgent(model=CapturingModel(), retriever=retriever)
        agent.ask("Primeira")
        agent.ask("Segunda")
        self.assertIn("Primeira", prompts[1])

    def test_rag_agent_respects_max_history(self):
        retriever = SimpleRetriever([Document("1", "assunto.")])
        prompts: list[str] = []

        class CapturingModel:
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return "ok"

        agent = RAGAgent(model=CapturingModel(), retriever=retriever, max_history=1)
        agent.ask("Pergunta A")
        agent.ask("Pergunta B")
        agent.ask("Pergunta C")
        # Only last 1 turn should appear in the prompt for C
        self.assertNotIn("Pergunta A", prompts[2])
        self.assertIn("Pergunta B", prompts[2])

    def test_rag_agent_with_zero_history_does_not_include_previous_turns(self):
        retriever = SimpleRetriever([Document("1", "assunto.")])
        prompts: list[str] = []

        class CapturingModel:
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return "ok"

        agent = RAGAgent(model=CapturingModel(), retriever=retriever, max_history=0)
        agent.ask("Pergunta A")
        agent.ask("Pergunta B")
        self.assertNotIn("Pergunta A", prompts[1])

    def test_rag_agent_system_prompt_present(self):
        retriever = SimpleRetriever([Document("1", "assunto.")])
        prompts: list[str] = []

        class CapturingModel:
            def generate(self, prompt: str) -> str:
                prompts.append(prompt)
                return "ok"

        agent = RAGAgent(model=CapturingModel(), retriever=retriever)
        agent.ask("Teste")
        self.assertIn("agente de IA", prompts[0])

    def test_azure_openai_model_success(self):
        model = AzureOpenAIModel(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4o",
        )
        with patch("um_agente_de_ia.agent.urlopen") as mock_urlopen:
            response = mock_urlopen.return_value.__enter__.return_value
            response.read.return_value = b'{"choices":[{"message":{"content":"resposta azure"}}]}'
            response.headers.get_content_charset.return_value = "utf-8"

            answer = model.generate("Teste Azure")

        self.assertEqual(answer, "resposta azure")
        request = mock_urlopen.call_args.args[0]
        self.assertIn("gpt-4o", request.full_url)
        self.assertIn("chat/completions", request.full_url)
        self.assertEqual(request.headers["Api-key"], "test-key")

    def test_azure_openai_model_raises_on_empty_key(self):
        with self.assertRaises(ValueError):
            AzureOpenAIModel(api_key="  ", endpoint="https://ep.com", deployment_name="dep")

    def test_azure_openai_model_raises_on_bad_endpoint(self):
        with self.assertRaises(ValueError):
            AzureOpenAIModel(api_key="k", endpoint="not-a-url", deployment_name="dep")

    def test_azure_openai_model_raises_on_empty_deployment(self):
        with self.assertRaises(ValueError):
            AzureOpenAIModel(api_key="k", endpoint="https://ep.com", deployment_name="")

    def test_azure_openai_model_wraps_transport_errors(self):
        model = AzureOpenAIModel(
            api_key="key",
            endpoint="https://my-resource.openai.azure.com",
            deployment_name="gpt-4o",
        )
        with patch("um_agente_de_ia.agent.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(RuntimeError) as ctx:
                model.generate("Teste")
        self.assertIn("Azure OpenAI", str(ctx.exception))

    def test_bedrock_model_success_response(self):
        mock_boto3 = MagicMock()
        mock_client = mock_boto3.client.return_value
        response_payload = json.dumps({
            "content": [{"type": "text", "text": "resposta bedrock"}]
        }).encode("utf-8")
        mock_client.invoke_model.return_value = {"body": io.BytesIO(response_payload)}

        with patch("um_agente_de_ia.agent.boto3", mock_boto3):
            model = BedrockModel(model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")
            answer = model.generate("Teste Bedrock")

        self.assertEqual(answer, "resposta bedrock")
        mock_boto3.client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")
        call_kwargs = mock_client.invoke_model.call_args.kwargs
        self.assertEqual(call_kwargs["modelId"], "anthropic.claude-3-haiku-20240307-v1:0")
        body_sent = json.loads(call_kwargs["body"])
        self.assertEqual(body_sent["messages"][0]["content"], "Teste Bedrock")
        self.assertEqual(body_sent["anthropic_version"], "bedrock-2023-05-31")

    def test_bedrock_model_wraps_client_errors(self):
        mock_boto3 = MagicMock()
        mock_client = mock_boto3.client.return_value
        mock_client.invoke_model.side_effect = Exception("AccessDenied")

        with patch("um_agente_de_ia.agent.boto3", mock_boto3):
            model = BedrockModel()
            with self.assertRaises(RuntimeError) as ctx:
                model.generate("Teste")
        self.assertIn("Amazon Bedrock", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, Exception)

    def test_bedrock_model_raises_when_boto3_unavailable(self):
        with patch("um_agente_de_ia.agent.boto3", None):
            with self.assertRaises(ImportError) as ctx:
                BedrockModel()
        self.assertIn("boto3", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
