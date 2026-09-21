# Um Agente de IA (RAG + Modelos Prontos)

Projeto base para criar uma IA do zero com estrutura completa: sistemas, design, infraestrutura, segurança, código de conduta, licença, linguagem, bibliotecas, treinamento e avaliação.

## Objetivo
Construir um agente de IA modular para apoiar projetos em tempo real, usando **RAG** (Retrieval-Augmented Generation) e integração com **modelos prontos** (OpenAI, Azure OpenAI, Ollama, etc).

## Stack (linguagem e bibliotecas)
- **Linguagem:** Python 3.12+
- **Bibliotecas padrão usadas no código atual:** `dataclasses`, `typing`, `json`, `pathlib`, `re`, `urllib`
- **Modelos prontos:** via adapter (`ReadyModel`), permitindo trocar o provedor sem alterar o núcleo do agente
- **Empacotamento:** `pyproject.toml` com metadados do projeto e requisito mínimo de Python

## Estrutura
```text
um_agente_de_ia/
  agent.py        # Núcleo RAG (ingestão, recuperação, prompt e resposta)
  main.py         # Exemplo de execução via CLI
  training.py     # Pipeline simples de preparo de dados
  evaluation.py   # Métricas básicas de avaliação
  projects.py     # Modelos e gestão de projetos
  problems.py     # Modelos e gestão de problemas

src/
  ...             # Compatibilidade com imports antigos
```

## Design e arquitetura
1. **Camada de conhecimento**: documentos em memória (`Document`) + índice simples.
2. **Camada de recuperação**: `SimpleRetriever` por sobreposição de termos.
3. **Camada de geração**: `ReadyModel` (interface) + implementação de exemplo (`EchoReadyModel`).
4. **Orquestração RAG**: `RAGAgent.ask()` monta contexto + prompt + chamada ao modelo.

## Integração com n8n
- O projeto agora inclui `N8NWebhookModel` em `um_agente_de_ia/agent.py`.
- Essa integração envia `{"prompt": "...texto..."}` para um webhook n8n via `POST` e usa o retorno para responder.
- Campos de resposta priorizados no JSON retornado: `response`, `answer`, `text`, `output`.
- Exemplo de uso:
```python
from um_agente_de_ia.agent import Document, N8NWebhookModel, RAGAgent, SimpleRetriever

docs = [Document("1", "Contexto do projeto")]
model = N8NWebhookModel(
    "https://n8n.example.com/webhook/rota",
    token="SEU_TOKEN_OPCIONAL",  # opcional: use apenas se o webhook exigir autenticação
)
agent = RAGAgent(model=model, retriever=SimpleRetriever(docs))
print(agent.ask("Me atualize sobre o projeto"))
```

## Integração com AWS

### Amazon Bedrock (modelos de linguagem)
- `BedrockModel` em `um_agente_de_ia/agent.py` usa o Amazon Bedrock Runtime para invocar modelos de IA (padrão: Anthropic Claude 3 Haiku).
- Autenticação via cadeia de credenciais padrão do boto3 (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, perfil `~/.aws/credentials`, IAM Role, etc.).
- Exemplo de uso:
```python
from um_agente_de_ia.agent import BedrockModel, Document, RAGAgent, SimpleRetriever

docs = [Document("1", "Contexto do projeto")]
model = BedrockModel(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",  # padrão
    region_name="us-east-1",  # padrão
)
agent = RAGAgent(model=model, retriever=SimpleRetriever(docs))
print(agent.ask("Resuma o projeto"))
```

### Amazon S3 (persistência de projetos)
- `ProjectManager.export_csv_s3(bucket, key)` — exporta projetos como CSV para um bucket S3.
- `ProjectManager.import_csv_s3(bucket, key)` — importa projetos de um CSV armazenado no S3.
- Exemplo de uso:
```python
from um_agente_de_ia.projects import Project, ProjectManager

manager = ProjectManager([
    Project("p1", "Site institucional", "Atualizar landing page", "em_andamento"),
])

# Exportar para S3
manager.export_csv_s3("meu-bucket", "projetos/export.csv", region_name="sa-east-1")

# Importar do S3 para um novo gerenciador
novo_manager = ProjectManager()
novo_manager.import_csv_s3("meu-bucket", "projetos/export.csv", region_name="sa-east-1")
```

### Configuração de credenciais AWS
```bash
# Opção 1: variáveis de ambiente
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# Opção 2: arquivo de credenciais
aws configure
```

### Dependências
```bash
python -m pip install -r requirements.txt
```

`boto3` é usado nas integrações com AWS (Bedrock e S3). O restante do projeto funciona só com a biblioteca padrão do Python.

Para instalação editável com suporte às integrações AWS:

```bash
python -m pip install -e ".[aws]"
```

## Infraestrutura sugerida
- Executar localmente com Python.
- Evolução natural:
  - API REST (FastAPI)
  - Banco vetorial (FAISS/Qdrant/pgvector)
  - Observabilidade (OpenTelemetry + logs estruturados)
  - CI/CD (lint/test/security scan)

## Segurança
- Ver `SECURITY.md`.
- Práticas aplicadas neste baseline:
  - separação entre recuperação e geração
  - sanitização básica de texto antes de recuperação
  - prompt com contexto controlado
  - sem hardcode de segredos

## Treinamento
- `um_agente_de_ia/training.py` descreve base para curadoria, limpeza e split de dados.
- Estratégia recomendada:
  - usar fine-tuning somente quando necessário
  - priorizar RAG para reduzir custo e risco de alucinação

## Avaliação
- `um_agente_de_ia/evaluation.py` inclui métricas iniciais:
  - `exact_match`
  - `context_recall`
- Evoluir com avaliações humanas e testes de segurança (prompt injection e data leakage).

## Sistema de controle de projetos
- `um_agente_de_ia/projects.py` adiciona um gerenciador de projetos com:
  - cadastro e remoção de projetos
  - atualização de status (`planejado`, `em_andamento`, `pausado`, `concluido`, `cancelado`)
  - busca por nome/descrição
  - perguntas ao agente com contexto dos projetos cadastrados

## Modelos de problemas
- `um_agente_de_ia/problems.py` adiciona modelos para registrar problemas do projeto com:
  - cadastro e remoção de problemas
  - atualização de status (`aberto`, `investigando`, `resolvido`, `arquivado`)
  - controle de severidade (`baixa`, `media`, `alta`, `critica`)
  - busca por título/descrição
  - perguntas ao agente com contexto dos problemas cadastrados

## Como executar
```bash
python -m um_agente_de_ia.main
```

Se preferir usar instalação editável com a estrutura moderna do projeto:

```bash
python -m pip install -e .
python -m um_agente_de_ia.main
```

## Testes
```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Compatibilidade
- Desenvolvimento local validado em Python 3.12
- CI configurada para Python 3.12 e 3.13

## Página inicial
- Foi adicionada uma página estática em `index.html`.
- Para abrir localmente no navegador, execute um servidor local:
```bash
python -m http.server 8000
```
Depois acesse: `http://localhost:8000`

## Governança
- **Código de Conduta:** `CODE_OF_CONDUCT.md`
- **Licença:** `LICENSE`
