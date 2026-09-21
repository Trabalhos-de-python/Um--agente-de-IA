# Wiki — Um Agente de IA

## 1) Visão geral
O **Um Agente de IA** é um projeto em Python para construção de um agente modular com arquitetura **RAG (Retrieval-Augmented Generation)**, focado em apoiar projetos em tempo real com contexto controlado.

## 2) Objetivo
- Permitir ingestão e recuperação de contexto.
- Integrar provedores de modelos sem alterar o núcleo.
- Oferecer base para evolução com API, observabilidade e pipelines de avaliação.

## 3) Estrutura do projeto
- `um_agente_de_ia/agent.py`: núcleo do agente RAG e integrações de modelo.
- `um_agente_de_ia/main.py`: execução via CLI.
- `um_agente_de_ia/training.py`: preparo/curadoria de dados.
- `um_agente_de_ia/evaluation.py`: métricas de avaliação.
- `um_agente_de_ia/projects.py`: gestão de projetos.
- `um_agente_de_ia/problems.py`: gestão de problemas.
- `tests/`: suíte de testes automatizados.

## 4) Arquitetura (RAG)
1. **Conhecimento**: documentos representados em memória.
2. **Recuperação**: busca por sobreposição de termos.
3. **Geração**: interface de modelo com adaptadores de provedores.
4. **Orquestração**: composição de contexto + prompt + resposta.

## 5) Integrações disponíveis
### n8n
- Envio de prompt para webhook via `POST`.
- Consumo de campos como `response`, `answer`, `text` e `output` no retorno.

### AWS
- **Amazon Bedrock** para inferência com modelos de linguagem.
- **Amazon S3** para importação/exportação de projetos em CSV.

## 6) Como executar
Pré-requisitos:
- Python 3.11+

Instalação:
```bash
python -m pip install -r requirements.txt
```

Execução:
```bash
python -m um_agente_de_ia.main
```

## 7) Testes
```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## 8) Segurança
- Não versionar segredos, tokens ou credenciais.
- Consulte `SECURITY.md` para políticas e reporte responsável.

## 9) Contribuição
- Siga `CONTRIBUTING.md`.
- Faça mudanças pequenas e focadas.
- Execute testes antes de abrir PR.

## 10) Governança
- Código de conduta: `CODE_OF_CONDUCT.md`
- Licença: `LICENSE`

## 11) Próximos passos sugeridos
- Expor API REST (ex.: FastAPI).
- Integrar banco vetorial.
- Ampliar observabilidade e segurança de prompts.
