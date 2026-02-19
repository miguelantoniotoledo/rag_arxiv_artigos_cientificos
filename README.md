# RAG Jogos de Inverno - Pipeline arXiv

Este projeto realiza o pipeline de busca, extração, processamento e embedding de artigos científicos da arXiv, com foco em temas parametrizáveis via YAML. O objetivo é preparar dados para sistemas de Recuperação Aumentada por Geração (RAG).

---

## Instalação de Dependências

1. **Crie um ambiente virtual (recomendado):**
   ```bash
   python -m venv .venv
   # Ative o ambiente:
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```
2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Parametrização via YAML

O arquivo de configuração principal é `config/arxiv_config.yaml`. Parâmetros principais:

- `terms`: Lista de termos de busca (AND entre eles)
- `max_results`: Máximo de artigos a buscar
- `start_date` / `end_date`: Intervalo de datas (YYYY-MM-DD)
- `output_dir`: Pasta de saída dos dados
- `extract_mode`: 'abstract' ou 'fulltext' (define o tipo de extração)
- `chunk_size` / `chunk_overlap`: Tamanho e sobreposição dos chunks para embedding
- `embedding_model`: Nome do modelo HuggingFace para embeddings

Exemplo:
```yaml
terms:
  - "data warehouse"
  - "artificial intelligence"
max_results: 100
start_date: "2019-01-01"
end_date: "2026-12-31"
output_dir: "data"
extract_mode: "fulltext"
chunk_size: 1000
chunk_overlap: 200
embedding_model: "BAAI/bge-m3"
```

---

## Escopo e Detalhamento das Funções (`src/`)

### 1. `arxiv_fetch.py`
- **Função:** Busca artigos na arXiv via API, baixa PDFs e salva metadados em `data/metadata.json`.
- **Principais funções:**
  - `fetch(config_path)`: Executa a busca, download e salva metadados.
  - `load_config`, `build_search_query`, `entry_pdf_url`, `download_file`.
- **Execução direta:**
  ```bash
  python src/arxiv_fetch.py [config/arxiv_config.yaml]
  ```

### 2. `arxiv_extract.py`
- **Função:** Extrai texto dos PDFs baixados (abstract ou fulltext), limpa e salva em `.txt` para cada artigo em `data/rag_texts/`.
- **Principais funções:**
  - `main(cfg_path)`: Percorre metadados, extrai texto e salva.
  - `extract_abstract`, `extract_fulltext`, `clean_text`, `extract_sections`.
- **Execução direta:**
  ```bash
  python src/arxiv_extract.py [config/arxiv_config.yaml]
  ```

### 3. `rag_chunk_embed.py`
- **Função:** Realiza chunking dos textos extraídos, gera embeddings usando modelo HuggingFace e salva em `data/rag_chunks.jsonl`.
- **Principais funções:**
  - `main(cfg_path)`: Percorre textos, faz chunking, gera embeddings e salva.
  - `chunk_text_by_sentence`, `mean_pooling`.
- **Execução direta:**
  ```bash
  python src/rag_chunk_embed.py [config/arxiv_config.yaml]
  ```

---

## Instruções para Execução do Processo

1. **Configure o arquivo `config/arxiv_config.yaml` conforme desejado.**
2. **Crie e ative o ambiente virtual:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # ou
   source .venv/bin/activate  # Linux/Mac
   ```
3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Execute as etapas na ordem:**
   1. **Buscar e baixar artigos:**
      ```bash
      python src/arxiv_fetch.py
      ```
   2. **Extrair texto dos PDFs:**
      ```bash
      python src/arxiv_extract.py
      ```
   3. **Gerar embeddings dos chunks:**
      ```bash
      python src/rag_chunk_embed.py
      ```

Os arquivos de saída principais estarão em `data/metadata.json`, `data/rag_texts/` e `data/rag_chunks.jsonl`.

---

## Observações
- O pipeline pode ser customizado alterando o YAML.
- O modelo de embedding pode ser trocado por qualquer compatível com HuggingFace Transformers.
- Para grandes volumes, recomenda-se GPU para acelerar embeddings.

---

Dúvidas ou sugestões: abra uma issue!
