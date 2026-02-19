import os
import yaml
import json
import numpy as np
import faiss
from typing import List, Dict

# Carrega configuração YAML
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Carrega chunks e embeddings
def load_chunks(path: str) -> List[Dict]:
    chunks = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

# Indexa embeddings no FAISS
def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Busca top-k

def search(index, query_emb, k):
    D, I = index.search(query_emb, k)
    return D[0], I[0]

# Salva log dos chunks recuperados
def save_log(log_path, results):
    with open(log_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Main

def main(cfg_path='config/arxiv_config.yaml', query_text=None):
    cfg = load_config(cfg_path)
    output_dir = cfg.get('output_dir', 'data')
    chunk_file = os.path.join(output_dir, 'rag_chunks.jsonl')
    log_file = os.path.join(output_dir, 'retriever_log.jsonl')
    k = int(cfg.get('retriever_top_k', 5))
    embedding_model = cfg.get('embedding_model', 'BAAI/bge-m3')

    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    chunks = load_chunks(chunk_file)
    embeddings = np.array([c['embedding'] for c in chunks], dtype=np.float32)
    index = build_faiss_index(embeddings)

    if query_text is None:
        query_text = input('Digite sua consulta: ')

    encoded_input = tokenizer(query_text, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    token_embeddings = model_output[0]
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    query_emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    query_emb = query_emb.cpu().numpy().astype(np.float32)

    scores, indices = search(index, query_emb, k)
    results = []
    for score, idx in zip(scores, indices):
        chunk = chunks[idx]
        result = {
            'score': float(score),
            'chunk_id': chunk['chunk_id'],
            'id': chunk['id'],
            'text': chunk['text'][:200],  # preview
        }
        results.append(result)
        print(f"Score: {score:.4f} | ID: {chunk['chunk_id']} | Preview: {chunk['text'][:100]}")
    save_log(log_file, results)
    print(f"Log salvo em {log_file}")

if __name__ == '__main__':
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config/arxiv_config.yaml'
    query = sys.argv[2] if len(sys.argv) > 2 else None
    main(cfg, query)
