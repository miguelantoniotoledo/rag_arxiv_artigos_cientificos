import os
import sys
import glob
import json
import yaml
import re
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def chunk_text_by_sentence(text: str, tokenizer, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_len = 0
    last_sentences = []
    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        n_tokens = len(tokens)
        if current_len + n_tokens > chunk_size and current:
            chunks.append(' '.join(current).strip())
            if overlap > 0 and last_sentences:
                current = last_sentences[-overlap:]
                current_len = sum(len(tokenizer.tokenize(s)) for s in current)
            else:
                current = []
                current_len = 0
        current.append(sent)
        current_len += n_tokens
        last_sentences.append(sent)
        if len(last_sentences) > overlap:
            last_sentences = last_sentences[-overlap:]
    if current:
        chunks.append(' '.join(current).strip())
    return [c for c in chunks if c.strip()]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main(cfg_path='config/arxiv_config.yaml'):
    cfg = load_config(cfg_path)
    output_dir = cfg.get('output_dir', 'data')
    chunk_size = int(cfg.get('chunk_size', 512))
    overlap = int(cfg.get('chunk_overlap', 64))
    text_dir = os.path.join(output_dir, 'rag_texts')
    out_path = os.path.join(output_dir, 'rag_chunks.jsonl')
    model_name = cfg.get('embedding_model', 'BAAI/bge-m3')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("[INFO] Usando GPU (CUDA) para embeddings.")
    else:
        print("[INFO] Usando CPU para embeddings.")
    model = model.to(device)
    try:
        model_max_length = tokenizer.model_max_length
        if model_max_length is None or model_max_length > 8192:
            model_max_length = 8192
    except Exception:
        model_max_length = 8192
    txt_files = glob.glob(os.path.join(text_dir, '*.txt'))
    total_files = len(txt_files)
    all_chunks = []
    total_chunks = 0
    chunk_counts = []
    # Prepass para contar chunks
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunk_text_by_sentence(text, tokenizer, chunk_size, overlap)
        chunk_counts.append(len(chunks))
        total_chunks += len(chunks)
    processed_chunks = 0
    for file_idx, txt_file in enumerate(txt_files):
        aid = os.path.splitext(os.path.basename(txt_file))[0]
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunk_text_by_sentence(text, tokenizer, chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            encode_len = min(chunk_size, model_max_length)
            encoded_input = tokenizer(chunk, padding=True, truncation=True, max_length=encode_len, return_tensors='pt')
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            with torch.no_grad():
                model_output = model(**encoded_input)
            emb = mean_pooling(model_output, encoded_input['attention_mask'])
            emb = emb[0].cpu().numpy().tolist()
            all_chunks.append({
                'id': aid,
                'chunk_id': f'{aid}_{idx}',
                'text': chunk,
                'embedding': emb
            })
            processed_chunks += 1
            percent = (processed_chunks / total_chunks) * 100
            print(f"Processando: arquivo {file_idx+1}/{total_files}, chunk {idx+1}/{len(chunks)} ({percent:.2f}%)")
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in all_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Saved {len(all_chunks)} chunks with embeddings to {out_path}')

if __name__ == '__main__':
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config/arxiv_config.yaml'
    main(cfg)
import os
import sys
import glob
import json
import yaml
import re
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def chunk_text_by_sentence(text: str, tokenizer, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    # Divide em sentenças (simples, pode ser melhorado com nltk/spacy)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_len = 0
    last_sentences = []
    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        n_tokens = len(tokens)
        if current_len + n_tokens > chunk_size and current:
            chunks.append(' '.join(current).strip())
            # Prepara overlap
            if overlap > 0 and last_sentences:
                current = last_sentences[-overlap:]
                current_len = sum(len(tokenizer.tokenize(s)) for s in current)
            else:
                current = []
                current_len = 0
        current.append(sent)
        current_len += n_tokens
        last_sentences.append(sent)
        if len(last_sentences) > overlap:
            last_sentences = last_sentences[-overlap:]
    if current:
        chunks.append(' '.join(current).strip())
    return [c for c in chunks if c.strip()]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element: last hidden state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main(cfg_path='config/arxiv_config.yaml'):
    cfg = load_config(cfg_path)
    output_dir = cfg.get('output_dir', 'data')
    chunk_size = int(cfg.get('chunk_size', 512))
    overlap = int(cfg.get('chunk_overlap', 64))
    text_dir = os.path.join(output_dir, 'rag_texts')
    out_path = os.path.join(output_dir, 'rag_chunks.jsonl')
    model_name = cfg.get('embedding_model', 'BAAI/bge-m3')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("[INFO] Usando GPU (CUDA) para embeddings.")
    else:
        print("[INFO] Usando CPU para embeddings.")
    model = model.to(device)
    # Descobre o max_length do modelo
    try:
        model_max_length = tokenizer.model_max_length
        if model_max_length is None or model_max_length > 8192:
            model_max_length = 8192  # para bge-m3
    except Exception:
        model_max_length = 8192
    all_chunks = []
    for txt_file in glob.glob(os.path.join(text_dir, '*.txt')):
        aid = os.path.splitext(os.path.basename(txt_file))[0]
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        # Chunking por sentença
        chunks = chunk_text_by_sentence(text, tokenizer, chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            encode_len = min(chunk_size, model_max_length)
            encoded_input = tokenizer(chunk, padding=True, truncation=True, max_length=encode_len, return_tensors='pt')
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            with torch.no_grad():
                model_output = model(**encoded_input)
            emb = mean_pooling(model_output, encoded_input['attention_mask'])
            emb = emb[0].cpu().numpy().tolist()
            all_chunks.append({
                'id': aid,
                'chunk_id': f'{aid}_{idx}',
                'text': chunk,
                'embedding': emb
            })
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in all_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Saved {len(all_chunks)} chunks with embeddings to {out_path}')

if __name__ == '__main__':
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config/arxiv_config.yaml'
    main(cfg)
