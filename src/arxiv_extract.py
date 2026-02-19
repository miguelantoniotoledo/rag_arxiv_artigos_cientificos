import os
import sys
import json
import yaml
from pathlib import Path
from PyPDF2 import PdfReader

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_metadata(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_abstract(meta):
    return meta.get('summary', '')


import re

def clean_text(text):
    # Remove headers/footers: lines with page numbers, running titles, etc.
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        l = line.strip()
        # Remove lines that are just numbers or short (likely page numbers)
        if re.match(r"^\d{1,3}$", l):
            continue
        # Remove lines that look like 'arXiv preprint ...' or similar
        if re.search(r'arxiv|preprint|doi|copyright', l, re.I):
            continue
        # Remove image captions or figure/table codes
        if re.match(r'^(figure|fig\.|table|tab\.|image)[\s\d:.-]*', l, re.I):
            continue
        # Remove lines that are all uppercase and short (section headers will be handled separately)
        if l.isupper() and len(l) < 30:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def extract_sections(text):
    # Split by common section headers (case-insensitive)
    # e.g., Introduction, Methods, Results, Discussion, Conclusion, References
    section_pattern = re.compile(r"^\s*(\d+\.?\d*\s+)?([A-Z][A-Za-z\s-]{2,40})\s*$", re.M)
    sections = []
    last = 0
    for m in section_pattern.finditer(text):
        start = m.start()
        if last < start:
            sections.append(text[last:start].strip())
        last = start
    sections.append(text[last:].strip())
    return [s for s in sections if s.strip()]

def extract_fulltext(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text() or ''
            text.append(page_text)
        full = '\n'.join(text)
        # Clean up headers/footers, image codes
        full = clean_text(full)
        # Find abstract and references
        abstract_match = re.search(r'(?i)abstract[\s\n]*([\s\S]{20,2000}?)\n\s*\n', full)
        if abstract_match:
            abs_end = abstract_match.end()
        else:
            abs_end = 0
        ref_match = re.search(r'(?i)\n\s*(references|bibliography)\s*\n', full)
        ref_start = ref_match.start() if ref_match else len(full)
        # Extract only the main text (after abstract, before references)
        main_text = full[abs_end:ref_start].strip()
        # Split into sections
        sections = extract_sections(main_text)
        # Join sections with clear separator
        return '\n\n---\n\n'.join(sections)
    except Exception as e:
        print(f"[WARN] Failed to extract {pdf_path}: {e}")
        return ''

def main(cfg_path='config/arxiv_config.yaml'):
    cfg = load_config(cfg_path)
    output_dir = cfg.get('output_dir', 'data')
    extract_mode = cfg.get('extract_mode', 'abstract')
    meta_path = os.path.join(output_dir, 'metadata.json')
    out_text_dir = os.path.join(output_dir, 'rag_texts')
    os.makedirs(out_text_dir, exist_ok=True)
    metadata = load_metadata(meta_path)
    for meta in metadata:
        aid = meta.get('id')
        if not aid:
            continue
        if extract_mode == 'abstract':
            text = extract_abstract(meta)
        else:
            pdf_path = meta.get('local_pdf')
            if not pdf_path or not os.path.exists(pdf_path):
                print(f"[WARN] PDF not found for {aid}")
                continue
            text = extract_fulltext(pdf_path)
        # Save as .txt for RAG
        out_path = os.path.join(out_text_dir, f"{aid}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved {extract_mode} for {aid} -> {out_path}")
if __name__ == '__main__':
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config/arxiv_config.yaml'
    main(cfg)
