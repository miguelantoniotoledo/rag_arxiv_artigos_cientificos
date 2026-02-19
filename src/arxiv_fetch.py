import os
import sys
import time
import json
import requests
import feedparser
import yaml
from datetime import datetime

ARXIV_API = 'http://export.arxiv.org/api/query'


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_search_query(terms):
    if not terms:
        return 'all:"")'
    parts = []
    for t in terms:
        # search across all fields for the term
        parts.append(f'all:{t}')
    return '+AND+'.join(parts)


def parse_date(datestr):
    # arXiv dates look like '2024-01-02T12:34:56Z'
    try:
        return datetime.strptime(datestr, '%Y-%m-%dT%H:%M:%SZ').date()
    except Exception:
        try:
            return datetime.strptime(datestr, '%Y-%m-%d').date()
        except Exception:
            return None


def entry_pdf_url(entry):
    for link in entry.get('links', []):
        if link.get('type') == 'application/pdf':
            return link.get('href')
        href = link.get('href', '')
        if href.endswith('.pdf'):
            return href
    # fallback: construct from id
    eid = entry.get('id', '')
    if eid:
        # id often like http://arxiv.org/abs/xxxx
        aid = eid.rsplit('/', 1)[-1]
        return f'https://arxiv.org/pdf/{aid}.pdf'
    return None


def sanitize_filename(s):
    return ''.join(c for c in s if c.isalnum() or c in ('-', '_', '.')).rstrip()


def download_file(url, path, timeout=30):
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def fetch(config_path='config/arxiv_config.yaml'):
    cfg = load_config(config_path)
    terms = cfg.get('terms', [])
    max_results = int(cfg.get('max_results', 50))
    start_date = cfg.get('start_date')
    end_date = cfg.get('end_date')
    output_dir = cfg.get('output_dir', 'data')

    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    q = build_search_query(terms)
    params = {
        'search_query': q,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }

    print('Querying arXiv:', q)
    r = requests.get(ARXIV_API, params=params, timeout=30)
    r.raise_for_status()

    feed = feedparser.parse(r.text)
    entries = feed.get('entries', [])
    print(f'Fetched {len(entries)} entries, filtering by date range')

    os.makedirs(output_dir, exist_ok=True)
    pdf_dir = os.path.join(output_dir, 'pdfs')
    os.makedirs(pdf_dir, exist_ok=True)

    results = []
    for e in entries:
        pub = parse_date(e.get('published', ''))
        if pub is None:
            continue
        if start_date and pub < start_date:
            continue
        if end_date and pub > end_date:
            continue

        pdf_url = entry_pdf_url(e)
        aid = e.get('id', '').rsplit('/', 1)[-1]
        fname = sanitize_filename(aid) + '.pdf'
        fpath = os.path.join(pdf_dir, fname)
        meta = {
            'id': aid,
            'title': e.get('title'),
            'authors': [a.get('name') for a in e.get('authors', [])],
            'published': e.get('published'),
            'summary': e.get('summary'),
            'pdf_url': pdf_url,
            'local_pdf': fpath if pdf_url else None
        }

        if pdf_url:
            try:
                if not os.path.exists(fpath):
                    print('Downloading', pdf_url)
                    download_file(pdf_url, fpath)
                    # be polite
                    time.sleep(1)
                else:
                    print('Already have', fpath)
            except Exception as ex:
                print('Failed to download', pdf_url, ex)
                meta['local_pdf'] = None

        results.append(meta)

    # save metadata
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as mf:
        json.dump(results, mf, indent=2, ensure_ascii=False)

    print(f'Done. Saved {len(results)} metadata entries to', meta_path)


if __name__ == '__main__':
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config/arxiv_config.yaml'
    fetch(cfg)
