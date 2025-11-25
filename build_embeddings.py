#!/usr/bin/env python3
"""
build_embeddings.py:- Builds FAISS + BM25 indices from catalog JSON
#run command for mac:      OMP_NUM_THREADS=1 python build_embeddings.py --input catalog_clean.json --outdir vector_store \
        --chunk-tokens 512 --overlap-tokens 160 --batch 64
"""

import argparse
import json
import os
import re
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# ---------- Optional BM25 ----------
try:
    from rank_bm25 import BM25Okapi
except Exception as e:
    BM25Okapi = None
    _bm25_import_error = e

# ---------- Chunking Helpers ----------
def approx_chars_for_tokens(tokens, chars_per_token=4):
    return int(tokens * chars_per_token)

def _is_heading_line(s):
    s_stripped = s.strip()
    if not s_stripped:
        return False
    if len(s_stripped) > 80:
        return False
    if s_stripped.endswith('.') or s_stripped.endswith('?') or s_stripped.endswith('!'):
        return False
    if re.match(r'^[A-Z0-9][A-Za-z0-9\s&\-\(\)\/]+$', s_stripped):
        if len(s_stripped.split()) <= 6:
            return True
    return False

def chunk_text_on_sentences(text, max_tokens=512, overlap_tokens=160):
    if not text:
        return []
    approx_chunk_chars = approx_chars_for_tokens(max_tokens)
    overlap_chars = approx_chars_for_tokens(overlap_tokens)
    text = text.replace("\r\n", "\n").strip()

    raw_paras = []
    current_para = []
    for line in text.split("\n"):
        if line.strip() == "":
            if current_para:
                raw_paras.append("\n".join(current_para).strip())
                current_para = []
        else:
            current_para.append(line)
    if current_para:
        raw_paras.append("\n".join(current_para).strip())

    paras = []
    i = 0
    while i < len(raw_paras):
        para = raw_paras[i]
        lines = para.split("\n")
        if len(lines) == 1 and _is_heading_line(lines[0]) and i + 1 < len(raw_paras):
            combined = lines[0].strip() + ". " + raw_paras[i+1]
            paras.append(combined.strip())
            i += 2
        else:
            paras.append(para)
            i += 1

    sentences = []
    for p in paras:
        sents = re.split(r'(?<=[\.\?\!])\s+|\n+', p)
        for s in sents:
            s = s.strip()
            if s:
                sentences.append(s)

    chunks = []
    cur = []
    cur_len = 0
    for s in sentences:
        s_len = len(s)
        if cur_len + s_len + 1 <= approx_chunk_chars or not cur:
            cur.append(s)
            cur_len += s_len + 1
        else:
            chunks.append(" ".join(cur).strip())
            if overlap_chars > 0:
                tail = []
                tail_len = 0
                for token in reversed(cur):
                    token_len = len(token) + 1
                    if tail_len + token_len <= overlap_chars:
                        tail.insert(0, token)
                        tail_len += token_len
                    else:
                        break
                cur = tail[:] if tail else []
                cur_len = tail_len
            else:
                cur = []
                cur_len = 0
            cur.append(s)
            cur_len += s_len + 1

    if cur:
        chunks.append(" ".join(cur).strip())

    final = []
    for c in chunks:
        if len(c) > approx_chunk_chars * 2:
            i = 0
            L = len(c)
            step = max(approx_chunk_chars - overlap_chars, approx_chunk_chars // 2)
            while i < L:
                final.append(c[i:i+approx_chunk_chars].strip())
                i += step
        else:
            final.append(c)
    return final

# ---------- BM25 Helpers ----------
_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")

def _tokenize_for_bm25(text):
    if not text:
        return []
    text = text.lower()
    tokens = _WORD_RE.findall(text)
    return tokens

def _build_and_save_bm25(corpus_texts, outpath_prefix):
    if BM25Okapi is None:
        raise ImportError(f"rank_bm25 not available: {_bm25_import_error}. Install with `pip install rank_bm25`")
    tokenized = [_tokenize_for_bm25(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    with open(outpath_prefix + ".bm25.pkl", "wb") as f:
        pickle.dump({"tokenized": tokenized, "bm25": bm25}, f)
    return tokenized, bm25

# ---------- Build Pipeline ----------
def build_indices(products, outdir, model_name="BAAI/bge-base-en-v1.5",
                  chunk_tokens=512, overlap_tokens=160, batch_size=64,
                  hnsw_m=32):
    os.makedirs(outdir, exist_ok=True)

    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    model = SentenceTransformer(model_name)

    prod_ids = []
    prod_texts = []
    products_meta = {}
    for p in products:
        pid = p.get("product_id") or p.get("url")
        pid = str(pid)
        short_text = " ".join(filter(None, [p.get("name",""), p.get("search_text","")]))
        prod_ids.append(pid)
        prod_texts.append(short_text)
        products_meta[pid] = {
            "product_id": pid,
            "url": p.get("url"),
            "name": p.get("name"),
            "description": p.get("description"),
            "job_levels": p.get("job_levels"),
            "languages": p.get("languages"),
            "duration": p.get("duration"),
            "adaptive_support": p.get("adaptive_support"),
            "remote_support": p.get("remote_support"),
            "test_type": p.get("test_type"),
            "test_type_names": p.get("test_type_names")
        }

    # ---------- Short-field embeddings ----------
    print("Embedding short-field product texts (title + search_text)...")
    short_embs = []
    for i in tqdm(range(0, len(prod_texts), batch_size)):
        batch = prod_texts[i:i+batch_size]
        em = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(em)
        short_embs.append(em)
    if short_embs:
        short_embs = np.vstack(short_embs).astype("float32")
    else:
        short_embs = np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    dim = short_embs.shape[1] if short_embs.size else model.get_sentence_embedding_dimension()

    print("Building FAISS index for product short-field vectors (IndexFlatIP)...")
    short_index = faiss.IndexFlatIP(dim)
    if short_embs.shape[0] > 0:
        short_index.add(short_embs)
    faiss.write_index(short_index, os.path.join(outdir, "short.index"))

    prod_map = {"ids": prod_ids}
    with open(os.path.join(outdir, "products_meta.json"), "w", encoding="utf-8") as f:
        json.dump(products_meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, "prod_map.json"), "w", encoding="utf-8") as f:
        json.dump(prod_map, f, indent=2, ensure_ascii=False)

    # ---------- Short-field BM25 ----------
    try:
        print("Building BM25 index for short-field texts (title + search_text)...")
        short_tokenized, short_bm25 = _build_and_save_bm25(prod_texts, os.path.join(outdir, "short_texts"))
        print("Short-field BM25 built and saved.")
    except ImportError as e:
        print("WARNING: Could not build short-field BM25 index:", e)
        short_tokenized, short_bm25 = None, None

    # ---------- Chunk generation ----------
    print("Generating chunks from descriptions (title weighted into chunks)...")
    chunk_texts = []
    chunk_meta = []
    for p in products:
        pid = p.get("product_id") or p.get("url")
        pid = str(pid)
        title = p.get("name") or ""
        desc = p.get("description") or ""
        full = (title + ". " + desc).strip()
        chunks = chunk_text_on_sentences(full, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        if not chunks:
            chunks = [title + " " + p.get("search_text","")]
        for idx, c in enumerate(chunks):
            chunk_texts.append(c)
            chunk_meta.append({
                "product_id": pid,
                "chunk_index": idx,
                "chunk_text": c[:4000],
                "title": title,
                "test_type": p.get("test_type"),
                "duration": p.get("duration")
            })

    print(f"Total chunks: {len(chunk_texts)}")

    # ---------- Chunk BM25 ----------
    try:
        print("Building BM25 index for chunk texts...")
        chunk_tokenized, chunk_bm25 = _build_and_save_bm25(chunk_texts, os.path.join(outdir, "chunks"))
        print("Chunk BM25 built and saved.")
    except ImportError as e:
        print("WARNING: Could not build chunk BM25 index:", e)
        chunk_tokenized, chunk_bm25 = None, None

    # ---------- Chunk embeddings ----------
    print("Embedding chunks...")
    chunk_embs_batches = []
    for i in tqdm(range(0, len(chunk_texts), batch_size)):
        batch = chunk_texts[i:i+batch_size]
        em = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(em)
        chunk_embs_batches.append(em)
    if chunk_embs_batches:
        chunk_embs = np.vstack(chunk_embs_batches).astype("float32")
    else:
        chunk_embs = np.zeros((0, dim), dtype="float32")
    dim_chunk = chunk_embs.shape[1] if chunk_embs.size else dim

    # ---------- Chunk FAISS index (HNSW with fallback) ----------
    chunks_index_path = os.path.join(outdir, "chunks.index")
    try:
        print("Attempting to build FAISS HNSW index for chunk embeddings...")
        try:
            faiss.omp_set_num_threads(1)
        except Exception:
            pass
        hnsw_index = faiss.IndexHNSWFlat(dim_chunk, hnsw_m)
        try:
            hnsw_index.hnsw.efConstruction = 200
        except Exception:
            pass
        if chunk_embs.shape[0] > 0:
            hnsw_index.add(chunk_embs)
        faiss.write_index(hnsw_index, chunks_index_path)
        print("HNSW chunk index built and saved.")
    except Exception as e:
        print("WARNING: HNSW build failed with exception:", repr(e))
        print("Falling back to IndexFlatIP for chunk embeddings (exact search).")
        fallback_index = faiss.IndexFlatIP(dim_chunk)
        if chunk_embs.shape[0] > 0:
            fallback_index.add(chunk_embs)
        faiss.write_index(fallback_index, chunks_index_path)
        print("Fallback chunk index built and saved as IndexFlatIP.")

    # ---------- Save metadata & manifest ----------
    with open(os.path.join(outdir, "chunks_meta.json"), "w", encoding="utf-8") as f:
        json.dump(chunk_meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, "model_name.txt"), "w", encoding="utf-8") as f:
        f.write(model_name)

    bm25_info = {
        "short_bm25_pickle": os.path.join(outdir, "short_texts.bm25.pkl") if short_bm25 is not None else None,
        "chunk_bm25_pickle": os.path.join(outdir, "chunks.bm25.pkl") if chunk_bm25 is not None else None
    }
    with open(os.path.join(outdir, "bm25_info.json"), "w", encoding="utf-8") as f:
        json.dump(bm25_info, f, indent=2, ensure_ascii=False)

    print("Done. Indices and metadata saved to:", outdir)
    return outdir

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="catalog_clean.json", help="clean catalog json")
    parser.add_argument("--outdir", default="vector_store", help="output directory")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=160)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hnsw-m", type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        products = json.load(f)
    build_indices(products, args.outdir, model_name="BAAI/bge-base-en-v1.5",
                  chunk_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens,
                  batch_size=args.batch, hnsw_m=args.hnsw_m)
