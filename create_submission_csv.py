#!/usr/bin/env python3
"""
create_submission_csv.py
"""
import argparse
import csv
import os
from pathlib import Path
from typing import List

from improved_query_engine import ImprovedSHLQueryEngine

def read_queries_from_text(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [l for l in lines if l]

def read_queries_from_csv(path: Path) -> List[str]:
    queries = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if 'Query' in reader.fieldnames:
            key = 'Query'
        else:
            key = None
            for fn in reader.fieldnames:
                if fn.lower() == 'query':
                    key = fn
                    break
        if not key:
            raise SystemExit("Input CSV must contain a 'Query' column")
        for row in reader:
            q = row.get(key, "").strip()
            if q:
                queries.append(q)
    return queries

def make_submission(engine: ImprovedSHLQueryEngine, queries: List[str], out_csv: Path, per_query: int = 3):
    per_query = max(1, min(per_query, 10))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Assessment_url"])

        for q in queries:
            recs = engine.recommend(q, top_k=per_query, use_reranking=True, balance_types=True, apply_patterns=True, use_expansion=True)
            if not recs:
                writer.writerow([q, ""])
                continue
            count = 0
            for r in recs:
                if count >= per_query:
                    break
                url = r.get("url") or r.get("Assessment_url") or ""
                writer.writerow([q, url])
                count += 1

    print(f"Saved submission CSV to: {out_csv} (queries: {len(queries)}, per-query: {per_query})")

def main():
    parser = argparse.ArgumentParser(description="Create submission CSV in required Appendix 3 format")
    parser.add_argument("--input", required=True, help="Input queries file: plain .txt (one per line) or .csv with 'Query' column")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path (default: submission.csv)")
    parser.add_argument("--per-query", type=int, default=3, help="Number of recommendations per query (1-10). Default 3")
    parser.add_argument("--vector-store", default="vector_store", help="Vector store directory (passed to engine)")
    parser.add_argument("--train-data", default=None, help="Optional training data path (Excel) - same param used by engine")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    if inp.suffix.lower() in (".csv",):
        queries = read_queries_from_csv(inp)
    else:
        queries = read_queries_from_text(inp)

    if not queries:
        raise SystemExit("No queries found in input")
    engine = ImprovedSHLQueryEngine(vector_store_dir=args.vector_store, train_data_path=args.train_data)
    outp = Path(args.output)
    make_submission(engine, queries, outp, per_query=args.per_query)

if __name__ == "__main__":
    main()
