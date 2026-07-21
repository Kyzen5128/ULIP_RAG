#!/usr/bin/env python3
"""Build a content-addressed MiniLM/FAISS index for a generated IKEA corpus."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_json
else:
    from .common import DEFAULT_ROOT, DEFAULT_VERSION, read_jsonl, sha256_file, snapshot_root, write_json


ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = snapshot_root(args.version, args.root)
    suffix = "" if args.split == "all" else f"_{args.split}"
    corpus_path = (args.corpus or root / "corpus" / f"product_corpus{suffix}.jsonl").resolve()
    output_path = (args.output or root / "corpus" / f"corpus_index{suffix}.faiss").resolve()
    rows = read_jsonl(corpus_path)
    if not rows:
        raise ValueError(f"corpus is empty: {corpus_path}")
    if args.split != "all":
        wrong = [row.get("doc_id") for row in rows if row.get("split") != args.split]
        if wrong:
            raise ValueError(f"corpus contains {len(wrong)} documents outside split={args.split}")
    texts = [str(row.get("text") or "").strip() for row in rows]
    if any(not text for text in texts):
        raise ValueError("every corpus row must have non-empty text")

    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER, device=args.device)
    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms <= 1e-12):
        raise ValueError("encoder produced non-finite or zero-length vectors")
    vectors /= norms
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    if index.ntotal != len(rows) or index.d != 384:
        raise ValueError(f"unexpected index shape d={index.d}, rows={index.ntotal}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    rebuilt = faiss.read_index(str(temporary))
    if rebuilt.ntotal != len(rows) or rebuilt.d != vectors.shape[1]:
        temporary.unlink(missing_ok=True)
        raise ValueError("written FAISS index failed read-back validation")
    temporary.replace(output_path)
    report = {
        "schema_version": "ikea-rag-index/2.0",
        "snapshot_version": args.version,
        "split": args.split,
        "encoder": ENCODER,
        "normalization": "l2",
        "metric": "inner_product_cosine_for_unit_vectors",
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "index_path": str(output_path),
        "index_sha256": sha256_file(output_path),
        "rows": int(index.ntotal),
        "dimension": int(index.d),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "manifests" / f"rag_index_{args.split}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
