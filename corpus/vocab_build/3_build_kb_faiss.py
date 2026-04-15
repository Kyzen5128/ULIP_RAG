# -*- coding: utf-8 -*-

import json, numpy as np, faiss
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA = Path("data")
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SIM_DROP = 0.92

if __name__ == "__main__":
    records = [json.loads(l) for l in (DATA/"generated_sentences.jsonl").open("r", encoding="utf-8")]
    texts = [r["text"] for r in records]
    meta  = [{"concept": r["concept"], "evidence": r["evidence"], "source": r["source"]} for r in records]

    model = SentenceTransformer(ENCODER)
    X = model.encode(texts, batch_size=128, show_progress_bar=True).astype("float32")

    # 簡單去重：逐步挑選，不與已選集合 cosine>SIM_DROP
    keep_idx = []
    if len(X) > 0:
        keep_idx.append(0)
        for i in tqdm(range(1, len(X)), desc="去重"):
            sims = cosine_similarity(X[i:i+1], X[keep_idx])[0]
            if sims.max() < SIM_DROP:
                keep_idx.append(i)

    X = X[keep_idx]
    texts = [texts[i] for i in keep_idx]
    meta  = [meta[i]  for i in keep_idx]

    faiss.normalize_L2(X)
    np.save(DATA/"kb_vectors.npy", X)
    (DATA/"kb_texts.json").write_text(json.dumps(
        [{"text": t, **m} for t, m in zip(texts, meta)], indent=2, ensure_ascii=False
    ))

    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
    faiss.write_index(index, str(DATA/"kb_index.faiss"))
    print(f"✓ KB完成：texts={len(texts)}, vec={X.shape}")
