# -*- coding: utf-8 -*-
"""重建 RAG 檢索索引(MiniLM 384d 版)—— 2026-07-10 第二階段整理新增。

背景:
  現役索引 core/rag_corpus/minilm_corpus_index.faiss(舊名 clip_corpus_index.faiss)
  實測為 384 維 MiniLM(all-MiniLM-L6-v2)+ IndexFlatIP + L2-normalized,
  但 repo 內原本只有 CLIP 512d 版建索引腳本(build_rag_index.py,已 DEPRECATED),
  導致現役索引一度是「不可再生資產」。本腳本補上正確的再生路徑。

驗證邏輯(--verify,預設開):
  重建後與現有索引逐列比對餘弦相似度、並抽樣查詢比對 top-k 結果;
  一致才建議替換。預設輸出到 *.rebuilt.faiss,不覆蓋現役檔。

用法:
  cd /home/kyzen/ULIP_RAG
  conda run -n ulip python core/scripts/rebuild_rag_index_minilm.py            # 重建+驗證(不覆蓋)
  conda run -n ulip python core/scripts/rebuild_rag_index_minilm.py --no-verify
"""
import argparse
import json
import os

import faiss
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_DIR = os.path.join(REPO, "core", "rag_corpus")
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    ap = argparse.ArgumentParser("Rebuild MiniLM 384d RAG index")
    ap.add_argument("--corpus", default=os.path.join(CORPUS_DIR, "rag_corpus.jsonl"))
    ap.add_argument("--existing", default=os.path.join(CORPUS_DIR, "corpus_index.faiss"),
                    help="現役索引(經 symlink),用於驗證比對")
    ap.add_argument("--out", default=os.path.join(CORPUS_DIR, "minilm_corpus_index.rebuilt.faiss"),
                    help="輸出路徑(預設帶 .rebuilt,不覆蓋現役檔)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]
    texts = [x["text"] for x in corpus]
    print(f"corpus: {len(texts)} 條")

    model = SentenceTransformer(ENCODER)
    X = model.encode(texts, batch_size=args.batch_size,
                     convert_to_numpy=True, show_progress_bar=True).astype("float32")
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # 與 RAGRetriever 查詢端一致(L2-norm + IP)

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, args.out)
    print(f"已寫出 {args.out}(dim={X.shape[1]}, ntotal={index.ntotal})")

    if args.no_verify:
        return

    if not os.path.exists(args.existing):
        print("⚠️ 找不到現役索引可比對,跳過驗證")
        return
    old = faiss.read_index(args.existing)
    if old.ntotal != index.ntotal or old.d != index.d:
        print(f"❌ 維度/筆數不符: 現役 (d={old.d}, n={old.ntotal}) vs 重建 (d={index.d}, n={index.ntotal})")
        return

    # 1) 逐列餘弦(兩邊都已 normalize,內積即餘弦)
    old_vecs = old.reconstruct_n(0, old.ntotal)
    cos = (old_vecs * X).sum(axis=1)
    print(f"逐列餘弦: mean={cos.mean():.6f}  min={cos.min():.6f}  (<0.999 的列數: {(cos < 0.999).sum()})")

    # 2) 抽樣查詢 top-5 一致率
    rng = np.random.default_rng(0)
    qidx = rng.choice(len(texts), size=min(50, len(texts)), replace=False)
    Q = X[qidx]
    _, I_old = old.search(Q, 5)
    _, I_new = index.search(Q, 5)
    agree = (I_old == I_new).mean()
    print(f"抽樣 50 查詢 top-5 逐位一致率: {agree * 100:.2f}%")

    if cos.min() > 0.999 and agree > 0.99:
        print("✅ 驗證通過:重建索引與現役等價。可將 .rebuilt 檔替換現役檔(手動確認後執行)。")
    else:
        print("⚠️ 與現役索引不完全一致(可能是 sentence-transformers 版本差異)。"
              "檢索行為以現役檔為準,勿貿然替換;此重建檔僅供災難復原。")


if __name__ == "__main__":
    main()
