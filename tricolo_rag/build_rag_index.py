# 檔案路徑: build_rag_index.py

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

def build_index(args):
    """
    讀取原始的 ULIP/TriCoLo 文本資料，並建立 RAG 所需的語料庫和 FAISS 索引。
    """
    print("--- Step 1: 讀取原始文本資料 ---")
    
    # 讀取訓練集的文本資料來當作我們的知識庫
    input_json_path = args.input_json
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"找不到輸入的 JSON 檔案: {input_json_path}")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有不重複的 caption
    corpus_texts = list(set([item['caption'].strip() for item in data]))
    print(f"從 {len(data)} 筆資料中，提取到 {len(corpus_texts)} 筆不重複的文本作為語料庫。")

    print("\n--- Step 2: 載入 sentence-transformer 模型 ---")
    # 使用與 Retriever 中相同的模型來確保編碼一致
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    embedding_dim = encoder.get_sentence_embedding_dimension()
    print(f"模型 '{encoder.__class__.__name__}' 載入成功，向量維度: {embedding_dim}")

    print("\n--- Step 3: 將語料庫編碼成向量 ---")
    corpus_embeddings = encoder.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=True)
    
    # 正規化向量，這對於使用內積 (inner product) 的 FAISS 索引很重要
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    print(f"編碼完成，產生 {corpus_embeddings.shape[0]} 個向量。")

    print("\n--- Step 4: 建立並訓練 FAISS 索引 ---")
    # 我們使用 IndexFlatIP，因為正規化後的向量，內積等價於餘弦相似度
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(corpus_embeddings.astype('float32'))
    print(f"FAISS 索引建立完成，總共包含 {index.ntotal} 個向量。")

    print("\n--- Step 5: 儲存語料庫和 FAISS 索引 ---")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. 儲存語料庫 (rag_corpus.jsonl)
    corpus_path = os.path.join(output_dir, "rag_corpus.jsonl")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(corpus_texts):
            # 我們可以儲存更多元數據，但此處只存 text
            f.write(json.dumps({'doc_id': i, 'text': text}) + '\n')
    print(f"語料庫已儲存至: {corpus_path}")
    
    # 2. 儲存 FAISS 索引 (corpus_index.faiss)
    index_path = os.path.join(output_dir, "corpus_index.faiss")
    faiss.write_index(index, index_path)
    print(f"FAISS 索引已儲存至: {index_path}")

    print("\n--- RAG 索引庫建立完成！ ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="建立 RAG 的 FAISS 索引庫")
    parser.add_argument(
        '--input_json', 
        type=str, 
        default='data/text2shape-data/chair_table/preprocessed/exp_data/train_map.json',
        help='包含文本資料的來源 JSON 檔案路徑'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/text2shape-data/chair_table/rag_corpus',
        help='儲存 RAG 語料庫和索引的目標資料夾'
    )
    args = parser.parse_args()
    
    build_index(args)