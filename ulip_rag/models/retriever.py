# ULIP/rag/retriever.py
import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from typing import List

class DenseRetriever:
    """
    一個基於 FAISS 和 SentenceTransformer 的稠密檢索器。
    它負責將查詢編碼，並從索引中快速找到最相似的文件。
    """
    def __init__(self, corpus_path: str, index_path: str, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cuda'):
        """
        初始化檢索器。

        Args:
            corpus_path (str): 知識語料庫檔案的路徑 (rag_corpus.jsonl)。
            index_path (str): FAISS 索引檔案的路徑 (corpus_index.faiss)。
            model_name (str): 用於編碼查詢的 Sentence Transformer 模型名稱。
            device (str): 執行模型的設備 ('cuda' or 'cpu')。
        """
        print("正在初始化 DenseRetriever...")
        if not os.path.exists(corpus_path) or not os.path.exists(index_path):
            raise FileNotFoundError(f"找不到語料庫或索引檔案。請先執行 build_rag_index.py。")

        # 載入 Sentence Transformer 模型
        self.model = SentenceTransformer(model_name, device=device)
        print(f"'{model_name}' 模型已載入到 {device}。")

        # 載入 FAISS 索引
        self.index = faiss.read_index(index_path)
        print(f"FAISS 索引已從 '{index_path}' 載入，包含 {self.index.ntotal} 個向量。")

        # 載入語料庫以進行內容映射
        print(f"正在從 '{corpus_path}' 載入語料庫內容...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = [json.loads(line) for line in f]
        
        print("DenseRetriever 初始化完成。")

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        對單一查詢執行檢索。

        Args:
            query (str): 用於查詢的文字字串。
            k (int): 要返回的 top-k 文件數量。

        Returns:
            List[str]: 包含 top-k 相關文件文字內容的列表。
        """
        if not query:
            return []
        
        # 1. 將查詢編碼為向量
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # 2. 標準化查詢向量 (因為索引中的向量也是標準化的)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # 3. 在 FAISS 索引中執行搜索
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # 4. 根據索引ID，從語料庫中提取對應的文件內容
        results = [self.corpus[i]['text'] for i in indices[0] if i != -1]
        
        return results

# 可選的測試程式碼
if __name__ == '__main__':
    # 假設您已在 ULIP/data/rag_corpus/ 中生成了檔案
    retriever = DenseRetriever(
        corpus_path='/home/klooom/cheng/3d_retrival/ULIP/data/rag_corpus/rag_corpus.jsonl',
        index_path='/home/klooom/cheng/3d_retrival/ULIP/data/rag_corpus/corpus_index.faiss'
    )
    
    test_query = "a modern wooden chair with armrests"
    search_results = retriever.search(test_query, k=3)
    
    print(f"\n查詢: '{test_query}'")
    print("檢索結果:")
    for i, doc in enumerate(search_results):
        print(f"{i+1}. {doc}")
