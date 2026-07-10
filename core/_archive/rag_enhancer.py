# =====================================================================
# ⚠️ DEPRECATED(2026-07-10 標記):本檔已無人 import(rg 全 repo 實測)。
# 獨立版 RAGEnhancer。現役版本(含 key_padding_mask 修復)在
# core/models/ULIP_models.py:517-547。僅供歷史參考,勿修改勿引用。
# =====================================================================
# ULIP/models/rag_enhancer.py (最終修正版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import json
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    def __init__(self, corpus_dir, top_k=5, device='cuda'):
        self.device = device
        self.top_k = top_k
        corpus_path = f"{corpus_dir}/rag_corpus.jsonl"
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = [json.loads(line) for line in f]
        self.index = faiss.read_index(f"{corpus_dir}/corpus_index.faiss")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    @torch.no_grad()
    def retrieve(self, queries):
        flat_queries = [q[0] for q in queries if q]
        if not flat_queries:
            return [[] for _ in queries]
        
        query_embeddings = self.encoder.encode(
            flat_queries, convert_to_tensor=True, show_progress_bar=False, device=self.device
        )
        query_embeddings = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)
        scores, indices = self.index.search(query_embeddings.cpu().numpy(), self.top_k)
        
        all_retrieved_docs = []
        for i in range(len(flat_queries)):
            retrieved_docs = [self.corpus[idx]['text'] for idx in indices[i]]
            all_retrieved_docs.append(retrieved_docs)
        return all_retrieved_docs

class RAGEnhancer(nn.Module):
    def __init__(self, corpus_dir, embed_dim=512, top_k=5, device='cuda'):
        super().__init__()
        self.device =device
        self.retriever = RAGRetriever(corpus_dir, top_k=top_k, device=device)
        doc_embed_dim = self.retriever.encoder.get_sentence_embedding_dimension()

        # 它的所有內部維度，都應該與 CLIP Base Transformer 的輸出維度 (transformer_width) 匹配
        # 我們將在主模型中傳入這個維度
        self.doc_projection = nn.Linear(doc_embed_dim, embed_dim)
        
        self.attention_weights = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.enhancement_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 確保所有模組都在正確的設備上
        self.to(device)

    def forward(self, original_text_features, raw_queries):
        batch_size = original_text_features.shape[0]
        
        # 1. 檢索相關文檔
        retrieved_docs = self.retriever.retrieve(raw_queries)
        
        # print("raw_queries:", raw_queries[0], "retrieved_docs:", retrieved_docs[0] )
        # assert False
        # 2. 編碼檢索到的文檔
        # --- START: 關鍵修正 ---
        # 使用正確的嵌套列表推導式來展平列表
        flat_docs = [doc for docs_list in retrieved_docs for doc in docs_list if docs_list]
        # --- END: 關鍵修正 ---

        if not flat_docs:
            return original_text_features

        with torch.no_grad():
            doc_embeddings = self.retriever.encoder.encode(
                flat_docs, convert_to_tensor=True, device='cpu'  # 先編碼到CPU
            )
            # 立即移動到目標設備
            doc_embeddings = doc_embeddings.to(original_text_features.device)
        
        doc_features_proj = self.doc_projection(doc_embeddings)
        
        doc_features_list = []
        current_pos = 0
        for docs in retrieved_docs:
            if docs:
                num_docs = len(docs)
                doc_features_list.append(doc_features_proj[current_pos : current_pos + num_docs])
                current_pos += num_docs
            else:
                # 確保佔位符也在正確的設備上
                placeholder = torch.zeros(
                    self.retriever.top_k, 
                    original_text_features.shape[-1], 
                    device=original_text_features.device  # 使用正確的設備
                )
                doc_features_list.append(placeholder)

        # 使用 padding 來處理長度不一的檢索結果
        padded_doc_features = torch.nn.utils.rnn.pad_sequence(doc_features_list, batch_first=True, padding_value=0.0)

        query_features = original_text_features.unsqueeze(1)
        
        context_features, _ = self.attention_weights(
            query=query_features,
            key=padded_doc_features,
            value=padded_doc_features
        )
        context_features = context_features.squeeze(1)
        
        enhanced_features = self.fusion_norm(original_text_features + context_features)
        final_features = enhanced_features + self.enhancement_mlp(enhanced_features)
        
        return final_features