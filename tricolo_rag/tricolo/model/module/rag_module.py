# 檔案路徑: tricolo/model/module/rag_module.py

import torch
import torch.nn as nn
import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    def __init__(self, corpus_dir, top_k=5, device='cuda'):
        print(f"INFO: Initializing RAGRetriever with corpus directory '{corpus_dir}' and top_k={top_k}...")
        self.top_k = top_k
        index_path = os.path.join(corpus_dir, 'corpus_index.faiss')
        print(f"INFO: RAGRetriever - Loading FAISS index from '{index_path}'...")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run build_rag_index.py first.")
        self.index = faiss.read_index(index_path)
        self.query_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("INFO: RAGRetriever is ready.")

    @torch.no_grad()
    def retrieve_docs(self, raw_text_queries, corpus):
        if not raw_text_queries: return [[] for _ in raw_text_queries]
        processed_queries = [q[0] if isinstance(q, list) and len(q) > 0 else q if isinstance(q, str) else "" for q in raw_text_queries]
        if not any(q.strip() for q in processed_queries): return [[] for _ in raw_text_queries]
            
        query_embeddings = self.query_encoder.encode(processed_queries, convert_to_numpy=True, show_progress_bar=False)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        _, indices = self.index.search(query_embeddings.astype('float32'), self.top_k)
        
        return [[corpus[idx]['text'] for idx in q_indices if idx != -1] for q_indices in indices]

class RAGEnhancer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))

    def forward(self, original_feature, doc_features):
        original_expanded = original_feature.unsqueeze(1)
        context = torch.cat([original_expanded, doc_features], dim=1)
        attn_output, _ = self.attention(original_expanded, context, context)
        fused = self.norm1(original_expanded + attn_output)
        fused = self.norm2(fused + self.ffn(fused))
        return fused.squeeze(1)