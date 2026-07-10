# =====================================================================
# ⚠️ DEPRECATED(2026-07-10 標記):本檔已無人 import(rg 全 repo 實測)。
# 獨立版 RAG adapter 實驗。現役 RAG 實作在 core/models/ULIP_models.py:473-883。
# 僅供歷史參考,勿修改勿引用。
# =====================================================================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import faiss
# import json
# from sentence_transformers import SentenceTransformer

# class RAGRetriever:
#     """保持原有的檢索器，但用於特徵增強而非文本生成"""
#     def __init__(self, corpus_dir, top_k=5, device='cuda'):
#         self.device = device
#         self.top_k = top_k
#         corpus_path = f"{corpus_dir}/rag_corpus.jsonl"
#         with open(corpus_path, 'r', encoding='utf-8') as f:
#             self.corpus = [json.loads(line) for line in f]
#         self.index = faiss.read_index(f"{corpus_dir}/corpus_index.faiss")
#         self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

#     @torch.no_grad()
#     def retrieve(self, queries):
#         flat_queries = [q[0] if isinstance(q, list) and q else str(q) for q in queries]
#         if not flat_queries:
#             return [[] for _ in queries]
#         query_embeddings = self.encoder.encode(
#             flat_queries, convert_to_tensor=True, show_progress_bar=False, device=self.device
#         )
#         query_embeddings = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)
#         scores, indices = self.index.search(query_embeddings.cpu().numpy(), self.top_k)
#         all_retrieved_docs = []
#         for i in range(len(flat_queries)):
#             retrieved_docs = [self.corpus[idx]['text'] for idx in indices[i]]
#             all_retrieved_docs.append(retrieved_docs)
#         return all_retrieved_docs

# class RAGEnhancer(nn.Module):
#     """
#     新的 RAG 增強器：直接在特徵空間進行增強，而不是生成文本
#     這樣可以保證梯度流動，並且計算效率更高
#     """
#     def __init__(self, corpus_dir, clip_embed_dim=512, top_k=5, device='cuda'):
#         super().__init__()
#         self.retriever = RAGRetriever(corpus_dir, top_k=top_k, device=device)
        
#         # 獲取檢索到的文檔嵌入的維度
#         doc_embed_dim = self.retriever.encoder.get_sentence_embedding_dimension()
        
#         # 可訓練的組件
#         self.doc_projection = nn.Linear(doc_embed_dim, clip_embed_dim)
#         self.attention_weights = nn.MultiheadAttention(
#             embed_dim=clip_embed_dim,
#             num_heads=8,
#             batch_first=True,
#             dropout=0.1
#         )
#         self.fusion_norm = nn.LayerNorm(clip_embed_dim)
#         self.enhancement_mlp = nn.Sequential(
#             nn.Linear(clip_embed_dim, clip_embed_dim * 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(clip_embed_dim * 2, clip_embed_dim)
#         )

#     def forward(self, original_text_features, raw_queries):
#         """
#         直接增強文本特徵，而不需要重新編碼
        
#         Args:
#             original_text_features: [batch_size, embed_dim] - 原始 CLIP 文本特徵
#             raw_queries: list of strings - 原始查詢文本
        
#         Returns:
#             enhanced_features: [batch_size, embed_dim] - 增強後的文本特徵
#         """
#         batch_size = original_text_features.shape[0]
        
#         # 1. 檢索相關文檔
#         retrieved_docs = self.retriever.retrieve(raw_queries)
        
#         # 2. 編碼檢索到的文檔
#         all_doc_embeddings = []
#         for docs in retrieved_docs:
#             if docs:
#                 with torch.no_grad():
#                     doc_embs = self.retriever.encoder.encode(
#                         docs, convert_to_tensor=True, device=original_text_features.device
#                     )
#                 all_doc_embeddings.append(doc_embs)
#             else:
#                 # 如果沒有檢索到文檔，使用零向量
#                 dummy_emb = torch.zeros(
#                     (self.retriever.top_k, self.retriever.encoder.get_sentence_embedding_dimension()),
#                     device=original_text_features.device
#                 )
#                 all_doc_embeddings.append(dummy_emb)
        
#         # 3. 將文檔嵌入投影到 CLIP 空間
#         doc_embeddings = torch.stack(all_doc_embeddings)  # [batch_size, top_k, doc_dim]
#         doc_features = self.doc_projection(doc_embeddings)  # [batch_size, top_k, clip_dim]
        
#         # 4. 使用注意力機制融合原始特徵和檢索特徵
#         query_features = original_text_features.unsqueeze(1)  # [batch_size, 1, clip_dim]
#         enhanced_features, _ = self.attention_weights(
#             query=query_features,
#             key=doc_features,
#             value=doc_features
#         )
#         enhanced_features = enhanced_features.squeeze(1)  # [batch_size, clip_dim]
        
#         # 5. 殘差連接和進一步增強
#         enhanced_features = self.fusion_norm(original_text_features + enhanced_features)
#         final_features = enhanced_features + self.enhancement_mlp(enhanced_features)
        
#         return final_features