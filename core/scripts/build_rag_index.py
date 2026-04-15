# --- 原 build_rag_index.py 的同一路徑下，把這段覆蓋掉 ------------------
import open_clip
import torch
import faiss
import json
import numpy as np
import os
from tqdm import tqdm

def build_rag_index(corpus_dir='/home/kyzen/ULIP_RAG/ulip_rag/rag_corpus',
                    clip_model_name='ViT-B-32',
                    pretrained_tag='laion2b_s34b_b79k',
                    batch_size=64,
                    device='cuda'):

    corpus_path = os.path.join(corpus_dir, 'rag_corpus.jsonl')
    index_path  = os.path.join(corpus_dir, 'clip_corpus_index.faiss')

    # 1. 讀語料
    print(f'🔍  Loading corpus from {corpus_path} …')
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = [json.loads(l) for l in f]
    texts = [x['text'] for x in corpus]
    print(f'✓  {len(texts)} chunks')

    # 2. 取 CLIP text encoder
    print(f'\n⏳  Loading CLIP model: {clip_model_name} ({pretrained_tag})')
    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=pretrained_tag, device=torch.device(device))
    clip_model.eval()

    # 3. 批次編碼文字 → embedding
    print('⚙️  Encoding text with CLIP …')
    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            tokens = open_clip.tokenize(batch).to(device)
            feats  = clip_model.encode_text(tokens)
            feats  = feats / feats.norm(dim=-1, keepdim=True)  # L2-norm
            all_embeds.append(feats.cpu())

    embeddings = torch.cat(all_embeds, 0).numpy().astype('float32')

    # 4. 建 FAISS Index - cosine = inner-product (因為已 L2-norm)
    print(f'\n🚀  Building FAISS index  (dim = {embeddings.shape[1]})')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # 5. 儲存
    faiss.write_index(index, index_path)
    print(f'💾  Saved to {index_path}')

if __name__ == '__main__':
    build_rag_index()
