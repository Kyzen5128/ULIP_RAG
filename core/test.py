"""
ULIP RAG 模型評估腳本
修正版本 - 與 RAG 架構保持一致，支援 ShapeNet 測試
包含 NDCG 和 MRR 指標計算
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import json
import os
from pathlib import Path

import models.ULIP_models as models
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))  # 2026-07-10 消除 cwd=core 依賴
from utils.tokenizer import SimpleTokenizer
from utils import utils
from data.dataset_3d import Dataset_3D, rag_collate_fn, customized_collate_fn


def simple_cross_modal_collate_fn(batch):
    """
    同時支援：
      • (pc, label, name)          —— ModelNet40 舊格式
      • (taxonomy_id, model_id, text, pc, img) —— ShapeNet-55
    """
    if isinstance(batch[0][0], torch.Tensor):     # 舊格式
        pc  = torch.stack([b[0] for b in batch], 0)
        lbl = torch.LongTensor([b[1] for b in batch])
        name = [b[2] for b in batch]
        return pc, lbl, name

    # ---- ShapeNet-55 新格式 ----
    pc  = torch.stack([b[3] for b in batch], 0)         # 第 4 個才是點雲
    lbl = [b[0] for b in batch]                         # taxonomy_id (str)
    name = [b[1] for b in batch]                        # model_id
    return pc, lbl, name


def calculate_ndcg_at_k(scores, relevance, k):
    """
    計算 NDCG@K
    """
    if len(scores) == 0:
        return 0.0
    
    sorted_indices = np.argsort(scores)[::-1]
    sorted_relevance = relevance[sorted_indices]
    
    dcg = 0.0
    for i in range(min(k, len(sorted_relevance))):
        if sorted_relevance[i] > 0:
            dcg += (2 ** sorted_relevance[i] - 1) / np.log2(i + 2)
    
    ideal_relevance = np.sort(relevance)[::-1]
    idcg = 0.0
    for i in range(min(k, len(ideal_relevance))):
        if ideal_relevance[i] > 0:
            idcg += (2 ** ideal_relevance[i] - 1) / np.log2(i + 2) 
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_mrr(scores, relevance):
    """
    計算 MRR (Mean Reciprocal Rank)
    """
    if len(scores) == 0:
        return 0.0
    
    sorted_indices = np.argsort(scores)[::-1]
    sorted_relevance = relevance[sorted_indices]
    
    for i, rel in enumerate(sorted_relevance):
        if rel > 0:
            return 1.0 / (i + 1)
    
    return 0.0


def calculate_retrieval_metrics(query_features, doc_features, labels, k_values=[1, 5, 10]):
    """
    計算檢索指標
    """
    batch_size = query_features.shape[0]
    ndcg_scores = {f'NDCG@{k}': [] for k in k_values}
    mrr_scores = []
    
    similarity_matrix = F.cosine_similarity(
        query_features.unsqueeze(1),
        doc_features.unsqueeze(0),
        dim=2
    )
    
    for i in range(batch_size):
        scores = similarity_matrix[i].cpu().numpy()
        relevance = labels[i].cpu().numpy()
        
        mrr = calculate_mrr(scores, relevance)
        mrr_scores.append(mrr)
        
        for k in k_values:
            ndcg = calculate_ndcg_at_k(scores, relevance, k)
            ndcg_scores[f'NDCG@{k}'].append(ndcg)
    
    metrics = {}
    metrics['MRR'] = np.mean(mrr_scores) if mrr_scores else 0.0
    
    for k in k_values:
        metrics[f'NDCG@{k}'] = np.mean(ndcg_scores[f'NDCG@{k}']) if ndcg_scores[f'NDCG@{k}'] else 0.0
    
    return metrics 

def calculate_retrieval_metrics_streaming(pc_feat, text_feat, labels, ks=(1, 5, 10)):
    n_query  = pc_feat.size(0)
    text_feat = text_feat.float().cpu()
    text_feat_t = text_feat.t()

    ndcg = {k: [] for k in ks}
    mrr  = []

    for i in range(n_query):
        score = (pc_feat[i].unsqueeze(0) @ text_feat_t).squeeze(0)   # (C,)
        rel   = labels[i]

        rank  = (-score).argsort()
        hit   = (rel[rank] > 0).nonzero(as_tuple=True)[0]
        mrr.append(1.0 / (hit[0].item()+1) if len(hit) else 0.0)

        # --------- 修補點：k_eff = min(k, #classes) ----------
        C = rel.size(0)
        for k in ks:
            k_eff  = min(k, C)                       # ****
            topk   = rank[:k_eff]                   # ****
            gains  = (2**rel[topk].float()-1) / torch.log2(torch.arange(k_eff)+2)   # ****
            dcg    = gains.sum().item()

            ideal  = torch.sort(rel, descending=True)[0][:k_eff]                    # ****
            idcg   = ((2**ideal.float()-1) / torch.log2(torch.arange(k_eff)+2)).sum().item()
            ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)

    out = {'MRR': np.mean(mrr)}
    out.update({f'NDCG@{k}': np.mean(ndcg[k]) for k in ks})
    return out

def evaluate_rag_quality(model, test_loader, args):
    """
    評估 RAG 質量 - 分析檢索到的文檔與查詢的相關性
    """
    model.eval()
    print("=> Evaluating RAG retrieval quality...")
    
    total_queries = 0
    total_docs_retrieved = 0
    all_similarities = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if batch_data is None:
                continue
                
            try:
                # 處理 RAG 格式的 batch_data: (pc, text_data, image)
                if len(batch_data) == 3:
                    pc, text_data, image = batch_data
                else:
                    print(f"Unexpected batch structure with {len(batch_data)} elements")
                    continue
                
                # 處理 RAG 格式的文本輸入
                if isinstance(text_data, tuple) and len(text_data) == 2:
                    tokenized_text, raw_text = text_data
                    if tokenized_text.dim() == 3:
                        tokenized_text = tokenized_text.squeeze(1)
                else:
                    print("Warning: Expected RAG format (tokenized, raw_text) but got different format")
                    continue
                
                tokenized_text = tokenized_text.cuda(args.gpu, non_blocking=True)
                
                # 獲取原始和增強的文本特徵
                original_features = utils.get_model(model).encode_text(tokenized_text)
                enhanced_features = utils.get_model(model).encode_text_with_rag(tokenized_text, raw_text)
                
                # 檢索文檔用於分析
                retrieved_docs = utils.get_model(model).retriever.retrieve_docs(raw_text, utils.get_model(model).rag_corpus)
                
                # 計算增強效果
                for j in range(len(raw_text)):
                    if j < len(retrieved_docs) and retrieved_docs[j]:
                        total_queries += 1
                        total_docs_retrieved += len(retrieved_docs[j])
                        
                        # 計算原始查詢與檢索文檔的相似度
                        query_text = raw_text[j] if isinstance(raw_text[j], str) else raw_text[j][0]
                        for doc in retrieved_docs[j]:
                            # 簡單的詞匯重疊計算
                            query_words = set(query_text.lower().split())
                            doc_words = set(doc.lower().split())
                            if query_words and doc_words:
                                similarity = len(query_words & doc_words) / len(query_words | doc_words)
                                all_similarities.append(similarity)
                
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
                
            if i % args.print_freq == 0:
                print(f"Processed {i+1}/{len(test_loader)} batches")

    # 計算統計數據
    avg_docs_per_query = total_docs_retrieved / total_queries if total_queries > 0 else 0
    avg_similarity = np.mean(all_similarities) if all_similarities else 0
    
    print("RAG quality evaluation completed.")
    return {
        'total_queries': total_queries,
        'avg_docs_per_query': avg_docs_per_query,
        'avg_query_doc_similarity': avg_similarity,
        'similarity_std': np.std(all_similarities) if all_similarities else 0,
        'max_similarity': np.max(all_similarities) if all_similarities else 0,
        'min_similarity': np.min(all_similarities) if all_similarities else 0
    }

def evaluate_cross_modal_retrieval(model, test_loader, args):
    """
    評估跨模態檢索性能 (3D ↔ Text)
    正確使用 ShapeNet taxonomy_id 作為真實標籤
    """
    model.eval()
    print("=> Evaluating cross-modal retrieval...")

    all_pc_features, all_labels = [], []

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if batch_data is None:
                continue

            try:
                # 對應 rag_collate_fn 輸出 (pc_tensor, text_batch, img_tensor, taxonomy_id) — 若已擴充
                if len(batch_data) == 4:
                    pc, text_data, image, labels = batch_data
                elif len(batch_data) == 3:
                    pc, text_data, image = batch_data
                    labels = [0] * pc.shape[0]  # 假標籤（僅作為 fallback）
                else:
                    print(f"Unexpected batch length: {len(batch_data)}")
                    continue

                pc = pc.cuda(args.gpu, non_blocking=True)
                pc_features = utils.get_model(model).encode_pc(pc)
                pc_features = F.normalize(pc_features, dim=-1)
                all_pc_features.append(pc_features.cpu())

                all_labels.extend(labels)  # taxonomy_id 應是 list[str] 或 list[int]

            except Exception as e:
                print(f"[ERROR @ Batch {i}] {e}")
                continue

            if i % args.print_freq == 0:
                print(f"Processed {i+1}/{len(test_loader)} batches")

    all_pc_features = torch.cat(all_pc_features, dim=0)

    # 取得 taxonomy 列表與 label 名稱
    with open('/mnt/P300/data/ULIP/ULIP_Shapenet_Triplets/taxonomy.json', 'r') as f:
        taxonomy_data = json.load(f)
    synset_to_name = {item['synsetId']: item['name'] for item in taxonomy_data}
    unique_labels = list(synset_to_name.keys())  # synsetId = e.g., '02691156'
    label_names = [synset_to_name[sid].split(',')[0] for sid in unique_labels]

    # 編碼 label 對應的描述文字
    tokenizer = SimpleTokenizer()
    with open(os.path.join(_CORE_DIR, "data/configs/templates.json")) as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    text_features = []
    print("=> Encoding class text features...")
    for label_name in label_names:
        prompts = [t.format(label_name) for t in templates]
        tokenized = tokenizer(prompts).cuda(args.gpu, non_blocking=True)
        if tokenized.ndim < 2:
            tokenized = tokenized[None, ...]
        if args.use_rag_adapter:
            raw_text = [[p] for p in prompts]
            class_emb = utils.get_model(model).encode_text_with_rag(tokenized, raw_text)
        else:
            class_emb = utils.get_model(model).encode_text(tokenized)
        class_emb = F.normalize(class_emb, dim=-1).mean(dim=0)
        text_features.append(class_emb.cpu())

    text_features = torch.stack(text_features, dim=0)

    # 建立 one-hot relevance matrix
    label_to_index = {sid: idx for idx, sid in enumerate(unique_labels)}
    relevance_matrix = torch.zeros(all_pc_features.shape[0], len(unique_labels))

    for i, label in enumerate(all_labels):
        if label in label_to_index:
            relevance_matrix[i, label_to_index[label]] = 1

    print("=> Calculating retrieval metrics...")
    return calculate_retrieval_metrics_streaming(all_pc_features, text_features, relevance_matrix)







def main():
    parser = argparse.ArgumentParser(description='ULIP RAG Model Evaluation')
    
    # 基本參數
    parser.add_argument('--model', default='ULIP_PointBERT_RAG', type=str, help="Model name")
    parser.add_argument('--test_ckpt_addr', required=True, type=str, help='Checkpoint path for testing')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for evaluation')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--print-freq', default=50, type=int, help='Print frequency')
    
    # 數據集參數
    parser.add_argument('--validate_dataset_name', default='shapenet', type=str, 
                        help='Dataset name: shapenet, modelnet40, objaverse')
    parser.add_argument('--validate_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--npoints', default=8192, type=int)
    
    # RAG 參數
    parser.add_argument('--use_rag_adapter', action='store_true', help='Use RAG model')
    parser.add_argument('--training_strategy', type=str, default='staged_2', choices=['staged_1', 'staged_2'])
    parser.add_argument('--rag_corpus_dir', type=str, default=None)  # None→自動用 core/rag_corpus
    parser.add_argument('--rag_top_k', type=int, default=5)
    
    # 評估選項
    parser.add_argument('--eval_rag_quality', action='store_true', help='Evaluate RAG retrieval quality')
    parser.add_argument('--eval_cross_modal', action='store_true', help='Evaluate cross-modal retrieval')
    parser.add_argument('--eval_zero_shot', action='store_true', help='Evaluate zero-shot classification')
    
    # 從 main.py 補全的參數
    parser.add_argument('--evaluate_3d', action='store_true', default=True, help='Set to True for testing.')
    parser.add_argument('--use_height', action='store_true', help='Whether to use height normalization.')
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--stage1_ckpt_path', type=str, default=None, help='Stage 1 checkpoint path (not used in testing)')

    args = parser.parse_args()
    
    # 強制設定評估模式
    args.evaluate_3d = True

    torch.cuda.set_device(args.gpu)
    
    # 載入模型
    print(f"=> Loading checkpoint '{args.test_ckpt_addr}'")
    checkpoint = torch.load(args.test_ckpt_addr, map_location='cpu', weights_only=False)
    
    # 處理可能的 'module.' 前綴
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    if isinstance(state_dict, dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        print("Warning: Could not find state_dict in checkpoint")
        state_dict = {}
    
    # 創建模型
    if args.use_rag_adapter:
        model = models.ULIP_PointBERT_RAG(args)
    else:
        model = getattr(models, args.model)(args)
    
    model.cuda(args.gpu)
    
    # 載入權重
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # 分析 missing keys
    rag_missing_keys = [k for k in missing_keys if 'rag_enhancer' in k]
    other_missing_keys = [k for k in missing_keys if 'rag_enhancer' not in k]
    
    if rag_missing_keys and args.use_rag_adapter:
        print(f"INFO: RAG enhancer weights not found in checkpoint (expected for non-RAG checkpoints)")
        print(f"RAG enhancer will use randomly initialized weights.")
    
    if other_missing_keys:
        print(f"Warning - Missing non-RAG keys: {other_missing_keys}")
    
    if unexpected_keys:
        print(f"Warning - Unexpected keys: {unexpected_keys}")
        
    model.eval()
    print("=> Model loaded successfully")
    
    # 準備數據
    tokenizer = SimpleTokenizer()
    
    # 為測試準備 transform
    import torchvision.transforms as transforms
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 根據評估任務和數據集選擇正確的 collate_fn
    if args.eval_rag_quality and args.use_rag_adapter:
        print("Using RAG collate function for RAG quality evaluation.")
        collate_fn = rag_collate_fn
    elif args.eval_cross_modal:
        if args.use_rag_adapter:
            print("Using RAG collate function for cross-modal evaluation.")
            collate_fn = rag_collate_fn
        else:
            print("Using simple collate function for cross-modal evaluation.")
            collate_fn = simple_cross_modal_collate_fn
    elif args.eval_zero_shot:
        # 2026-07-10 修:zero-shot 走 main.test_zeroshot_3d_core,其期望「預設 collate」
        # 的 (pc, target, target_name) 批次(見 main.py 內 len(data_tuple)==3 分支;
        # main.py 的 val_loader 本來就不掛 collate_fn)。原本對 use_rag_adapter 硬掛
        # rag_collate_fn,會把 ModelNet 的 int32 label 當文字 torch.stack 而 TypeError。
        print("Using default collate for zero-shot evaluation (same as main.py val_loader).")
        collate_fn = None
    else:
        print("Using standard collate function as fallback.")
        collate_fn = customized_collate_fn
    
    test_dataset = utils.get_dataset(test_transform, tokenizer, args, 'val')  # 傳入 test_transform！
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True, 
        collate_fn=collate_fn,
        drop_last=False
    )
    print(f"=> Dataset loaded: {len(test_dataset)} samples")
    
    # 執行評估
    results = {}
    
    if args.eval_rag_quality and args.use_rag_adapter:
        results['RAG_Quality'] = evaluate_rag_quality(model, test_loader, args)
    
    if args.eval_cross_modal:
        results['Cross_Modal_Retrieval'] = evaluate_cross_modal_retrieval(model, test_loader, args)
    
    if args.eval_zero_shot:
        from main import test_zeroshot_3d_core
        results['Zero_Shot_Classification'] = test_zeroshot_3d_core(test_loader, model, tokenizer, args)

    # 如果沒有指定任何評估類型，默認執行零樣本分類
    if not (args.eval_rag_quality or args.eval_cross_modal or args.eval_zero_shot):
        print("No specific evaluation type specified, running zero-shot classification...")
        from main import test_zeroshot_3d_core
        results['Zero_Shot_Classification'] = test_zeroshot_3d_core(test_loader, model, tokenizer, args)

    # 打印和保存結果
    if results:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for eval_type, metrics in results.items():
            if metrics:
                print(f"\n=== {eval_type.replace('_', ' ')} ===")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"{key:25}: {value:.4f}")
                    else:
                        print(f"{key:25}: {value}")
        
        # 保存結果
        output_file = f"evaluation_results_{args.validate_dataset_name}_{args.model}.json"
        if args.use_rag_adapter:
            output_file = output_file.replace('.json', '_RAG.json')
            
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=> Results saved to {output_file}")
        print("="*60)
    else:
        print("\nNo evaluation was performed.")


if __name__ == '__main__':
    main()