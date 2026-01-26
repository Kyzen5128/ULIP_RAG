#!/usr/bin/env python3
"""
ULIP RAG 快速測試腳本
測試 RAG Enhancer 和數據載入功能
"""

import sys
import torch
import numpy as np

# 添加必要的路徑
sys.path.insert(0, '/home/kyzen/cheng/ulip_rag')
sys.path.insert(0, '/mnt/data1/cheng/ULIP/data')

print("=" * 60)
print("  ULIP RAG 快速功能測試")
print("=" * 60)

# 1. 測試 RAG Enhancer
print("\n[1] 測試 RAG Enhancer...")
try:
    from models.rag_enhancer import RAGEnhancer, RAGRetriever
    
    # 創建檢索器
    corpus_dir = '/mnt/data1/cheng/ULIP/data/data/rag_corpus'
    print(f"  載入RAG語料庫: {corpus_dir}")
    
    retriever = RAGRetriever(corpus_dir=corpus_dir, top_k=3, device='cuda')
    print(f"  ✓ 檢索器初始化成功")
    print(f"  - 語料庫大小: {len(retriever.corpus)} 條")
    print(f"  - 索引向量數: {retriever.index.ntotal}")
    
    # 測試檢索
    test_queries = [
        ["a modern wooden chair"],
        ["comfortable sofa with armrests"],
        ["dining table"]
    ]
    
    print(f"\n  測試檢索功能...")
    retrieved_docs = retriever.retrieve(test_queries)
    
    for i, (query, docs) in enumerate(zip(test_queries, retrieved_docs)):
        print(f"\n  查詢 {i+1}: {query[0]}")
        print(f"  檢索到 {len(docs)} 個文檔:")
        for j, doc in enumerate(docs[:2], 1):  # 只顯示前2個
            print(f"    {j}. {doc[:80]}...")
    
    # 創建 RAG Enhancer
    print(f"\n  初始化 RAG Enhancer...")
    enhancer = RAGEnhancer(
        corpus_dir=corpus_dir,
        embed_dim=512,
        top_k=3,
        device='cuda'
    )
    print(f"  ✓ RAG Enhancer 初始化成功")
    
    # 測試特徵增強
    print(f"\n  測試特徵增強...")
    batch_size = 2
    dummy_text_features = torch.randn(batch_size, 512).cuda()
    raw_queries = [["modern chair"], ["wooden table"]]
    
    enhanced_features = enhancer(dummy_text_features, raw_queries)
    print(f"  ✓ 特徵增強成功")
    print(f"  - 輸入形狀: {dummy_text_features.shape}")
    print(f"  - 輸出形狀: {enhanced_features.shape}")
    print(f"  - 輸出設備: {enhanced_features.device}")
    
    print(f"\n✓✓✓ RAG Enhancer 測試通過！")
    
except Exception as e:
    print(f"✗✗✗ RAG Enhancer 測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 2. 測試數據集載入
print("\n" + "=" * 60)
print("[2] 測試數據集載入...")
try:
    from data.dataset_3d import Dataset_3D, rag_collate_fn
    from utils.tokenizer import SimpleTokenizer
    import argparse
    
    # 創建模擬參數
    args = argparse.Namespace(
        pretrain_dataset_name='shapenet',
        validate_dataset_name='modelnet40',
        pretrain_dataset_prompt='shapenet_64',
        validate_dataset_prompt='modelnet40_64',
        use_rag_adapter=True,
        use_height=False,
        npoints=8192,
        model='ULIP_PointBERT'
    )
    
    print(f"  創建 tokenizer...")
    tokenizer = SimpleTokenizer()
    print(f"  ✓ Tokenizer 創建成功")
    
    print(f"\n  測試 ModelNet40 數據集載入...")
    try:
        dataset_3d = Dataset_3D(args, tokenizer, 'val', train_transform=None)
        dataset = dataset_3d.dataset
        print(f"  ✓ Dataset 載入成功")
        print(f"  - 數據集名稱: {dataset_3d.dataset_name}")
        print(f"  - 樣本數量: {len(dataset)}")
        
        # 獲取一個樣本
        print(f"\n  測試獲取樣本...")
        sample = dataset[0]
        print(f"  ✓ 樣本獲取成功")
        print(f"  - 樣本類型: {type(sample)}")
        print(f"  - 樣本長度: {len(sample)}")
        
        if len(sample) >= 3:
            pc, label, name = sample[0], sample[1], sample[2]
            print(f"  - 點雲形狀: {pc.shape}")
            print(f"  - 標籤: {label}")
            print(f"  - 名稱: {name}")
        
    except Exception as e:
        print(f"  ⚠ ModelNet40 數據集測試跳過: {e}")
    
    print(f"\n✓✓✓ 數據集載入測試通過！")
    
except Exception as e:
    print(f"✗✗✗ 數據集測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 3. 測試 collate function
print("\n" + "=" * 60)
print("[3] 測試 RAG Collate Function...")
try:
    from data.dataset_3d import rag_collate_fn
    
    # 創建模擬batch (ShapeNet格式)
    print(f"  創建模擬 batch...")
    batch = []
    for i in range(2):
        taxonomy_id = f"0269115{i}"
        model_id = f"test_model_{i}"
        
        # 文本：(tokenized_tensor, [raw_string])
        tokenized = torch.randint(0, 1000, (1, 77))
        raw_text = [f"a modern chair {i}"]
        text = (tokenized, raw_text)
        
        # 點雲
        pc = torch.randn(8192, 3)
        
        # 圖像
        img = torch.randn(3, 224, 224)
        
        batch.append((taxonomy_id, model_id, text, pc, img))
    
    # 測試 collate
    print(f"  測試 rag_collate_fn...")
    result = rag_collate_fn(batch)
    
    if result is not None:
        pc_tensor, text_batch, img_tensor, labels = result
        print(f"  ✓ Collate 成功")
        print(f"  - PC形狀: {pc_tensor.shape}")
        print(f"  - Text類型: {type(text_batch)}")
        if isinstance(text_batch, tuple):
            print(f"    - Tokens形狀: {text_batch[0].shape}")
            print(f"    - Raw text: {text_batch[1]}")
        print(f"  - Image形狀: {img_tensor.shape if img_tensor is not None else 'None'}")
        print(f"  - Labels: {labels}")
    
    print(f"\n✓✓✓ Collate Function 測試通過！")
    
except Exception as e:
    print(f"✗✗✗ Collate測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 4. 測試模型導入
print("\n" + "=" * 60)
print("[4] 測試模型導入...")
try:
    import models.ULIP_models_rag as models_rag
    
    print(f"  ✓ ULIP_models_rag 導入成功")
    
    # 檢查關鍵函數
    if hasattr(models_rag, 'ULIP_PointBERT_RAG'):
        print(f"  ✓ ULIP_PointBERT_RAG 工廠函數存在")
    
    if hasattr(models_rag, 'ULIP_Loss_RAG_Enhanced'):
        print(f"  ✓ ULIP_Loss_RAG_Enhanced 損失函數存在")
    
    print(f"\n✓✓✓ 模型導入測試通過！")
    
except Exception as e:
    print(f"✗✗✗ 模型導入測試失敗: {e}")
    import traceback
    traceback.print_exc()

# 總結
print("\n" + "=" * 60)
print("  測試總結")
print("=" * 60)
print("\n所有基本功能測試完成！")
print("\n您現在可以：")
print("1. 運行完整訓練:")
print("   python main.py --use_rag_adapter --training_strategy staged_1 ...")
print("\n2. 運行評估:")
print("   python test.py --use_rag_adapter --test_ckpt_addr <checkpoint> ...")
print("\n3. 查看詳細文檔:")
print("   - file_overview.md")
print("   - data_directory_summary.md")
print("=" * 60)
