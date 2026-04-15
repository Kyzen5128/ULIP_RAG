#!/usr/bin/env python3
"""
ULIP RAG 環境檢查腳本
檢查所有必要的依賴是否已安裝並且可以正常運作
"""

import sys
import subprocess

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_import(module_name, package_name=None):
    """檢查模組是否可以導入"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name:30s} OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name:30s} FAILED: {e}")
        return False

def check_version(module_name, min_version=None):
    """檢查模組版本"""
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        status = f"✓ {module_name:20s} version: {version}"
        if min_version and version != 'unknown':
            print(status)
        else:
            print(status)
        return True
    except:
        return False

def main():
    print_section("ULIP RAG 環境檢查")
    
    print(f"\nPython版本: {sys.version}")
    print(f"Python路徑: {sys.executable}")
    
    # 1. 核心依賴
    print_section("1. 核心深度學習框架")
    all_ok = True
    all_ok &= check_import('torch', 'PyTorch')
    all_ok &= check_import('torchvision', 'TorchVision')
    all_ok &= check_import('numpy', 'NumPy')
    
    # 檢查 CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        device_count = torch.cuda.device_count() if cuda_available else 0
        print(f"  CUDA可用: {cuda_available}")
        print(f"  CUDA版本: {cuda_version}")
        print(f"  GPU數量: {device_count}")
        if cuda_available:
            for i in range(device_count):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    except:
        print("  ✗ 無法檢查CUDA狀態")
    
    # 2. CLIP 和視覺模型
    print_section("2. CLIP 和視覺相關")
    all_ok &= check_import('open_clip', 'Open-CLIP')
    all_ok &= check_import('timm', 'TIMM')
    all_ok &= check_import('PIL', 'Pillow')
    
    # 3. RAG 相關
    print_section("3. RAG 檢索相關")
    all_ok &= check_import('sentence_transformers', 'Sentence-Transformers')
    all_ok &= check_import('faiss', 'FAISS')
    all_ok &= check_import('transformers', 'Transformers')
    
    # 4. 專案特定依賴
    print_section("4. 專案特定依賴")
    all_ok &= check_import('easydict', 'EasyDict')
    all_ok &= check_import('h5py', 'H5Py')
    all_ok &= check_import('open3d', 'Open3D')
    all_ok &= check_import('wandb', 'WandB')
    all_ok &= check_import('termcolor', 'Termcolor')
    all_ok &= check_import('lmdb', 'LMDB')
    all_ok &= check_import('yaml', 'PyYAML')
    all_ok &= check_import('tqdm', 'TQDM')
    
    # 5. 檢查data模組
    print_section("5. 檢查數據模組")
    try:
        sys.path.insert(0, '/home/kyzen/ULIP_RAG/ulip_rag')
        from data.dataset_3d import Dataset_3D, rag_collate_fn, customized_collate_fn
        print("✓ dataset_3d 模組導入成功")
        print("✓ Dataset_3D 類別可用")
        print("✓ rag_collate_fn 函數可用")
        print("✓ customized_collate_fn 函數可用")
    except Exception as e:
        print(f"✗ dataset_3d 模組導入失敗: {e}")
        all_ok = False
    
    # 6. 檢查模型模組
    print_section("6. 檢查模型模組")
    try:
        sys.path.insert(0, '/home/kyzen/ULIP_RAG/ulip_rag')
        import models.ULIP_models as models
        print("✓ ULIP_models 模組導入成功")
        
        # 檢查RAG模組
        from models.rag_enhancer import RAGEnhancer, RAGRetriever
        print("✓ RAG Enhancer 模組可用")
        
    except Exception as e:
        print(f"✗ 模型模組導入失敗: {e}")
        all_ok = False
    
    # 7. 檢查工具模組
    print_section("7. 檢查工具模組")
    try:
        from utils.tokenizer import SimpleTokenizer
        from utils import utils
        print("✓ Tokenizer 可用")
        print("✓ Utils 模組可用")
    except Exception as e:
        print(f"✗ 工具模組導入失敗: {e}")
        all_ok = False
    
    # 8. 檢查關鍵檔案
    print_section("8. 檢查關鍵檔案和目錄")
    import os
    
    paths_to_check = [
        ("/home/kyzen/ULIP_RAG/core/data/configs/dataset_3d.py", "dataset_3d.py"),
        ("/home/kyzen/ULIP_RAG/core/data/configs/rag_corpus/rag_corpus.jsonl", "RAG語料庫"),
        ("/home/kyzen/ULIP_RAG/core/data/configs/rag_corpus/corpus_index.faiss", "FAISS索引"),
        ("/home/kyzen/ULIP_RAG/core/data/configs/dataset_catalog.json", "數據集目錄"),
        ("/home/kyzen/ULIP_RAG/core/data/configs/templates.json", "文本模板"),
        ("/home/kyzen/ULIP_RAG/core/data/configs/labels.json", "標籤文件"),
        ("/home/kyzen/ULIP_RAG/ulip_rag/main.py", "主訓練腳本"),
        ("/home/kyzen/ULIP_RAG/ulip_rag/test.py", "測試腳本"),
    ]
    
    for path, name in paths_to_check:
        if os.path.exists(path):
            size = os.path.getsize(path) if os.path.isfile(path) else "DIR"
            print(f"✓ {name:30s} 存在 ({size})")
        else:
            print(f"✗ {name:30s} 不存在")
            all_ok = False
    
    # 9. 檢查版本
    print_section("9. 已安裝套件版本")
    check_version('torch')
    check_version('torchvision')
    check_version('numpy')
    check_version('transformers')
    check_version('sentence_transformers')
    check_version('timm')
    check_version('open_clip')
    
    # 最終結果
    print_section("檢查結果")
    if all_ok:
        print("✓✓✓ 所有檢查通過！環境已準備就緒！")
        print("\n您可以開始訓練或測試了：")
        print("  訓練: python main.py --use_rag_adapter ...")
        print("  測試: python test.py --use_rag_adapter ...")
        return 0
    else:
        print("✗✗✗ 部分檢查失敗，請檢查上述錯誤")
        print("\n建議安裝缺失的依賴：")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
