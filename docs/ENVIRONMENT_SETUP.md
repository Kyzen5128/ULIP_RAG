> ⚠️ **過時警告(2026-07-10)**:本檔為 4090 時代文件。專案位置/env 名/路徑均已失效(現為 /home/kyzen/ULIP_RAG + conda ulip + /mnt/P300/data)。現況請看 docs/INDEX.md。

# ULIP_RAG 環境架設指南

本文檔記錄了 ULIP_RAG 項目的完整環境架設步驟和所需依賴。

## 系統需求

- **作業系統**: Linux (Ubuntu 推薦)
- **Python 版本**: 3.9.25
- **CUDA 版本**: 12.1
- **GPU**: 支援 CUDA 的 NVIDIA GPU

## 環境架設步驟

### 1. 安裝 Miniconda

如果尚未安裝 Miniconda，請先下載並安裝：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 2. 創建 Conda 環境

```bash
conda create -n ulip_rag python=3.9.25 -y
conda activate ulip_rag
```

### 3. 安裝 PyTorch 及相關套件

安裝 PyTorch 2.1.0 with CUDA 12.1 支援：

```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 4. 安裝 FAISS-GPU (用於 RAG 檢索)

```bash
pip install faiss-gpu==1.7.2
```

### 5. 安裝 PointNet2 相關套件

```bash
cd Pointnet2_PyTorch/pointnet2_ops_lib
pip install -e .
cd ../..
```

安裝其他 3D 點雲處理套件：

```bash
pip install open3d==0.16.0
pip install git+https://github.com/unlimblue/KNN_CUDA.git
```

### 6. 安裝 Vision-Language 模型套件

```bash
pip install open-clip-torch==2.24.0
pip install transformers==4.35.2
pip install timm==0.4.12
pip install sentence-transformers==2.2.2
pip install tokenizers==0.15.2
pip install safetensors==0.7.0
```

### 7. 安裝 NLP 套件

```bash
pip install spacy==3.8.11
python -m spacy download en_core_web_sm
pip install nltk==3.9.2
pip install sentencepiece==0.2.1
```

### 8. 安裝科學計算套件

```bash
pip install numpy==1.26.4
pip install scipy==1.13.1
pip install pandas==2.3.3
pip install scikit-learn==1.6.1
pip install matplotlib==3.9.4
```

### 9. 安裝資料處理套件

```bash
pip install pillow==11.3.0
pip install h5py==3.6.0
pip install lmdb==1.3.0
pip install pyquaternion==0.9.9
```

### 10. 安裝實驗追蹤與工具

```bash
pip install wandb==0.13.3
pip install huggingface-hub==0.16.4
pip install tqdm==4.67.1
pip install easydict==1.9
pip install ConfigArgParse==1.7.1
```

### 11. 安裝視覺化套件

```bash
pip install plotly==6.5.2
pip install dash==3.4.0
```

### 12. 安裝其他依賴

```bash
pip install pydantic==2.12.5
pip install wikipedia==1.4.0
pip install beautifulsoup4==4.14.3
pip install ipython==8.18.1
```

## 快速安裝（使用 requirements.txt）

您也可以使用提供的 `requirements.txt` 文件一次安裝所有依賴：

```bash
conda create -n ulip_rag python=3.9.25 -y
conda activate ulip_rag
pip install -r requirements.txt

# 特別安裝 PyTorch (需要指定 CUDA 版本)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# 安裝 PointNet2 ops (本地編譯)
cd Pointnet2_PyTorch/pointnet2_ops_lib
pip install -e .
cd ../..

# 下載 spaCy 語言模型
python -m spacy download en_core_web_sm
```

## 資料目錄設置

### 符號連結設置

為節省空間，數據目錄使用符號連結指向較大的儲存裝置：

```bash
# 當前配置
/home/kyzen/cheng/ulip_rag/data -> /mnt/data1/cheng/ULIP/data/data
```

### 預訓練模型

下載並放置預訓練模型到 `data/initialize_models/`：

- **point_bert_pretrained.pt** (527MB) - PointBERT 預訓練權重
- **slip_base_100ep.pt** (2.0GB) - SLIP 視覺-語言預訓練模型

## 驗證安裝

驗證關鍵套件是否正確安裝：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"
python -c "import open_clip; print('OpenCLIP: OK')"
python -c "import pointnet2_ops; print('PointNet2 Ops: OK')"
```

## 已安裝套件清單

### 核心深度學習框架
- PyTorch: 2.1.0+cu121
- TorchVision: 0.16.0+cu121
- Triton: 2.1.0

### 3D 點雲處理
- pointnet2_ops: 3.0.0
- KNN_CUDA: 0.2
- open3d: 0.16.0
- pyquaternion: 0.9.9

### RAG & 檢索
- faiss-gpu: 1.7.2
- sentence-transformers: 2.2.2

### Vision-Language 模型
- open-clip-torch: 2.24.0
- transformers: 4.35.2
- timm: 0.4.12

### NLP 相關
- spacy: 3.8.11
- en_core_web_sm: 3.8.0
- nltk: 3.9.2
- sentencepiece: 0.2.1

### 科學計算
- numpy: 1.26.4
- scipy: 1.13.1
- pandas: 2.3.3
- scikit-learn: 1.6.1

### 實驗追蹤
- wandb: 0.13.3
- huggingface-hub: 0.16.4

## 常見問題

### Q: CUDA 版本不匹配
A: 確保您的 NVIDIA 驅動支援 CUDA 12.1。可以使用 `nvidia-smi` 檢查。

### Q: PointNet2 編譯失敗
A: 確保已安裝 `ninja` 和 CUDA toolkit：
```bash
pip install ninja
```

### Q: FAISS 安裝失敗
A: 如果 GPU 版本安裝失敗，可以暫時使用 CPU 版本：
```bash
pip install faiss-cpu==1.7.2
```

### Q: 磁碟空間不足
A: 考慮將大型模型文件和資料集移到 `/mnt/data1` 等較大的儲存裝置，並使用符號連結。

## 參考資訊

- **項目路徑**: `/home/kyzen/cheng/ulip_rag`
- **資料路徑**: `/mnt/data1/cheng/ULIP/data/data` (透過符號連結)
- **Conda 環境路徑**: `/home/kyzen/miniconda3/envs/ulip_rag`

---

**最後更新**: 2026-02-02
