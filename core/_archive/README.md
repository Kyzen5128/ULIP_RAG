# core/_archive/ — 已封存的歷史實驗檔

> 建立:2026-07-10(第二階段整理)。這裡的檔案 **無人 import**(rg 全 repo 實測),
> 從原位置(core/models/、core/data/)移入,保留作研究演進的歷史證據。**勿引用、勿修改。**

| 檔案 | 原位置 | 現役對應 |
|---|---|---|
| `ULIP_models_rag.py` | core/models/ | 舊版 RAG 模型(ULIP_RAG_ADAPTER、ULIP_Loss_RAG_Decoupled 實驗);現役在 `core/models/ULIP_models.py:473-883` |
| `rag_adapter.py` | core/models/ | 獨立版 adapter 實驗;無現役對應 |
| `rag_generator.py` | core/models/ | 獨立版 generator 實驗;無現役對應 |
| `rag_enhancer.py` | core/models/ | 現役 RAGEnhancer(含 key_padding_mask 修復)在 `ULIP_models.py:517-547` |
| `retriever.py` | core/models/ | 現役 RAGRetriever 在 `ULIP_models.py:473-515` |
| `dataset_3d_rag.py` | core/data/ | 現役 rag_collate_fn 在 `core/data/dataset_3d.py:856` |

**注意**:`core/check_environment.py:119` 與 `core/test_rag_functions.py:27` 仍寫著
`from models.rag_enhancer import ...` —— 那兩支本身就是 DEPRECATED(4090 遺留,檔頭有註記),
移檔後它們會 ImportError,屬預期行為,不修。
