# -*- coding: utf-8 -*-
"""
RAG 知識庫建構（逐句 caption 搜 Wikipedia 版）

流程：
 1. 讀取 ShapeNet captions（只保留椅子 synset 03001627）
 2. 每句 caption 向 Wikipedia 搜索，抓摘要 (page.summary)
    - 具備 JSON 快取，避免重複呼叫 API
 3. 產生少量合成描述 
 4. 清洗 → 語意去重 (FAISS) → 屬性抽取
 5. 儲存向量 .npy、純文本 .json、結構化 .jsonl
"""

import json, re, time, random
from pathlib import Path
from typing import List, Dict, Set, Optional

import numpy as np
import wikipedia
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ----------------------------
# 資料結構
# ----------------------------
class CorpusEntry:
    def __init__(
        self,
        text: str,
        category: str,
        source: str,
        confidence: float,
        attributes: Optional[Dict[str, List[str]]] = None,
    ):
        self.text       = text
        self.category   = category
        self.source     = source
        self.confidence = confidence
        self.attributes = attributes or {}

    def to_dict(self):
        return {
            "text":       self.text,
            "category":   self.category,
            "source":     self.source,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }


# ----------------------------
# Builder
# ----------------------------
class OpenKnowledgeBaseBuilder:
    def __init__(self,
                 shapenet_path: str = "data/shapenet",
                 output_dir: str    = "data/rag_corpus"):

        self.shapenet_path = Path(shapenet_path)
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        wikipedia.set_lang("en")                          # Wiki 取英文
        print(" 載入嵌入模型 all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # --------- 屬性辭典 ---------
        self.attr_vocab: Dict[str, Set[str]] = {
            "colors": {
                "red","blue","green","yellow","orange","purple","pink","black",
                "white","gray","grey","brown","beige","cream","navy","maroon",
                "gold","silver","bronze","copper"
            },
            "materials": {
                "wood","wooden","oak","pine","mahogany","metal","steel","iron",
                "aluminum","plastic","acrylic","glass","leather","fabric",
                "cotton","velvet","marble","stone","upholstered"
            },
            "styles": {
                "modern","contemporary","vintage","antique","traditional",
                "minimalist","industrial","scandinavian","mid-century",
                "rustic","art deco","bohemian"
            },
            "shapes": {
                "round","square","rectangular","oval","circular","linear",
                "curved","straight","angular","geometric","organic"
            },
        }

    # ------------------------
    # 主流程
    # ------------------------
    def build(self):
        print("\n=== 開始建構知識庫 ===")

        # 1) ShapeNet captions
        print("\n[1/6] 讀取 ShapeNet caption …")
        shapenet_entries = self._collect_shapenet()
        print(f"  ↳ {len(shapenet_entries)} 句")

        captions = [e.text for e in shapenet_entries]

        # 2) Wikipedia by caption
        print("\n[2/6] Caption → Wikipedia 擷取 …")
        wiki_entries = self._collect_wikipedia_by_caption(captions)
        print(f"  ↳ {len(wiki_entries)} 條摘要")

        # 3) Synthetic
        print("\n[3/6] 產生合成描述 …")
        synth_entries = self._generate_synthetic()
        print(f"  ↳ {len(synth_entries)} 條")

        all_entries = shapenet_entries + wiki_entries + synth_entries

        # 4) 清洗 + 去重
        print("\n[4/6] 清洗與語意去重 …")
        clean_entries = self._clean_and_dedup(all_entries)
        print(f"  ↳ 剩下 {len(clean_entries)} 條")

        # 5) 屬性抽取
        print("\n[5/6] 抽取屬性 …")
        self._extract_attributes(clean_entries)
        print("  ↳ 完成")

        # 6) 儲存
        print("\n[6/6] 儲存檔案 …")
        self._save(clean_entries)
        print("\n✅ 知識庫已完成\n")

    # ------------------------
    # 1) 讀取 ShapeNet 椅子 caption
    # ------------------------
    def _collect_shapenet(self) -> List[CorpusEntry]:
        file = self.shapenet_path / "vocab" / "augmented_shapenet.json"
        if not file.exists():
            raise FileNotFoundError(file)

        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        CHAIR_SYNSET = "03001627"               # 椅子 synset
        entries = []
        for item in data["captions"]:
            if item["category"] != CHAIR_SYNSET:
                continue
            text = " ".join(item["caption"])
            entries.append(CorpusEntry(text, "chair", "shapenet", 0.9))
        return entries

    # ------------------------
    # 2) 每句 caption 查 Wikipedia
    # ------------------------
    def _collect_wikipedia_by_caption(self,
                                      captions: List[str],
                                      max_hits: int = 1) -> List[CorpusEntry]:

        cache_path = self.output_dir / "wiki_cache.json"
        cache: Dict[str, str] = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())

        uniq_caps = list(dict.fromkeys(
            re.sub(r"\s+", " ", c.strip()) for c in captions if c.strip()
        ))
        print(f"   將查詢 {len(uniq_caps)} 句 caption")

        entries: list[CorpusEntry] = []
        for idx, query in enumerate(tqdm(uniq_caps, desc="WikiCapt")):

            if query in cache:                     # 快取命中
                summary = cache[query]
                if summary:
                    entries.append(CorpusEntry(summary, "chair",
                                               "wiki_caption", 0.7))
                continue

            try:
                titles = wikipedia.search(query, results=max_hits)
            except Exception:
                cache[query] = ""
                continue

            grabbed = False
            for title in titles:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = page.summary.strip()
                except (wikipedia.exceptions.PageError,
                        wikipedia.exceptions.DisambiguationError):
                    continue

                if 15 < len(summary.split()) < 150:
                    entries.append(CorpusEntry(summary, "chair",
                                               "wiki_caption", 0.7))
                    cache[query] = summary
                    grabbed = True
                    break

            if not grabbed:
                cache[query] = ""                 # 標示失敗

            if (idx + 1) % 200 == 0:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False))
                time.sleep(1.0)                   # 降低頻率

        cache_path.write_text(json.dumps(cache, ensure_ascii=False))
        return entries

    # ------------------------
    # 3) Synthetic
    # ------------------------
    def _generate_synthetic(self) -> List[CorpusEntry]:
        templates = [
            "A {style} chair with a {shape} silhouette, crafted from {material}.",
            "This {color} chair adopts a {style} vibe and uses premium {material}.",
            "Featuring clean lines, the chair comes in {color} and {material}.",
            "An elegant chair marrying {material} and {material2} for a unique presence."
        ]
        entries = []
        for _ in range(100):                      # 產生 100 條
            mat1, mat2 = random.sample(list(self.attr_vocab["materials"]), 2)
            tpl  = random.choice(templates)
            sent = tpl.format(
                style     = random.choice(list(self.attr_vocab["styles"])),
                shape     = random.choice(list(self.attr_vocab["shapes"])),
                material  = mat1,
                material2 = mat2,
                color     = random.choice(list(self.attr_vocab["colors"]))
            )
            entries.append(CorpusEntry(sent, "chair", "synthetic", 0.6))
        return entries

    # ------------------------
    # 4) 清洗 + 去重（FAISS）
    # ------------------------
    def _clean_and_dedup(self,
                         entries: List[CorpusEntry],
                         thresh: float = 0.90) -> List[CorpusEntry]:

        tmp: Dict[str, CorpusEntry] = {}
        for e in entries:
            t = re.sub(r'\s+', ' ', e.text).strip()
            if 8 < len(t.split()) < 200:
                k = t.lower()
                if k not in tmp or e.confidence > tmp[k].confidence:
                    e.text = t
                    tmp[k] = e
        pool = list(tmp.values())
        print(f"  字面去重後 {len(pool)} 條")

        if len(pool) < 2:
            return pool

        texts = [e.text for e in pool]
        embs  = self.embedder.encode(texts, batch_size=128,
                                     show_progress_bar=True).astype("float32")
        faiss.normalize_L2(embs)

        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        k = 5
        D, I = index.search(embs, k)

        drop: set[int] = set()
        for i, (d_row, idx_row) in enumerate(zip(D, I)):
            if i in drop: continue
            for d, j in zip(d_row[1:], idx_row[1:]):
                if d > thresh:
                    if pool[i].confidence >= pool[j].confidence:
                        drop.add(j)
                    else:
                        drop.add(i)
                        break
        return [e for i, e in enumerate(pool) if i not in drop]

    # ------------------------
    # 5) 屬性抽取
    # ------------------------
    def _extract_attributes(self, entries: List[CorpusEntry]):
        for e in tqdm(entries, desc="Attr"):
            low = f" {e.text.lower()} "
            for typ, vocab in self.attr_vocab.items():
                found = {
                    w for w in vocab
                    if re.search(rf'\b{re.escape(w)}s?\b', low)
                }
                if found:
                    e.attributes[typ] = sorted(found)

    # ------------------------
    # 6) 儲存
    # ------------------------
    def _save(self, entries: List[CorpusEntry]):
        texts = [e.text for e in entries]
        vecs  = self.embedder.encode(texts, batch_size=128,
                                     show_progress_bar=True).astype("float32")
        assert vecs.shape[0] == len(texts)

        vec_path  = self.output_dir / "kb_vectors.npy"
        text_path = self.output_dir / "kb_texts.json"
        jsonl_path= self.output_dir / "kb_structured.jsonl"

        np.save(vec_path, vecs)
        text_path.write_text(json.dumps(texts, ensure_ascii=False))
        with jsonl_path.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")

        print(f"\n 已儲存：向量 {vecs.shape}  →  {vec_path}")
        print(f"          純文本           →  {text_path}")
        print(f"          結構化 JSONL    →  {jsonl_path}")


# ----------------------------
# 執行
# ----------------------------
if __name__ == "__main__":
    builder = OpenKnowledgeBaseBuilder(
        shapenet_path="data/shapenet",
        output_dir="data/rag_corpus"
    )
    builder.build()
