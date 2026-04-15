"""
開放式 RAG 語料庫建構系統
從多個開放數據源收集和清洗家具/物品的多維度描述
"""

import json
import os
import requests
import time
import re
from typing import List, Dict, Set
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import spacy
import wikipedia
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class CorpusEntry:
    """語料庫條目結構"""
    text: str
    category: str
    attributes: Dict[str, List[str]]  # {"color": ["red", "brown"], "style": ["modern"], ...}
    source: str
    confidence: float

class HFEmbeddingModel:
    """Hugging Face 句子嵌入模型封裝"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    def encode(self, texts: List[str], batch_size=32) -> np.ndarray:
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                model_output = self.model(**encoded_input)
                
                token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                attention_mask = encoded_input.attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

class OpenCorpusBuilder:
    """開放式語料庫建構器"""
    
    def __init__(self, output_dir="/home/kyzen/ULIP_RAG/ulip_rag/rag_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用 Hugging Face Transformers 句子嵌入模型
        self.semantic_model = HFEmbeddingModel('sentence-transformers/all-MiniLM-L6-v2')
        
        # 載入 spaCy 模型（用於 NER 和語法分析）
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy en_core_web_sm not found. Some features disabled.")
            self.nlp = None
        
        # 預定義的屬性詞彙表
        self.attribute_vocab = self._load_attribute_vocabulary()
        
        # 目標類別（可從 ShapeNet/ModelNet 類別擴展）
        self.target_categories = [
            "chair", "table", "sofa", "bed", "cabinet", "shelf", "lamp", 
            "desk", "stool", "bench", "dresser", "wardrobe", "bookshelf",
            "coffee table", "dining table", "armchair", "nightstand",
            # 新增
            "recliner", "ottoman", "loveseat", "futon", "couch",
            "sideboard", "credenza", "console table", "bar stool",
            "office chair", "rocking chair", "accent chair"
        ]
    
    def _load_attribute_vocabulary(self) -> Dict[str, Set[str]]:
        """載入屬性詞彙表"""
        return {
            # 顏色 (Colors)
            "colors": {
                "red",          # 紅色
                "blue",         # 藍色
                "green",        # 綠色
                "yellow",       # 黃色
                "orange",       # 橘色
                "purple",       # 紫色
                "pink",         # 粉紅
                "black",        # 黑色
                "white",        # 白色
                "gray",         # 灰色
                "grey",         # 灰色
                "brown",        # 棕色
                "beige",        # 米色
                "cream",        # 奶油色
                "navy",         # 海軍藍
                "maroon",       # 栗色
                "gold",         # 金色
                "silver",       # 銀色
                "bronze",       # 青銅色
                "copper",       # 銅色
                "walnut-toned", # 胡桃木色
                "cherry",       # 櫻桃色
                "ebony",        # 烏木色
                "natural",      # 原木色
                "light",        # 淺色
                "dark"          # 深色
            },
            # 材質 (Materials)
            "materials": {
                "wood",         # 木頭
                "wooden",       # 木製
                "oak",          # 橡木
                "pine",         # 松木
                "mahogany",     # 桃花心木
                "teak",         # 柚木
                "walnut",       # 胡桃木
                "metal",        # 金屬
                "steel",        # 鋼
                "iron",         # 鐵
                "aluminum",     # 鋁
                "brass",        # 黃銅
                "copper",       # 銅
                "plastic",      # 塑膠
                "acrylic",      # 亞克力
                "glass",        # 玻璃
                "leather",      # 皮革
                "fabric",       # 布料
                "cotton",       # 棉
                "velvet",       # 天鵝絨
                "silk",         # 絲綢
                "linen",        # 亞麻
                "marble",       # 大理石
                "stone",        # 石材
                "ceramic",      # 陶瓷
                "bamboo",       # 竹子
                "rattan",       # 藤
                "wicker",       # 柳條編織
                "upholstered",  # 軟包/沙發布
                "plywood",      # 夾板
                "MDF"           # 密集板
            },
            "styles": {
                # === Modern / Contemporary ===
                "modern",                   # 現代
                "contemporary",             # 當代
                "minimalist",               # 極簡
                "scandinavian",             # 北歐
                "industrial",               # 工業風
                "mid_century_modern",       # 中世紀現代
                "modern_luxury",            # 現代輕奢

                # === Natural / Lifestyle ===
                "japanese",                 # 日式
                "zen",                      # 禪風
                "rustic",                   # 鄉村
                "farmhouse",                # 農舍風
                "coastal",                  # 海岸風
                "bohemian",                 # 波希米亞

                # === Classic / Historical ===
                "traditional",              # 傳統
                "vintage",                  # 復古（近代）
                "antique",                  # 古董
                "victorian",                # 維多利亞
                "baroque",                  # 巴洛克
                "rococo",                   # 洛可可
                "art_deco",                 # 裝飾藝術
                "colonial",                 # 殖民風
                "french_country",           # 法式鄉村

                # === Design-driven / Mixed ===
                "eclectic",                 # 折衷主義
                "urban",                    # 都市風
                "loft",                     # 工業風
                "futuristic",               # 未來風

                # === Regional / Market-driven ===
                "korean",                   # 韓式
                "italian_modern",           # 義式現代
                "american_modern",          # 美式現代

                # === Premium / Statement ===
                "luxury"                    # 奢華
            },
            # 功能 (Functions)
            "functions": {
                "storage",       # 儲物
                "seating",       # 座位
                "sleeping",      # 睡眠
                "dining",        # 用餐
                "working",       # 工作
                "reading",       # 閱讀
                "display",       # 展示
                "lighting",      # 照明
                "decorative",    # 裝飾
                "entertainment", # 娛樂
                "organization",  # 整理
                "relaxation",    # 放鬆
                "gathering",     # 聚會
                "conversation"   # 交談
            },
            # 形狀 (Shapes)
            "shapes": {
                "round",        # 圓形
                "square",       # 方形
                "rectangular",  # 長方形
                "oval",         # 橢圓形
                "circular",     # 圓形
                "linear",       # 線性
                "curved",       # 曲線
                "straight",     # 直線
                "angular",      # 有角度的
                "geometric",    # 幾何形
                "organic",      # 有機形態
                "L-shaped",     # L型
                "U-shaped",     # U型
                "modular",      # 模組化
                "sectional"     # 組合式
            },
            # 尺寸 (Sizes)
            "sizes": {
                "small",        # 小
                "medium",       # 中
                "large",        # 大
                "compact",      # 緊湊
                "oversized",    # 超大
                "miniature",    # 迷你
                "full-size",    # 全尺寸
                "queen",        # 雙人（床墊）
                "king",         # 加大雙人
                "twin",         # 單人
                "single",       # 單人
                "double",       # 雙人
                "spacious",     # 寬敞
                "narrow",       # 窄
                "wide",         # 寬
                "tall",         # 高
                "low"           # 低
            },
            # 容納人數 (Capacities)
            "capacities": {
                "single-person",  # 單人
                "two-person",     # 雙人
                "three-person",   # 三人
                "four-person",    # 四人
                "six-person",     # 六人
                "eight-person",   # 八人
                "family-sized",   # 家庭尺寸
                "couples",        # 情侶
                "individual",     # 個人
                "group",          # 群體
                "multi-person"    # 多人
            },
            # 擺放位置 (Placements)
            "placements": {
                "corner",                    # 角落
                "center",                    # 中央
                "wall-mounted",              # 壁掛
                "freestanding",              # 獨立式
                "against the wall",          # 靠牆
                "by the window",             # 窗邊
                "near the entrance",         # 入口附近
                "beside the bed",            # 床邊
                "under the window",          # 窗下
                "in the center of the room", # 房間中央
                "next to the sofa",          # 沙發旁
                "facing the TV"              # 對著電視
            },
            # 房間 (Rooms)
            "rooms": {
                "living room",     # 客廳
                "dining room",     # 餐廳
                "bedroom",         # 臥室
                "office",          # 辦公室
                "study",           # 書房
                "kitchen",         # 廚房
                "bathroom",        # 浴室
                "hallway",         # 走廊
                "balcony",         # 陽台
                "patio",           # 露台
                "guest room",      # 客房
                "children's room", # 兒童房
                "master bedroom",  # 主臥
                "home office"      # 居家辦公室
            }
        }
    
    def build_corpus(self):
        """建構完整的開放式語料庫"""
        print(" 開始建構開放式 RAG 語料庫...")
        
        all_entries = []
        
        # 1. 從 Wikipedia 收集描述
        print("\n 從 Wikipedia 收集家具描述...")
        wiki_entries = self.collect_from_wikipedia()
        all_entries.extend(wiki_entries)
        print(f" Wikipedia: 收集到 {len(wiki_entries)} 條記錄")
        
        # 2. 從 ProductHunt/家具網站 API 收集（此處可補充）
        # ...
        
        # 3. 合成描述生成
        print("\n 生成合成描述...")
        synthetic_entries = self.generate_synthetic_descriptions()
        all_entries.extend(synthetic_entries)
        print(f" Synthetic: 生成 {len(synthetic_entries)} 條記錄")
        
        # 4. 數據清洗和去重
        print("\n 數據清洗和去重...")
        cleaned_entries = self.clean_and_deduplicate(all_entries)
        print(f"✓ 清洗後: {len(cleaned_entries)} 條記錄")
        
        # 5. 屬性提取和標註
        print("\n 提取和標註屬性...")
        annotated_entries = self.extract_attributes(cleaned_entries)
        print(f"✓ 標註完成: {len(annotated_entries)} 條記錄")
        
        # 6. 品質評估和過濾
        print("\n 品質評估和過濾...")
        high_quality_entries = self.quality_filter(annotated_entries)
        print(f"✓ 高品質語料: {len(high_quality_entries)} 條記錄")
        
        # 7. 保存語料庫
        print("\n 保存語料庫...")
        self.save_corpus(high_quality_entries)
        
        print(f"\n 語料庫建構完成！最終包含 {len(high_quality_entries)} 條記錄")
    
    def collect_from_wikipedia(self) -> List[CorpusEntry]:
        """從 Wikipedia 收集家具描述"""
        entries = []
        
        for category in tqdm(self.target_categories, desc="Wikipedia extraction"):
            try:
                # 原本
                # search_results = wikipedia.search(f"{category} furniture", results=5)
                # 改成
                search_results = wikipedia.search(f"{category} furniture", results=10)  # 增加到 10

                
                for page_title in search_results:
                    try:
                        page = wikipedia.page(page_title, auto_suggest=False)
                        content = page.content
                        
                        paragraphs = self._extract_relevant_paragraphs(content, category)
                        
                        for paragraph in paragraphs:
                            if len(paragraph.split()) >= 10:
                                entries.append(CorpusEntry(
                                    text=paragraph,
                                    category=category,
                                    attributes={},
                                    source=f"wikipedia:{page_title}",
                                    confidence=0.8
                                ))
                        
                        time.sleep(0.1)
                        
                    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                        continue
                        
            except Exception as e:
                print(f"Error processing {category}: {e}")
                continue
        
        return entries
    
    def _extract_relevant_paragraphs(self, content: str, category: str) -> List[str]:
        paragraphs = content.split('\n\n')
        relevant = []
        keywords = [category, "design", "material", "style", "construction", "use"]
        
        for para in paragraphs:
            if len(para.split()) < 5:
                continue
                
            para_lower = para.lower()
            if any(keyword in para_lower for keyword in keywords):
                cleaned = self._clean_paragraph(para)
                if cleaned:
                    relevant.append(cleaned)
        
        # 原本只取前 3 段
        #return relevant[:3]
        # 改成取前 5 段
        return relevant[:5]
    
    def _clean_paragraph(self, paragraph: str) -> str:
        cleaned = re.sub(r'\[\d+\]', '', paragraph)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned.split()) < 10 or len(cleaned.split()) > 200:
            return ""
        return cleaned
    
    def generate_synthetic_descriptions(self) -> List[CorpusEntry]:
        entries = []
        templates = [
            # 基本描述（材質+顏色+風格）
            "This {category} features a {style} design with {material} construction and {color} finish.",
            "A {style} {category} made from {material}, perfect for {function} in {room} spaces.",
            "Contemporary {category} with {shape} form, crafted in {material} with {color} accents.",
            
            # 場景+擺放位置
            "This {size} {category} is ideal for placement {placement}, adding {style} charm to your {room}.",
            "A {color} {material} {category} designed to be placed {placement} in your {room}.",
            "Perfect for {room}, this {category} fits beautifully {placement}.",
            
            # 容納人數
            "A {capacity} {category} with {style} design, perfect for {function} and {function2}.",
            "This {size} {category} comfortably accommodates {capacity} use, made from premium {material}.",
            "Designed for {capacity} comfort, this {color} {category} features {material} construction.",
            
            # 綜合描述（多維度）
            "This {style} {category} in {color} {material} is sized for {capacity} use, ideal {placement} in your {room}.",
            "A {size} {shape} {category} with {color} finish, crafted from {material} for {function} in the {room}.",
            "Elegant {capacity} {category} featuring {material} frame with {color} upholstery, perfect {placement}.",
            
            # 功能性描述
            "This {category} offers excellent {function} capabilities with its {shape} {material} design.",
            "A {style} {category} optimized for {function}, featuring {size} dimensions and {color} tones.",
            "Multi-functional {category} suitable for {function} and {function2}, available in {color} {material}.",
            
            # 空間描述
            "Transform your {room} with this {style} {category}, designed for placement {placement}.",
            "This {size} {category} maximizes space efficiency when placed {placement} in your {room}.",
            "Ideal for {room} corners or {placement}, this {color} {category} adds {style} elegance."
        ]
        
        import random
        for category in self.target_categories:
            for template in templates:
                for _ in range(20):
                    description = self._fill_template(template, category)
                    if description:
                        entries.append(CorpusEntry(
                            text=description,
                            category=category,
                            attributes={},
                            source="synthetic",
                            confidence=0.6
                        ))
        return entries
    
    def _fill_template(self, template: str, category: str) -> str:
        try:
            import random
            
            attrs = {
                "category": category,
                "style": random.choice(list(self.attribute_vocab["styles"])),
                "material": random.choice(list(self.attribute_vocab["materials"])),
                "material2": random.choice(list(self.attribute_vocab["materials"])),
                "color": random.choice(list(self.attribute_vocab["colors"])),
                "shape": random.choice(list(self.attribute_vocab["shapes"])),
                "function": random.choice(list(self.attribute_vocab["functions"])),
                "function2": random.choice(list(self.attribute_vocab["functions"])),
                "size": random.choice(list(self.attribute_vocab["sizes"])),
                "capacity": random.choice(list(self.attribute_vocab["capacities"])),
                "placement": random.choice(list(self.attribute_vocab["placements"])),
                "room": random.choice(list(self.attribute_vocab["rooms"]))
            }
            
            used_keys = re.findall(r'\{(\w+)\}', template)
            filtered_attrs = {k: v for k, v in attrs.items() if k in used_keys}
            return template.format(**filtered_attrs)
        except KeyError:
            return ""
    
    def clean_and_deduplicate(self, entries: List[CorpusEntry]) -> List[CorpusEntry]:
        cleaned = []
        for entry in entries:
            text = entry.text.strip()
            text = re.sub(r'\s+', ' ', text)
            
            if (len(text.split()) >= 8 and
                len(text) <= 500 and
                not text.lower().startswith(('http', 'www')) and
                any(cat in text.lower() for cat in self.target_categories)):
                
                entry.text = text
                cleaned.append(entry)
        
        if len(cleaned) > 1000:
            cleaned = self._semantic_deduplication(cleaned)
        
        return cleaned
    
    def _semantic_deduplication(self, entries: List[CorpusEntry], threshold=0.85) -> List[CorpusEntry]:
        if not entries:
            return entries
        
        texts = [entry.text for entry in entries]
        embeddings = self.semantic_model.encode(texts)
        
        similarity_matrix = cosine_similarity(embeddings)
        
        to_remove = set()
        for i in range(len(entries)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(entries)):
                if j in to_remove:
                    continue
                if similarity_matrix[i][j] > threshold:
                    if entries[i].confidence >= entries[j].confidence:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        
        return [entry for i, entry in enumerate(entries) if i not in to_remove]
    
    def extract_attributes(self, entries: List[CorpusEntry]) -> List[CorpusEntry]:
        for entry in tqdm(entries, desc="Extracting attributes"):
            for attr_type, vocab in self.attribute_vocab.items():
                found_attrs = []
                text_lower = entry.text.lower()
                
                for attr_value in vocab:
                    if attr_value in text_lower:
                        found_attrs.append(attr_value)
                
                if found_attrs:
                    entry.attributes[attr_type] = found_attrs
        
        return entries
    
    def quality_filter(self, entries: List[CorpusEntry], min_confidence=0.5) -> List[CorpusEntry]:
        high_quality = []
        for entry in entries:
            quality_score = self._calculate_quality_score(entry)
            if quality_score >= min_confidence:
                entry.confidence = quality_score
                high_quality.append(entry)
        return high_quality
    
    def _calculate_quality_score(self, entry: CorpusEntry) -> float:
        score = entry.confidence
        
        if entry.attributes:
            attr_count = sum(len(attrs) for attrs in entry.attributes.values())
            score += min(attr_count * 0.05, 0.2)
        
        word_count = len(entry.text.split())
        if 15 <= word_count <= 50:
            score += 0.1
        elif word_count < 8:
            score -= 0.2
        
        if entry.source == "synthetic":
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def save_corpus(self, entries: List[CorpusEntry]):
        jsonl_path = self.output_dir / "rag_corpus.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                json_record = {
                    "text": entry.text,
                    "obj_id": entry.category,
                    "category": entry.category,
                    "attributes": entry.attributes,
                    "source": entry.source,
                    "confidence": entry.confidence
                }
                f.write(json.dumps(json_record, ensure_ascii=False) + '\n')
        
        json_path = self.output_dir / "rag_corpus_structured.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            corpus_data = {
                "metadata": {
                    "total_entries": len(entries),
                    "categories": list(set(entry.category for entry in entries)),
                    "sources": list(set(entry.source for entry in entries)),
                    "average_confidence": sum(entry.confidence for entry in entries) / len(entries)
                },
                "entries": [
                    {
                        "text": entry.text,
                        "category": entry.category,
                        "attributes": entry.attributes,
                        "source": entry.source,
                        "confidence": entry.confidence
                    }
                    for entry in entries
                ]
            }
            json.dump(corpus_data, f, indent=2, ensure_ascii=False)
        
        self._save_statistics_report(entries)
        
        print(f"✅ 語料庫已保存到: {self.output_dir}")
        print(f"  - JSONL 格式: {jsonl_path}")
        print(f"  - JSON 格式: {json_path}")
        print(f"  - 統計報告: {self.output_dir / 'corpus_statistics.txt'}")
    
    def _save_statistics_report(self, entries: List[CorpusEntry]):
        report_path = self.output_dir / "corpus_statistics.txt"
        
        total_entries = len(entries)
        categories = {}
        sources = {}
        avg_confidence = sum(entry.confidence for entry in entries) / total_entries
        
        for entry in entries:
            categories[entry.category] = categories.get(entry.category, 0) + 1
            sources[entry.source] = sources.get(entry.source, 0) + 1
        
        attr_stats = {}
        for entry in entries:
            for attr_type, attrs in entry.attributes.items():
                if attr_type not in attr_stats:
                    attr_stats[attr_type] = {}
                for attr in attrs:
                    attr_stats[attr_type][attr] = attr_stats[attr_type].get(attr, 0) + 1
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== RAG 語料庫統計報告 ===\n\n")
            f.write(f"總條目數: {total_entries}\n")
            f.write(f"平均信心度: {avg_confidence:.3f}\n\n")
            
            f.write("=== 類別分布 ===\n")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{cat}: {count} ({count/total_entries*100:.1f}%)\n")
            
            f.write("\n=== 來源分布 ===\n")
            for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{src}: {count} ({count/total_entries*100:.1f}%)\n")
            
            f.write("\n=== 屬性統計 ===\n")
            for attr_type, attr_counts in attr_stats.items():
                f.write(f"\n{attr_type.upper()}:\n")
                top_attrs = sorted(attr_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                for attr, count in top_attrs:
                    f.write(f"  {attr}: {count}\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="建構開放式 RAG 語料庫")
    parser.add_argument("--output_dir", default="/home/kyzen/ULIP_RAG/ulip_rag/rag_corpus", 
                       help="輸出目錄")
    parser.add_argument("--categories", nargs='+', 
                       help="目標類別列表（可選）")
    
    args = parser.parse_args()
    
    builder = OpenCorpusBuilder(output_dir=args.output_dir)
    
    if args.categories:
        builder.target_categories = args.categories
    
    builder.build_corpus()

if __name__ == "__main__":
    main()
