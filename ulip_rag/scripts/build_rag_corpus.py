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
    
    def __init__(self, output_dir="/home/kyzen/cheng/ulip_rag/rag_corpus"):
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
            "coffee table", "dining table", "armchair", "nightstand"
        ]
    
    def _load_attribute_vocabulary(self) -> Dict[str, Set[str]]:
        """載入屬性詞彙表"""
        return {
            "colors": {
                "red", "blue", "green", "yellow", "orange", "purple", "pink",
                "black", "white", "gray", "grey", "brown", "beige", "cream",
                "navy", "maroon", "gold", "silver", "bronze", "copper"
            },
            "materials": {
                "wood", "wooden", "oak", "pine", "mahogany", "teak", "walnut",
                "metal", "steel", "iron", "aluminum", "brass", "copper",
                "plastic", "acrylic", "glass", "leather", "fabric", "cotton",
                "velvet", "silk", "linen", "marble", "stone", "ceramic",
                "bamboo", "rattan", "wicker", "upholstered"
            },
            "styles": {
                "modern", "contemporary", "vintage", "antique", "traditional",
                "minimalist", "industrial", "scandinavian", "mid-century",
                "rustic", "shabby chic", "art deco", "victorian", "baroque",
                "colonial", "country", "urban", "zen", "bohemian", "eclectic"
            },
            "functions": {
                "storage", "seating", "sleeping", "dining", "working", "reading",
                "display", "lighting", "decorative", "entertainment", "organization"
            },
            "shapes": {
                "round", "square", "rectangular", "oval", "circular", "linear",
                "curved", "straight", "angular", "geometric", "organic"
            },
            "sizes": {
                "small", "medium", "large", "compact", "oversized", "miniature",
                "full-size", "queen", "king", "twin", "single", "double"
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
                search_results = wikipedia.search(f"{category} furniture", results=5)
                
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
        
        return relevant[:3]
    
    def _clean_paragraph(self, paragraph: str) -> str:
        cleaned = re.sub(r'\[\d+\]', '', paragraph)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned.split()) < 10 or len(cleaned.split()) > 200:
            return ""
        return cleaned
    
    def generate_synthetic_descriptions(self) -> List[CorpusEntry]:
        entries = []
        templates = [
            "This {category} features a {style} design with {material} construction and {color} finish.",
            "A {style} {category} made from {material}, perfect for {function} in {setting} spaces.",
            "Contemporary {category} with {shape} form, crafted in {material} with {color} accents.",
            "Classic {category} design featuring {material} frame and {color} upholstery, ideal for {setting}.",
            "Modern {category} combining {material} and {material2} for a {style} aesthetic."
        ]
        
        settings = ["living room", "dining room", "bedroom", "office", "study"]
        
        import random
        for category in self.target_categories:
            for template in templates:
                for _ in range(20):
                    description = self._fill_template(template, category, settings)
                    if description:
                        entries.append(CorpusEntry(
                            text=description,
                            category=category,
                            attributes={},
                            source="synthetic",
                            confidence=0.6
                        ))
        return entries
    
    def _fill_template(self, template: str, category: str, settings: List[str]) -> str:
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
                "setting": random.choice(settings)
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
    parser.add_argument("--output_dir", default="/home/kyzen/cheng/ulip_rag/rag_corpus", 
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
