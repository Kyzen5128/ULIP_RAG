import json
from transformers import AutoTokenizer
from tqdm import tqdm

input_path = "/home/kyzen/ULIP_RAG/ulip_rag/rag_corpus/rag_corpus.jsonl"
output_path = "/home/kyzen/ULIP_RAG/ulip_rag/rag_corpus/rag_corpus_tokenized.jsonl"

# === 選擇模型分詞器 ===
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def detect_category(text):
    t = text.lower()
    if "chair" in t: return "chair"
    if "table" in t or "desk" in t: return "table"
    return "other"

# === 讀取原始資料 ===
with open(input_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} samples")

tokenized_data = []

# === 分詞處理 ===
for entry in tqdm(data, desc="Tokenizing"):
    text = entry["text"].strip()

    encoding = tokenizer(text, truncation=True, max_length=128)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    tokenized_entry = {
        "doc_id": entry["doc_id"],
        "text": text,
        "tokens": tokens,
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "category": detect_category(text)
    }
    tokenized_data.append(tokenized_entry)

# === 輸出 ===
with open(output_path, "w", encoding="utf-8") as f:
    for item in tokenized_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Done! Saved tokenized corpus to {output_path}")
