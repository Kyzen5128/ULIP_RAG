# 檔案路徑: tricolo/data/tokenizer.py

import json
import os
import re
from functools import lru_cache

@lru_cache()
def default_word_to_idx(shapenet_json_path: str):
    """
    從 shapenet.json 檔案中讀取並快取詞彙表。
    """
    if not os.path.exists(shapenet_json_path):
        raise FileNotFoundError(
            f"找不到詞彙表檔案: {shapenet_json_path}. "
            "請確保 Text2Shape 資料集已正確下載。"
        )
    with open(shapenet_json_path, 'r') as f:
        data = json.load(f)
    # word_to_idx 包含了從單詞到索引的映射
    return data['word_to_idx']

def whitespace_clean(text):
    text = re.sub(r'\s\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, shapenet_json_path="data/text2shape-data/chair_table/shapenet.json"):
        self.word_to_idx = default_word_to_idx(shapenet_json_path)
        self.unk_token_id = self.word_to_idx.get('<unk>', 0) # 假設 <unk> 對應 0

    def __call__(self, texts, context_length: int = 77):
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        for text in texts:
            text = whitespace_clean(text).lower()
            tokens = text.split(' ')
            
            # 將單詞轉換為索引
            token_ids = [self.word_to_idx.get(word, self.unk_token_id) for word in tokens]
            
            # 處理填充和截斷
            if len(token_ids) > context_length:
                token_ids = token_ids[:context_length]
            else:
                token_ids += [0] * (context_length - len(token_ids)) # 用 0 來填充
            
            all_tokens.append(token_ids)
        
        # 轉換為 torch tensor
        import torch
        return torch.tensor(all_tokens, dtype=torch.long)