# -*- coding: utf-8 -*-

import json, re, random
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers import pipeline

DATA = Path("data")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# 生成設定
SENTS_PER_CONCEPT = 6
MAX_RETRY = 2
MIN_LEN, MAX_LEN = 24, 220


STYLE_FALLBACK = ["Victorian","Scandinavian","Art-Deco","Rustic","Industrial","Minimalist","Mid-century"]
MATERIAL_FALLBACK = ["mahogany","oak","walnut","rattan","acrylic","marble","leather","velvet","linen","bronze","steel","glass"]
APPEARANCE_BANK = ["sleek","ornate","tufted","barrel-back","low-profile","carved","fluted","beveled","scalloped","tapered"]

pipe = pipeline("text-generation", model=MODEL_ID, torch_dtype="auto", device_map="auto")

def pick_k(arr: List[str], k: int) -> List[str]:
    arr = list(dict.fromkeys(arr))
    random.shuffle(arr)
    return arr[:k]

def build_prompt(rec: Dict) -> Dict:
    c = rec["concept"]
    cn = rec.get("conceptnet", {})
    mats = cn.get("materials", []) or MATERIAL_FALLBACK
    props= cn.get("properties", [])
    uses = cn.get("uses", [])
    parts= cn.get("parts", []) or rec.get("wordnet_parts_hint", [])[:10]
    styles = [s for s in cn.get("related", []) if re.search(r"(style|design|victorian|modern|scandinavian|rustic|baroque)", s)] \
             or STYLE_FALLBACK

    need = {
        "materials": pick_k(mats, 3),
        "styles":    pick_k(styles, 3),
        "appearance": pick_k(APPEARANCE_BANK, 3),
        "parts":     pick_k(parts, 3)
    }

    sys = "You are a furniture design describer. Write concise, precise, domain-faithful sentences."
    usr = (
        f"Concept: '{c}'.\n"
        f"Write {SENTS_PER_CONCEPT} DIFFERENT single-sentence descriptions (one per line). "
        f"Each sentence MUST include AT LEAST TWO tokens drawn from the union of these lists:\n"
        f"- materials: {need['materials']}\n"
        f"- styles: {need['styles']}\n"
        f"- appearance: {need['appearance']}\n"
        f"- parts: {need['parts']}\n"
        f"Focus on geometry, parts, craft terms, finishes; avoid generic fluff. No numbering."
    )
    return {"system": sys, "user": usr, "need": need}

def llm_once(system: str, user: str) -> List[str]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user}
    ]
    out = pipe(
        messages,
        max_new_tokens=420,
        temperature=0.8,
        do_sample=True,
        return_full_text=False          
    )

    # 取得純文字
    gen = out[0]["generated_text"]
    if isinstance(gen, list):                   # 新版格式：list of messages
        gen = " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in gen
        )

    txt = gen.strip()
    lines = [re.sub(r"^[\-\d\.\s]+", "", l).strip() for l in txt.split("\n")]
    lines = [l for l in lines if MIN_LEN <= len(l) <= MAX_LEN]
    # 取最後 N 句，避開把提示內容回顯進來
    return lines[-(SENTS_PER_CONCEPT + 3):]


def passes_guard(s: str, need: Dict[str, List[str]]) -> bool:
    bank = set(sum(need.values(), []))
    hit = sum(1 for w in bank if re.search(rf"\b{re.escape(w)}\b", s, flags=re.I))
    return hit >= 2

if __name__ == "__main__":
    out = (DATA / "generated_sentences.jsonl").open("w", encoding="utf-8")
    with (DATA / "semantic_bank.jsonl").open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="LLM 受控生成"):
            rec = json.loads(line)
            prompt = build_prompt(rec)
            cand = []
            tries = 0
            while len(cand) < SENTS_PER_CONCEPT and tries <= MAX_RETRY:
                tries += 1
                raw = llm_once(prompt["system"], prompt["user"])
                for s in raw:
                    if passes_guard(s, prompt["need"]):
                        cand.append(s)
                    if len(cand) >= SENTS_PER_CONCEPT:
                        break
            # 回寫
            for s in cand[:SENTS_PER_CONCEPT]:
                out.write(json.dumps({
                    "concept": rec["concept"],
                    "text": s,
                    "evidence": prompt["need"],   # 生成時要求的關鍵詞子集
                    "source": {
                        "wordnet": True,
                        "conceptnet": True,
                        "model": MODEL_ID
                    }
                }, ensure_ascii=False) + "\n")
    out.close()
    print(" 生成完成：generated_sentences.jsonl")
