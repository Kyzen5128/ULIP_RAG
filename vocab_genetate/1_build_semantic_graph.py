
import json, re, time, random, requests
from pathlib import Path
from typing import Dict, List, Set
from nltk.corpus import wordnet as wn
from tqdm import tqdm

DATA = Path("data")
DATA.mkdir(exist_ok=True)

# ---------- 1) WordNet 種子（可自由增刪） ----------
SEED_SYNSETS = [
    # Seating
    "chair.n.01", "sofa.n.01", "bench.n.01", "stool.n.01",
    # Tables & Desks
    "table.n.02", "desk.n.01", "coffee_table.n.01",
    # Beds
    "bed.n.01", "bunk.n.01",
    # Storage
    "cabinet.n.01", "shelf.n.01", "drawer.n.01",
    # Lighting & Decor
    "lamp.n.02", "chandelier.n.01", "mirror.n.01", "screen.n.03"
]

# ---------- 2) ConceptNet API ----------
CN_ENDPOINT = "https://api.conceptnet.io/query"

# 取用的關係，對應我們的語義欄位
REL_MAP = {
    "/r/HasProperty":  "properties",   # 形容詞傾向
    "/r/MadeOf":       "materials",
    "/r/UsedFor":      "uses",
    "/r/HasA":         "parts",
    "/r/PartOf":       "super_parts",
    "/r/IsA":          "isa",
    "/r/RelatedTo":    "related"
}

def clean_label(s: str) -> str:
    s = s.replace("_", " ").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def crawl_wordnet(synsets: List[wn.synset], relation: str, visited: Set[wn.synset]):
    nxt = set()
    for s in synsets:
        try:
            for n in getattr(s, relation)():
                if n not in visited:
                    visited.add(n); nxt.add(n)
        except AttributeError:
            pass
    if not nxt: return set()
    return nxt | crawl_wordnet(list(nxt), relation, visited)

def wordnet_skeleton():
    roots = [wn.synset(x) for x in SEED_SYNSETS]
    hypos = crawl_wordnet(roots, "hyponyms", set()) | crawl_wordnet(roots, "instance_hyponyms", set())
    parts = crawl_wordnet(roots+list(hypos), "part_meronyms", set())

    def nm(s): return s.name().split(".")[0].replace("_", " ")
    boring = {"entity","object","physical entity","instrumentality"}
    concepts = sorted({nm(s) for s in hypos if nm(s) not in boring})
    parts_   = sorted({nm(s) for s in parts if nm(s) not in boring})
    return concepts, parts_

# ---------- ConceptNet 快取 ----------
CACHE_PATH = DATA / "conceptnet_cache.json"
if CACHE_PATH.exists():
    CN_CACHE: Dict[str, Dict] = json.loads(CACHE_PATH.read_text())
else:
    CN_CACHE = {}

def conceptnet_query(term: str, limit=200) -> List[Dict]:
    key = f"{term}:{limit}"
    if key in CN_CACHE:
        return CN_CACHE[key]["edges"]
    params = {"node": f"/c/en/{term}", "limit": str(limit)}
    try:
        r = requests.get(CN_ENDPOINT, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        edges = data.get("edges", [])
    except Exception:
        edges = []
    CN_CACHE[key] = {"edges": edges}
    if len(CN_CACHE) % 50 == 0:  # 週期性寫盤
        CACHE_PATH.write_text(json.dumps(CN_CACHE))
    time.sleep(0.1)  # 延遲
    return edges

def extract_attributes_from_cn(term: str) -> Dict[str, List[str]]:
    term = clean_label(term)
    edges = conceptnet_query(term, limit=300)
    bucket = {v: set() for v in set(REL_MAP.values())}
    for e in edges:
        rel = e.get("rel", {}).get("@id")
        if rel not in REL_MAP: continue
        tgt_field = REL_MAP[rel]
        start = e.get("start", {}).get("label", "") or e.get("start", {}).get("@id","")
        end   = e.get("end", {}).get("label", "") or e.get("end", {}).get("@id","")
        s = clean_label(start); t = clean_label(end)
        # 僅納入與 term 直接相連的另一端
        other = t if s == term else (s if t == term else None)
        if not other: continue
        # 過濾掉過泛詞
        if other in {"object","thing","entity"}: continue
        # 簡單限制詞長
        if 2 <= len(other) <= 40:
            bucket[tgt_field].add(other)
    return {k: sorted(list(v)) for k, v in bucket.items() if v}

if __name__ == "__main__":
    # 1) WordNet骨架
    concepts, parts = wordnet_skeleton()
    (DATA / "ontology_skeleton.json").write_text(
        json.dumps({"concepts": concepts, "parts": parts}, indent=2, ensure_ascii=False)
    )
    print(f"WordNet concepts={len(concepts)}, parts={len(parts)}")

    # 2) ConceptNet 擴展並寫 semantic_bank.jsonl
    out = DATA / "semantic_bank.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in tqdm(concepts, desc="ConceptNet擴展"):
            attrs = extract_attributes_from_cn(c)
            rec = {
                "concept": c,
                "wordnet_parts_hint": parts[:50],    # 作為共用部件提示的一部分（可裁剪）
                "conceptnet": attrs                  # 來源：ConceptNet
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 收尾：寫出快取
    CACHE_PATH.write_text(json.dumps(CN_CACHE))
    print("✓ semantic_bank.jsonl 已建立；conceptnet_cache.json 已更新")
