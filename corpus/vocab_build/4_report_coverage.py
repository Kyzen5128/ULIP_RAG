# -*- coding: utf-8 -*-

import json, re, math
from collections import Counter
from pathlib import Path

DATA = Path("data")

def entropy(counter: Counter) -> float:
    n = sum(counter.values()); 
    if n == 0: return 0.0
    H = 0.0
    for c in counter.values():
        p = c / n
        H -= p * math.log(p + 1e-12)
    return H

if __name__ == "__main__":
    rows = json.loads((DATA/"kb_texts.json").read_text())
    n = len(rows)
    hit_any = 0
    materials, styles, parts = Counter(), Counter(), Counter()

    for r in rows:
        s = r["text"].lower()
        ev = r["evidence"]
        bank = {"materials":0, "styles":0, "appearance":0, "parts":0}
        for k, lst in ev.items():
            for w in lst:
                if re.search(rf"\b{re.escape(w.lower())}\b", s):
                    bank[k]+=1
        if sum(bank.values())>=2: hit_any+=1
        for w in ev.get("materials", []):
            if w in s: materials[w]+=1
        for w in ev.get("styles", []):
            if w in s: styles[w]+=1
        for w in ev.get("parts", []):
            if w in s: parts[w]+=1

    print(f"句數：{n}")
    print(f"命中≥2關鍵詞比例：{hit_any/n:.2%}")
    print(f"材料熵：{entropy(materials):.3f}；風格熵：{entropy(styles):.3f}；部件熵：{entropy(parts):.3f}")
    print("材料Top10：", materials.most_common(10))
    print("風格Top10：", styles.most_common(10))
    print("部件Top10：", parts.most_common(10))
