import os
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import torch
torch.backends.cudnn.enabled = False  # cuDNN 9.0.1 incompatible on this machine

# 2026-07-10 統一連線(mongo 已啟用 --auth,舊無認證 URI 已失效)
from mongo_conn import get_client

client = get_client()
col = client["furniture_db"]["ikea_product"]

def sample_top_per_category(col, top_n=50, min_count=30):
    selected = []
    # 先統計各類別數量
    category_counts = list(col.aggregate([
        {"$group": {"_id": "$main_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]))

    print("=== Sampling Plan (top by clip_confidence) ===")
    for cat_doc in category_counts:
        cat = cat_doc["_id"]
        count = cat_doc["count"]
        if cat is None or count < min_count:
            print(f"Skip {cat}: only {count} items")
            continue

        # 依分數排序取前 top_n
        docs = list(col.find(
            {"main_type": cat, "clip_confidence": {"$type": "double"}},
            {
                "_id": 1, "image": 1, "images": 1, "description": 1,
                "main_type": 1, "clip_confidence": 1,
                "brand": 1, "price": 1, "url": 1, "converted_3d": 1
            }
        ).sort("clip_confidence", -1).limit(top_n))

        # 排除已完成 'y' 的資料，避免重複跑
        docs = [d for d in docs if d.get("converted_3d") != "y"]

        print(f"{cat:<18} -> {len(docs)} selected (from {count})")
        selected.extend(docs)

    print(f"Total selected: {len(selected)}")
    return selected

batch = sample_top_per_category(col, top_n=50, min_count=30)



import json
from PIL import Image
from tqdm import tqdm
import requests

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# 只跑取樣 batch
if not batch:
    raise SystemExit("No documents to process. Check sampling conditions.")

# Trellis 初始化
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

# 輸出目錄
OUT_DIR = "./ulip_output"
os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUT_DIR}/ply", exist_ok=True)
os.makedirs(f"{OUT_DIR}/glb", exist_ok=True)
os.makedirs(f"{OUT_DIR}/json", exist_ok=True)

for doc in tqdm(batch, desc="Trellis 3D & ULIP export"):
    _id = doc["_id"]

    # 若已完成則略過（雙重保護）
    if doc.get("converted_3d") == "y":
        continue

    # 先標記未完成狀態（可續跑）
    col.update_one({"_id": _id}, {"$set": {"converted_3d": "n"}})

    img_url = doc.get("image") or (doc.get("images") or [None])[0]
    if not img_url:
        # 無圖，維持 'n'
        continue

    try:
        # 下載圖片到本地
        img_path = os.path.join(OUT_DIR, "images", f"{_id}.jpg")
        if not os.path.exists(img_path):
            r = requests.get(img_url, timeout=15)
            r.raise_for_status()
            with open(img_path, "wb") as f:
                f.write(r.content)

        # Trellis 生成
        image = Image.open(img_path).convert("RGB")
        outputs = pipeline.run(image, seed=1)

        # 存點雲 PLY（Gaussians）
        ply_path = os.path.join(OUT_DIR, "ply", f"{_id}.ply")
        outputs["gaussian"][0].save_ply(ply_path)

        # 存網格 GLB（Mesh + Gaussian 融合烘焙）
        glb_path = os.path.join(OUT_DIR, "glb", f"{_id}.glb")
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,
            texture_size=1024
        )
        glb.export(glb_path)

        # 輸出 ULIP 相容 JSON
        json_path = os.path.join(OUT_DIR, "json", f"{_id}.json")
        ulip_entry = {
            "image": img_path,
            "pointcloud": ply_path,
            "mesh": glb_path,
            "text": doc.get("description", ""),
            "category": doc.get("main_type", "Unknown"),
            "meta": {
                "source_id": str(_id),
                "clip_confidence": doc.get("clip_confidence"),
                "brand": doc.get("brand"),
                "price": doc.get("price"),
                "url": doc.get("url")
            }
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(ulip_entry, jf, ensure_ascii=False, indent=2)

        # 成功標記
        col.update_one({"_id": _id}, {"$set": {"converted_3d": "y"}})

    except Exception as e:
        # 保留 'n' 以便日後重試
        col.update_one({"_id": _id}, {"$set": {"converted_3d_error": str(e)}})
