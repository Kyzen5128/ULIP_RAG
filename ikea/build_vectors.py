#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_vectors.py — 用已訓練好的 ULIP checkpoint 對 IKEA-ULIP 輸出做「離線一次性」編碼
輸出三種模態的向量與對應 meta（jsonl）供檢索用。

本版已去除「預採樣/重採樣流程」：假設點雲已是固定點數（預設 8192；若你的資料是 8092，請用 --npoints 8092）。
只做：讀檔 ->（必要時輕量對齊長度）-> normalize -> encode。
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# === ULIP ===
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils as U  # for get_model()


# ==========================
# Argument parsing
# ==========================
def get_args():
    ap = argparse.ArgumentParser("Build vectors for IKEA ULIP dataset")

    ap.add_argument("--json_dir", required=True, type=str,
                    help="資料集 JSON 目錄，內含多個 *.json（一個產品一檔）")
    ap.add_argument("--out_dir", required=True, type=str,
                    help="輸出向量與索引的資料夾")
    ap.add_argument("--ckpt", required=True, type=str,
                    help="ULIP checkpoint 檔（.pt）")
    ap.add_argument("--model", default="ULIP_PointBERT", type=str,
                    help="若 ckpt 沒帶 args.model，則用這個名稱建模")

    ap.add_argument("--modalities", nargs="+", default=["pc", "img", "txt"],
                    choices=["pc", "img", "txt"],
                    help="要抽哪些模態的向量")
    ap.add_argument("--max_images", type=int, default=8,
                    help="每個樣本最多採用多少張渲染圖做聚合；0=不限")

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size_img", type=int, default=64)
    ap.add_argument("--batch_size_txt", type=int, default=256)
    ap.add_argument("--batch_size_pc", type=int, default=32)

    ap.add_argument("--npoints", type=int, default=8192,
                    help="點雲固定點數（若你的資料是 8092，請設為 8092）")
    ap.add_argument("--pc_sampler", default="fps", choices=["fps", "truncate", "random"],
                    help="點數過多時的降採樣方式。fps=與訓練/serving 一致（預設，建議）；"
                         "truncate=直接截斷（舊版行為，與部署向量不一致，勿用）；random=隨機抽")
    ap.add_argument("--use_height", action="store_true",
                    help="若模型需要 height channel（PointNeXt 用），平常關閉即可")

    # 路徑 / 除錯
    ap.add_argument("--root_hint", default=None,
                    help="repo 或資料根目錄（例如 /home/.../ULIP），用來修正 ./ulip_output/... 相對路徑")
    ap.add_argument("--debug_paths", action="store_true",
                    help="逐筆印出解析後的圖片/點雲路徑與存在性（只印前 debug_limit 筆）")
    ap.add_argument("--debug_limit", type=int, default=20,
                    help="debug_paths 模式下最多印幾筆")

    # 執行裝置
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--amp", action="store_true", help="使用 autocast 加速（影像/文字）")

    return ap.parse_args()


# ==========================
# 智能路徑解析
# ==========================
class PathResolver:
    """
    解析 JSON 內的相對路徑；支援把 `./ulip_output/...` / `ulip_output/...` 轉址到實際的 ulip_output 根目錄；
    也會嘗試 `/data/ulip_output` ↔ `/ulip_output` 的互換；可選擇性加上 root_hint。
    """
    def __init__(self, json_dir: str, root_hint: Optional[str] = None, debug_limit: int = 8):
        self.json_dir = Path(json_dir).resolve()
        # 猜 repo_root：.../ULIP/data/ulip_output/json -> repo_root=.../ULIP
        self.repo_root = self.json_dir
        for p in self.json_dir.parents:
            if p.name == "data":
                self.repo_root = p.parent
                break

        cand = [
            self.repo_root / "ulip_output",          # .../ULIP/ulip_output
            self.json_dir.parent,                    # .../ULIP/data/ulip_output
            self.repo_root / "data" / "ulip_output", # .../ULIP/data/ulip_output
        ]
        if root_hint:
            cand.append(Path(root_hint) / "ulip_output")
            cand.append(Path(root_hint) / "data" / "ulip_output")

        seen, ulip_roots = set(), []
        for c in cand:
            c = c.resolve()
            if c.is_dir() and str(c) not in seen:
                seen.add(str(c))
                ulip_roots.append(c)

        # 優先挑有 ply/ 的根目錄
        pref = None
        for c in ulip_roots:
            if (c / "ply").is_dir():
                pref = c
                break
        self.ulip_roots = [pref] + [c for c in ulip_roots if c != pref] if pref else ulip_roots

        self._debug_budget = debug_limit

    def _try_candidates(self, tail: str) -> Optional[str]:
        for root in self.ulip_roots:
            cand = (root / tail).resolve()
            if cand.is_file():
                return str(cand)
        return None

    def _maybe_debug_print(self, original: str, tried: list):
        if self._debug_budget <= 0:
            return
        self._debug_budget -= 1
        shown = " | ".join(tried[:4])
        print(f"[path-resolve] miss: '{original}' ; tried: {shown}")

    def resolve(self, p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        p = p.strip()

        tried = []

        # 1) 絕對路徑
        if os.path.isabs(p) and os.path.isfile(p):
            return p

        # 2) 以 ./ 或 ../ 起頭
        if p.startswith("./") or p.startswith("../"):
            cand = Path(p).expanduser().resolve()
            tried.append(str(cand))
            if cand.is_file():
                return str(cand)

        # 3) 明確處理 ulip_output 前綴
        norm = p.lstrip("./")
        if norm.startswith("ulip_output/"):
            tail = norm.split("ulip_output/", 1)[1]  # e.g. "ply/xxx.ply"
            hit = self._try_candidates(tail)
            if hit:
                return hit
            tried.extend([str((r / tail).resolve()) for r in self.ulip_roots])

        # 4) 以 repo_root 為基準
        cand = (self.repo_root / p).resolve()
        tried.append(str(cand))
        if cand.is_file():
            return str(cand)

        # 5) 以 json_dir 為基準
        cand = (self.json_dir / p).resolve()
        tried.append(str(cand))
        if cand.is_file():
            return str(cand)

        # 6) 把 "/data/ulip_output" 換成 "/ulip_output"
        if "/data/ulip_output" in p:
            tail = p.split("/data/ulip_output", 1)[1].lstrip("/\\")
            hit = self._try_candidates(tail)
            if hit:
                return hit
            tried.extend([str((r / tail).resolve()) for r in self.ulip_roots])

        # 全部失敗，列出嘗試
        self._maybe_debug_print(p, tried)
        return None


# ==========================
# JSON 掃描 + 路徑解析
# ==========================
def collect_items(json_dir: str,
                  max_images: int,
                  resolver: PathResolver,
                  debug_paths: bool = False,
                  debug_limit: int = 20) -> List[Dict]:
    """
    回傳 items 清單：
      item = {
         "id": <檔名無副檔名>,
         "json_path": <JSON 絕對路徑>,
         "img_paths": [已解析存在的圖像路徑...],
         "pc_path": <已解析存在的 PLY/NPY 路徑 or ''>,
         "caption": <llava_caption_en 或 text 或 ''>,
         "category": <或 ''>
      }
    """
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    print(f"Scan JSON: {len(json_files)} files")

    dbg_cnt = 0
    items = []

    for jp in json_files:
        try:
            d = json.load(open(jp, "r", encoding="utf-8"))
        except Exception as e:
            print(f"[JSON ERR] {jp}: {e}")
            continue

        _id = os.path.splitext(os.path.basename(jp))[0]
        raw_img = (d.get("image") or "").strip()
        render_list = d.get("render_images") or []
        raw_pc = (d.get("pointcloud") or "").strip()

        # 解析圖片：優先 render_images（最多 max_images 張），否則退回單張 image
        img_paths = []
        if render_list:
            limit = max_images if max_images > 0 else len(render_list)
            for r in render_list[:limit]:
                rp = resolver.resolve(r)
                if rp and os.path.exists(rp):
                    img_paths.append(rp)
        if not img_paths and raw_img:
            ip = resolver.resolve(raw_img)
            if ip and os.path.exists(ip):
                img_paths.append(ip)

        # 解析點雲
        pc_path = ""
        if raw_pc:
            pp = resolver.resolve(raw_pc)
            if pp and os.path.exists(pp):
                pc_path = pp

        # caption / 類別
        caption = (d.get("llava_caption_en") or d.get("text") or "").strip()
        category = (d.get("category") or "").strip()

        if debug_paths and (dbg_cnt < debug_limit):
            print("\n" + "=" * 90)
            print(f"[ID] { _id }")
            print(f"[JSON] { jp }")
            print(f"[IMAGE(raw)] {raw_img}")
            if render_list:
                print(f"[RENDERS(raw)] {len(render_list)} (show first 5):")
                for i, r in enumerate(render_list[:5]):
                    rp = resolver.resolve(r)
                    print(f"  - [{i:02d}] {r} -> {rp} | exists={os.path.exists(rp) if rp else False}")
                if len(render_list) > 5:
                    print("  ...")
            print(f"[IMG(paths)] {len(img_paths)} (show first 5):")
            for i, p in enumerate(img_paths[:5]):
                print(f"  - [{i:02d}] {p} | exists={os.path.exists(p)}")
            res_pc = resolver.resolve(raw_pc) if raw_pc else ""
            print(f"[PC] raw={raw_pc} -> {res_pc} | exists={os.path.exists(res_pc) if res_pc else False}")
            print(f"[CAPTION] present={bool(caption)} | len={len(caption)} | category={category}")
            dbg_cnt += 1

        items.append({
            "id": _id,
            "json_path": str(Path(jp).resolve()),
            "img_paths": img_paths,
            "pc_path": pc_path,
            "caption": caption,
            "category": category,
        })

    return items


# ==========================
# 圖像轉換
# ==========================
def build_image_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])


def pil_loader(path: str):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


# ==========================
# 模型載入
# ==========================
def load_ulip_model(ckpt_path: str, fallback_model_name: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only= False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    if "args" in ckpt and hasattr(ckpt["args"], "model"):
        mdl_name = ckpt["args"].model
        setattr(ckpt["args"], "evaluate_3d", True)
        model = getattr(models, mdl_name)(args=ckpt["args"])
        print(f"Using model from checkpoint: {mdl_name}")
    else:
        from argparse import Namespace
        fake = Namespace(model=fallback_model_name)
        model = getattr(models, fallback_model_name)(args=fake)
        print(f"Using fallback model from args: {fallback_model_name}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[load_state_dict] Missing keys:", missing)
    if unexpected:
        print("[load_state_dict] Unexpected keys:", unexpected)

    model = model.to(device).eval()
    print("Model loaded.")
    return model


# ==========================
# 抽向量：TXT
# ==========================
@torch.no_grad()
def encode_texts(model, tokenizer, items: List[Dict],
                 device: torch.device, bs: int, amp: bool = False):
    texts, metas = [], []
    for it in items:
        cap = (it["caption"] or "").strip()
        if not cap:
            continue
        texts.append(cap)
        metas.append({
            "id": it["id"],
            "json_path": it["json_path"],
            "category": it["category"],
            "caption": cap,
            "preview_image": it["img_paths"][0] if it["img_paths"] else ""
        })

    if len(texts) == 0:
        print("[TXT] 無可用文字 caption，略過。")
        return np.zeros((0, 512), np.float32), []

    feats = []
    for i in tqdm(range(0, len(texts), bs), desc="Encode TXT"):
        batch = texts[i:i+bs]
        tok = tokenizer(batch).to(device)
        if tok.dim() == 1:
            tok = tok.unsqueeze(0)
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                z = U.get_model(model).encode_text(tok)
        else:
            z = U.get_model(model).encode_text(tok)
        z = F.normalize(z, dim=-1)
        feats.append(z.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    return feats.astype(np.float32), metas


# ==========================
# 抽向量：IMG（多張聚合 -> mean -> normalize）
# ==========================
@torch.no_grad()
def encode_images(model, transform, items: List[Dict],
                  device: torch.device, bs: int, amp: bool = False):
    metas, img_lists = [], []
    for it in items:
        paths = it["img_paths"]
        if not paths:
            continue
        img_lists.append(paths)
        metas.append({
            "id": it["id"],
            "json_path": it["json_path"],
            "category": it["category"],
            "caption": it["caption"],
            "preview_image": paths[0] if paths else ""
        })

    if len(img_lists) == 0:
        print("[IMG] 無可用圖像，略過。")
        return np.zeros((0, 512), np.float32), []

    flat_imgs, counts = [], []
    for paths in img_lists:
        counts.append(len(paths))
        flat_imgs.extend(paths)

    feats_each, batch_imgs = [], []
    for p in tqdm(flat_imgs, desc="Encode IMG per item"):
        try:
            img = pil_loader(p)
            img = transform(img)
            batch_imgs.append(img)
            if len(batch_imgs) == bs:
                x = torch.stack(batch_imgs, 0).to(device)
                if amp and device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        z = U.get_model(model).encode_image(x)
                else:
                    z = U.get_model(model).encode_image(x)
                z = F.normalize(z, dim=-1)
                feats_each.append(z.cpu().numpy())
                batch_imgs = []
        except Exception as e:
            print(f"[IMG read ERR] {p}: {e}")

    if batch_imgs:
        x = torch.stack(batch_imgs, 0).to(device)
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                z = U.get_model(model).encode_image(x)
        else:
            z = U.get_model(model).encode_image(x)
        z = F.normalize(z, dim=-1)
        feats_each.append(z.cpu().numpy())

    if len(feats_each) == 0:
        print("[IMG] 圖像讀取/編碼全失敗。")
        return np.zeros((0, 512), np.float32), []

    feats_each = np.concatenate(feats_each, axis=0)

    # 聚合成每個 item 一個向量
    agg, cursor = [], 0
    for c in counts:
        if c <= 0:
            agg.append(None)
            continue
        block = feats_each[cursor: cursor + c]
        cursor += c
        if block.shape[0] == 0:
            agg.append(None)
        else:
            v = block.mean(axis=0, keepdims=True)
            v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
            agg.append(v.astype(np.float32))

    final_vecs, final_metas = [], []
    for vec, meta in zip(agg, metas):
        if vec is None:
            continue
        final_vecs.append(vec)
        final_metas.append(meta)

    if len(final_vecs) == 0:
        print("[IMG] 聚合後無有效向量。")
        return np.zeros((0, 512), np.float32), []

    final_vecs = np.concatenate(final_vecs, axis=0)
    return final_vecs, final_metas


# ==========================
# 點雲讀取 & encode（無重採樣版）
# ==========================
def load_pointcloud_auto(path: str) -> np.ndarray:
    """
    讀 PLY/NPY/NPZ → 回傳 Nx3（若是 N×C 只取前三欄）。
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            if arr.dtype == object:  # 可能是 dict
                try:
                    d = arr.item()
                    for k in ["xyz", "points", "pc"]:
                        if k in d:
                            a = np.asarray(d[k], dtype=np.float32)
                            return a[:, :3] if a.ndim == 2 else a.reshape(-1, 3)
                except Exception:
                    pass
            return arr[:, :3] if arr.ndim == 2 else arr.reshape(-1, 3)
        return np.asarray(arr, dtype=np.float32)[:, :3]

    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        for k in ["xyz", "points", "pc"]:
            if k in data:
                a = np.asarray(data[k], dtype=np.float32)
                return a[:, :3] if a.ndim == 2 else a.reshape(-1, 3)
        first_key = list(data.files)[0]
        a = np.asarray(data[first_key], dtype=np.float32)
        return a[:, :3] if a.ndim == 2 else a.reshape(-1, 3)

    # 優先 open3d
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.size > 0:
            return pts[:, :3]
    except Exception:
        pass

    # 退回 plyfile
    try:
        from plyfile import PlyData
        ply = PlyData.read(path)
        v = ply["vertex"]
        xs = np.asarray(v["x"], dtype=np.float32)
        ys = np.asarray(v["y"], dtype=np.float32)
        zs = np.asarray(v["z"], dtype=np.float32)
        return np.stack([xs, ys, zs], axis=1)
    except Exception:
        pass

    # 最後用 trimesh
    try:
        import trimesh
        tm = trimesh.load(path, process=False)
        if hasattr(tm, "vertices"):
            return np.asarray(tm.vertices, dtype=np.float32)[:, :3]
        elif isinstance(tm, trimesh.Scene):
            buf = []
            for geom in tm.geometry.values():
                if hasattr(geom, "vertices"):
                    buf.append(np.asarray(geom.vertices, dtype=np.float32))
            if buf:
                return np.vstack(buf).astype(np.float32)[:, :3]
    except Exception:
        pass

    raise RuntimeError(f"Failed to read point cloud: {path}")


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """中心化 + 單位球正規化（與 ULIP 一致）"""
    c = np.mean(pc[:, :3], axis=0)
    pc[:, :3] = pc[:, :3] - c
    scale = np.max(np.linalg.norm(pc[:, :3], axis=1)) + 1e-9
    pc[:, :3] = pc[:, :3] / scale
    return pc


def sample_points(pts: np.ndarray, npoints: int, sampler: str, device: torch.device) -> np.ndarray:
    """
    N >= npoints 時的降採樣。
    fps      : 與訓練 (ikea_ulip.py) / serving (app sample_xyz) 一致；
               優先用 pointnet2_ops 的 GPU FPS（毫秒級），失敗才退回 numpy FPS（每筆數十秒）。
    truncate : 直接切前 npoints（舊版行為；與部署向量不一致，僅保留供對照）。
    random   : 均勻隨機抽。
    """
    N = pts.shape[0]
    if N <= npoints:
        return pts
    if sampler == "truncate":
        return pts[:npoints]
    if sampler == "random":
        idx = np.random.choice(N, npoints, replace=False)
        return pts[idx]
    # --- fps ---
    if device.type == "cuda":
        try:
            from pointnet2_ops import pointnet2_utils as _p2
            xyz = torch.from_numpy(np.ascontiguousarray(pts[:, :3])).float().unsqueeze(0).to(device)
            idx = _p2.furthest_point_sample(xyz, npoints)  # [1, npoints] int32
            return pts[idx[0].long().cpu().numpy()]
        except Exception as e:
            print(f"[FPS] GPU FPS 失敗（{type(e).__name__}），退回 numpy FPS（很慢）: {e}")
    from data.dataset_3d import farthest_point_sample as _np_fps
    return _np_fps(pts, npoints)


@torch.no_grad()
def encode_pointclouds(model, items: List[Dict], device: torch.device,
                       npoints: int, bs: int, use_height: bool,
                       sampler: str = "fps"):
    pcs, metas = [], []
    for it in tqdm(items, desc=f"Sample PC ({sampler})"):
        p = it["pc_path"]
        if not p:
            continue
        try:
            pts = load_pointcloud_auto(p)  # [N,3]
            if pts.ndim != 2 or pts.shape[1] < 3:
                continue
            pts = pts[:, :3].astype(np.float32)

            # 對齊長度：多 → 依 --pc_sampler 降採樣（預設 FPS，與訓練/serving 一致）；少 → 重複補
            N = pts.shape[0]
            if N >= npoints:
                pts = sample_points(pts, npoints, sampler, device)
            else:
                rep = np.random.choice(N, npoints - N, replace=True)
                pts = np.concatenate([pts, pts[rep]], axis=0)

            # normalize
            pts = pc_normalize(pts)

            if use_height:
                h = (pts[:, 1:2] - pts[:, 1:2].min())
                pts = np.concatenate([pts, h], axis=1)

            pcs.append(torch.from_numpy(pts).float())
            metas.append({
                "id": it["id"],
                "json_path": it["json_path"],
                "category": it["category"],
                "caption": it["caption"],
                "preview_image": it["img_paths"][0] if it["img_paths"] else ""
            })
        except Exception as e:
            print(f"[PC read ERR] {p}: {e}")

    if len(pcs) == 0:
        print("[PC] 無可用點雲，略過。")
        return np.zeros((0, 512), np.float32), []

    feats = []
    for i in tqdm(range(0, len(pcs), bs), desc="Encode PC"):
        batch = torch.stack(pcs[i:i+bs], 0).to(device)  # [B,N,3(+1)]
        z = U.get_model(model).encode_pc(batch)
        z = F.normalize(z, dim=-1)
        feats.append(z.cpu().numpy())

    feats = np.concatenate(feats, axis=0).astype(np.float32)
    return feats, metas


# ==========================
# Save helpers
# ==========================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_vectors(out_dir: str, tag: str, vecs: np.ndarray, metas: List[Dict]):
    ensure_dir(out_dir)
    npy_path = os.path.join(out_dir, f"vectors_{tag}.npy")
    meta_path = os.path.join(out_dir, f"meta_{tag}.jsonl")
    np.save(npy_path, vecs)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[SAVE] {tag}: vecs={vecs.shape} -> {npy_path}")
    print(f"[SAVE] {tag}: meta -> {meta_path}")


# ==========================
# Main
# ==========================
def main():
    args = get_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Device: {device}")

    # 1) 掃描 JSON + 解析路徑（會顯示實際路徑/存在性）
    resolver = PathResolver(args.json_dir, root_hint=args.root_hint, debug_limit=args.debug_limit)
    items = collect_items(
        json_dir=args.json_dir,
        max_images=args.max_images,
        resolver=resolver,
        debug_paths=args.debug_paths,
        debug_limit=args.debug_limit
    )

    # 摘要一下 usable 統計
    stats = defaultdict(int)
    for it in items:
        stats["img_usable"] += 1 if it["img_paths"] else 0
        stats["pc_usable"]  += 1 if it["pc_path"] else 0
        stats["txt_usable"] += 1 if it["caption"] else 0
    print("\n==== Summary of parsed items ====")
    print(f"JSON files    : {len(items)}")
    print(f"IMG usable    : {stats['img_usable']}  | no_images : {len(items)-stats['img_usable']}")
    print(f"PC  usable    : {stats['pc_usable']}   | no_pc     : {len(items)-stats['pc_usable']}")
    print(f"TXT usable    : {stats['txt_usable']}  | no_txt    : {len(items)-stats['txt_usable']}")

    # 2) 載入 ULIP 模型
    model = load_ulip_model(args.ckpt, args.model, device)
    tokenizer = SimpleTokenizer()
    img_tfm = build_image_transform(args.img_size)

    ensure_dir(args.out_dir)

    # 3) 逐模態抽向量
    if "txt" in args.modalities:
        txt_vecs, txt_meta = encode_texts(model, tokenizer, items, device, args.batch_size_txt, amp=args.amp)
        save_vectors(args.out_dir, "txt", txt_vecs, txt_meta)

    if "img" in args.modalities:
        img_vecs, img_meta = encode_images(model, img_tfm, items, device, args.batch_size_img, amp=args.amp)
        save_vectors(args.out_dir, "img", img_vecs, img_meta)

    if "pc" in args.modalities:
        pc_vecs, pc_meta = encode_pointclouds(
            model, items, device,
            npoints=args.npoints,
            bs=args.batch_size_pc,
            use_height=args.use_height,
            sampler=args.pc_sampler,
        )
        save_vectors(args.out_dir, "pc", pc_vecs, pc_meta)

    # 4) schema / 總覽
    schema = {
        "modalities": args.modalities,
        "dims": {
            m: (int(np.load(os.path.join(args.out_dir, f"vectors_{m}.npy")).shape[1])
                if os.path.exists(os.path.join(args.out_dir, f"vectors_{m}.npy")) else 0)
            for m in ["pc", "img", "txt"]
        }
    }
    with open(os.path.join(args.out_dir, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"\nDone. wrote vectors to {args.out_dir}")


if __name__ == "__main__":
    main()
