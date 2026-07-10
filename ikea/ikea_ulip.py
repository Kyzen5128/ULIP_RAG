#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob, random, logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
import trimesh

from utils.build import DATASETS

# 直接複用 ULIP 現成的點雲工具（與官方流程一致）
from data.dataset_3d import (
    pc_normalize as ulip_pc_normalize,
    farthest_point_sample,
    random_point_dropout,
    random_scale_point_cloud,
    shift_point_cloud,
    jitter_point_cloud,
    rotate_perturbation_point_cloud,
    rotate_point_cloud,
)

logger = logging.getLogger("IkeaULIP")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)


def _pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
    return img


def _make_placeholder_image(res: int = 224) -> Image.Image:
    return Image.fromarray(np.full((res, res, 3), 128, dtype=np.uint8))


def _safe_isfile(p: Optional[str]) -> bool:
    try:
        return bool(p) and os.path.isfile(p)
    except Exception:
        return False


class _PathResolver:
    """
    解析 JSON 內的相對路徑；支援把 `./ulip_output/...` / `ulip_output/...`
    轉址到實際的 ulip_output 根目錄；也試 `.../data/ulip_output/...`。
    """
    def __init__(self, json_dir: str, debug_limit: int = 8):
        self.json_dir = Path(json_dir).resolve()
        # 猜 repo_root：.../ULIP/data/ulip_output/json -> repo_root=.../ULIP
        self.repo_root = self.json_dir
        for p in self.json_dir.parents:
            if p.name == "data":
                self.repo_root = p.parent
                break

        # 候選 ulip_output 根目錄（依優先序）
        cand = [
            self.repo_root / "ulip_output",          # .../ULIP/ulip_output
            self.json_dir.parent,                    # .../ULIP/data/ulip_output
            self.repo_root / "data" / "ulip_output"  # .../ULIP/data/ulip_output（保險再試一次）
        ]

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
        logger.info(f"[path-resolve] miss: '{original}' ; tried: {shown}")

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


@DATASETS.register_module(name="IkeaULIP")
class IkeaULIP(data.Dataset):
    """
    從 /.../ulip_output/json/*.json 讀：
      - image / render_images[*]
      - pointcloud(.ply)
      - category, llava_caption_en
    影像可缺（REQUIRE_IMAGE=False 時用佔位圖），點雲必須要有且點數 >= MIN_POINTS。
    支援 hash split（TRAIN_RATIO）。
    與 ULIP 其他資料集的點雲處理完全對齊（normalize + FPS + train 增強）。
    """

    def __init__(self, config, subset: str = "train"):
        # 來自 YAML（大小寫都支援）
        def _get(cfg, *keys, default=None):
            for k in keys:
                if hasattr(cfg, k):
                    return getattr(cfg, k)
            return default

        self.json_dir     = _get(config, "JSON_DIR", "json_dir", default="./ulip_output/json")
        self.split_method = _get(config, "SPLIT_METHOD", "split_method", default="hash")
        self.train_ratio  = float(_get(config, "TRAIN_RATIO", "train_ratio", default=0.9))
        self.require_image= bool(_get(config, "REQUIRE_IMAGE", "require_image", default=False))
        self.render_pick  = _get(config, "RENDER_PICK", "render_pick", default="random")
        self.min_points   = int(_get(config, "MIN_POINTS", "min_points", default=128))

        # 與 ULIP 對齊的開關
        self.pc_sampler   = str(_get(config, "PC_SAMPLER", "pc_sampler", default="fps")).lower()  # "fps"|"random"
        self.augment      = bool(_get(config, "AUGMENT", "augment", default=(subset == "train")))
        self.pad_if_short = bool(_get(config, "PAD_IF_SHORT", "pad_if_short", default=True))

        # 由外層注入
        self.subset          = subset if subset in ("train", "val", "test") else _get(config, "subset", default="train")
        self.tokenizer       = _get(config, "tokenizer", default=None)
        self.train_transform = _get(config, "train_transform", default=None)
        self.npoints         = int(_get(config, "npoints", default=8192))
        self.use_height      = bool(_get(config, "use_height", default=False))

        # 影像 eval 時的預設轉換
        self._eval_fallback_transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # 路徑轉成絕對，建立 resolver
        self.json_dir = str(Path(self.json_dir).resolve())
        self.resolver = _PathResolver(self.json_dir, debug_limit=12)

        # 掃描與解析
        all_jsons = self._scan_all_jsons(self.json_dir)
        samples, skip_stats = self._build_samples(all_jsons)

        # split（hash）
        self.samples = self._apply_split(samples, self.subset, self.train_ratio)

        cats = sorted(list({s["category"] for s in self.samples}))
        logger.info(f"[IkeaULIP] {self.subset} = {len(self.samples)} | cats={len(cats)} | json_dir={self.json_dir}")

        bad_preview = "; ".join([f"{k}:{v}" for k, v in list(skip_stats.items())[:6]])
        if bad_preview:
            logger.info(f"[IkeaULIP] skipped_reasons (head): {bad_preview}")

        if len(self.samples) == 0:
            logger.warning("目前資料筆數是 0。多半是路徑或點數不足所致。")

    # ---------- 掃描與解析 ----------
    def _scan_all_jsons(self, root: str) -> List[str]:
        pats = [os.path.join(root, "*.json"), os.path.join(root, "**", "*.json")]
        files = []
        for pat in pats:
            files.extend(glob.glob(pat, recursive=True))
        files = sorted(list({str(Path(f).resolve()) for f in files}))
        return files

    def _build_samples(self, json_paths: List[str]) -> Tuple[List[dict], dict]:
        samples, skip_stats = [], {}
        for jp in json_paths:
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                skip_stats[f"json_read:{type(e).__name__}"] = skip_stats.get(f"json_read:{type(e).__name__}", 0) + 1
                continue

            _id = Path(jp).stem
            cat = data.get("category") or "Unknown"

            # 文本（缺就用類別名備援）
            cap = (data.get("llava_caption_en") or data.get("text") or "").strip()
            if not cap:
                cap = cat if isinstance(cat, str) else "furniture"

            # render 全部解析（供隨機挑選）；同時保留一張後備 image
            render_list = data.get("render_images") or []
            render_paths = []
            for rp in render_list:
                r = self.resolver.resolve(rp)
                if _safe_isfile(r):
                    render_paths.append(r)

            img_path = render_paths[0] if render_paths else self.resolver.resolve(data.get("image"))
            if self.require_image and not _safe_isfile(img_path):
                skip_stats["no_image"] = skip_stats.get("no_image", 0) + 1
                continue

            # 點雲（必要）
            pc_path = self.resolver.resolve(data.get("pointcloud"))
            # 優先使用預採樣版（ikea/presample_ply.py 產出的 <ply 同層>/ply_8192/<id>.ply）。
            # 原始 gaussian PLY 每顆 33~67 萬點，numpy FPS 每筆 20-40s；
            # 預採樣檔 N==npoints 會直接跳過 FPS，訓練 epoch 從小時級降到分鐘級。
            # 若 ply_8192/ 不存在則維持原行為。
            if pc_path:
                _pre = os.path.join(os.path.dirname(os.path.dirname(pc_path)),
                                    "ply_8192", os.path.basename(pc_path))
                if _safe_isfile(_pre) and os.path.getsize(_pre) > 0:
                    pc_path = _pre
                    skip_stats["presampled_used"] = skip_stats.get("presampled_used", 0) + 1
            if not _safe_isfile(pc_path):
                skip_stats["no_pointcloud"] = skip_stats.get("no_pointcloud", 0) + 1
                continue

            # 只為了檢查點數是否足夠（實際抽樣放在 __getitem__）
            try:
                m = trimesh.load(pc_path, process=False)
                if hasattr(m, "vertices"):
                    pts = np.asarray(m.vertices, dtype=np.float32)
                elif isinstance(m, trimesh.Scene):
                    buf = []
                    for geom in m.geometry.values():
                        if hasattr(geom, "vertices"):
                            buf.append(np.asarray(geom.vertices, dtype=np.float32))
                    pts = np.vstack(buf) if buf else np.zeros((0, 3), dtype=np.float32)
                else:
                    pts = np.zeros((0, 3), dtype=np.float32)
            except Exception as e:
                skip_stats[f"ply_read:{type(e).__name__}"] = skip_stats.get(f"ply_read:{type(e).__name__}", 0) + 1
                continue

            if pts.shape[0] < self.min_points:
                skip_stats["too_few_points"] = skip_stats.get("too_few_points", 0) + 1
                continue

            samples.append({
                "id": _id,
                "json_path": jp,
                "category": str(cat),
                "caption": cap,
                "image_path": img_path,        # 後備單張
                "render_paths": render_paths,  # 全部 render（可隨機挑）
                "pointcloud_path": pc_path,
                "num_points": int(pts.shape[0]),
            })

        return samples, skip_stats

    def _apply_split(self, samples: List[dict], subset: str, ratio: float) -> List[dict]:
        if self.split_method != "hash" or subset not in ("train", "val", "test"):
            return samples

        import hashlib
        train_set, val_set = [], []
        thr = int(ratio * 10000)

        for s in samples:
            key = s["id"]
            h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10000
            (train_set if h < thr else val_set).append(s)

        return train_set if subset == "train" else val_set

    # ---------- Dataset 介面 ----------
    def __len__(self) -> int:
        return len(self.samples)

    def _load_pointcloud(self, path: str) -> torch.Tensor:
        m = trimesh.load(path, process=False)
        if hasattr(m, "vertices"):
            pts = np.asarray(m.vertices, dtype=np.float32)
        elif isinstance(m, trimesh.Scene):
            buf = []
            for geom in m.geometry.values():
                if hasattr(geom, "vertices"):
                    buf.append(np.asarray(geom.vertices, dtype=np.float32))
            pts = np.vstack(buf) if buf else np.zeros((0, 3), dtype=np.float32)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)

        # 只取 xyz 並做 ULIP 同款 normalize
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        pts = ulip_pc_normalize(pts)

        N, n = pts.shape[0], self.npoints

        if N > n:
            if self.pc_sampler == "fps":
                pts = farthest_point_sample(pts, n)  # [n, 3]
            else:
                idx = np.random.choice(N, n, replace=False)
                pts = pts[idx, :]
        else:
            if self.pad_if_short and N > 0:
                rep_idx = np.random.choice(N, n - N, replace=True)
                pts = np.concatenate([pts, pts[rep_idx]], axis=0)
            # 否則就維持 < n 的長度（多數 backbone 仍偏好固定長度，建議保留 pad）

        # 訓練增強（和 ULIP 一致）
        if self.augment and self.subset == "train":
            pts_b = pts[None, ...]  # [1, n, 3]
            pts_b = random_point_dropout(pts_b)
            pts_b = random_scale_point_cloud(pts_b)
            pts_b = shift_point_cloud(pts_b)
            pts_b = rotate_perturbation_point_cloud(pts_b)
            pts_b = rotate_point_cloud(pts_b)
            pts = pts_b.squeeze(0)

        if self.use_height:
            h = pts[:, 1:2] - pts[:, 1:2].min()
            pts = np.concatenate([pts, h], axis=1)

        return torch.from_numpy(pts).float()

    def _load_image(self, path: Optional[str]) -> torch.Tensor:
        if _safe_isfile(path):
            img = _pil_loader(path)
        else:
            img = _make_placeholder_image(224)
        tfm = self.train_transform or self._eval_fallback_transform
        return tfm(img) if tfm is not None else T.ToTensor()(img)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        taxonomy_id = s["category"]
        model_id    = s["id"]

        # 文本（單句）
        caption = s["caption"] or taxonomy_id
        if self.tokenizer is None:
            raise RuntimeError("tokenizer 未注入")
        tokenized = torch.stack([self.tokenizer(caption)])

        # 點雲
        pc = self._load_pointcloud(s["pointcloud_path"])

        # 影像：若有 render_paths 且 render_pick=random，每次隨機挑一張
        img_path = s["image_path"]
        if self.render_pick == "random":
            rlist = s.get("render_paths") or []
            if rlist:
                img_path = random.choice(rlist)
        elif isinstance(self.render_pick, str) and self.render_pick.startswith("fixed_"):
            deg = self.render_pick.split("_", 1)[1]
            rlist = s.get("render_paths") or []
            hit = None
            for p in rlist:
                # 檔名內含 _r_XXX 視角（若無就用第一張）
                if f"_r_{deg}" in os.path.basename(p):
                    hit = p
                    break
            if hit:
                img_path = hit

        img = self._load_image(img_path)

        return (taxonomy_id, model_id, tokenized, pc, img)
