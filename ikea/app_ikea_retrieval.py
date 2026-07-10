# app_ikea_retrieval.py
#132
import os
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import torch
torch.backends.cudnn.enabled = False  # cuDNN 9.0.1 incompatible on this machine

from typing import List, Optional, Dict, Any
from datetime import datetime
import mimetypes, json, tempfile, logging, shutil, uuid
import numpy as np
import torch
import torch.nn.functional as F

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from PIL import Image
import shutil

# 下載工具（僅在需要時使用）
import requests
from io import BytesIO

# 去背
from rembg import remove
# CLIP 分類
from transformers import CLIPProcessor, CLIPModel
# Trellis 3D
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
# ===== NEW: FPS 需要 =====
import trimesh
from data.dataset_3d import pc_normalize as ulip_pc_normalize, farthest_point_sample
import logging
from collections import defaultdict
import re


logger = logging.getLogger("uvicorn.error")


# --------- MIME 註冊 ---------
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")

# ULIP bits
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils as U  # has get_model()

# --------- config via env ---------
# 2026-07-10 第三階段:預設值從 4090 舊路徑(/mnt/data1、/home/klooom)改為 3090 現行路徑。
# env 仍可覆寫(run_app_3090.sh 會設);忘掛 env 時不再靜默指向不存在的 4090 路徑。
VEC_DIR = os.getenv("VEC_DIR", "/mnt/P300/data/ikea_data/vectors")
CKPT    = os.getenv("CKPT",    "/mnt/P300/data/ULIP/checkpoint_last.pt")
DEVICE  = "cuda" if (os.getenv("DEVICE", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu"

# 兩個根資料夾（取代以前的 ASSET_BASE/ulip_output & test_output）
ULIP_OUTPUT   = os.getenv("ULIP_OUTPUT",   "/mnt/P300/data/ikea_data")    # 以前的 ulip_output
CUSTOM_OUTPUT = os.getenv("CUSTOM_OUTPUT", "/mnt/P300/data/custom_data")  # 以前的 test_output

# 臨時、上傳與輸出都放在 CUSTOM_OUTPUT
TMP_DIR    = os.getenv("TMP_DIR",    os.path.join(CUSTOM_OUTPUT, "tmp"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(CUSTOM_OUTPUT, "uploads"))
IMG_DIR    = os.path.join(CUSTOM_OUTPUT, "images")
PLY_DIR    = os.path.join(CUSTOM_OUTPUT, "ply")
GLB_DIR    = os.path.join(CUSTOM_OUTPUT, "glb")
JSON_DIR   = os.path.join(CUSTOM_OUTPUT, "json")
for d in [TMP_DIR, UPLOAD_DIR, IMG_DIR, PLY_DIR, GLB_DIR, JSON_DIR]:
    os.makedirs(d, exist_ok=True)

# 可選 MongoDB
# 2026-07-10:優先 env MONGO_URL,再讀 ~/.ulip_mongo.env(mongo 已啟用 --auth);
# 都沒有才落回舊 URI(僅為相容,實際會被 --auth 拒絕)
try:
    from mongo_conn import get_mongo_uri  # PYTHONPATH 含 ikea/ 時可用
    MONGO_URL = get_mongo_uri(required=False) or "mongodb://127.0.0.1:27017/"
except ImportError:
    MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
MONGO_DB  = os.getenv("MONGO_DB", "furniture_db")
MONGO_COL = os.getenv("MONGO_COL", "ikea_product")
USE_MONGO = os.getenv("USE_MONGO", "1")  # "1" 啟用，其他視為停用

mongo_client = None
mongo_col = None
if USE_MONGO == "1":
    try:
        from pymongo import MongoClient
        from bson import ObjectId
        mongo_client = MongoClient(MONGO_URL)
        mongo_col = mongo_client[MONGO_DB][MONGO_COL]
    except Exception as _:
        mongo_client = None
        mongo_col = None

# --------- CLIP / Trellis 懶載入 ---------
_CLIP = {"model": None, "proc": None, "device": "cuda" if torch.cuda.is_available() else "cpu"}
_TRELLIS = {"pipe": None}

CATEGORIES = [
    "Sofa", "Dining Table", "Wardrobe", "Bookshelf", "Bed", "Kitchen Island",
    "Office Desk", "Shoe Rack", "Sideboard", "Bench", "Nightstand", "Bar Stool",
    "Recliner", "TV Stand", "Coffee Table", "Filing Cabinet", "Bunk Bed",
    "Dining Chair", "Vanity Table", "Storage Ottoman"
]

TS_RE = re.compile(r"^\d{8}T\d{9,}$")  # 例如 20250909T134507858
PROMPTS = [f"A product photo of a {c}" for c in CATEGORIES]
def _ts_id_ms() -> str:
    # 以 UTC 毫秒時間戳做 ID（避免同秒撞名）
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
def _lazy_clip():
    if _CLIP["model"] is None:
        name = "openai/clip-vit-large-patch14"
        _CLIP["model"] = CLIPModel.from_pretrained(name).to(_CLIP["device"]).eval()
        _CLIP["proc"] = CLIPProcessor.from_pretrained(name)
    return _CLIP["model"], _CLIP["proc"], _CLIP["device"]

def _lazy_trellis():
    if _TRELLIS["pipe"] is None:
        _TRELLIS["pipe"] = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        if DEVICE == "cuda":
            _TRELLIS["pipe"].cuda()
    return _TRELLIS["pipe"]

# --------- 向量與 meta 載入 ---------
def _load_meta_jsonl(path: str) -> List[Dict]:
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def _ensure_normalized(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / n

pc_vec_path   = os.path.join(VEC_DIR, "vectors_pc.npy")
pc_meta_path  = os.path.join(VEC_DIR, "meta_pc.jsonl")
if not (os.path.isfile(pc_vec_path) and os.path.isfile(pc_meta_path)):
    raise RuntimeError(f"Missing vectors/meta under {VEC_DIR}")

PC_VECS  = np.load(pc_vec_path).astype(np.float32)
PC_VECS  = _ensure_normalized(PC_VECS)
PC_META  = _load_meta_jsonl(pc_meta_path)

# id -> meta
ID2META: Dict[str, Dict[str, Any]] = {m.get("id"): m for m in PC_META if m.get("id")}

# --------- 工具函式 ---------
IMG_KEYS      = {"image", "preview_image", "img", "thumbnail", "thumb", "jpeg", "jpg", "png"}
GLB_KEYS      = {"mesh", "glb_path", "model_path", "pc_path", "glb", "gltf"}
IMG_KEY_ORDER = ["image", "preview_image", "img", "thumbnail", "thumb", "jpg", "jpeg", "png"]
GLB_KEY_ORDER = ["mesh", "glb_path", "model_path", "pc_path", "glb", "gltf"]
IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
GLB_EXTS = [".glb", ".gltf", ".ply", ".obj"]

def _select_fields(meta: Dict[str, Any], wanted: set) -> Dict[str, Any]:
    picked = {}
    for k, v in meta.items():
        kk = k.lower()
        if kk in wanted or any(w in kk for w in wanted):
            picked[k] = v
    return picked

def _abs_if_exists(p: str) -> str:
    if not p:
        return ""
    p = os.path.normpath(p)
    # 絕對路徑：直接檢查
    if os.path.isabs(p):
        return p if os.path.isfile(p) else ""
    # 相對路徑：依序嘗試兩個根
    for root in (CUSTOM_OUTPUT, ULIP_OUTPUT):
        cand = os.path.normpath(os.path.join(root, p))
        if os.path.isfile(cand):
            return cand
    return ""

def _find_file_by_keys(meta: Dict[str, Any], key_order: List[str]) -> str:
    for k in key_order:
        for mk, mv in meta.items():
            mk_l = mk.lower()
            if mk_l == k or k in mk_l:
                path = _abs_if_exists(str(mv))
                if path:
                    return path
    return ""

def _find_file_by_ext(meta: Dict[str, Any], exts: List[str]) -> str:
    for ext in exts:
        for v in meta.values():
            if isinstance(v, str) and v.lower().endswith(ext):
                path = _abs_if_exists(v)
                if path:
                    return path
    return ""

def _fallback_scan_by_id(id_: str, exts: List[str], kind: str) -> str:
    candidates = []

    search_roots = [ULIP_OUTPUT, CUSTOM_OUTPUT]


    if kind == "img":
        for root in search_roots:
            base = os.path.join(root, "images")
            for ext in exts:
                candidates.append(os.path.join(base, f"{id_}{ext}"))
    else:
        for root in search_roots:
            for sub in ["glb", "gltf", "ply"]:
                base = os.path.join(root, sub)
                for ext in exts:
                    candidates.append(os.path.join(base, f"{id_}{ext}"))
            # 舊的 ply_old 資料夾只在 ulip_output 可能有
            candidates.append(os.path.join(root, "ply_old", f"{id_}.ply"))

    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isfile(c):
            return c
    return ""

def _guess_mime_from_path(path: str, kind_hint: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".glb":
        return "model/gltf-binary"
    if ext == ".gltf":
        return "model/gltf+json"
    if ext in [".ply", ".obj"]:
        return "application/octet-stream"
    if ext == ".png":
        return "image/png"
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    m, _ = mimetypes.guess_type(path)
    return m or ("application/octet-stream" if kind_hint != "img" else "image/*")

# ===== NEW: PLY→(N,3) / FPS / 儲存 =====
def load_xyz(ply_path: str) -> np.ndarray:
    """讀 .ply → (N,3) xyz，支援 Scene"""
    m = trimesh.load(ply_path, process=False)
    if hasattr(m, "vertices"):
        pts = np.asarray(m.vertices, dtype=np.float32)
    elif isinstance(m, trimesh.Scene):
        buf = []
        for g in m.geometry.values():
            if hasattr(g, "vertices"):
                buf.append(np.asarray(g.vertices, dtype=np.float32))
        pts = np.vstack(buf) if buf else np.zeros((0, 3), dtype=np.float32)
    else:
        pts = np.zeros((0, 3), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    if pts.shape[1] > 3:
        pts = pts[:, :3]
    return pts

def sample_xyz(pts: np.ndarray, npoints: int = 8192, sampler: str = "fps", seed: Optional[int] = None) -> np.ndarray:
    """
    先 normalize 再抽樣到 npoints。
    N>=npoints：FPS or random；N<npoints：重複補齊。
    """
    if pts.size == 0:
        return np.zeros((npoints, 3), dtype=np.float32)

    # normalize 與 ULIP 一致
    pts = ulip_pc_normalize(pts.astype(np.float32, copy=False))
    N = pts.shape[0]
    if N >= npoints:
        if sampler == "fps":
            pts = farthest_point_sample(pts, npoints)  # [n,3]
        else:
            rng = np.random.default_rng(seed)
            idx = rng.choice(N, npoints, replace=False)
            pts = pts[idx, :]
    else:
        rng = np.random.default_rng(seed)
        rep_idx = rng.choice(N, npoints - N, replace=True)
        pts = np.concatenate([pts, pts[rep_idx]], axis=0)
    return pts.astype(np.float32, copy=False)

def save_ply_xyz(pts: np.ndarray, out_path: str, binary: bool = True):
    """只存頂點的 PLY（無 faces/顏色）"""
    mesh = trimesh.Trimesh(vertices=pts, faces=None, process=False)
    mesh.export(
        out_path,
        file_type="ply",
        encoding="binary_little_endian" if binary else "ascii",
    )
# --------- Pydantic schema ---------
class TextSearchReq(BaseModel):
    query: str
    top_k: int = 10
    category: Optional[str] = None

class SearchHit(BaseModel):
    id: str
    score: float
    category: Optional[str] = None
    caption_en: Optional[str] = None
    preview_image: Optional[str] = None
    json_path: Optional[str] = None

class TextSearchResp(BaseModel):
    count: int
    results: List[SearchHit]

class FileGetResp(BaseModel):
    id: str
    type: str
    category: Optional[str] = None
    caption_en: Optional[str] = None
    data: Dict[str, Any]

# --------- ULIP text encoder ---------
def load_ulip_text_encoder(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    if "args" in ckpt and hasattr(ckpt["args"], "model"):
        mdl_name = ckpt["args"].model
        setattr(ckpt["args"], "evaluate_3d", True)
        model = getattr(models, mdl_name)(args=ckpt["args"])
        print(f"[ULIP] Using model from checkpoint: {mdl_name}")
    else:
        from argparse import Namespace
        fake = Namespace(model="ULIP_PointBERT")
        model = getattr(models, "ULIP_PointBERT")(args=fake)
        print("[ULIP] Fallback model: ULIP_PointBERT")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print("[ULIP] Missing keys:", missing[:5], "...")
    if unexpected: print("[ULIP] Unexpected keys:", unexpected[:5], "...")
    model = model.to(device).eval()
    tok   = SimpleTokenizer()
    return model, tok

MODEL, TOKENIZER = load_ulip_text_encoder(CKPT, DEVICE)
print(f"[READY] vectors={PC_VECS.shape}, device={DEVICE}")

print(f"[PATHS] VEC_DIR={VEC_DIR}")

def _tokenize_safe(texts, device: str):
    toks = TOKENIZER(texts)
    if not torch.is_tensor(toks):
        toks = torch.as_tensor(toks, dtype=torch.long)
    else:
        if getattr(toks, "is_sparse", False):
            toks = toks.to_dense()
        toks = toks.to(dtype=torch.long)
    if toks.dim() == 1:
        toks = toks.unsqueeze(0)
    return toks.to(device).contiguous()

@torch.no_grad()
def encode_text(query: str) -> np.ndarray:
    tok = _tokenize_safe([query], DEVICE)
    feats = U.get_model(MODEL).encode_text(tok)  # [1, D]
    feats = F.normalize(feats, dim=-1)
    return feats.squeeze(0).detach().cpu().numpy().astype(np.float32)

def search_text(query: str, top_k: int, category: Optional[str]) -> List[Dict]:
    q = encode_text(query)
    if category:
        idx = [i for i, m in enumerate(PC_META) if (m.get("category") == category)]
        if not idx:
            return []
        vecs = PC_VECS[idx]
        metas = [PC_META[i] for i in idx]
    else:
        vecs = PC_VECS
        metas = PC_META

    scores = vecs @ q
    if top_k <= 0:
        top_k = 10
    k = min(top_k, scores.shape[0])
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for i in top_idx:
        m = metas[i]
        results.append({
            "id": m.get("id"),
            "score": float(scores[i]),
            "category": m.get("category"),
            "caption_en": m.get("caption"),
            "preview_image": m.get("preview_image"),
            "json_path": m.get("json_path"),
        })
    return results
def _scan_ids_under(root: str) -> List[str]:
    """掃描指定 root 下的 images/ply/glb/json 取得檔名前綴（ID）。"""
    if not root or not os.path.isdir(root):
        return []
    subdirs = ["images", "ply", "glb", "json"]
    exts = {".jpg", ".jpeg", ".png", ".webp", ".ply", ".glb", ".gltf", ".json"}

    seen: set[str] = set()
    for sub in subdirs:
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        try:
            for ent in os.scandir(d):
                if not ent.is_file():
                    continue
                name = ent.name
                _, ext = os.path.splitext(name)
                if ext.lower() not in exts:
                    continue
                base = os.path.splitext(name)[0]
                # if not TS_RE.match(base): 
                #     continue
                seen.add(base)
        except FileNotFoundError:
            continue
    return sorted(seen)

def _scan_ids(kind: int) -> List[str]:
    """
    kind=0: ULIP_OUTPUT + CUSTOM_OUTPUT
    kind=1: 只 ULIP_OUTPUT（爬蟲）
    kind=2: 只 CUSTOM_OUTPUT（custom 上傳）
    """
    if kind == 1:
        roots = [ULIP_OUTPUT]
    elif kind == 2:
        roots = [CUSTOM_OUTPUT]
    else:
        roots = [ULIP_OUTPUT, CUSTOM_OUTPUT]

    all_ids: set[str] = set()
    for r in roots:
        all_ids.update(_scan_ids_under(r))
    # 依照時間戳字串倒序
    return sorted(all_ids, reverse=True)

class IdListResp(BaseModel):
    count: int
    ids: List[str]
# --------- FastAPI app ---------
app = FastAPI(title="IKEA ULIP Retrieval", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post("/search/text", response_model=TextSearchResp)
def text_search(req: TextSearchReq):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is empty")
    hits = search_text(q, req.top_k, req.category)
    return TextSearchResp(count=len(hits), results=hits)

# --------- 檔案查詢 ---------
@app.get("/file", response_model=FileGetResp)
def get_file_meta(type: str, id: str):
    t = type.strip().lower()
    if t not in {"img", "glb"}:
        raise HTTPException(status_code=400, detail="type must be 'img' or 'glb'")

    meta = ID2META.get(id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"id '{id}' not found")

    base = {"id": meta.get("id"), "category": meta.get("category"), "caption_en": meta.get("caption")}

    if t == "img":
        data = _select_fields(meta, IMG_KEYS)
        if not data and "image" in meta:
            data = {"image": meta.get("image")}
    else:
        data = _select_fields(meta, GLB_KEYS)

    if not data:
        raise HTTPException(status_code=404, detail=f"id '{id}' has no fields for type '{t}'")

    return FileGetResp(id=base["id"], type=t, category=base["category"], caption_en=base["caption_en"], data=data)

@app.get("/get/file")
def get_file_bytes(type: str, id: str):
    t = type.strip().lower()
    if t not in {"img", "glb"}:
        raise HTTPException(status_code=400, detail="type must be 'img' or 'glb'")

    meta = ID2META.get(id)  # 可能為 None

    if t == "img":
        path = _fallback_scan_by_id(id, IMG_EXTS, "img")
    else:
        path = ""
        if meta:  # 如果有 meta 就優先用
            path = _find_file_by_keys(meta, GLB_KEY_ORDER) or _find_file_by_ext(meta, GLB_EXTS)
        if not path:
            path = _fallback_scan_by_id(id, GLB_EXTS, "glb")

    print("[BYTES-DEBUG]", f"id={id}", f"type={t}", f"picked={path or 'NOT_FOUND'}")

    if not path:
        raise HTTPException(status_code=404, detail=f"id '{id}' not found anywhere")

    mime = _guess_mime_from_path(path, t)
    filename = os.path.basename(path)
    headers = {"Cache-Control": "public, max-age=86400", "Content-Disposition": f'inline; filename=\"{filename}\"'}
    return FileResponse(path, media_type=mime, filename=filename, headers=headers)

# --------- 上傳->去背->分類->Trellis-> ULIP JSON ---------
def _save_upload_to_disk(upload: UploadFile) -> tuple[str, str]:
    ctype = (upload.content_type or "").lower()
    if not ctype.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"只接受影像，收到 content-type={ctype or '未知'}")
    guessed_ext = mimetypes.guess_extension(ctype) or ""
    orig_ext = os.path.splitext(upload.filename or "")[1]
    ext = (guessed_ext or orig_ext or ".bin").lower()
    fname = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex}{ext}"
    out_path = os.path.join(UPLOAD_DIR, fname)
    return fname, out_path

def _remove_bg(in_path: str) -> str:
    img = Image.open(in_path).convert("RGBA")
    res = remove(img)
    out_path = in_path.rsplit(".", 1)[0] + "_nobg.png"
    res.save(out_path, format="PNG")
    return out_path

def _classify_clip(image_path: str) -> tuple[str, float]:
    model, proc, dev = _lazy_clip()
    im = Image.open(image_path).convert("RGB")
    inputs = proc(text=PROMPTS, images=im, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
        probs = logits.softmax(dim=1)
    idx = probs.argmax(dim=1).item()
    return CATEGORIES[idx], float(probs[0, idx])

def _lookup_mongo_fields(source_id: Optional[str]) -> Dict[str, Any]:
    out = {"description": "", "brand": None, "price": None, "url": None}
    if not (mongo_col and source_id):
        return out
    # 嘗試以 ObjectId 查詢，失敗則用字串比對
    doc = None
    try:
        doc = mongo_col.find_one({"_id": ObjectId(source_id)})
    except Exception:
        doc = mongo_col.find_one({"_id": source_id})
    if not doc:
        return out
    out["description"] = doc.get("description", "") or ""
    out["brand"] = doc.get("brand")
    out["price"] = doc.get("price")
    out["url"] = doc.get("url")
    return out

def _build_ulip_json(image_path: str, ply_path: str, glb_path: str,
                     description: str, category: str, meta: dict) -> dict:
    return {
        "image": image_path,
        "pointcloud": ply_path,
        "mesh": glb_path,
        "text": description,
        "category": category,
        "meta": meta
    }

@app.post("/post/upload")
async def post_upload(
    file: UploadFile = File(...),
    #  表單欄位（需求：要放進 meta）
    length_mm: float = Form(..., description="實體長度，單位 mm"),
    width_mm:  float = Form(..., description="實體寬度，單位 mm"),
    height_mm: float = Form(..., description="實體高度，單位 mm"),
    #  其餘可選欄位
    description_in: Optional[str] = Form(default=None),
    brand:        Optional[str]   = Form(default=None),
    price:        Optional[float] = Form(default=None),
    url:          Optional[str]   = Form(default=None),
):
    # ---------- A) 生成時間戳 ID 與臨時工作區 ----------
    base_id = _ts_id_ms()            # ← 「source_id = 檔名 = 時間戳」
    work_dir = tempfile.mkdtemp(prefix=f"{base_id}_", dir=TMP_DIR)

    # 推斷副檔名，用於原始上傳檔（臨時存放）
    ctype = (file.content_type or "").lower()
    ext_guess = mimetypes.guess_extension(ctype) or os.path.splitext(file.filename or "")[1] or ".jpg"
    raw_tmp = os.path.join(work_dir, f"{base_id}{ext_guess.lower()}")

    # staging 檔案
    img_tmp     = os.path.join(work_dir, f"{base_id}.jpg")
    nobg_tmp    = os.path.join(work_dir, f"{base_id}_nobg.png")
    ply8192_tmp = os.path.join(work_dir, f"{base_id}.ply")   # 只留 8192 版，檔名就叫 base_id.ply
    glb_tmp     = os.path.join(work_dir, f"{base_id}.glb")
    json_tmp    = os.path.join(work_dir, f"{base_id}.json")

    # 最終目的地
    img_final  = os.path.join(IMG_DIR,  f"{base_id}.jpg")
    ply_final  = os.path.join(PLY_DIR,  f"{base_id}.ply")     # 只存 8192 版
    glb_final  = os.path.join(GLB_DIR,  f"{base_id}.glb")
    json_final = os.path.join(JSON_DIR, f"{base_id}.json")

    try:
        # ---------- B) 寫入原始上傳檔（到 staging） ----------
        try:
            with open(raw_tmp, "wb") as fout:
                while True:
                    chunk = await file.read(1 << 20)
                    if not chunk:
                        break
                    fout.write(chunk)
        finally:
            await file.close()

        # ---------- C) 複製一份原圖 → images 用 ----------
        try:
            with Image.open(raw_tmp) as im:
                im.convert("RGB").save(img_tmp, format="JPEG", quality=95)
        except Exception:
            shutil.copy2(raw_tmp, img_tmp)

        # ---------- D) 去背（staging） ----------
        try:
            with Image.open(raw_tmp).convert("RGBA") as im_rgba:
                nobg = remove(im_rgba)
                nobg.save(nobg_tmp, format="PNG")
        except Exception as e:
            logger.exception("去背失敗")
            raise HTTPException(status_code=500, detail=f"去背失敗: {e}")

        # ---------- E) CLIP 分類 ----------
        try:
            label, conf = _classify_clip(nobg_tmp)
        except Exception as e:
            logger.exception("分類失敗")
            raise HTTPException(status_code=500, detail=f"分類失敗: {e}")

        # ---------- F) Trellis → 產 GLB 與 PLY(8192) ----------
        try:
            pipe = _lazy_trellis()
            img_rgb = Image.open(nobg_tmp).convert("RGB")
            outputs = pipe.run(img_rgb, seed=1)

            if "gaussian" not in outputs or not outputs["gaussian"]:
                raise RuntimeError("Trellis 沒產出 gaussian")
            if "mesh" not in outputs or not outputs["mesh"]:
                raise RuntimeError("Trellis 沒產出 mesh")

            # 先把 Trellis 的點雲暫存成 raw PLY（存在 staging 內），再 FPS → 8192，最後丟掉 raw
            raw_ply_tmp = os.path.join(work_dir, f"{base_id}__raw.ply")
            outputs["gaussian"][0].save_ply(raw_ply_tmp)

            # GLB（staging）
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=0.95,
                texture_size=1024
            )
            glb.export(glb_tmp)

            # FPS 固定 8192（staging），然後立刻刪掉 raw
            pts = load_xyz(raw_ply_tmp)
            pts_8192 = sample_xyz(pts, npoints=8192, sampler="fps", seed=None)
            save_ply_xyz(pts_8192, ply8192_tmp, binary=True)
            try:
                os.remove(raw_ply_tmp)
            except Exception:
                pass

        except RuntimeError as e:
            logger.exception("Trellis 3D 失敗 (RuntimeError)")
            raise HTTPException(status_code=500, detail=f"Trellis 3D 失敗: {e}")
        except Exception as e:
            logger.exception("Trellis 3D 失敗")
            raise HTTPException(status_code=500, detail=f"Trellis 3D 失敗: {e}")

        # ---------- G) 檢查 staging 產物 ----------
        for p in (img_tmp, ply8192_tmp, glb_tmp):
            if (not os.path.isfile(p)) or os.path.getsize(p) <= 0:
                raise HTTPException(status_code=500, detail=f"產物不完整: {os.path.basename(p)}")

        # ---------- H) 組 ULIP JSON（JSON 內路徑寫最終位置） ----------
        description = (description_in or "").strip()
        ulip_obj = _build_ulip_json(
            image_path=img_final,
            ply_path=ply_final,    # 若函式參數名是 ply_path 就用下面那行
            # ply_path=ply_final,
            glb_path=glb_final,
            description=description,
            category=label,
            meta={
                "source_id": base_id,                              # ← 時間戳
                "clip_confidence": round(float(conf), 4),
                "brand": brand,
                "price": price,
                "url": url,
                "dimensions_mm": {                                 # ← 使用者輸入尺寸
                    "length": float(length_mm),
                    "width":  float(width_mm),
                    "height": float(height_mm),
                },
            }
        )
        # 若 _build_ulip_json 參數名固定為 ply_path，請把上面 pointcloud_path 換成 ply_path

        # 先寫 JSON 到 staging
        with open(json_tmp, "w", encoding="utf-8") as jf:
            json.dump(ulip_obj, jf, ensure_ascii=False, indent=2)

        # ---------- I) 一次性 commit 到正式目錄 ----------
        for d in (IMG_DIR, PLY_DIR, GLB_DIR, JSON_DIR):
            os.makedirs(d, exist_ok=True)

        os.replace(img_tmp,  img_final)
        os.replace(ply8192_tmp, ply_final)   # 只留下 8192 版
        os.replace(glb_tmp,  glb_final)
        os.replace(json_tmp, json_final)

        # 回傳 ULIP JSON（其中 meta.source_id = base_id）
        return ulip_obj

    finally:
        # 清掉 staging
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

@app.get("/get/ids", response_model=IdListResp)
def list_ids(type: int = 0):
    """
    type=0 → ULIP_OUTPUT + CUSTOM_OUTPUT
    type=1 → 只 ULIP_OUTPUT（爬蟲）
    type=2 → 只 CUSTOM_OUTPUT（custom 上傳）
    """
    if type not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="type must be 0(all) | 1(crawler) | 2(custom)")
    ids = _scan_ids(type)
    return IdListResp(count=len(ids), ids=ids)