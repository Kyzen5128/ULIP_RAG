# parts2words/models/clip_backend.py
"""
Thin wrapper  around HuggingFace CLIP, plus FAISS indexes
"""

import torch, faiss, json, numpy as np
from transformers import CLIPModel, CLIPProcessor
from torchvision.transforms import ToTensor
import torch.nn.functional as F


def _faiss_norm(mat):
    """L2‑normalize (N,D) numpy in‑place then用 InnerProduct index等同 cosine"""
    faiss.normalize_L2(mat)


class CLIPRetriever:
    """
    1. encode_text(list[str])  -> (B,512) torch
    2. encode_image(list[PIL]) -> (B,512) torch
    3. 已內建 text 與 image FAISS (IP) index 供 .tindex / .iindex
    """
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 text_kb_vec: str = "/home/klooom/cheng/3d_retrival/Parts2Words/data/rag_corpus/text_kb_vectors.npy",
                 img_kb_vec: str = "/home/klooom/cheng/3d_retrival/Parts2Words/data/rag_corpus/img_kb_vectors.npy"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: CLIPModel = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc  : CLIPProcessor = CLIPProcessor.from_pretrained(model_name)

        # ---------- TEXT KB ----------
        tv = np.load(text_kb_vec).astype("float32")
        _faiss_norm(tv)
        self.tindex = faiss.IndexFlatIP(tv.shape[1])
        self.tindex.add(tv)
        self.texts = json.load(open(text_kb_vec.replace("_vectors.npy", "_texts.json"), "r"))

        # ---------- IMAGE KB ----------
        iv = np.load(img_kb_vec).astype("float32")
        _faiss_norm(iv)
        self.iindex = faiss.IndexFlatIP(iv.shape[1])
        self.iindex.add(iv)
        self.img_paths = json.load(open(img_kb_vec.replace("_vectors.npy", "_paths.json"), "r"))

        self.to_tensor = ToTensor()

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode_text(self, str_list):
        inputs = self.proc(text=str_list, return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)          # (B,512)

    @torch.no_grad()
    def encode_image(self, pil_list):
        inputs = self.proc(images=pil_list, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)          # (B,512)
