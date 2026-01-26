# file: parts2words/models/rag_model.py
# -*- coding: utf-8 -*-

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# -------------------------------------------------
# 從原模型檔案載入基礎組件（若已經放在同一檔可忽略這行）
# -------------------------------------------------
from .model import (
    EncoderText, PointNetDenseCls, EncoderImage,
    EMDLoss, feature_transform_regularizer, l2norm
)

# ======================== 小工具 ========================
def tokenize(sent: str):
    """極簡 tokenizer：依空白切詞，可替換成自訂 BPE/WordPiece。"""
    return sent.lower().strip().split()

def pad_ids(seqs, pad_idx: int = 0):
    """將不等長序列補 PAD，回傳 tensor / 長度 / 最長長度。"""
    max_len = max(len(s) for s in seqs)
    padded  = [s + [pad_idx] * (max_len - len(s)) for s in seqs]
    lengths = [len(s) for s in seqs]
    return torch.tensor(padded).long(), lengths, max_len

# ======================== 詞級 Cross‑Attention ========================
class FusionAttentionWord(nn.Module):
    """Q 來自 caption 詞向量；K/V 來自檢索句子的詞向量。"""
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out    = nn.Linear(dim, dim)
        self.dp     = nn.Dropout(0.1)

    def forward(self, Q, K, V):
        # Q: (B, Lq, D)  K/V: (B, Lk, D)
        attn = torch.bmm(self.q_proj(Q), self.k_proj(K).transpose(1, 2))
        attn = attn / math.sqrt(K.size(-1))
        w    = F.softmax(attn, dim=-1)
        w    = self.dp(w)
        ctx  = torch.bmm(w, self.v_proj(V))           # (B, Lq, D)
        return self.out(ctx)
    
    
class FusionAttentionStack(nn.Module):
    """
    可堆疊的詞級 Cross-Attention：
    - 每層：Pre-Norm → Cross-Attn → 殘差 → (可選) FFN → 殘差
    - Q 來自 caption 詞向量；K/V 來自檢索上下文的詞向量
    參數：
      num_layers: 疊幾層 cross-attn
      ffn_ratio : FFN 隱層維度倍率，D -> (ffn_ratio*D) -> D
      use_ffn   : 是否在每層 attn 後加 FFN
      dropout   : attn/ffn 的 dropout
    """
    def __init__(self, dim: int, num_layers: int = 1,
                 ffn_ratio: int = 4, use_ffn: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.use_ffn    = use_ffn
        self.dropout    = nn.Dropout(dropout)

        self.attn_layers = nn.ModuleList([FusionAttentionWord(dim) for _ in range(num_layers)])
        self.norm_q = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.norm_m = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])  # 給 FFN 前用

        if use_ffn:
            hid = ffn_ratio * dim
            self.ffn = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, hid),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hid, dim),
                ) for _ in range(num_layers)
            ])
        else:
            self.ffn = None

    def forward(self, H_cap, H_ctx):
        """
        H_cap: (B, Lq, D)  來自 caption 的詞級表示
        H_ctx: (B, Lk, D)  來自 top-k 檢索句子的詞級表示（已串接）
        """
        H = H_cap
        for l in range(self.num_layers):
            # Pre-Norm + Cross-Attn
            H_norm = self.norm_q[l](H)
            attn_out = self.attn_layers[l](H_norm, H_ctx, H_ctx)  # (B, Lq, D)
            H = H + self.dropout(attn_out)

            # (可選) FFN 子層
            if self.use_ffn:
                M = self.norm_m[l](H)
                H = H + self.dropout(self.ffn[l](M))
        return H

    
class FusionConcat(nn.Module):
    def __init__(self, dim: int, pooler: str = 'mean', dropout: float = 0.1):
        super().__init__()
        self.pooler = pooler  # 'mean' or 'max'
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, H_cap, H_ctx):
        # H_cap: (B, Lcap, D), H_ctx: (B, Lctx, D)
        if self.pooler == 'max':
            ctx_pool, _ = H_ctx.max(dim=1, keepdim=True)   # (B,1,D)
        else:
            ctx_pool = H_ctx.mean(dim=1, keepdim=True)     # (B,1,D)
        ctx_tiled = ctx_pool.expand(-1, H_cap.size(1), -1) # (B,Lcap,D)
        cat = torch.cat([H_cap, ctx_tiled], dim=-1)        # (B,Lcap,2D)
        fused = self.fuse_mlp(cat)                         # (B,Lcap,D)
        g = self.gate(cat)                                 # (B,Lcap,D)
        return self.norm(H_cap + fused)                # (B,Lcap,D)    
    

# ======================== RAG 文字編碼器 ========================
class RAGTextEncoder(nn.Module):
    """
    Stage‑1：只用純文字 EncoderText。
    Stage‑2：3D part embedding 經線性投射到與 Sentence‑BERT 相同的空間後，
             與文字向量加權平均組合成檢索 query。
    """
    def __init__(self, opt):
        super().__init__()
        self.opt   = opt
        self.alpha = opt.alpha_query                # 文字佔比（0~1）
        
        self.fuse_type = getattr(opt, 'fuse_type', 'cross')

        # ---------- 詞彙表 ----------
        with open('data/shapenet/vocab/augmented_shapenet.json', 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.w2i = vocab_data['word_to_idx']
        self.pad = self.w2i.get('<pad>', 0)
        self.unk = self.w2i.get('<unk>', 0)

        # ---------- Caption Encoder ----------
        self.base_txt_enc = EncoderText(
            opt.vocab_size, opt.word_dim, opt.embed_size,
            opt.num_layers, use_bi_gru=opt.bi_gru,
            no_txtnorm=opt.no_txtnorm
        )

        # ---------- Sentence‑BERT ----------
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2').cuda()
        self.retriever.eval()
        self.sen_dim = self.retriever.get_sentence_embedding_dimension()  # = 384

        # ---------- 3D → Sentence‑BERT 投射 ----------
        in_dim = opt.img_dim + 3 if opt.precomp_enc_type == 'rgb' else opt.img_dim
        self.proj3d = nn.Linear(in_dim, self.sen_dim)

        # ---------- 詞級 Cross‑Attention ----------
        if self.fuse_type == 'cross':
            self.fusion = FusionAttentionStack(
                dim=opt.embed_size,
                num_layers=getattr(opt, 'fuse_layers', 1),
                ffn_ratio=getattr(opt, 'fuse_ffn_ratio', 4),
                use_ffn=getattr(opt, 'fuse_use_ffn', True),
                dropout=getattr(opt, 'fuse_dropout', 0.1),
            )
        elif self.fuse_type == 'concat':
            self.fusion = FusionConcat(
                dim=opt.embed_size,
                pooler=getattr(opt, 'ctx_pooler', 'mean'),
                dropout=getattr(opt, 'fuse_dropout', 0.1),
            )
        else:
            raise ValueError(f"Unknown fuse_type: {self.fuse_type}")

        self.norm = nn.LayerNorm(opt.embed_size)  # 可保留；cross/concat 都可再做一次 LN
        

        # ---------- 向量索引 ----------
        kb_vec = np.load('data/rag_corpus/kb_vectors.npy').astype('float32')  # (N, 384)
        faiss.normalize_L2(kb_vec)
        self.kb_idx = faiss.IndexFlatIP(self.sen_dim)
        self.kb_idx.add(kb_vec)

        with open('data/rag_corpus/kb_texts.json', 'r', encoding='utf-8') as f:
            self.kb_texts = json.load(f)

        self.top_k = 5  # 檢索句數

    # ===== 內部工具 =====
    def _sents_to_tensor(self, sentences):
        ids_batch = [
            [self.w2i.get(tok, self.unk) for tok in tokenize(s)]
            for s in sentences
        ]
        pad_tensor, lens, _ = pad_ids(ids_batch, self.pad)
        return pad_tensor.cuda(), lens

    # ===== forward =====
    def forward(self, captions, lengths, original_texts,
                part_embed=None, cur_epoch=0):
        """
        captions    : (B, L) caption token ids
        lengths     : list[int] caption 長度
        original_texts : list[str] 原始 caption 字串
        part_embed  : (B, Kp, C) 3D 部分特徵（point2part 後）
        cur_epoch   : 當前 epoch
        """
        # -------- Stage‑1：只用 caption 自身 --------
        if cur_epoch < self.opt.stage_1_epoch:
            H_cap, L_cap = self.base_txt_enc(captions, lengths)
            if not self.opt.no_txtnorm:
                H_cap = l2norm(H_cap, dim=-1)
            return H_cap, L_cap

        # -------- 1. 建 Query 向量 --------
        with torch.no_grad():
            txt_vec = self.retriever.encode(
                original_texts,
                convert_to_tensor=True,
                device='cuda',
                normalize_embeddings=True
            )  # (B, sen_dim)

        if part_embed is not None:
            B, Kp, C = part_embed.shape
            part_flat = part_embed.reshape(-1, C)          # (B*Kp, C)
            part_vec  = self.proj3d(part_flat).view(B, Kp, -1)  # (B, Kp, sen_dim)
            part_vec  = F.normalize(part_vec.mean(dim=1), p=2, dim=-1)  # (B, sen_dim)
            q_vec = self.alpha * txt_vec + (1 - self.alpha) * part_vec  # (B, sen_dim)
        else:
            q_vec = txt_vec  # (B, sen_dim)

        # -------- 2. FAISS 檢索 --------
        _, idxs = self.kb_idx.search(q_vec.detach().cpu().numpy(), self.top_k)

        # -------- 3. 取回句子詞向量 --------
        ctx_sents = [self.kb_texts[j] for i in idxs for j in i]      # (B*top_k,)
        ctx_ids, ctx_lens = self._sents_to_tensor(ctx_sents)         # (B*top_k, Lctx)
        H_ctx, _ = self.base_txt_enc(ctx_ids, ctx_lens)              # (B*top_k, Lctx, D)

        H_ctx = H_ctx.view(len(idxs), self.top_k, -1, self.opt.embed_size) \
                     .reshape(len(idxs), -1, self.opt.embed_size)    # (B, top_k*Lctx, D)

        # -------- 4. Caption self encoding --------
        H_cap, L_cap = self.base_txt_enc(captions, lengths)          # (B, Lcap, D)

        # -------- 5. 融合 --------
        H_out = self.fusion(H_cap, H_ctx)   # 無論 cross/concat 都走這裡
        H_out = self.norm(H_out)
        if not self.opt.no_txtnorm:
            H_out = l2norm(H_out, dim=-1)

        return H_out, L_cap



# ======================== 整合到 SCAN ========================
class RAGSCAN(object):
    def __init__(self, opt):
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.max_k_part = opt.K
        self.min_point_rate = opt.min_point_rate
        self.stage_1_epoch = opt.stage_1_epoch
        self.alpha = opt.alpha
        self.precomp_enc_type = opt.precomp_enc_type
        self.SEG_NUM = opt.SEG_NUM

        # 3D segmentation & embedding
        self.pointnet = PointNetDenseCls(
            opt.inp_size, opt.SEG_NUM, opt.img_dim, feature_transform=True
        )

        # image / part encoder
        self.img_enc = EncoderImage(
            opt.img_dim, opt.embed_size,
            precomp_enc_type=opt.precomp_enc_type,
            no_imgnorm=opt.no_imgnorm
        )

        self.txt_enc = RAGTextEncoder(opt)

        if torch.cuda.is_available():
            self.pointnet.cuda()
            self.img_enc.cuda()
            self.txt_enc.cuda()

        self.criterion = EMDLoss(opt=opt, margin=opt.margin)
        params = list(self.pointnet.parameters()) + \
                 list(self.img_enc.parameters()) + \
                 list(self.txt_enc.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0


    # ---------- 儲存 / 載入 ----------
    def state_dict(self):
        return [
            self.img_enc.state_dict(),
            self.txt_enc.base_txt_enc.state_dict(),
            self.txt_enc.fusion.state_dict(),
            self.txt_enc.proj3d.state_dict(),
            self.pointnet.state_dict()
        ]

    def load_state_dict(self, state):
        self.img_enc.load_state_dict(state[0])
        self.txt_enc.base_txt_enc.load_state_dict(state[1])
        self.txt_enc.fusion.load_state_dict(state[2])
        self.txt_enc.proj3d.load_state_dict(state[3])
        self.pointnet.load_state_dict(state[4])

    # ---------- mode ----------
    def train_start(self):
        self.img_enc.train()
        self.txt_enc.train()
        self.pointnet.train()

    def val_start(self):
        self.img_enc.eval()
        self.txt_enc.eval()
        self.pointnet.eval()

    # ---------- forward ----------
    def forward_emb(self, part_emb, captions, lengths,
                    original_texts, cur_epoch=None):
        if cur_epoch is None:
            cur_epoch = self.stage_1_epoch + 1

        img_emb = self.img_enc(part_emb)
        cap_emb, cap_lens = self.txt_enc(
            captions, lengths, original_texts,
            part_embed=part_emb, cur_epoch=cur_epoch
        )

        # Debug
        #print(f"[FORWARD_EMB] img_emb {img_emb.shape} {img_emb.dtype}")
        #print(f"[FORWARD_EMB] cap_emb {cap_emb.shape} {cap_emb.dtype}")

        return img_emb, cap_emb, cap_lens

    
    
    def forward_semantic_loss(self, pred, target, trans_feat):
        pred = pred.view(-1, self.SEG_NUM)
        target = target.view(-1, 1)[:, 0]
        loss = F.nll_loss(pred, target)
        loss += feature_transform_regularizer(trans_feat) * 0.001
        self.logger.update('Semantic Loss', loss.item(), pred.size(0))

        return loss

    def forward_loss(self, img_emb, cap_emb, cap_len, k_part, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, hardest = self.criterion(img_emb, cap_emb, cap_len, k_part)
        self.logger.update('Hardest Loss', hardest.item(), img_emb.size(0))
        self.logger.update('Matching Loss (Semi hard)', loss.item(), img_emb.size(0))
        return loss
    
    def forward_pointnet(self, points):
        points = points.transpose(2, 1)
        pred, embed, trans, trans_feat = self.pointnet(points)
        return pred, embed, trans_feat              # <─ 只回傳 3 個

    # ---------- mask / part 相關 ----------
    def point2part(self, point_embed, seg_idx):
        """將點雲嵌入平均成部分嵌入。"""
        from torch.nn.utils.rnn import pad_sequence
        seg_idx = seg_idx.detach()
        B, Pts  = seg_idx.shape
        D       = point_embed.size(-1)

        M = torch.zeros(B, self.SEG_NUM, Pts, device=point_embed.device)
        M.scatter_(1, seg_idx.unsqueeze(1), 1)
        M_mean  = F.normalize(M, p=1, dim=-1)
        all_part_embed = torch.bmm(M_mean, point_embed)   # (B, SEG_NUM, D)

        seg_cnt   = M.sum(dim=2)
        kth_cnt   = torch.kthvalue(
            seg_cnt, self.SEG_NUM + 1 - self.max_k_part, dim=1, keepdim=True
        ).values
        mask      = torch.logical_and(
            seg_cnt >= Pts * self.min_point_rate, seg_cnt > kth_cnt
        )
        k_part    = mask.sum(dim=1).tolist()

        packed = torch.masked_select(
            all_part_embed, mask.unsqueeze(-1)
        ).reshape(-1, D)

        if packed.numel() == 0:
            part_embed = torch.zeros(B, self.max_k_part, D,
                                     device=point_embed.device)
        else:
            part_embed = pad_sequence(packed.split(k_part, 0),
                                      batch_first=True)
        return part_embed, k_part

    def mask_emb(self, emb, n_len):
        if len(n_len) == 0:
            return emb
        B, max_n, _ = emb.shape
        mask = torch.arange(max_n, device=emb.device).expand(B, max_n)
        mask = mask >= torch.tensor(n_len, device=emb.device).unsqueeze(1)
        return emb.masked_fill(mask.unsqueeze(-1), 0)

    # ---------- 訓練一步 ----------
    def train_emb(self, batch, epoch):
        self.Eiters += 1
        self.optimizer.zero_grad()

        shapes          = batch["shapes"].cuda()
        captions        = batch["captions"].cuda()
        semantic_labels = batch["semantic_labels"].cuda()
        lengths         = batch["lengths"]
        original_texts  = batch["original_texts"]

        # 3D segmentation
        sem_pred, point_emb, trans_feat = self.forward_pointnet(shapes)
        loss = F.nll_loss(
            sem_pred.view(-1, self.SEG_NUM),
            semantic_labels.view(-1, 1)[:, 0]
        )
        loss += feature_transform_regularizer(trans_feat) * 0.001

        # Stage‑2 進行對齊損失
        if epoch >= self.stage_1_epoch:
            if self.precomp_enc_type == 'rgb':
                rgb = shapes[:, :, -3:]
                point_emb = torch.cat([point_emb, rgb], dim=-1)

            part_emb, k_part = self.point2part(point_emb, sem_pred.argmax(-1))
            img_emb, cap_emb, cap_len = self.forward_emb(
                part_emb, captions, lengths, original_texts, epoch
            )
            img_emb = self.mask_emb(img_emb, k_part)
            loss += self.alpha * self.criterion(
                img_emb, cap_emb, cap_len, k_part
            )[0]

        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()


# ==== 兼容舊 eval：轉出 EMD 工具 =====
from .model import (
    get_emd_score as _get_emd_score,
    lambda_softmax_func as _lambda_softmax_func,
    lambda_softmax_func_i2t as _lambda_softmax_func_i2t,
)

def get_emd_score(im, s, s_l, k_part, opt):
    """包一層轉呼叫，讓 eval.py 能找到舊接口"""
    return _get_emd_score(im, s, s_l, k_part, opt)

# 若你的程式其他地方也直接 import 這兩個 softmax，可以一併 re‑export
lambda_softmax_func = _lambda_softmax_func
lambda_softmax_func_i2t = _lambda_softmax_func_i2t
