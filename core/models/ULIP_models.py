'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
 

 ## This file will add the RAG mechanism to change the original CLIP model
 ## TODO: implement the RAG mechanism


'''

# Modified from github.com/openai/CLIP
import os as _os_anchor
# 2026-07-10 第三階段:錨定 core/ 根目錄,消除 cwd=core 依賴
_CORE_DIR = _os_anchor.path.dirname(_os_anchor.path.dirname(_os_anchor.path.abspath(__file__)))
from collections import OrderedDict

import timm
from torch import nn
from models.pointnet2.pointnet2 import Pointnet2_Ssg
from data.dataset_3d import  *

from models import losses
from torch.nn.parameter import Parameter
from easydict import EasyDict
import open_clip
import torch

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ULIP_WITH_IMAGE(nn.Module):
    def __init__(self, point_encoder, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.point_encoder = point_encoder

        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 512))
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

    def encode_image(self, image): ## clip image encoder
        x = self.visual(image)
        x = x @ self.image_projection

        return x

    def encode_text(self, text): ## clip text encoder
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def forward(self, pc, text, image=None):
        
        text_embed_all = []
        for i in range(text.shape[0]):
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)
        pc_embed = self.encode_pc(pc)
        if image is not None:
            image_embed = self.encode_image(image)
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'logit_scale': self.logit_scale.exp()}
            
            
class ULIP2_WITH_OPENCLIP(nn.Module):
    def __init__(self, point_encoder, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)

        self.open_clip_model = kwargs.open_clip_model

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.point_encoder = point_encoder
        
        self.tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 1280))
        nn.init.normal_(self.pc_projection, std=1280 ** -0.5)

    def encode_image(self, image):
        x = self.open_clip_model.encode_image(image)

        return x

    def encode_text(self, text):
        x = self.open_clip_model.encode_text(text)

        return x

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def forward(self, pc, text, image=None):

        text_embed_all = []
        for i in range(text.shape[0]):
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)
        pc_embed = self.encode_pc(pc)
        if image is not None:
            image_embed = self.encode_image(image)
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'logit_scale': self.logit_scale.exp()}


def get_loss(args):
    return losses.ULIPWithImageLoss()


def get_metric_names(model):
    return ['loss', 'ulip_loss', 'ulip_pc_image_acc', 'ulip_pc_text_acc']


def ULIP_PN_SSG(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    point_encoder = Pointnet2_Ssg()
    pc_feat_dims = 256
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('/mnt/P300/data/ULIP/ULIP-1/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model

def ULIP_PN_MLP(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointmlp.pointMLP import pointMLP
    point_encoder = pointMLP()
    pc_feat_dims = 256
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('/mnt/P300/data/ULIP/ULIP-1/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model

def ULIP_PointBERT(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointbert.point_encoder import PointTransformer
    config_addr = _os_anchor.path.join(_CORE_DIR, 'models/pointbert/PointTransformer_8192point.yaml')
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)
    pc_feat_dims = 768
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('/mnt/P300/data/ULIP/ULIP-1/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'), weights_only=False)
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model

def ULIP2_PointBERT_Colored(args):
    print("Get openclip model:")
    open_clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14',
                                                                          pretrained='laion2b_s39b_b160k')
    open_clip_model.eval()
    print("Finished loading the openclip model.")

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointbert.point_encoder import PointTransformer, PointTransformer_Colored
    config_addr = _os_anchor.path.join(_CORE_DIR, 'models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml')
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer_Colored(config.model, args=args)
    pc_feat_dims = 768
    # =====================================================================

    model = ULIP2_WITH_OPENCLIP(open_clip_model=open_clip_model, point_encoder=point_encoder, pc_feat_dims=pc_feat_dims)

    return model

def ULIP_PN_NEXT(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointnext.pointnext import PointNEXT
    point_encoder = PointNEXT()
    pc_feat_dims = 256
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('/mnt/P300/data/ULIP/ULIP-1/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'), weights_only=False)
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model


def ULIP_CUSTOMIZED(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # This is a sample template to pre-train your customized 3D backbones, please modify this part accordingly!
    from models.customized_backbone.customized_backbone import CUSTOMIZED_BACKBONE
    point_encoder = CUSTOMIZED_BACKBONE()
    # We assume you might have different point cloud output feature dimension,
    # we added a projecting layer to unify the point cloud output dimension before doing the multimodal alignment,
    # please change the output feature dimension here.
    pc_feat_dims = 512
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('/mnt/P300/data/ULIP/ULIP-1/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'), weights_only=False)
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model





# =====================================================================
# ============ START: RAG   =================
# =====================================================================

import torch.nn.functional as F

import torch
import torch.nn as nn
import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    """這個類別只負責從 FAISS 索引中檢索文件ID,不進行神經網路編碼"""
    def __init__(self, corpus_dir, top_k=5, device='cuda'):
        self.top_k = top_k
        index_path = os.path.join(corpus_dir, 'corpus_index.faiss')
        print(f"INFO: RAGRetriever - Loading FAISS index from '{index_path}'...")
        self.index = faiss.read_index(index_path)
        # 用於查詢的輕量級編碼器
        self.query_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("INFO: RAGRetriever is ready.")

    @torch.no_grad()
    def retrieve_docs(self, raw_text_queries, corpus):
        if not raw_text_queries: 
            return [[] for _ in raw_text_queries]
        
        processed_queries = []
        for query in raw_text_queries:
            if isinstance(query, list) and len(query) > 0:
                processed_queries.append(query[0])  # 取第一個元素
            elif isinstance(query, str):
                processed_queries.append(query)
            else:
                processed_queries.append("")  # 空字符串作為默認值
        
        
        # 檢查是否有有效的查詢
        if not processed_queries or all(not q.strip() for q in processed_queries):
            print("WARNING: No valid queries found, returning empty results")
            return [[] for _ in raw_text_queries]
        
        #  現在傳遞處理過的字符串列表給 sentence_transformers
        query_embeddings = self.query_encoder.encode(processed_queries, convert_to_numpy=True, show_progress_bar=False)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        _, indices = self.index.search(query_embeddings.astype('float32'), self.top_k)
        
        # 返回每個查詢對應的文檔列表
        retrieved_docs = []
        for i, query_indices in enumerate(indices):
            query_docs = [corpus[idx]['text'] for idx in query_indices if idx != -1]
            retrieved_docs.append(query_docs)
        
        return retrieved_docs

class RAGEnhancer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU() if hasattr(nn, 'GELU') else nn.ReLU(), # 兼容舊版 torch
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, original_feature_eot, doc_features, doc_padding_mask=None):
        """
        doc_padding_mask: [B, K] bool，True = 該位是補零 padding（檢索不足 top_k 時）。
        修正：原版沒傳 key_padding_mask，全零 key 會參與 softmax、稀釋注意力。
        不影響權重相容性（無新參數；舊 checkpoint 照常載入）。
        """
        original_expanded = original_feature_eot.unsqueeze(1)
        context = torch.cat([original_expanded, doc_features], dim=1)
        key_padding_mask = None
        if doc_padding_mask is not None:
            # context 第 0 位是 original 本身，永不遮罩；後 K 位依 doc_padding_mask
            never_mask = torch.zeros(doc_padding_mask.size(0), 1,
                                     dtype=torch.bool, device=doc_padding_mask.device)
            key_padding_mask = torch.cat([never_mask, doc_padding_mask], dim=1)
        attn_output, _ = self.attention(original_expanded, context, context,
                                        key_padding_mask=key_padding_mask)
        fused = self.norm1(original_expanded + attn_output)
        fused = self.norm2(fused + self.ffn(fused))
        return fused.squeeze(1)


# --- 1. Loss 函數 ---
class ULIP_Loss_RAG_Enhanced(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.training_strategy = args.training_strategy

    def _generate_labels(self, batch_size, device):
        return torch.arange(batch_size, device=device)

    def _contrastive_loss_with_acc(self, feat_a, feat_b, logit_scale, labels):
        """
        計算雙向對比損失和準確率 
        
        Args:
            feat_a: 特徵A [B, D]
            feat_b: 特徵B [B, D] 
            logit_scale: 溫度參數
            labels: 標籤 [B] = [0, 1, 2, ..., B-1]
            
        Returns:
            loss: 雙向對比損失
            acc: 平均準確率
        """
        # 計算相似度矩陣
        logits_a_to_b = logit_scale * feat_a @ feat_b.t()  # [B, B]
        logits_b_to_a = logit_scale * feat_b @ feat_a.t()  # [B, B]
        
        # 對比學習損失 (InfoNCE)
        loss_a_to_b = F.cross_entropy(logits_a_to_b, labels)
        loss_b_to_a = F.cross_entropy(logits_b_to_a, labels)
        loss = (loss_a_to_b + loss_b_to_a) / 2
        
        # 準確率計算
        with torch.no_grad():
            pred_a = torch.argmax(logits_a_to_b, dim=-1)
            acc_a = 100 * pred_a.eq(labels).sum().float() / labels.size(0)
            
            pred_b = torch.argmax(logits_b_to_a, dim=-1)
            acc_b = 100 * pred_b.eq(labels).sum().float() / labels.size(0)
            
            avg_acc = (acc_a + acc_b) / 2
        
        return loss, avg_acc

    def forward(self, outputs, **kwargs):
        """
        主要的forward函數,根據訓練階段選擇不同的損失計算
        """
        # 確定batch size和設備
        first_tensor = next(iter(outputs.values()))
        batch_size = first_tensor.size(0)
        device = first_tensor.device
        labels = self._generate_labels(batch_size, device)
        
        if self.training_strategy in ['staged_1', 'stage_1']:
            return self._stage1_loss(outputs, labels)
        elif self.training_strategy in ['staged_2', 'stage_2']:
            return self._stage2_loss(outputs, labels)
        else:
            raise ValueError(f"Unknown training strategy: {self.training_strategy}")

    def _stage1_loss(self, outputs, labels):
        """
        Stage 1: RAG 一致性訓練
        目標: 讓RAG增強的文本特徵學會利用外部知識,同時保持與原始文本的一致性
        """
        enhanced_text = F.normalize(outputs['enhanced_text_embed'], dim=-1, p=2)
        original_text = F.normalize(outputs['original_text_embed'], dim=-1, p=2)
        image_embed = F.normalize(outputs['image_embed'], dim=-1, p=2)
        logit_scale = outputs['logit_scale']
        
        # 1. RAG一致性損失: 增強文本應該保持原始文本的語義
        consistency_loss = F.mse_loss(enhanced_text, original_text.detach())
        
        # 2. 增強文本與圖像的對比學習
        enhanced_contrast_loss, enhanced_image_acc = self._contrastive_loss_with_acc(
            enhanced_text, image_embed, logit_scale, labels
        )
        
        # 3. 原始文本與圖像的對比學習 (作為參考基準)
        original_contrast_loss, original_image_acc = self._contrastive_loss_with_acc(
            original_text, image_embed, logit_scale, labels
        )
        
        # 總損失: 主要優化一致性,輔助優化對比學習
        total_loss = consistency_loss + 0.5 * enhanced_contrast_loss
        
        return {
            'loss': total_loss,
            'consistency_loss': consistency_loss,
            'enhanced_contrast_loss': enhanced_contrast_loss,
            'original_contrast_loss': original_contrast_loss,
            'enhanced_image_acc': enhanced_image_acc,
            'original_image_acc': original_image_acc,
            # 為了與訓練腳本兼容,添加這些標準指標
            'ulip_loss': total_loss,
            'ulip_pc_image_acc': enhanced_image_acc,
            'ulip_pc_text_acc': enhanced_image_acc
        }

    def _stage2_loss(self, outputs, labels):
        """
        Stage 2: 3D-Text 對齊訓練
        目標: 讓3D點雲特徵對齊到RAG增強的文本特徵空間
        """
        pc_embed = F.normalize(outputs['pc_embed'], dim=-1, p=2)
        enhanced_text = F.normalize(outputs['enhanced_text_embed'], dim=-1, p=2)
        image_embed = F.normalize(outputs['image_embed'], dim=-1, p=2)
        logit_scale = outputs['logit_scale']
        
        # 1. 3D點雲與RAG增強文本的對比學習 (主要目標)
        pc_text_loss, pc_text_acc = self._contrastive_loss_with_acc(
            pc_embed, enhanced_text, logit_scale, labels
        )
        
        # 2. 3D點雲與圖像的對比學習 (保持原有的多模態能力)
        pc_image_loss, pc_image_acc = self._contrastive_loss_with_acc(
            pc_embed, image_embed, logit_scale, labels
        )
        
        # 總損失: 主要優化3D-文本對齊,輔助保持3D-圖像對齊
        total_loss = pc_text_loss + 0.5 * pc_image_loss
        
        return {
            'loss': total_loss,
            'pc_text_loss': pc_text_loss,
            'pc_image_loss': pc_image_loss,
            # 與原始ULIP保持一致的指標名稱
            'ulip_loss': total_loss,
            'ulip_pc_text_acc': pc_text_acc,
            'ulip_pc_image_acc': pc_image_acc
        }

# --- 2. RAG 特徵增強模型 ---
class ULIP_with_RAG_Enhancer(ULIP_WITH_IMAGE):
    def __init__(self, *args, **kwargs):
        rag_corpus_dir = kwargs.pop('rag_corpus_dir', None) or _os_anchor.path.join(_CORE_DIR, 'rag_corpus')  # 舊預設 data/rag_corpus 在 3090 不存在
        rag_top_k = kwargs.pop('rag_top_k', 5)
        # 必須在 super().__init__ 之前 pop,避免傳給父類
        super().__init__(*args, **kwargs)
        
        self.retriever = RAGRetriever(corpus_dir=rag_corpus_dir, top_k=rag_top_k)
        self.rag_enhancer = RAGEnhancer(embed_dim=self.transformer.width)
        from utils.tokenizer import SimpleTokenizer
        self.tokenizer = SimpleTokenizer() # 需要一個 tokenizer 來處理檢索到的文件
        # 將語料庫載入到模型中,以便在 forward 中使用
        corpus_path = os.path.join(rag_corpus_dir, 'rag_corpus.jsonl')
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.rag_corpus = [json.loads(line) for line in f]

    def encode_text_base(self, text_tokens):
        """一個輔助函數,返回投影前的基礎特徵"""
        x = self.token_embedding(text_tokens)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        return x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]

    def encode_text_with_rag(self, text_tokens, raw_text_queries):
        # 1. 得到原始文本的基礎特徵 (投影前)
        base_feature_eot = self.encode_text_base(text_tokens)

        # 2. 檢索文件
        retrieved_docs = self.retriever.retrieve_docs(raw_text_queries, self.rag_corpus)
        flat_docs = [doc for sublist in retrieved_docs for doc in sublist]

        if flat_docs:
            # 3.使用 同一個 編碼器來編碼檢索到的文件
            doc_tokens = self.tokenizer(flat_docs).to(text_tokens.device)
            base_doc_features = self.encode_text_base(doc_tokens)
            
            # 重新組織為 [B, K, D]，並記錄 padding 位置（True=補零）供 attention 遮罩
            k = self.retriever.top_k
            doc_features_padded = torch.zeros(len(raw_text_queries), k, base_doc_features.shape[1], device=base_doc_features.device)
            doc_padding_mask = torch.ones(len(raw_text_queries), k,
                                          dtype=torch.bool, device=base_doc_features.device)
            current_pos = 0
            for i, sublist in enumerate(retrieved_docs):
                if sublist:
                    num_docs = len(sublist)
                    doc_features_padded[i, :num_docs] = base_doc_features[current_pos : current_pos + num_docs]
                    doc_padding_mask[i, :num_docs] = False
                    current_pos += num_docs

            # 4. 使用 RAGEnhancer 注入知識,得到融合後的基礎特徵（padding 位不參與 softmax）
            fused_feature_eot = self.rag_enhancer(base_feature_eot, doc_features_padded, doc_padding_mask)
        else:
            # 如果沒有檢索到任何文件,則直接使用原始特徵
            fused_feature_eot = base_feature_eot
        
        # 5. 將融合後的特徵,通過【同一個】CLIP 最終投影層
        enhanced_text_embed = fused_feature_eot @ self.text_projection
        return enhanced_text_embed

    def forward(self, pc, text, image=None, **kwargs):
        # 解包 RAG 格式的文本輸入
        tokenized_text, raw_text = text
                
        if tokenized_text.dim() == 3: 
            tokenized_text = tokenized_text.squeeze(1)

        # 獲取 RAG 增強文本特徵
        enhanced_text_embed = self.encode_text_with_rag(tokenized_text, raw_text)
        
        # 獲取其他模態特徵
        pc_embed = self.encode_pc(pc) if pc is not None else None
        image_embed = self.encode_image(image) if image is not None else None

        result = {
            'enhanced_text_embed': enhanced_text_embed,
            'pc_embed': pc_embed,
            'image_embed': image_embed,
            'logit_scale': self.logit_scale.exp()
        }
        
        # 訓練時添加原始文本特徵（損失函數會根據階段自動使用）
        if self.training:
            result['original_text_embed'] = self.encode_text(tokenized_text)
        
        return result

# --- 3. 工廠函數 ---
def ULIP_PointBERT_RAG(args):
    from utils.config import cfg_from_yaml_file
    from models.pointbert.point_encoder import PointTransformer
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    config_addr = _os_anchor.path.join(_CORE_DIR, 'models/pointbert/PointTransformer_8192point.yaml')
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)

    model_kwargs = {
        'point_encoder': point_encoder, 'vision_model': vision_model,
        'embed_dim': 512, 'vision_width': 768, 'context_length': 77, 'vocab_size': 49408,
        'transformer_width': 512, 'transformer_heads': 8, 'transformer_layers': 12,
        'pc_feat_dims': 768, 'rag_corpus_dir': args.rag_corpus_dir, 'rag_top_k': args.rag_top_k
    }
    
    # 創建 RAG 增強版的模型
    model = ULIP_with_RAG_Enhancer(**model_kwargs)
    
    # 載入預訓練的 SLIP 權重到骨幹網路
    if not getattr(args, 'evaluate_3d', False) and not getattr(args, 'resume', ''):
        print("INFO: Loading SLIP pre-trained weights for backbone...")
        try:
            slip_weights = torch.load('/mnt/P300/data/ULIP/ULIP-1/initialize_models/slip_base_100ep.pt', map_location='cpu',weights_only=False)['state_dict']
            slip_weights = {k.replace('module.', ''): v for k, v in slip_weights.items()}
            model.load_state_dict(slip_weights, strict=False)
            print("INFO: SLIP weights loaded successfully.")
        except FileNotFoundError:
            print("WARNING: Pre-trained SLIP model not found. Skipping weight loading.")


    print(f"INFO: Applying RAG freezing strategy for '{args.training_strategy}'")
    
    # 首先,預設凍結所有參數
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    #  支援 staged_1/stage_1 和 staged_2/stage_2 兩種命名
    if args.training_strategy in ['staged_1', 'stage_1']:
        print("--- INFO: STAGE 1 - Training rag_enhancer only ---")
        for param in model.rag_enhancer.parameters():
            param.requires_grad = True
            
    elif args.training_strategy in ['staged_2', 'stage_2']:
        print("--- INFO: STAGE 2 - Training point_encoder and pc_projection only ---")
        for param in model.point_encoder.parameters():
            param.requires_grad = True
        model.pc_projection.requires_grad = True
        
        # 載入 Stage 1 訓練好的 rag_enhancer 權重
        if hasattr(args, 'stage1_ckpt_path') and args.stage1_ckpt_path and os.path.isfile(args.stage1_ckpt_path):
            print(f"--- INFO: Loading Stage 1 checkpoint from '{args.stage1_ckpt_path}' ---")
            ckpt = torch.load(args.stage1_ckpt_path, map_location='cpu', weights_only=False)
            
            #  rag_enhancer 權重的載入
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
            
            # 提取 rag_enhancer 的權重並移除前綴
            enhancer_weights = {}
            for k, v in state_dict.items():
                if 'rag_enhancer' in k:
                    # 移除 'module.' 和 'rag_enhancer.' 前綴
                    new_key = k.replace('module.', '').replace('rag_enhancer.', '')
                    enhancer_weights[new_key] = v
            
            if enhancer_weights:
                missing_keys, unexpected_keys = model.rag_enhancer.load_state_dict(enhancer_weights, strict=False)
                if missing_keys:
                    print(f"--- WARNING: Missing keys in rag_enhancer: {missing_keys} ---")
                if unexpected_keys:
                    print(f"--- WARNING: Unexpected keys in rag_enhancer: {unexpected_keys} ---")
                print("--- INFO: Stage 1 rag_enhancer weights loaded successfully. ---")
            else:
                print(f"WARNING: Could not find 'rag_enhancer' weights in '{args.stage1_ckpt_path}'.")
                print("Available keys:", [k for k in state_dict.keys() if 'rag_enhancer' in k])
        else:
            print("WARNING: Stage 2 started without a valid --stage1_ckpt_path. rag_enhancer will use initial weights.")
            
    else:
        #  更清楚的錯誤信息
        print(f"ERROR: Unknown training strategy '{args.training_strategy}'.")
        print("Supported strategies: 'staged_1', 'stage_1', 'staged_2', 'stage_2'")
        raise ValueError(f"Unsupported training strategy: {args.training_strategy}")

    # 檢查並報告可訓練的參數
    trainable_params = []
    total_params = 0
    trainable_count = 0
    
    print("\n--- Trainable Parameters ---")
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params.append(name)
            trainable_count += param.numel()
            #print(f"  ✓ {name} ({param.numel():,} params)")
    
    print(f"\nTrainable: {trainable_count:,} / {total_params:,} parameters ({100*trainable_count/total_params:.2f}%)")
    print("--------------------------\n")
    
    #  安全檢查：
    if trainable_count == 0:
        raise RuntimeError(f"No trainable parameters found! Strategy '{args.training_strategy}' resulted in all parameters being frozen.")
            
    return model
# =====================================================================
# ================= END: RAG  =========================================
# =====================================================================









