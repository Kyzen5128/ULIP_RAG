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
from collections import OrderedDict

import timm
from torch import nn
from models.pointnet2.pointnet2 import Pointnet2_Ssg
from data.dataset_3d import  *

from models import losses
from torch.nn.parameter import Parameter
from easydict import EasyDict
import open_clip

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
    config_addr = './models/pointbert/PointTransformer_8192point.yaml'
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
    config_addr = './models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml'
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


# =====================================================================
# ============ START: RAG Adapter Integration Code ====================
# == 請將以下所有程式碼複製到 ULIP/models/ULIP_models.py 的文件末尾 ==
# =====================================================================

# 檢查必要的導入，如果文件頂部沒有，則需要添加
try:
    from .rag_adapter import RAGAdapter
except ImportError:
    # 這是為了讓程式碼片段在沒有 rag_adapter.py 時也能被 python 直譯器讀取
    # 實際運行前請務必創建 models/rag_adapter.py
    print("Warning: 'models.rag_adapter' not found. Please create it before running.")
    RAGAdapter = None

from .losses import ClipLoss

# ---------------------------------------------------------------------
# 1. 新的解耦損失函數 (Decoupled Loss Function)
# ---------------------------------------------------------------------
class ULIP_Loss_RAG_Decoupled(nn.Module):
    """
    為 RAG Adapter 設計的解耦損失函數。
    - 3D-Text 損失使用 RAG 增強後的文字特徵。
    - 2D-Text 損失使用原始的 CLIP 文字特徵，以保持穩定性。
    """
    def __init__(self,):
        super().__init__()
        self.clip_loss = ClipLoss()

    def forward(self, outputs, **kwargs):
        alpha = 0.5 # 超參數，用來平衡 2D 和 3D 損失
        
        # 從模型輸出中獲取所有必要的特徵
        enhanced_text_embed = outputs.get('enhanced_text_embed')
        clip_text_embed = outputs.get('clip_text_embed')
        pc_embed = outputs.get('pc_embed')
        image_embed = outputs.get('image_embed')
        logit_scale = outputs.get('logit_scale')
        
        if enhanced_text_embed is None or clip_text_embed is None or pc_embed is None:
            raise ValueError("Missing required embeddings in model output for loss calculation.")

        # 計算 3D-Text 損失 (使用增強特徵)
        loss_3d = self.clip_loss(enhanced_text_embed, pc_embed, logit_scale)
        
        total_loss = loss_3d
        
        # 計算 2D-Text 損失 (使用原始CLIP特徵)
        if image_embed is not None:
            loss_2d = self.clip_loss(clip_text_embed, image_embed, logit_scale)
            total_loss = total_loss + alpha * loss_2d
            
        return total_loss


# ---------------------------------------------------------------------
# 2. 帶有 RAG 適配器的新 ULIP 模型類
# ---------------------------------------------------------------------
class ULIP_RAG_ADAPTER(ULIP_WITH_IMAGE):
    """
    繼承自 ULIP_WITH_IMAGE，並集成了並行的 RAG 適配器。
    """
    def __init__(self, *args, **kwargs):
        # 從 kwargs 中提前取出 RAG 相關參數，避免傳給父類
        rag_corpus_dir = kwargs.pop('rag_corpus_dir', 'data/rag_corpus')
        rag_top_k = kwargs.pop('rag_top_k', 5)
        
        # 調用父類的 __init__ 方法，完成標準 ULIP 模型的初始化
        super().__init__(*args, **kwargs)
        
        print("Initializing ULIP with RAG Adapter...")
        if RAGAdapter is None:
            raise ImportError("RAGAdapter class not found. Please ensure 'models/rag_adapter.py' is implemented.")
            
        self.rag_adapter = RAGAdapter(
            corpus_dir=rag_corpus_dir,
            clip_embed_dim=self.text_projection.shape[1], # 通常是 512
            top_k=rag_top_k,
        )
        # 注意：這裡我們將 loss 的計算移交給 main.py 中的 criterion，
        # 這是 ULIP 專案的標準做法，所以不在模型內部持有 loss 實例。
        
    def encode_text_original(self, tokenized_text):
        """
        執行原始的 CLIP 文字編碼，得到未經 RAG 增強的特徵。
        """
        x = self.token_embedding(tokenized_text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, pc, text, image=None, **kwargs):
        """
        模型的前向傳播，實現解耦邏輯。
        """
        # 假設來自數據加載器的 text 是一個元組 (tokenized_ids, raw_string)
        if not isinstance(text, (list, tuple)) or len(text) != 2:
            raise ValueError("Input 'text' for ULIP_RAG_ADAPTER must be a tuple of (tokenized_tensor, raw_string_list)")
        tokenized_text, raw_text = text

        # --- 路徑1: 獲取原始 CLIP 文字特徵 ---
        clip_text_embed = self.encode_text_original(tokenized_text)

        # --- 路徑2: 獲取 RAG 增強文字特徵 ---
        if self.training:
            # 只在訓練時啟用 RAG 增強
            enhanced_text_embed = self.rag_adapter(raw_text, clip_text_embed)
        else:
            # 在評估/推理時，預設使用原始特徵以加快速度並保持一致性
            enhanced_text_embed = clip_text_embed

        # --- 編碼其他模態 ---
        pc_embed = self.encode_pc(pc)
        image_embed = self.encode_image(image) if image is not None else None

        # --- 構建輸出字典 ---
        # 返回所有需要的特徵，交給外部的損失函數進行計算
        return {
            'pc_embed': pc_embed,
            'image_embed': image_embed,
            'clip_text_embed': clip_text_embed,
            'enhanced_text_embed': enhanced_text_embed,
            'logit_scale': self.logit_scale.exp()
        }

# ---------------------------------------------------------------------
# 3. 完整的 RAG 模型工廠函數 (Factory Function)
# ---------------------------------------------------------------------
def ULIP_PointBERT_RAG(args):
    """
    創建帶有 RAG 適配器的 ULIP-PointBERT 模型的完整工廠函數。
    """
    # --- 這部分是從原始 ULIP_PointBERT 函數中複製的 ---
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    from models.pointbert.point_encoder import PointTransformer
    from utils.config import cfg_from_yaml_file
    
    config_addr = './models/pointbert/PointTransformer_8192point.yaml'
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)
    pc_feat_dims = 768
    # --- 原始邏輯結束 ---

    # 準備傳遞給 ULIP_RAG_ADAPTER 的所有參數
    model_kwargs = {
        'point_encoder': point_encoder,
        'vision_model': vision_model,
        'embed_dim': 512,
        'vision_width': 768,
        'context_length': 77,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12,
        'pc_feat_dims': pc_feat_dims,
        # RAG 特定參數
        'rag_corpus_dir': args.rag_corpus_dir,
        'rag_top_k': args.rag_top_k
    }

    # 創建我們的 RAG 模型
    model = ULIP_RAG_ADAPTER(**model_kwargs)

    # --- 權重載入邏輯 (與原始函數相同) ---
    if not args.evaluate_3d:
        print("Loading and freezing pre-trained SLIP weights...")
        pretrain_slip_model = torch.load('/mnt/P300/data/ULIP/ULIP-1/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            # 跳過 RAG 適配器和新的 PC 投影層，因為它們在預訓練模型中不存在
            if 'rag_adapter' in name or 'pc_projection' in name:
                continue

            if name not in pretrain_slip_model_params:
                print(f"Warning: parameter {name} not found in pre-trained model.")
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]
            
            # 凍結載入的參數
            param.requires_grad = False
            # print('load {} and freeze'.format(name))
            param.data.copy_(param_new)
    
    # --- (可選) 凍結主幹，只訓練 adapter ---
    if getattr(args, 'freeze_main_model', False):
        print("Freezing main model... Only training the RAG adapter.")
        for name, param in model.named_parameters():
            if 'rag_adapter' not in name:
                param.requires_grad = False
            else:
                # 確保 adapter 是可訓練的
                param.requires_grad = True
                print(f"Training enabled for: {name}")

    return model

# =====================================================================
# ============= END: RAG Adapter Integration Code =====================
# =====================================================================