# 檔案路徑: tricolo/loss/unified_rag_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedRAGLoss(nn.Module):
    def __init__(self, w_rag_contrast=0.5, w_orig_contrast=1.0):
        super().__init__()
        self.w_rag_contrast = w_rag_contrast
        self.w_orig_contrast = w_orig_contrast

    def _contrastive_loss(self, feat_a, feat_b, logit_scale):
        # 如果任一特徵為 None，則返回一個在正確設備上的零張量
        if feat_a is None or feat_b is None:
            # 確保返回的是張量，而不是 Python 的數字 0
            device = feat_a.device if feat_a is not None else feat_b.device if feat_b is not None else logit_scale.device
            return torch.tensor(0.0, device=device)
        
        labels = torch.arange(feat_a.shape[0], device=feat_a.device)
        logits = logit_scale * F.normalize(feat_a, dim=-1) @ F.normalize(feat_b, dim=-1).t()
        
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return (loss_a + loss_b) / 2

    def forward(self, outputs):
        # 從模型輸出中獲取所有需要的特徵和參數
        original_text = outputs.get('text_features')
        # 確保 total_loss 是一個在正確設備上的張量
        total_loss = torch.tensor(0.0, device=original_text.device)
        loss_dict = {}

        enhanced_text = outputs.get('enhanced_text_features')
        logit_scale = outputs.get('logit_scale')
        shape_modalities = {
            'image': outputs.get('image_features'),
            'voxel': outputs.get('voxel_features')
        }

        # --- RAG 增強文本特徵 vs 視覺特徵 ---
        # 只有在訓練階段且成功產生增強特徵時，enhanced_text 才不會是 None
        if enhanced_text is not None:
            for name, shape_feat in shape_modalities.items():
                rag_loss = self._contrastive_loss(enhanced_text, shape_feat, logit_scale)
                loss_dict[f'rag_text_{name}_loss'] = rag_loss
                total_loss += self.w_rag_contrast * rag_loss

        # --- 原始文本特徵 vs 視覺特徵 ---
        for name, shape_feat in shape_modalities.items():
            orig_loss = self._contrastive_loss(original_text, shape_feat, logit_scale)
            loss_dict[f'orig_text_{name}_loss'] = orig_loss
            total_loss += self.w_orig_contrast * orig_loss
        
        
        # 明確地檢查每個特徵是否「不是 None」，而不是對張量本身取布林值
        if all(feat is not None for feat in shape_modalities.values()):
        
            shape_loss = self._contrastive_loss(shape_modalities['image'], shape_modalities['voxel'], logit_scale)
            loss_dict['image_voxel_loss'] = shape_loss
            total_loss += self.w_orig_contrast * shape_loss

        loss_dict['loss'] = total_loss
        return loss_dict