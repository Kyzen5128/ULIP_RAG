from tricolo.evaluation.eval_retrieval import compute_metrics
from tricolo.model.module.rag_module import RAGRetriever, RAGEnhancer
from itertools import combinations
import lightning.pytorch as pl
import numpy as np
import pickle
import hydra
import clip
import os
import torch
import torch.nn as nn
import json
from tricolo.data.tokenizer import SimpleTokenizer
import random

class TriCoLoNet(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        # 檢查是否啟用 RAG
        self.use_rag = self.hparams.cfg.model.get('use_rag', False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # initialize model modules
        self.image_encoder = None
        self.voxel_encoder = None
        clip_model = None
        if cfg.model.image_encoder == "CLIPImageEncoder" or cfg.model.text_encoder == "CLIPTextEncoder":
            clip_model = clip.load(cfg.model.modules.clip_model, device=self.device)[0]
            # freeze CLIP
            for param in clip_model.parameters():
                param.requires_grad = False

        self.text_encoder = hydra.utils.instantiate(
            getattr(cfg.model.modules, cfg.model.text_encoder), clip_model=clip_model
        )

        if cfg.model.image_encoder is not None:
            self.image_encoder = hydra.utils.instantiate(
                getattr(cfg.model.modules, cfg.model.image_encoder), clip_model=clip_model
            )
        if cfg.model.voxel_encoder is not None:
            self.voxel_encoder = hydra.utils.instantiate(
                getattr(cfg.model.modules, cfg.model.voxel_encoder)
            )
            
        # --- 如果啟用 RAG，則初始化 RAG 相關模組 ---
        if self.use_rag:
            print("INFO: RAG mode is enabled.")
            self.retriever = RAGRetriever(
                corpus_dir=self.hparams.cfg.data.rag_corpus_dir, 
                top_k=self.hparams.cfg.model.rag_top_k
            )
            corpus_path = os.path.join(self.hparams.cfg.data.rag_corpus_dir, 'rag_corpus.jsonl')
            with open(corpus_path, 'r', encoding='utf-8') as f:
                self.rag_corpus = [json.loads(line) for line in f]
            
            self.rag_enhancer = RAGEnhancer(embed_dim=self.hparams.cfg.model.out_dim)
            self.rag_tokenizer = SimpleTokenizer()

        # initialize loss functions
        loss_cfg = cfg.loss.get(cfg.loss.name)
        self.loss_fn = hydra.utils.instantiate(loss_cfg)
        self.val_test_step_outputs = []

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.cfg.optimizer, params=self.parameters())


    def forward(self, data_dict):
        # 提取基礎特徵
        original_text_features = self.text_encoder(data_dict["tokens"], data_dict)
        
        output_dict = {
            "text_features": original_text_features
        }

        # --- RAG 邏輯：只在訓練時且啟用 RAG 時執行 ---
        if self.use_rag and self.training:
            raw_text_queries = data_dict["text"] # 假設 dataloader 提供原始文本
            
            # 1. 檢索文件
            retrieved_docs = self.retriever.retrieve_docs(raw_text_queries, self.rag_corpus)
            flat_docs = [doc for sublist in retrieved_docs for doc in sublist]

            doc_features_padded = torch.zeros(
                len(raw_text_queries), self.retriever.top_k, self.hparams.cfg.model.out_dim, 
                device=original_text_features.device
            )

            # 2. 如果成功檢索到文件，則編碼並填充
            if flat_docs:
                doc_tokens = self.rag_tokenizer(flat_docs).to(original_text_features.device)
                # 使用 text_encoder 來編碼檢索到的文件
                doc_features = self.text_encoder(doc_tokens, {})
                
                # --- 更健壯的填充邏輯 ---
                doc_idx_counter = 0
                for i, docs_for_query in enumerate(retrieved_docs):
                    num_docs = len(docs_for_query)
                    if num_docs > 0:
                        doc_features_padded[i, :num_docs] = doc_features[doc_idx_counter : doc_idx_counter + num_docs]
                        doc_idx_counter += num_docs
            
            # 3. 呼叫 Enhancer 進行特徵融合 (在迴圈外，只呼叫一次)
            enhanced_text_features = self.rag_enhancer(original_text_features, doc_features_padded)
            output_dict["enhanced_text_features"] = enhanced_text_features

        # --- 編碼其他模態 (保持不變) ---
        if self.image_encoder is not None:
            output_dict["image_features"] = self.image_encoder(data_dict["images"].flatten(end_dim=1), data_dict)
        if self.voxel_encoder is not None:
            output_dict["voxel_features"] = self.voxel_encoder(data_dict["voxels"], len(data_dict["model_id"]))

        output_dict["logit_scale"] = self.logit_scale.exp()
        return output_dict




    def _calculate_losses(self, output_dict, loss_prefix):
        loss_dict = {}
        # get all combinations to prepare contrastive losses
        all_combinations = combinations(output_dict.keys(), 2)
        # calculate contrastive losses
        for key_combination in all_combinations:
            loss_name = f"{loss_prefix}/{key_combination[0][:-9]}_{key_combination[1][:-9]}_loss"
            loss_dict[loss_name] = self.loss_fn(output_dict[key_combination[0]], output_dict[key_combination[1]])
        loss_dict[f"{loss_prefix}/total_loss"] = sum(loss_dict.values())
        return loss_dict

    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self.loss_fn(output_dict)

        log_dict = {f'train/{k}': v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True)

        return loss_dict["loss"]

    def validation_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict = self.loss_fn(output_dict)

        log_dict = {f'val/{k}': v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)

        for key, value in output_dict.items():
            output_dict[key] = value.cpu().numpy()

        reduced_data_dict = {
            "model_id": data_dict["model_id"],
            "category": data_dict["category"],
            "tokens": data_dict["tokens"].cpu().numpy()
        }

        self.val_test_step_outputs.append((reduced_data_dict, output_dict))


    def on_validation_epoch_end(self):
        embeddings_dict = self._collate_output()
        self.val_test_step_outputs.clear()
        pr_at_k = compute_metrics(self.hparams.cfg.data.dataset, embeddings_dict)
        self.log('val_eval/RR@1', pr_at_k["recall_rate"][0] *100 )
        self.log('val_eval/RR@5', pr_at_k["recall_rate"][4] *100)
        self.log('val_eval/NDCG@5', pr_at_k["ndcg"][4] * 100)
        self.log("val_eval/MRR", pr_at_k["mrr"] * 100)

    def test_step(self, data_dict, idx):
        output_dict = self(data_dict)
        # clean and move data to cpu for evaluation
        for key, value in output_dict.items():
            output_dict[key] = value.cpu().numpy()

        reduced_data_dict = {
            "model_id": data_dict["model_id"],
            "category": data_dict["category"],
            "tokens": data_dict["tokens"].cpu().numpy()
        }
        self.val_test_step_outputs.append((reduced_data_dict, output_dict))

    def on_test_epoch_end(self):
        embeddings_dict = self._collate_output()
        self.val_test_step_outputs.clear()
        if self.hparams.cfg.inference.evaluate:
            _ = compute_metrics(self.hparams.cfg.data.dataset, embeddings_dict, print_results=True)

        if self.hparams.cfg.inference.save_predictions:
            save_output_path = os.path.join(self.hparams.cfg.inference.output_dir, 'output.p')
            with open(save_output_path, 'wb') as f:
                pickle.dump(embeddings_dict, f)
            self.print(f"\nPredictions saved at {save_output_path}")


    def _collate_output(self):
        text_features_stacked = []
        shape_features_stacked = []
        model_ids_stacked = []
        category_list_stacked = []
        # caption_indices_stacked = []
        captions = []

        for data_dict, output_dict in self.val_test_step_outputs:
            text_features_stacked.append(output_dict["text_features"])
            shape_features = np.zeros_like(output_dict["text_features"])
            if "image_features" in output_dict:
                shape_features += output_dict["image_features"]
            if "voxel_features" in output_dict:
                shape_features += output_dict["voxel_features"]
            shape_features_stacked.append(shape_features)
            model_ids_stacked.extend(data_dict["model_id"])
            category_list_stacked.extend(data_dict["category"])
            # captions.append(data_dict["text"])
            # for cap in data_dict["tokens"]:
            #     caption_indices_stacked.append(cap)

        text_features_stacked = np.vstack(text_features_stacked)
        shape_features_stacked = np.vstack(shape_features_stacked)

        embeddings_dict = {"caption_embedding_tuples": []}
        for i in range(text_features_stacked.shape[0]):
            # TODO
            embeddings_dict["caption_embedding_tuples"].append(
                (
                    None, category_list_stacked[i], model_ids_stacked[i], text_features_stacked[i], shape_features_stacked[i]
                )
            )
        return embeddings_dict
