#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data.py
@Time    :   2020/11/25 20:34:57
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   Data provider
'''

import json
import os
import pickle
import platform
import random
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data

def add_vocab(i2w, w2i, word):
    idx = len(i2w)+1
    i2w[idx] = word
    w2i[word] = idx

def rotate_point(point, rotation_angle):
    point = np.array(point)
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, 0, -sin_theta],
                                [0, 1, 0],
                                [sin_theta, 0, cos_theta]])
    rotated_point = np.dot(point.reshape(-1, 3), rotation_matrix)
    return rotated_point

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_path, data_split, shapenet_path, vocab, part_num, cfg):
        self.vocab = vocab
        # cfg for point cloud
        self.npoints = cfg.num_points
        self.pkl_path = os.path.join(shapenet_path, cfg.pkl_path)
        self.seg_num = cfg.SEG_NUM

        self.part_num = part_num

        # 讀取split
        if data_split == "train":
            self.is_train = True
            file_split = cfg.data_split["train_data"]

        elif data_split == "test" or data_split == "val":
            self.is_train = False
            file_split = cfg.data_split["test_data"]

        # 讀取點雲數據
        self.data_dic  = {}
        with open(self.pkl_path, 'rb') as f:
            self.pkl_data = pickle.load(f)
        self.modelid_data = [k for k in self.pkl_data.keys()]

        # 取兩個数据集的交集
        mids = []
        for sp in file_split:
            m = np.loadtxt(sp, dtype=str).tolist()
            for i in m:
                if i in self.modelid_data:
                    mids.append(i)
        
        # 讀取vocab映射文件
        with open('/'.join((shapenet_path, 'vocab/augmented_shapenet.json')), encoding='utf-8') as f:
            self.vocab_json = json.load(f)
        self.vocab_mapping = self.vocab_json['word_to_idx']
        self.i2w = self.vocab_json['idx_to_word']
        add_vocab(self.i2w, self.vocab_mapping, '<start>')
        add_vocab(self.i2w, self.vocab_mapping, '<end>')

        # 建立字典
        self.mid_cap_data = [] #[[mid, caption], ..]
        self.mid_cap = defaultdict(list)
        caps = self.vocab_json['captions']
        # --- 修正開始 ---
        # 建立一個新的字典來儲存處理過的 caption
        self.mid_cap_processed = defaultdict(list)

        # 遍歷原始 caption 資料
        for item in caps:
            model_id = item['model']
            caption_data = item['caption']
            
            # **關鍵修正點**：
            # 無論 caption_data 是單字列表、數字列表，還是混合列表，
            # 都先將其中所有元素轉換成字串，然後再 join 成一個完整的句子。
            try:
                # 將列表中的所有元素（無論是 int 還是 str）都安全地轉換成 str
                safe_caption_list = [str(token) for token in caption_data]
                processed_caption = " ".join(safe_caption_list)
                self.mid_cap_processed[model_id].append(processed_caption)
            except TypeError:
                # 如果 caption_data 根本不是一個列表（例如，只是一個單一字串），也進行處理
                if isinstance(caption_data, str):
                    self.mid_cap_processed[model_id].append(caption_data)

        # 使用處理過的 mid_cap_processed 來建立最終的 mid_cap_data
        self.mid_cap_data = []
        for i in mids:
            for j in self.mid_cap_processed[i]:
                # 這裡的 j 現在保證是一個完整的句子字串
                self.mid_cap_data.append([i, j])
        # --- 修正結束 ---

        self.length = len(self.mid_cap_data)


    def __getitem__(self, index):
        # caption_text 現在就是我們需要的原始句子字串
        model_id, caption_text = self.mid_cap_data[index]

        # 獲取點雲以及語義標註
        xyz_data, _, seg_anno_data = self.pkl_data[model_id]
        choice = np.random.choice(
            seg_anno_data.shape[0], self.npoints, replace=True)
        xyz_data_ = xyz_data[choice]
        seg_anno_data_ = seg_anno_data[choice]
        if self.is_train:
            # scale
            xyz_data_[:, :3] = xyz_data_[:, :3] * np.random.uniform(0.9, 1.1)
            # rotate
            rotate_angle = np.random.uniform(-np.pi/2, np.pi/2)
            rot_xyz = rotate_point(xyz_data_[:, :3], rotate_angle)
            xyz_data_[:, :3] = rot_xyz

        # normalize
        xyz_data_[:, 3:] = xyz_data_[:, 3:] - 0.5
        xyz_data_ = torch.from_numpy(xyz_data_).float()
        seg_anno_data_ = torch.from_numpy(seg_anno_data_).long()

        # Tokenize 的過程，是基於 caption_text
        caption_tokens = []
        caption_tokens.append(self.vocab_mapping['<start>'])
        # 將 caption_text 切分成單詞列表來進行 tokenize
        # 使用 .get(token, 0) 來處理未登錄詞彙，避免 KeyError
        caption_tokens.extend([self.vocab_mapping.get(token, 0) for token in caption_text.split()])
        caption_tokens.append(self.vocab_mapping['<end>'])
        target = torch.as_tensor(caption_tokens, dtype=torch.long)
        
        # 在回傳值中，新增 caption_text
        return xyz_data_, target, seg_anno_data_, index, model_id, caption_text

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    
    # 將 caption_texts 也從 data 中解壓縮出來
    xyzrgbs, captions, semantic_labels, ids, model_ids, caption_texts = zip(*data)

    xyzrgbs = torch.stack(xyzrgbs, 0)
    semantic_labels = torch.stack(semantic_labels, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths), dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # 將收集到的原始文本列表加入回傳的字典
    return {
        "shapes": xyzrgbs,
        "captions": targets,
        "semantic_labels": semantic_labels,
        "lengths": lengths,
        "ids": ids,
        "model_ids": model_ids,
        "original_texts": caption_texts # 這是新增的關鍵欄位
    }

def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, opt.shapenet_path, vocab, opt.K, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(vocab, batch_size, workers, opt):
    dpath = opt.data_path

    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    # 根據原 val.py 腳本，驗證集也使用 'test' split
    val_loader = get_precomp_loader(dpath, 'test', vocab, opt,
                                    batch_size, False, workers) # 驗證時通常不需隨機排序
    
    return train_loader, val_loader