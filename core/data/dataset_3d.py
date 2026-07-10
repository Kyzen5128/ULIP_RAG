'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

import random

import torch
import numpy as np
import torch.utils.data as data

import yaml
from easydict import EasyDict

from utils.io import IO
from utils.build import DATASETS
from utils.logger import *
from utils.build import build_dataset_from_cfg
import json
from tqdm import tqdm
import pickle
from PIL import Image
import os as _os_anchor
# 2026-07-10 第三階段:以本檔位置錨定 core/ 根目錄,消除「必須 cwd=core」的隱性依賴
_CORE_DIR = _os_anchor.path.dirname(_os_anchor.path.dirname(_os_anchor.path.abspath(__file__)))

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

@DATASETS.register_module()
class ModelNet(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.generate_from_raw_data = False
        split = config.subset
        self.subset = config.subset
        self.use_10k_pc = config.use_10k_pc
        self.use_colored_pc = config.use_colored_pc

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger='ModelNet')

        if self.uniform:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                # make sure you have raw data in the path before you enable generate_from_raw_data=True.
                if self.generate_from_raw_data:
                    print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
                    self.list_of_points = [None] * len(self.datapath)
                    self.list_of_labels = [None] * len(self.datapath)

                    for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                        fn = self.datapath[index]
                        cls = self.classes[self.datapath[index][0]]
                        cls = np.array([cls]).astype(np.int32)
                        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                        if self.uniform:
                            point_set = farthest_point_sample(point_set, self.npoints)
                            print_log("uniformly sampled out {} points".format(self.npoints))
                        else:
                            point_set = point_set[0:self.npoints, :]

                        self.list_of_points[index] = point_set
                        self.list_of_labels[index] = cls

                    with open(self.save_path, 'wb') as f:
                        pickle.dump([self.list_of_points, self.list_of_labels], f)
                else:
                    # no pre-processed dataset found and no raw data found, then load 8192 points dataset then do fps after.
                    self.save_path = os.path.join(self.root,
                                                  'modelnet%d_%s_%dpts_fps.dat' % (
                                                  self.num_category, split, 8192))
                    print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                    if not self.use_10k_pc:
                        print_log('since no exact points pre-processed dataset found and no raw data found, load 8192 pointd dataset first, if downsampling with fps to {} happens later, the speed is excepted to be slower due to fps...'.format(self.npoints), logger='ModelNet')
                    with open(self.save_path, 'rb') as f:
                        self.list_of_points, self.list_of_labels = pickle.load(f)

            else:
                print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

        self.shape_names_addr = os.path.join(self.root, 'modelnet40_shape_names.txt')
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        self.shape_names = lines

        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        
        if self.use_10k_pc and self.use_colored_pc:
            self.modelnet_10k_colored_pc_file = '/mnt/P300/data/ULIP/ULIP-1/modelnet40_normal_resampled/modelnet40_colored_10k_pc.npy'
            self.modelnet_10k_rgb_data = np.load(self.modelnet_10k_colored_pc_file, allow_pickle=True)
            with open('/mnt/P300/data/ULIP/ULIP-1/modelnet40_normal_resampled/modelnet40_test_split_10k_colored.json', 'r') as f:
                self.cat_name = json.load(f)

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        if  self.npoints < point_set.shape[0]:
            point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height:
            self.gravity_dim = 1
            height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1] - point_set[:,
                                                                            self.gravity_dim:self.gravity_dim + 1].min()
            point_set = np.concatenate((point_set, height_array), axis=1)

        if self.use_10k_pc and self.use_colored_pc:
            point_set = self.modelnet_10k_rgb_data[index]['xyz']
            rgb_data = np.ones_like(point_set) * 0.4
            point_set = np.concatenate([point_set, rgb_data], axis=1)
            cat_name = self.cat_name[index]['category']
            label = [self.shape_names.index(cat_name)]
        elif self.use_colored_pc:
            rgb_data = np.ones_like(point_set) * 0.4
            point_set = np.concatenate([point_set, rgb_data], axis=1)

        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label_name = self.shape_names[int(label)]

        return current_points, label, label_name

# @DATASETS.register_module()
# class ShapeNet(data.Dataset):
#     def __init__(self, config):

#         self.data_root = config.DATA_PATH
#         self.pc_path = config.PC_PATH
#         self.subset = config.subset
#         self.npoints = config.npoints
#         self.tokenizer = config.tokenizer
#         self.train_transform = config.train_transform
#         self.id_map_addr = os.path.join(config.DATA_PATH, 'taxonomy.json')
#         self.rendered_image_addr = config.IMAGE_PATH
#         self.picked_image_type = ['', '_depth0001']
#         self.picked_rotation_degrees = list(range(0, 360, 12))
#         self.picked_rotation_degrees = [(3 - len(str(degree))) * '0' + str(degree) if len(str(degree)) < 3 else str(degree) for degree in self.picked_rotation_degrees]

#         with open(self.id_map_addr, 'r') as f:
#             self.id_map = json.load(f)

#         self.prompt_template_addr = os.path.join(_CORE_DIR, 'data/configs/templates.json')
#         with open(self.prompt_template_addr) as f:
#             self.templates = json.load(f)[config.pretrain_dataset_prompt]

#         self.synset_id_map = {}
#         for id_dict in self.id_map:
#             synset_id = id_dict["synsetId"]
#             self.synset_id_map[synset_id] = id_dict

#         self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
#         test_data_list_file = os.path.join(self.data_root, 'test.txt')

#         self.sample_points_num = self.npoints
#         self.whole = config.get('whole')

#         print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
#         print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
#         with open(self.data_list_file, 'r') as f:
#             lines = f.readlines()
#         if self.whole:
#             with open(test_data_list_file, 'r') as f:
#                 test_lines = f.readlines()
#             print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
#             lines = test_lines + lines
#         self.file_list = []
#         for line in lines:
#             line = line.strip()
#             taxonomy_id = line.split('-')[0]
#             model_id = line[len(taxonomy_id) + 1:].split('.')[0]
#             self.file_list.append({
#                 'taxonomy_id': taxonomy_id,
#                 'model_id': model_id,
#                 'file_path': line
#             })
#         print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

#         self.permutation = np.arange(self.npoints)

#         self.uniform = True
#         self.augment = True
#         self.use_caption_templates = False
#         # =================================================
#         # TODO: disable for backbones except for PointNEXT!!!
#         self.use_height = config.use_height
#         # =================================================

#         # +++ START: RAG MODIFICATION +++
#         # 根據 args 參數來決定是否啟用 RAG 模式
#         self.use_rag_adapter = getattr(config.args, 'use_rag_adapter', False)
#         if self.use_rag_adapter:
#             print_log('[DATASET] RAG Adapter mode is enabled. Dataset will return (tokens, raw_string) for text.', logger='ShapeNet-55')
#         # +++ END: RAG MODIFICATION +++

#         if self.augment:
#             print("using augmented point clouds.")

#     def pc_norm(self, pc):
#         """ pc: NxC, return NxC """
#         centroid = np.mean(pc, axis=0)
#         pc = pc - centroid
#         m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#         pc = pc / m
#         return pc

#     def random_sample(self, pc, num):
#         np.random.shuffle(self.permutation)
#         pc = pc[self.permutation[:num]]
#         return pc

#     def __getitem__(self, idx):
#         sample = self.file_list[idx]

#         data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

#         if self.uniform and self.sample_points_num < data.shape[0]:
#             data = farthest_point_sample(data, self.sample_points_num)
#         else:
#             data = self.random_sample(data, self.sample_points_num)
#         data = self.pc_norm(data)

#         if self.augment:
#             data = random_point_dropout(data[None, ...])
#             data = random_scale_point_cloud(data)
#             data = shift_point_cloud(data)
#             data = rotate_perturbation_point_cloud(data)
#             data = rotate_point_cloud(data)
#             data = data.squeeze()

#         if self.use_height:
#             self.gravity_dim = 1
#             height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
#                                                                        self.gravity_dim:self.gravity_dim + 1].min()
#             data = np.concatenate((data, height_array), axis=1)
#             data = torch.from_numpy(data).float()
#         else:
#             data = torch.from_numpy(data).float()
            
            
#         # # ------ START: 找到並刪除這整個區塊 ------
#         # captions = self.synset_id_map[sample['taxonomy_id']]['name']
#         # captions = [caption.strip() for caption in captions.split(',') if caption.strip()]
#         # caption = random.choice(captions)
#         # captions = []
#         # tokenized_captions = []
#         # if self.use_caption_templates:
#         #     for template in self.templates:
#         #         caption = template.format(caption)
#         #         captions.append(caption)
#         #         tokenized_captions.append(self.tokenizer(caption))
#         # else:
#         #     tokenized_captions.append(self.tokenizer(caption))

#         # tokenized_captions = torch.stack(tokenized_captions)
#         # # ------ END: 找到並刪除這整個區塊 ------
        
#         # +++ START: RAG MODIFICATION +++
#         # 處理文字，使其能夠同時返回 tokenized_ids 和 raw_string
#         captions = self.synset_id_map[sample['taxonomy_id']]['name']
#         captions = [caption.strip() for caption in captions.split(',') if caption.strip()]
#         base_caption = random.choice(captions)

#         # 原始 ULIP 在 `use_caption_templates=False` 時只返回一個 tokenized tensor,
#         # 這裡我們保持這個行為，同時準備好 RAG 需要的原始字串。
#         # 原始程式碼中 `tokenized_captions` 是一個堆疊的 tensor，我們也保持這個結構。
#         raw_caption = ""
#         if self.use_caption_templates:
#             raw_caption = self.templates[random.randint(0, len(self.templates)-1)].format(base_caption)
#         else:
#             raw_caption = base_caption

#         cap = self.tokenizer(raw_caption).unsqueeze(0) # 保持與原始輸出相同的維度

#         if self.use_rag_adapter:
#             # 如果啟用 RAG，返回元組 (tokens, raw_string)
#             # 注意：原始 ULIP 的 `forward` 迴圈會處理 captions 的第一個維度，
#             # 所以我們需要確保 `tokenized_captions` 的形狀是 [1, 77]
#             # raw_caption 則是一個簡單的字串，我們將其放入列表中以方便後續處理
#             text_output = (cap, [raw_caption])
#         else:
#             # 否則，保持原始行為，只返回 tokenized tensor
#             text_output = cap
            
#         tokenized_captions = text_output
#         # +++ END: RAG MODIFICATION +++


#         picked_model_rendered_image_addr = self.rendered_image_addr + '/' +\
#                                            sample['taxonomy_id'] + '-' + sample['model_id'] + '/'
#         picked_image_name = sample['taxonomy_id'] + '-' + sample['model_id'] + '_r_' +\
#                             str(random.choice(self.picked_rotation_degrees)) +\
#                             random.choice(self.picked_image_type) + '.png'
#         picked_image_addr = picked_model_rendered_image_addr + picked_image_name

#         try:
#             image = pil_loader(picked_image_addr)
#             image = self.train_transform(image)
#         except:
#             raise ValueError("image is corrupted: {}".format(picked_image_addr))

#         return sample['taxonomy_id'], sample['model_id'], tokenized_captions, data, image

#     def __len__(self):
#         return len(self.file_list)


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):

        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform
        self.id_map_addr = os.path.join(config.DATA_PATH, 'taxonomy.json')
        self.rendered_image_addr = config.IMAGE_PATH
        self.picked_image_type = ['', '_depth0001']
        self.picked_rotation_degrees = list(range(0, 360, 12))
        self.picked_rotation_degrees = [(3 - len(str(degree))) * '0' + str(degree) if len(str(degree)) < 3 else str(degree) for degree in self.picked_rotation_degrees]

        with open(self.id_map_addr, 'r') as f:
            self.id_map = json.load(f)

        self.prompt_template_addr = os.path.join(_CORE_DIR, 'data/configs/templates.json')
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]

        self.synset_id_map = {}
        for id_dict in self.id_map:
            synset_id = id_dict["synsetId"]
            self.synset_id_map[synset_id] = id_dict

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = self.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line[len(taxonomy_id) + 1:].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

        self.permutation = np.arange(self.npoints)

        self.uniform = True
        self.augment = True
        self.use_caption_templates = False
        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        # =================================================

        # +++ START: RAG MODIFICATION + ULIP Captions +++
        # 根據 args 參數來決定是否啟用 RAG 模式
        self.use_rag_adapter = getattr(config.args, 'use_rag_adapter', False)
        
        # 載入 ULIP-shapenet_triplets_captions.json
        self.triplets_captions_file = '/mnt/P300/data/ULIP/ULIP_Shapenet_Triplets/ULIP-shapenet_triplets_captions.json'
        self.model_to_captions = {}
        
        if os.path.exists(self.triplets_captions_file):
            print_log('[DATASET] Loading ULIP-shapenet_triplets_captions...', logger='ShapeNet-55')
            with open(self.triplets_captions_file, 'r', encoding='utf-8') as f:
                triplets_data = json.load(f)
            print_log(f'[DATASET] Loaded {len(triplets_data)} triplet captions', logger='ShapeNet-55')
            
            # 創建從 taxonomy_id-model_id 到 captions 的映射
            for image_path, captions in triplets_data.items():
                # 從路徑中提取 taxonomy_id 和 model_id
                # 路徑格式: /export/.../02691156-10155655850468db78d106ce0a280f87/02691156-10155655850468db78d106ce0a280f87_r_000.png
                path_parts = image_path.split('/')
                if len(path_parts) >= 2:
                    folder_name = path_parts[-2]  # 02691156-10155655850468db78d106ce0a280f87
                    if '-' in folder_name:
                        parts = folder_name.split('-', 1)
                        taxonomy_id = parts[0]
                        model_id = parts[1]
                        
                        model_key = f"{taxonomy_id}-{model_id}"
                        if model_key not in self.model_to_captions:
                            self.model_to_captions[model_key] = []
                        self.model_to_captions[model_key].extend(captions)
            
            print_log(f'[DATASET] Created caption mapping for {len(self.model_to_captions)} models', logger='ShapeNet-55')
        else:
            print_log(f'[DATASET] ULIP captions file not found: {self.triplets_captions_file}', logger='ShapeNet-55')
            print_log('[DATASET] Will use fallback to category names', logger='ShapeNet-55')
        
        if self.use_rag_adapter:
            print_log('[DATASET] RAG Adapter mode is enabled. Dataset will return (tokens, raw_string) for text.', logger='ShapeNet-55')
        # +++ END: RAG MODIFICATION + ULIP Captions +++

        if self.augment:
            print("using augmented point clouds.")

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        pc_file = os.path.join(self.pc_path, sample['file_path'])
        if not os.path.exists(pc_file):
            # Skip missing files by returning a random valid sample
            return self.__getitem__(np.random.randint(len(self.file_list)))
        data = IO.get(pc_file).astype(np.float32)

        if self.uniform and self.sample_points_num < data.shape[0]:
            data = farthest_point_sample(data, self.sample_points_num)
        else:
            data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)

        if self.augment:
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()

        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        # +++ START: Enhanced Text Processing with ULIP Captions +++
        # 1. 獲取基礎類別名稱（作為備用）
        captions = self.synset_id_map[sample['taxonomy_id']]['name']
        captions = [caption.strip() for caption in captions.split(',') if caption.strip()]
        base_caption = random.choice(captions)

        # 2. 嘗試從 ULIP triplets captions 獲取豐富描述
        model_key = f"{sample['taxonomy_id']}-{sample['model_id']}"
        raw_caption = ""
        
        if model_key in self.model_to_captions and len(self.model_to_captions[model_key]) > 0:
            # 如果找到 ULIP 的豐富描述，隨機選擇一個
            rich_captions = self.model_to_captions[model_key]
            raw_caption = random.choice(rich_captions)
        else:
            # 回退到原始的類別名稱處理
            if self.use_caption_templates:
                template = random.choice(self.templates)
                raw_caption = template.format(base_caption)
            else:
                raw_caption = base_caption

        # 3. Tokenize
        cap = self.tokenizer(raw_caption).unsqueeze(0)

        if self.use_rag_adapter:
            # RAG 模式：返回 (tokens, [raw_string])
            text_output = (cap, [raw_caption])
        else:
            # 標準模式：只返回 tokenized tensor
            text_output = cap
            
        tokenized_captions = text_output
        # +++ END: Enhanced Text Processing with ULIP Captions +++

        picked_model_rendered_image_addr = self.rendered_image_addr + '/' +\
                                           sample['taxonomy_id'] + '-' + sample['model_id'] + '/'
        picked_image_name = sample['taxonomy_id'] + '-' + sample['model_id'] + '_r_' +\
                            str(random.choice(self.picked_rotation_degrees)) +\
                            random.choice(self.picked_image_type) + '.png'
        picked_image_addr = picked_model_rendered_image_addr + picked_image_name

        try:
            image = pil_loader(picked_image_addr)
            image = self.train_transform(image)
        except:
            raise ValueError("image is corrupted: {}".format(picked_image_addr))

        return sample['taxonomy_id'], sample['model_id'], tokenized_captions, data, image

    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class Objaverse_Lvis_Colored(data.Dataset):
    def __init__(self, config):

        self.npoints = 10000
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform

        self.lvis_list_addr = os.path.join(_CORE_DIR, 'data/objaverse-lvis/lvis.json')
        self.lvis_metadata_addr = os.path.join(_CORE_DIR, 'data/objaverse-lvis/objaverse_lvis_metadata.json')

        with open(self.lvis_list_addr, 'r') as f:
            self.npy_file_map = json.load(f)

        self.file_list = list(self.npy_file_map.keys())

        with open(self.lvis_metadata_addr, 'r') as f:
            self.lvis_metadata = json.load(f)

        self.prompt_template_addr = os.path.join(_CORE_DIR, 'data/configs/templates.json')
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]

        self.sample_points_num = self.npoints

        print_log(f'Objaverse lvis {len(self.file_list)} instances were loaded', logger='objaverse_lvis')

        self.permutation = np.arange(self.npoints)

        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = False
        self.use_color = True
        
        self.objaverse_lvis_path = os.path.join(_CORE_DIR, 'data/objaverse-lvis')
        
        if self.use_color:
            print("use color")
        else:
            print("don't use color")

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):

        sample = self.file_list[idx]
        pc_addr = self.npy_file_map[sample]
        pc_addr = os.path.join(self.objaverse_lvis_path,self.npy_file_map[sample])
        data = np.load(pc_addr, allow_pickle=True)
        dict_data = data.item()
        xyz_data = dict_data['xyz']
        rgb_data = dict_data['rgb']

        data = self.pc_norm(xyz_data)
        if self.use_color:
            data = np.concatenate([data, rgb_data], axis=1)

        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        data = data.contiguous()

        name = self.lvis_metadata["value_to_key_mapping"][sample]
        label = self.lvis_metadata["key_to_id"][name]

        return data, label, name

    def __len__(self):
        return len(self.file_list)

import collections.abc as container_abcs
int_classes = int
# from torch._six import string_classes

import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')

def customized_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(batch, list):
        batch = [example for example in batch if example[4] is not None]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return customized_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customized_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customized_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [customized_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

# data/dataset_3d.py 中的 rag_collate_fn
import torch
from typing import List, Tuple, Any, Union

# -----------------------------------------------------------------------------
# 新版 rag_collate_fn
# -----------------------------------------------------------------------------
def rag_collate_fn(batch: List[Tuple[Any, ...]]
                   ) -> Union[Tuple[torch.Tensor,               # pc  (B, N, 3/6/…)
                                    Tuple[torch.Tensor, list],  # text (tokens , raw)
                                    torch.Tensor,               # image(B, 3, 224, 224) or None
                                    list                       # labels (str / int / None)
                                   ],
                               None]:
    """
    支援三種 sample 版型：
      1. ShapeNet-55  → len==5  (taxonomy_id, model_id, text, pc, img)
      2. 舊版  ULIP   → len==4  (idx, pc, text, img)          # idx 通常是 int
      3. 其他 (Objaverse…) → len==3  (pc, text, img)

    - 自動過濾 image == None 的樣本
    - 將 labels 以 list 形式額外返回：
        • ShapeNet  → taxonomy_id (str)，例："02691156"
        • len==4    → idx (int / str，依資料集而定)
        • len==3    → 無法取得 → None
    """
    if not batch:
        return None

    # ---------------- 過濾沒有影像的樣本 ----------------
    img_idx = {5: 4, 4: 3, 3: 2}[len(batch[0])]
    batch = [s for s in batch if s[img_idx] is not None]
    if not batch:
        return None

    # ---------------- 拆批次 ----------------
    pc_list, txt_list, img_list, label_list = [], [], [], []
    for s in batch:
        if len(s) == 5:                          # ShapeNet-55
            taxonomy_id, model_id, txt, pc, img = s
            label = taxonomy_id                  # 直接用 synsetId 當 label
        elif len(s) == 4:                        # 舊版 ULIP
            idx, pc, txt, img = s
            label = idx
        else:                                    # len == 3
            pc, txt, img = s
            label = None

        pc_list.append(pc)
        txt_list.append(txt)
        img_list.append(img)
        label_list.append(label)

    pc_tensor = torch.stack(pc_list, 0)

    # ---------------- 處理文字 ----------------
    # 兩種可能：
    #   (tokens, raw_string_list)  或者  單純 tokens
    if isinstance(txt_list[0], tuple):
        token_tensor = torch.stack([t[0] for t in txt_list], 0)          # (B, 77)
        raw_strings = [t[1] for t in txt_list]                          # List[List[str]] or List[str]
        text_batch = (token_tensor, raw_strings)
    else:
        text_batch = torch.stack(txt_list, 0)

    # ---------------- 處理影像 ----------------
    img_tensor = torch.stack(img_list, 0) if all(i is not None for i in img_list) else None

    # labels 保持 list 型別，讓上層自己決定要不要轉成 tensor
    return pc_tensor, text_batch, img_tensor, label_list



def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

class Dataset_3D():
    def __init__(self, args, tokenizer, dataset_type, train_transform=None):
        if dataset_type == 'train':
            self.dataset_name = args.pretrain_dataset_name
        elif dataset_type == 'val':
            self.dataset_name = args.validate_dataset_name
        else:
            raise ValueError("not supported dataset type.")
        with open(os.path.join(_CORE_DIR, 'data/configs/dataset_catalog.json'), 'r') as f:
            self.dataset_catalog = json.load(f)
            self.dataset_usage = self.dataset_catalog[self.dataset_name]['usage']
            self.dataset_split = self.dataset_catalog[self.dataset_name][self.dataset_usage]
            self.dataset_config_dir = self.dataset_catalog[self.dataset_name]['config']
        self.tokenizer = tokenizer
        self.train_transform = train_transform
        self.pretrain_dataset_prompt = getattr(args, 'pretrain_dataset_prompt', None)
        self.validate_dataset_prompt = args.validate_dataset_prompt
        if 'colored' in args.model.lower():
            self.use_colored_pc = True
        else:
            self.use_colored_pc = False
        if args.npoints == 10000:
            self.use_10k_pc = True
        else:
            self.use_10k_pc = False
        self.build_3d_dataset(args, self.dataset_config_dir)

    def build_3d_dataset(self, args, config):
        # catalog 內的 config 路徑多為 './data/configs/*.yaml',錨定到 core/ 根
        if not os.path.isabs(config):
            config = os.path.join(_CORE_DIR, config)
        config = cfg_from_yaml_file(config)
        config.tokenizer = self.tokenizer
        config.train_transform = self.train_transform
        config.pretrain_dataset_prompt = self.pretrain_dataset_prompt
        config.validate_dataset_prompt = self.validate_dataset_prompt
        config.args = args
        config.use_height = args.use_height
        config.npoints = args.npoints
        config.use_colored_pc = self.use_colored_pc
        config.use_10k_pc = self.use_10k_pc
        config_others = EasyDict({'subset': self.dataset_split, 'whole': True})
        self.dataset = build_dataset_from_cfg(config, config_others)
