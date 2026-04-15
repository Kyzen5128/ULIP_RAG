'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * Changed from SLIP
 * https://github.com/facebookresearch/SLIP
 * By Le Xue
'''
import argparse
from collections import OrderedDict
import math
import time
import wandb
import os
import sys
import numpy as np
import json
import torch
from pathlib import Path

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import collections
from tqdm import tqdm

from torch.nn.parameter import Parameter
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
import torch.distributed as dist


def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP training and evaluation', add_help=False)
    # Data
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='shapenet', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--validate_dataset_name', default='modelnet40', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='ULIP_PointBERT', type=str, help="Name of model to train.")
    # Training
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')

    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate_3d', action='store_true', help='eval ulip only')
    parser.add_argument('--evaluate_3d_ulip2', action='store_true', help='eval ulip2 only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    parser.add_argument('--test_ckpt_addr', default='', help='the ckpt to test 3d zero shot')
    
    # --- RAG 相關參數 ---
    parser.add_argument('--use_rag_adapter', action='store_true', help='Use Generative RAG model.')
    parser.add_argument('--training_strategy', type=str, default='staged_1',
                        choices=['staged_1', 'staged_2'],
                        help='Training strategy for Generative RAG.')
    parser.add_argument('--rag_corpus_dir', type=str, default='data/rag_corpus', help='Path to RAG corpus and index.')
    parser.add_argument('--rag_top_k', type=int, default=5, help='Number of documents to retrieve for RAG.')
    parser.add_argument('--freeze_backbone', action='store_true', help='(For RAG) Freeze backbone and train adapter only.')
    parser.add_argument('--stage1_ckpt_path', type=str, default=None,
                        help='[僅用於 Stage 2] Stage 1 訓練完成後的 checkpoint 路徑。')

    return parser

best_acc1 = 0

def main(args):
    utils.init_distributed_mode(args)

    global best_acc1

    if utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1] if args.output_dir else "ulip_rag_run"
        wandb.init(project='ULIP-RAG', id=wandb_id, config=args, reinit=True)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.evaluate_3d:
        zero_stats = test_zeroshot_3d(args)
        print(zero_stats)
        return
    elif args.evaluate_3d_ulip2:
        zero_stats = test_zeroshot_3d_ulip2(args)
        print(zero_stats)
        return

    # --- 1. 動態創建模型 ---
    if args.use_rag_adapter:
        # 為了保持原始 --model 參數的日誌記錄，我們先保存它
        args.model_name_for_metric = args.model
        # 強制使用我們的生成式 RAG 工廠函數
        print(f"=> creating model with Generative RAG: ULIP_PointBERT_RAG_GENERATIVE")
        model = models.ULIP_PointBERT_RAG(args) 
    else:
        args.model_name_for_metric = args.model
        print("=> creating model: {}".format(args.model))
        model = getattr(models, args.model)(args=args)
        
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200, find_unused_parameters=True)

    # --- 2. 動態定義損失函數 ---
    if args.use_rag_adapter:
        criterion = models.ULIP_Loss_RAG_Enhanced(args).cuda(args.gpu)
        print(f"INFO: Using Generative RAG Loss with strategy: {args.training_strategy}")
    else:
        # 使用原始檔案中的 get_loss
        from models import losses
        criterion = losses.ULIPWithImageLoss().cuda(args.gpu)


    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint.get('epoch', 0)
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"Loading model state dict result: {result}")
            if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint: scaler.load_state_dict(checkpoint['scaler'])
            best_acc1 = checkpoint.get('best_acc1', 0)
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True

    # --- 3. 動態數據加載 ---
    print("=> creating dataset")
    if args.use_rag_adapter:
        from data.dataset_3d import Dataset_3D, rag_collate_fn
        collate_function = rag_collate_fn
        print("INFO: Using RAG-specific collate function.")
    else:
        from data.dataset_3d import Dataset_3D, customized_collate_fn
        collate_function = customized_collate_fn
        print("INFO: Using standard collate function.")
    
    from utils.utils import get_dataset
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = get_dataset(train_transform, tokenizer, args, 'train')
    val_dataset = get_dataset(None, tokenizer, args, 'val')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        collate_fn=collate_function)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    print(args)
    print("=> beginning training")

    best_epoch = -1

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)
        val_stats = {"acc1": -1}

        if epoch % args.eval_freq == 0:
            val_stats = test_zeroshot_3d_core(val_loader, model, tokenizer, args)
            acc1 = val_stats["acc1"]
            print(val_stats)

            is_best = acc1 > best_acc1
            if is_best:
                best_epoch = epoch
            best_acc1 = max(acc1, best_acc1)

            if utils.is_main_process() and (is_best or (epoch > 0 and epoch % 50 == 0)):
                print("=> saving checkpoint")
                utils.save_on_master({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc1': best_acc1,
                        'args': args,
                    }, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'best_acc1': best_acc1,
                     'best_epoch': best_epoch}

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    
    # --- 5. 動態指標初始化 ---
    if args.training_strategy in ['staged_1', 'stage_1']:
        metric_names = [
            'loss', 
            'consistency_loss', 
            'enhanced_contrast_loss', 
            'enhanced_image_acc', 
            'original_image_acc',
            'ulip_loss',           # 為了兼容性
            'ulip_pc_image_acc',   # 為了兼容性
            'ulip_pc_text_acc'     # 為了兼容性
        ]
    elif args.training_strategy in ['staged_2', 'stage_2']:
        metric_names = [
            'loss',
            'pc_text_loss',
            'pc_image_loss', 
            'ulip_loss',
            'ulip_pc_text_acc',
            'ulip_pc_image_acc'
        ]
    
    metrics = OrderedDict([(name, AverageMeter(name, ':.4e')) for name in metric_names])
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    
    iters_per_epoch = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    for i, batch_data in enumerate(pbar):
        data_time.update(time.time() - end)
        if batch_data is None: continue
        
        # --- 6. 正確的資料解包與傳輸 ---
        if args.use_rag_adapter:
            # RAG collate 返回: (pc, text_data, image, labels)
            pc, text_data, image, labels = batch_data
            text_input_to_model = (text_data[0].cuda(args.gpu, non_blocking=True), text_data[1])
        else:
            _, _, text_input_to_model, pc, image = batch_data
            text_input_to_model = text_input_to_model.cuda(args.gpu, non_blocking=True)
            
        pc = pc.cuda(args.gpu, non_blocking=True)
        image = image.cuda(args.gpu, non_blocking=True)
        
        it = iters_per_epoch * epoch + i
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]

        with torch.cuda.amp.autocast(enabled=not args.disable_amp):
            outputs = model(pc=pc, text=text_input_to_model, image=image)
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        scaler.scale(loss).backward()

        if (i + 1) % args.update_freq == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if hasattr(utils.get_model(model), 'logit_scale'):
            utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        
        torch.cuda.synchronize()
        for k in loss_dict:
            if k in metrics:
                if utils.is_dist_avail_and_initialized():
                    t = loss_dict[k].data.clone()
                    dist.all_reduce(t)
                    t /= utils.get_world_size()
                    loss_value = t.item()
                else:
                    loss_value = loss_dict[k].item()
                metrics[k].update(loss_value, pc.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        mem.update(torch.cuda.max_memory_allocated() / 1e9)

        pbar.set_postfix(loss=f"{metrics['loss'].avg:.4f}",
                         acc=f"{metrics.get('enhanced_image_acc', metrics.get('ulip_pc_image_acc', AverageMeter('x'))).avg:.1f}%")
        if i % args.print_freq == 0:
            progress.display(i)
            
    progress.synchronize()
    return {k: v.avg for k, v in metrics.items() if v.count > 0}

def test_zeroshot_3d_core(test_loader, model, tokenizer, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    model.eval()
    print('=> encoding captions')
    
    # 根據 args.validate_dataset_name 獲取標籤
    if 'objaverse' in args.validate_dataset_name.lower():
        with open('data/objaverse-lvis/lvis.json', 'r') as f:
            lvis_data = json.load(f)
        labels = list(lvis_data.keys())
    else: # ModelNet40
        with open("./data/configs/labels.json") as f:
            labels = json.load(f)[args.validate_dataset_name]
            
    with open("./data/configs/templates.json") as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            
            if args.use_rag_adapter:
                # 為 RAG 模型準備正確的文本輸入格式
                raw_text_for_rag = [[t.format(l)] for t in templates]
                
                # 直接調用文本編碼方法，避免調用完整的 forward
                class_embeddings = utils.get_model(model).encode_text_with_rag(texts, raw_text_for_rag)
            else:
                # 標準 ULIP 模型
                class_embeddings = utils.get_model(model).encode_text(texts)

            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)

        for i, data_tuple in enumerate(test_loader):
            # 測試集的 dataloader 可能返回不同數量的項目
            if len(data_tuple) == 3:
                pc, target, target_name = data_tuple
            else: # 假設為4
                _, pc, target, target_name = data_tuple
                
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            pc_features = utils.get_model(model).encode_pc(pc)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            logits_per_pc = pc_features @ text_features.t()

            (acc1, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 5))
            if utils.is_dist_avail_and_initialized():
                acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            
            top1.update(acc1.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

    progress.synchronize()
    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return {'acc1': top1.avg, 'acc5': top5.avg}


def test_zeroshot_3d(args):
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    try:
        old_args = ckpt['args']
        model = getattr(models, old_args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> creating model: {}".format(old_args.model))
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    except:
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> creating model: {}".format(args.model))
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))

    tokenizer = SimpleTokenizer()

    test_dataset = get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )
    results = test_zeroshot_3d_core(test_loader, model, tokenizer, args)

    return results

def test_zeroshot_3d_ulip2(args):
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print("=> creating model: {}".format(args.model))

    model = getattr(models, args.model)(args=args)
    model.cuda()
    model.load_state_dict(state_dict, strict=False)
    print("=> loaded pretrained checkpoint '{}'".format(args.test_ckpt_addr))

    tokenizer = SimpleTokenizer()

    test_dataset = get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )
    results = test_zeroshot_3d_core(test_loader, model, tokenizer, args)

    return results

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = t[1]
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)