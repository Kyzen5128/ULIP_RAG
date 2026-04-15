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
import json
import collections
import numpy as np

import torch
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

# === 關鍵：註冊 IkeaULIP ===
import data.ikea_ulip  # 確保 @DATASETS.register_module() 被執行
from data.dataset_3d import *  # customized_collate_fn 等工具

from utils.utils import get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from data.dataset_3d import customized_collate_fn


def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP training and evaluation', add_help=False)
    # Data
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='ikea_ulip', type=str)      # ← 預設改 ikea_ulip
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--validate_dataset_name', default='ikea_ulip', type=str)      # ← 預設改 ikea_ulip
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true', help='whether to use height information (PointNeXt)')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='ULIP_PN_SSG', type=str)
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
    return parser


def _using_ikea_dataset(args):
    def _is_ikea(name):
        return name is not None and 'ikea_ulip' in str(name).lower()
    return _is_ikea(args.pretrain_dataset_name) or _is_ikea(args.validate_dataset_name)


best_acc1 = 0

def main(args):
    utils.init_distributed_mode(args)

    global best_acc1

    if utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='ULIP2', id=wandb_id, config=args, reinit=True, entity='hj6hki123-cpu')

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

    # create model
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(args=args)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200, find_unused_parameters=False)

    # define loss function (criterion) and optimizer
    criterion = models.get_loss(args).cuda(args.gpu)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            print('in optimizer freeze {}'.format(n))
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1']
            print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from the latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})".format(latest, latest_checkpoint['epoch']))

    cudnn.benchmark = True

    # ====================== Data loading code ======================
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = get_dataset(train_transform, tokenizer, args, 'train')
    val_dataset   = get_dataset(None,            tokenizer, args, 'val')

    use_ikea = _using_ikea_dataset(args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        collate_fn=customized_collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False,
        collate_fn=customized_collate_fn if use_ikea else None)

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    print(args)
    print("=> beginning training")

    best_epoch = -1
    retrieval_log = {}

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)
        val_stats = {"acc1": -1, "acc5": -1}
        retrieval_log = {}

        if epoch % 1 == 0:
            if use_ikea:
                # Ikea：使用六向檢索當驗證
                try:
                    if utils.is_main_process():
                        retrieval_log = eval_object_level_retrieval_6way(
                            model=model,
                            dataset=val_dataset,  # 驗證集
                            gpu_id=args.gpu if args.gpu is not None else 0,
                            batch_size=args.batch_size,
                            workers=args.workers,
                            k_ndcg=5,
                            chunk_size=512,
                            topk_chunk=1024
                        )
                        acc1 = retrieval_log.get("retr/s2t_r1", 0.0) * 100.0
                        acc5 = retrieval_log.get("retr/s2t_r5", 0.0) * 100.0
                        val_stats = {"acc1": acc1, "acc5": acc5}
                        print("[IKEA Eval]", val_stats)

                        def _p(tag):
                            return {k: retrieval_log[k] for k in retrieval_log if k.startswith(f"retr/{tag}_")}
                        print("[Retrieval] S2T:", _p("s2t"))
                        print("[Retrieval] T2S:", _p("t2s"))
                        print("[Retrieval] S2I:", _p("s2i"))
                        print("[Retrieval] I2S:", _p("i2s"))
                        print("[Retrieval] T2I:", _p("t2i"))
                        print("[Retrieval] I2T:", _p("i2t"))
                except Exception as e:
                    if utils.is_main_process():
                        print(f"[IKEA Eval] Skip due to error: {e}")
                    val_stats = {"acc1": -1, "acc5": -1}
            else:
                # 非 Ikea：沿用 zero-shot 分類驗證
                val_stats = test_zeroshot_3d_core(val_loader, model, tokenizer, args)

            acc1 = val_stats.get("acc1", -1)
            print(val_stats)

            is_best = acc1 > best_acc1
            if is_best:
                best_epoch = epoch
            best_acc1 = max(acc1, best_acc1)

            if is_best or epoch % 50 == 0:
                print("=> saving checkpoint")
                utils.save_on_master({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc1': best_acc1,
                        'args': args,
                    }, is_best, args.output_dir)

            if epoch + 1 == args.epochs:
                print("=> saving last checkpoint")
                utils.save_on_master({
                    'epoch': 'last',
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc1': best_acc1,
                    'args': args,
                }, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     **retrieval_log,
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
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]

        pc = inputs[3]
        texts = inputs[2]
        image = inputs[4]
        inputs = [pc, texts, image]

        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            import sys; sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale
        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                           'scaler': scaler.get_scale(),
                           'logit': logit_scale})
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}



def test_zeroshot_3d_core(test_loader, model, tokenizer, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    print('=> encoding captions')
    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    if 'objaverse' in args.validate_dataset_name.lower():
        labels = test_loader.dataset.lvis_metadata['all_keys']
    else:
        with open(os.path.join("./data", 'labels.json')) as f:
            labels = json.load(f)[args.validate_dataset_name]

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
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

        for i, (pc, target, target_name) in enumerate(test_loader):
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode pc
            pc_features = utils.get_model(model).encode_pc(pc)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_pc = pc_features @ text_features.t()

            # measure accuracy and record loss
            (acc1, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 5))
            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                if top5_accurate[idx].item():
                    per_class_correct_top5[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        top1_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
            top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]

        top1_accuracy_per_class = OrderedDict(top1_accuracy_per_class)
        top5_accuracy_per_class = OrderedDict(top5_accuracy_per_class)

    progress.synchronize()
    print(f'0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
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
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
        self.sum = int(t[0])
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


@torch.no_grad()
def eval_object_level_retrieval_6way(model, dataset, gpu_id=0, batch_size=64, workers=8, k_ndcg=5, chunk_size=512, topk_chunk=1024):
    """
    以 object_id = "<category>-<model_id>" 定義同一物件，計算六向檢索：
      S2T(PC→Text), T2S(Text→PC), S2I(PC→Image), I2S(Image→PC), T2I(Text→Image), I2T(Image→Text)
    回傳 dict，可直接 wandb.log
    """
    import torch.nn.functional as F
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=customized_collate_fn
    )

    model.eval()

    all_pc, all_txt, all_img = [], [], []
    all_tax, all_mdl = [], []

    # 先檢查資料形狀
    try:
        probe = next(iter(loader))
        if not (isinstance(probe, (list, tuple)) and len(probe) == 5):
            print("[Retrieval] Skip: dataset 非 (tax, model, txt, pc, img) 五元組。")
            return {}
    except StopIteration:
        print("[Retrieval] Skip: 驗證資料為空。")
        return {}

    # 評估期：固定只用 RGB、關閉增強（若資料集支援）
    if hasattr(dataset, "augment"):
        dataset.augment = False
    if hasattr(dataset, "picked_image_type"):
        dataset.picked_image_type = ['']  # 只用 RGB

    # 1) 特徵
    for batch in loader:
        taxonomy_ids = batch[0]
        model_ids    = batch[1]
        txt_tokens   = batch[2].to(device).long()
        pc           = batch[3].to(device).float()
        img          = batch[4].to(device)

        out = model(pc, txt_tokens, img)
        pc_emb   = F.normalize(out["pc_embed"],    dim=-1)
        txt_emb  = F.normalize(out["text_embed"],  dim=-1)
        img_emb  = F.normalize(out["image_embed"], dim=-1)

        all_pc.append(pc_emb.cpu())
        all_txt.append(txt_emb.cpu())
        all_img.append(img_emb.cpu())
        all_tax.extend(taxonomy_ids)
        all_mdl.extend(model_ids)

    pc_feats  = torch.cat(all_pc,  dim=0)
    txt_feats = torch.cat(all_txt, dim=0)
    img_feats = torch.cat(all_img, dim=0)

    object_ids = [f"{t}-{m}" for t, m in zip(all_tax, all_mdl)]
    from collections import defaultdict
    obj2idxs = defaultdict(list)
    for i, oid in enumerate(object_ids):
        obj2idxs[oid].append(i)

    def _best_ranks(query_feats, gallery_feats, name):
        N = query_feats.size(0)
        ranks = []
        for i in range(0, N, chunk_size):
            j = min(i + chunk_size, N)
            sims = query_feats[i:j].to(device) @ gallery_feats.to(device).t()
            sorted_idx = torch.argsort(sims, dim=1, descending=True)
            for local in range(j - i):
                gid = i + local
                oid = object_ids[gid]
                pos_idxs = obj2idxs[oid]
                row = sorted_idx[local]
                best = float("inf")
                for p in pos_idxs:
                    pos = (row == p).nonzero(as_tuple=True)[0]
                    if pos.numel() > 0:
                        r = int(pos.item()) + 1
                        if r < best:
                            best = r
                ranks.append(best)
        return ranks

    def _summarize(ranks):
        valid = [r for r in ranks if r != float("inf")]
        if len(valid) == 0:
            return dict(mrr=0., r1=0., r5=0., r10=0., mean=0., median=0., n=len(ranks))
        r = torch.tensor(valid, dtype=torch.float32)
        return dict(
            mrr=(1.0/r).mean().item(),
            r1=(r<=1).float().mean().item(),
            r5=(r<=5).float().mean().item(),
            r10=(r<=10).float().mean().item(),
            mean=r.mean().item(),
            median=r.median().item(),
            n=len(valid)
        )

    def _ndcg_at_k(query_feats, gallery_feats, k=5):
        N = query_feats.size(0)
        scores = []
        for i in range(0, N, topk_chunk):
            j = min(i + topk_chunk, N)
            sims = query_feats[i:j].to(device) @ gallery_feats.to(device).t()
            topk = torch.topk(sims, k=min(k, sims.size(1)), dim=1, largest=True, sorted=True).indices
            for row_idx in range(j - i):
                gid = i + row_idx
                oid = object_ids[gid]
                pos = set(obj2idxs[oid])
                if len(pos) == 0:
                    scores.append(0.0); continue
                dcg = 0.0
                for rank_pos in range(topk.size(1)):  # 0-based
                    idx = int(topk[row_idx, rank_pos])
                    if idx in pos:
                        dcg += 1.0 / np.log2(rank_pos + 2)
                P = min(len(pos), k)
                idcg = sum(1.0 / np.log2(t + 2) for t in range(P))
                scores.append(dcg / idcg if idcg > 0 else 0.0)
        return float(np.mean(scores))

    # 2) 六向檢索
    pairs = [
        ("s2t", pc_feats,  txt_feats),
        ("t2s", txt_feats, pc_feats),
        ("s2i", pc_feats,  img_feats),
        ("i2s", img_feats, pc_feats),
        ("t2i", txt_feats, img_feats),
        ("i2t", img_feats, txt_feats),
    ]
    final = {}
    for tag, q, g in pairs:
        ranks = _best_ranks(q, g, tag.upper())
        s = _summarize(ranks)
        ndcg5 = _ndcg_at_k(q, g, k=k_ndcg)
        final[f"retr/{tag}_mrr"]    = s["mrr"]
        final[f"retr/{tag}_r1"]     = s["r1"]
        final[f"retr/{tag}_r5"]     = s["r5"]
        final[f"retr/{tag}_r10"]    = s["r10"]
        final[f"retr/{tag}_ndcg5"]  = ndcg5
        final[f"retr/{tag}_mean"]   = s["mean"]
        final[f"retr/{tag}_median"] = s["median"]

    return final


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
