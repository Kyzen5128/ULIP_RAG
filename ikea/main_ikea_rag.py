"""Two-stage RAG fine-tuning entry point for the IKEA ULIP dataset.

The existing :mod:`ikea.main_ikea` remains the vanilla ULIP trainer.  This file
is intentionally separate so a RAG experiment cannot silently change the
production IKEA checkpoint or vector build path.

Stage 1 trains only ``rag_enhancer`` with text/image supervision.  Stage 2
requires a provenance-bound Stage 1 checkpoint and trains only PointBERT plus
``pc_projection`` against the frozen enhanced text space.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORE_DIR = _REPO_ROOT / "core"
for _path in (str(_CORE_DIR), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms

# Importing the module registers IkeaULIP with the Core dataset registry.
from ikea_ulip import IkeaULIP
from rag_training_utils import (
    PROVENANCE_SCHEMA,
    build_dataset_bundle_manifest,
    build_dataset_split_manifest,
    canonical_sha256,
    inspect_rag_artifacts,
    sha256_file,
    validate_stage1_checkpoint,
    write_json_atomic,
)
from data.dataset_3d import rag_collate_fn
import models.ULIP_models as models
from utils import utils
from utils.tokenizer import SimpleTokenizer


DEFAULT_DATASET_CONFIG = _REPO_ROOT / "ikea" / "ikea_ulip.yaml"
DEFAULT_RAG_CORPUS = _CORE_DIR / "rag_corpus" / "rag_corpus.jsonl"
DEFAULT_RAG_INDEX = _CORE_DIR / "rag_corpus" / "corpus_index.faiss"
DEFAULT_BASE_CHECKPOINT = Path("/mnt/P300/data/ULIP/checkpoint_last.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IKEA ULIP + Core RAG two-stage training (single GPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/ikea_rag"))
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Optional override; otherwise use TRAIN_RATIO from dataset config",
    )
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--rag-corpus", type=Path, default=DEFAULT_RAG_CORPUS)
    parser.add_argument("--rag-index", type=Path, default=DEFAULT_RAG_INDEX)
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument(
        "--require-rag-split",
        choices=("train", "val", "test"),
        help="Fail if any RAG document is not explicitly assigned to this split",
    )
    parser.add_argument(
        "--base-ulip-checkpoint",
        type=Path,
        default=DEFAULT_BASE_CHECKPOINT,
        help="Vanilla IKEA ULIP checkpoint used to warm-start both stages",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        help="Required for Stage 2; must have matching IKEA/RAG provenance",
    )
    parser.add_argument("--resume", type=Path, help="Resume the same stage")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--npoints", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-start", type=float, default=1e-6)
    parser.add_argument("--lr-end", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.98))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ULIP-IKEA-RAG")

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate corpus/index/checkpoints/dataset and exit before model construction",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="One epoch/one batch/eight validation items; still constructs the real model",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="0 means all batches (testing aid, not recommended for real runs)",
    )
    parser.add_argument(
        "--max-eval-items",
        type=int,
        default=0,
        help="0 means the complete validation split",
    )

    # Attributes retained for compatibility with the Core model factory.
    parser.set_defaults(
        model="ULIP_PointBERT_RAG",
        pretrain_dataset_name="ikea_ulip",
        validate_dataset_name="ikea_ulip",
        pretrain_dataset_prompt="shapenet_64",
        validate_dataset_prompt="modelnet40_64",
        evaluate_3d=False,
        evaluate_3d_ulip2=False,
        use_height=False,
        use_rag_adapter=True,
        distributed=False,
        world_size=1,
        rank=0,
        local_rank=0,
        dist_url="env://",
        dist_backend="nccl",
    )
    return parser


class IkeaRAGDatasetAdapter(torch.utils.data.Dataset):
    """Expose IKEA captions in the ``rag_collate_fn`` text tuple format."""

    def __init__(self, base: IkeaULIP, include_pointcloud: bool):
        self.base = base
        self.samples = base.samples
        self.include_pointcloud = include_pointcloud

    def __len__(self) -> int:
        return len(self.base)

    def _stage1_item(self, index: int):
        sample = self.samples[index]
        category = str(sample["category"])
        model_id = str(sample["id"])
        caption = str(sample.get("caption") or category or "furniture")
        tokenized = torch.stack([self.base.tokenizer(caption)])

        image_path = sample.get("image_path")
        render_paths = sample.get("render_paths") or []
        if self.base.render_pick == "random" and render_paths:
            image_path = random.choice(render_paths)
        elif isinstance(self.base.render_pick, str) and self.base.render_pick.startswith("fixed_"):
            degree = self.base.render_pick.split("_", 1)[1]
            image_path = next(
                (p for p in render_paths if f"_r_{degree}" in os.path.basename(p)),
                image_path,
            )
        image = self.base._load_image(image_path)
        # rag_collate_fn requires a stackable PC field. Stage 1 never forwards it.
        placeholder_pc = torch.zeros((1, 3), dtype=torch.float32)
        return category, model_id, (tokenized, [caption]), placeholder_pc, image

    def __getitem__(self, index: int):
        if not self.include_pointcloud:
            return self._stage1_item(index)
        category, model_id, tokenized, pointcloud, image = self.base[index]
        caption = str(self.samples[index].get("caption") or category or "furniture")
        return category, model_id, (tokenized, [caption]), pointcloud, image


class _Subset(torch.utils.data.Dataset):
    """Small deterministic prefix used only by ``--max-eval-items``."""

    def __init__(self, dataset: torch.utils.data.Dataset, count: int):
        self.dataset = dataset
        self.count = min(len(dataset), count)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        return self.dataset[index]


def _load_dataset_config(
    config_path: Path,
    tokenizer: SimpleTokenizer,
    transform: Optional[Any],
    args: argparse.Namespace,
    *,
    train: bool,
):
    config = utils.cfg_from_yaml_file(str(config_path))
    config.tokenizer = tokenizer
    config.train_transform = transform
    config.args = args
    config.use_height = False
    config.npoints = args.npoints
    if args.train_ratio is not None:
        config.TRAIN_RATIO = args.train_ratio
    if args.val_ratio is not None:
        config.VAL_RATIO = args.val_ratio
    if args.test_ratio is not None:
        config.TEST_RATIO = args.test_ratio
    config.AUGMENT = bool(train)
    config.RENDER_PICK = "random" if train else "fixed_000"
    return config


def build_datasets(args: argparse.Namespace):
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_config = _load_dataset_config(
        args.dataset_config, tokenizer, train_transform, args, train=True
    )
    val_config = _load_dataset_config(
        args.dataset_config, tokenizer, None, args, train=False
    )
    train_base = IkeaULIP(train_config, subset="train")
    val_base = IkeaULIP(val_config, subset="val")
    if not train_base.samples or not val_base.samples:
        raise ValueError(
            "The RAG fine-tune requires non-empty train and validation splits; "
            f"got train={len(train_base)}, val={len(val_base)}. "
            "Use a --train-ratio strictly between 0 and 1."
        )
    include_pointcloud = args.stage == "stage2"
    return (
        IkeaRAGDatasetAdapter(train_base, include_pointcloud),
        IkeaRAGDatasetAdapter(val_base, include_pointcloud),
        train_base,
        val_base,
    )


def _git_metadata() -> Dict[str, Any]:
    def run(*parts: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *parts], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _prepare_runtime_rag_view(output_dir: Path, rag: Mapping[str, Any]) -> Path:
    """Adapt arbitrary artifact filenames to the Core model's fixed filenames."""
    runtime_dir = output_dir / ".rag_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    links = {
        "rag_corpus.jsonl": Path(rag["corpus_path"]),
        "corpus_index.faiss": Path(rag["index_path"]),
    }
    for name, target in links.items():
        destination = runtime_dir / name
        if os.path.lexists(destination):
            if destination.is_symlink() and destination.resolve() == target.resolve():
                continue
            if destination.is_symlink():
                destination.unlink()
            else:
                raise FileExistsError(
                    f"Refusing to replace non-symlink RAG runtime artifact: {destination}"
                )
        destination.symlink_to(target.resolve())
    return runtime_dir


def _normalise_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        (key[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state_dict.items()
    }


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_base_ulip(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Base ULIP checkpoint has no valid state_dict: {checkpoint_path}")
    state_dict = _normalise_state_dict(state_dict)
    required = ("pc_projection", "text_projection", "image_projection")
    missing_required = [key for key in required if key not in state_dict]
    if missing_required or not any(k.startswith("point_encoder.") for k in state_dict):
        raise ValueError(
            "Base checkpoint is not an IKEA ULIP PointBERT checkpoint; "
            f"missing={missing_required or ['point_encoder.*']}"
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        key for key in incompatible.missing_keys if not key.startswith("rag_enhancer.")
    ]
    if unexpected or disallowed_missing:
        raise ValueError(
            "Base checkpoint/model mismatch: "
            f"unexpected={unexpected}, missing={disallowed_missing}"
        )


def _assert_enhancer_loaded(model: torch.nn.Module, expected: Mapping[str, Any]) -> None:
    actual = model.rag_enhancer.state_dict()
    for key, expected_tensor in expected.items():
        if key not in actual or not torch.equal(actual[key].cpu(), expected_tensor.cpu()):
            raise RuntimeError(f"Stage 1 enhancer weight was not loaded exactly: {key}")


def _assert_resume_provenance(
    checkpoint: Mapping[str, Any], provenance: Mapping[str, Any]
) -> None:
    recorded = checkpoint.get("provenance")
    if not isinstance(recorded, Mapping):
        raise ValueError("Resume checkpoint has no IKEA RAG provenance")
    checks = {
        "schema_version": (recorded.get("schema_version"), provenance["schema_version"]),
        "training_strategy": (
            recorded.get("training_strategy"),
            provenance["training_strategy"],
        ),
        "corpus_sha256": (
            recorded.get("rag", {}).get("corpus_sha256"),
            provenance["rag"]["corpus_sha256"],
        ),
        "index_sha256": (
            recorded.get("rag", {}).get("index_sha256"),
            provenance["rag"]["index_sha256"],
        ),
        "dataset_bundle_sha256": (
            recorded.get("dataset", {}).get("bundle_sha256"),
            provenance["dataset"]["bundle_sha256"],
        ),
    }
    mismatches = {key: values for key, values in checks.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(f"Resume checkpoint provenance mismatch: {mismatches}")


def _make_model(
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
    stage1_info: Optional[Mapping[str, Any]],
) -> torch.nn.Module:
    args.training_strategy = "staged_1" if args.stage == "stage1" else "staged_2"
    args.stage1_ckpt_path = (
        str(args.stage1_checkpoint.resolve()) if args.stage1_checkpoint else None
    )
    args.rag_corpus_dir = str(_prepare_runtime_rag_view(args.output_dir, provenance["rag"]))
    # Avoid the Core factory skipping its base construction because --resume is set;
    # this trainer performs a strict resume load below.
    requested_resume = args.resume
    args.resume = ""
    model = models.ULIP_PointBERT_RAG(args)
    args.resume = requested_resume
    _load_base_ulip(model, args.base_ulip_checkpoint.resolve())
    if stage1_info is not None:
        _assert_enhancer_loaded(model, stage1_info["enhancer_state"])

    if requested_resume:
        checkpoint = _load_checkpoint(requested_resume.resolve())
        _assert_resume_provenance(checkpoint, provenance)
        state_dict = _normalise_state_dict(checkpoint["state_dict"])
        model.load_state_dict(state_dict, strict=True)
    return model


def _optimizer(model: torch.nn.Module, args: argparse.Namespace):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or any(tag in name for tag in ("bias", "ln", "bn", "norm")):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay and not no_decay:
        raise RuntimeError("RAG freezing policy produced no trainable parameters")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
    )


def _schedule(
    base: float, final: float, start: float, epochs: int, steps: int, warmup_epochs: int
) -> np.ndarray:
    total = epochs * steps
    warmup = min(total, warmup_epochs * steps)
    values = []
    for step in range(total):
        if warmup and step < warmup:
            value = start + (base - start) * (step + 1) / warmup
        else:
            denominator = max(1, total - warmup - 1)
            progress = (step - warmup) / denominator
            value = final + 0.5 * (base - final) * (1 + math.cos(math.pi * progress))
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _set_train_modes(model: torch.nn.Module, stage: str) -> None:
    model.train()
    # Text/image backbones are frozen in both stages.  Keep their stochastic
    # training-time behavior disabled so Stage 1 sees stable supervision and
    # Stage 2 aligns points against a fixed target space.
    model.visual.eval()
    model.transformer.eval()
    model.point_encoder.eval()
    if stage == "stage2":
        # Stage 2 treats the Stage 1 text target as fixed; disable enhancer dropout.
        model.rag_enhancer.eval()
        # PointBERT is the trainable branch in Stage 2.  Restore only it to
        # training mode after freezing the other modalities above.
        model.point_encoder.train()


def train_one_epoch(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    schedule: np.ndarray,
    epoch: int,
    steps_per_epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    _set_train_modes(model, args.stage)
    optimizer.zero_grad(set_to_none=True)
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    accumulated = 0
    optimizer_step = 0
    start = time.time()
    total_batches = min(len(loader), args.max_train_batches or len(loader))

    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break
        if batch is None:
            continue
        pc, text_data, image, _labels = batch
        tokenized, raw_text = text_data
        tokenized = tokenized.cuda(args.gpu, non_blocking=True)
        image = image.cuda(args.gpu, non_blocking=True)
        pointcloud = None
        if args.stage == "stage2":
            pointcloud = pc.cuda(args.gpu, non_blocking=True)

        schedule_index = epoch * steps_per_epoch + min(optimizer_step, steps_per_epoch - 1)
        for group in optimizer.param_groups:
            group["lr"] = float(schedule[schedule_index])

        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(
                pc=pointcloud,
                text=(tokenized, raw_text),
                image=image,
            )
            loss_dict = criterion(outputs)
            loss = loss_dict["loss"] / args.update_freq
        if not math.isfinite(float(loss.item())):
            raise FloatingPointError(f"Non-finite loss at batch {batch_index}: {loss.item()}")
        scaler.scale(loss).backward()
        accumulated += 1

        should_step = accumulated == args.update_freq or batch_index + 1 == total_batches
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            model.logit_scale.data.clamp_(0, 4.6052)
            accumulated = 0
            optimizer_step += 1

        batch_size = int(image.size(0))
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                sums[key] = sums.get(key, 0.0) + float(value.detach().item()) * batch_size
                counts[key] = counts.get(key, 0) + batch_size
        if batch_index % args.print_freq == 0:
            print(
                f"epoch={epoch + 1} batch={batch_index + 1}/{total_batches} "
                f"loss={float(loss_dict['loss'].item()):.5f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )

    result = {key: sums[key] / counts[key] for key in sums if counts.get(key)}
    result["seconds"] = time.time() - start
    result["lr"] = float(optimizer.param_groups[0]["lr"])
    return result


def _retrieval_metrics(query: torch.Tensor, gallery: torch.Tensor, prefix: str):
    similarity = query @ gallery.t()
    order = similarity.argsort(dim=1, descending=True)
    target = torch.arange(order.size(0)).unsqueeze(1)
    rank = (order == target).nonzero(as_tuple=False)[:, 1] + 1
    rank = rank.float()
    return {
        f"retr/{prefix}_mrr": float((1.0 / rank).mean()),
        f"retr/{prefix}_r1": float((rank <= 1).float().mean()),
        f"retr/{prefix}_r5": float((rank <= 5).float().mean()),
        f"retr/{prefix}_r10": float((rank <= 10).float().mean()),
        f"retr/{prefix}_mean_rank": float(rank.mean()),
        f"retr/{prefix}_median_rank": float(rank.median()),
    }


@torch.no_grad()
def evaluate(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    stage: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    text_features, image_features, point_features = [], [], []
    for batch in loader:
        if batch is None:
            continue
        pc, text_data, image, _labels = batch
        tokens, raw_text = text_data
        tokens = tokens.cuda(args.gpu, non_blocking=True)
        image = image.cuda(args.gpu, non_blocking=True)
        pointcloud = pc.cuda(args.gpu, non_blocking=True) if stage == "stage2" else None
        outputs = model(pc=pointcloud, text=(tokens, raw_text), image=image)
        text_features.append(F.normalize(outputs["enhanced_text_embed"], dim=-1).cpu())
        image_features.append(F.normalize(outputs["image_embed"], dim=-1).cpu())
        if stage == "stage2":
            point_features.append(F.normalize(outputs["pc_embed"], dim=-1).cpu())

    text = torch.cat(text_features)
    image = torch.cat(image_features)
    result = {}
    result.update(_retrieval_metrics(text, image, "t2i"))
    result.update(_retrieval_metrics(image, text, "i2t"))
    if stage == "stage2":
        point = torch.cat(point_features)
        result.update(_retrieval_metrics(point, text, "s2t"))
        result.update(_retrieval_metrics(text, point, "t2s"))
        result.update(_retrieval_metrics(point, image, "s2i"))
        result.update(_retrieval_metrics(image, point, "i2s"))
    return result


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    best_metric: float,
    best_epoch: int,
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
) -> None:
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "args": vars(args),
        "provenance": dict(provenance),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.epochs = 1
    args.batch_size = min(args.batch_size, 2)
    args.workers = 0
    args.max_train_batches = 1
    args.max_eval_items = 8
    args.print_freq = 1


def _preflight(args: argparse.Namespace):
    for name in ("train_ratio", "val_ratio", "test_ratio"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.rag_top_k < 1:
        raise ValueError("--rag-top-k must be at least 1")
    if args.update_freq < 1:
        raise ValueError("--update-freq must be at least 1")
    for label, path in (
        ("dataset config", args.dataset_config),
        ("base ULIP checkpoint", args.base_ulip_checkpoint),
    ):
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.stage == "stage2" and args.stage1_checkpoint is None:
        raise ValueError("Stage 2 strictly requires --stage1-checkpoint")

    args.dataset_config = args.dataset_config.expanduser().resolve()
    args.base_ulip_checkpoint = args.base_ulip_checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.stage1_checkpoint:
        args.stage1_checkpoint = args.stage1_checkpoint.expanduser().resolve()
    if args.resume:
        args.resume = args.resume.expanduser().resolve()

    rag = inspect_rag_artifacts(args.rag_corpus, args.rag_index)
    rag["top_k"] = args.rag_top_k
    if args.require_rag_split:
        mismatches = []
        with Path(rag["corpus_path"]).open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                document = json.loads(raw_line)
                if document.get("split") != args.require_rag_split:
                    mismatches.append((line_number, document.get("doc_id"), document.get("split")))
                    if len(mismatches) >= 10:
                        break
        if mismatches:
            raise ValueError(
                f"RAG corpus violates required split={args.require_rag_split}: {mismatches}"
            )
        rag["required_split"] = args.require_rag_split
    train, val, train_base, val_base = build_datasets(args)
    args.train_ratio = float(train_base.train_ratio)
    args.val_ratio = float(train_base.val_ratio)
    args.test_ratio = float(train_base.test_ratio)
    if not 0.0 < args.train_ratio < 1.0 or args.val_ratio <= 0.0:
        raise ValueError(
            "RAG training requires non-empty train and validation ratio ranges; "
            f"got train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}"
        )
    train_manifest = build_dataset_split_manifest(
        train_base,
        split="train",
        dataset_config_path=args.dataset_config,
        train_ratio=args.train_ratio,
    )
    val_manifest = build_dataset_split_manifest(
        val_base,
        split="val",
        dataset_config_path=args.dataset_config,
        train_ratio=args.train_ratio,
    )
    dataset_manifest = build_dataset_bundle_manifest(train_manifest, val_manifest)

    stage1_info = None
    if args.stage == "stage2":
        stage1_info = validate_stage1_checkpoint(
            args.stage1_checkpoint,
            expected_rag=rag,
            expected_dataset_bundle_sha256=dataset_manifest["bundle_sha256"],
            require_pipeline_provenance=True,
        )

    base_checkpoint = {
        "path": str(args.base_ulip_checkpoint),
        "sha256": sha256_file(args.base_ulip_checkpoint),
    }
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "pipeline": "ikea_rag",
        "training_strategy": "stage_1" if args.stage == "stage1" else "stage_2",
        "created_at_unix": int(time.time()),
        "rag": rag,
        "dataset": {
            "manifest_file": "dataset_manifest.json",
            "bundle_sha256": dataset_manifest["bundle_sha256"],
            "train_samples": train_manifest["sample_count"],
            "val_samples": val_manifest["sample_count"],
            "train_ordered_ids_sha256": train_manifest["ordered_ids_sha256"],
            "val_ordered_ids_sha256": val_manifest["ordered_ids_sha256"],
            "config_path": train_manifest["dataset_config_path"],
            "config_sha256": train_manifest["dataset_config_sha256"],
        },
        "base_ulip_checkpoint": base_checkpoint,
        "stage1_checkpoint": (
            {"path": stage1_info["path"], "sha256": stage1_info["sha256"]}
            if stage1_info
            else None
        ),
        "git": _git_metadata(),
    }
    return train, val, dataset_manifest, provenance, stage1_info


def main(args: argparse.Namespace) -> None:
    _apply_smoke_defaults(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset, val_dataset, dataset_manifest, provenance, stage1_info = _preflight(args)
    summary = {
        "stage": args.stage,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "rag_corpus_rows": provenance["rag"]["corpus_rows"],
        "rag_index_dimension": provenance["rag"]["index_dimension"],
        "dataset_bundle_sha256": provenance["dataset"]["bundle_sha256"],
        "corpus_sha256": provenance["rag"]["corpus_sha256"],
        "index_sha256": provenance["rag"]["index_sha256"],
        "base_checkpoint_sha256": provenance["base_ulip_checkpoint"]["sha256"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.validate_only:
        print("Preflight validation passed; no model was constructed and no training ran.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("IKEA RAG training requires CUDA")
    torch.cuda.set_device(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "dataset_manifest.json", dataset_manifest)
    write_json_atomic(args.output_dir / "run_provenance.json", provenance)

    model = _make_model(args, provenance, stage1_info).cuda(args.gpu)
    criterion = models.ULIP_Loss_RAG_Enhanced(args).cuda(args.gpu)
    optimizer = _optimizer(model, args)
    scaler = amp.GradScaler(enabled=not args.disable_amp)
    start_epoch, best_metric, best_epoch = 0, float("-inf"), -1
    if args.resume:
        resume = _load_checkpoint(args.resume)
        start_epoch = int(resume.get("epoch", 0))
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        best_metric = float(resume.get("best_metric", float("-inf")))
        best_epoch = int(resume.get("best_epoch", -1))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=rag_collate_fn,
    )
    eval_dataset: torch.utils.data.Dataset = val_dataset
    if args.max_eval_items:
        eval_dataset = _Subset(val_dataset, args.max_eval_items)
    val_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=rag_collate_fn,
    )
    batches = min(len(train_loader), args.max_train_batches or len(train_loader))
    steps_per_epoch = max(1, math.ceil(batches / args.update_freq))
    lr_schedule = _schedule(
        args.lr,
        args.lr_end,
        args.lr_start,
        args.epochs,
        steps_per_epoch,
        args.warmup_epochs,
    )
    selection_metric = "retr/t2i_r1" if args.stage == "stage1" else "retr/s2t_r1"

    wandb_run = None
    if args.wandb and utils.is_main_process():
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.output_dir.name,
            config={**vars(args), "provenance": provenance},
        )

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scaler,
            lr_schedule,
            epoch,
            steps_per_epoch,
            args,
        )
        eval_stats: Dict[str, float] = {}
        if (epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs:
            eval_stats = evaluate(val_loader, model, args.stage, args)
        current = eval_stats.get(selection_metric, float("-inf"))
        is_best = current > best_metric
        if is_best:
            best_metric, best_epoch = current, epoch + 1

        _save_checkpoint(
            args.output_dir / "checkpoint_last.pt",
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            best_metric=best_metric,
            best_epoch=best_epoch,
            args=args,
            provenance=provenance,
        )
        if is_best:
            shutil.copyfile(
                args.output_dir / "checkpoint_last.pt",
                args.output_dir / "checkpoint_best.pt",
            )

        log = {
            "epoch": epoch + 1,
            "selection_metric": selection_metric,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            **{f"train/{key}": value for key, value in train_stats.items()},
            **eval_stats,
        }
        with (args.output_dir / "log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log, ensure_ascii=False) + "\n")
        print(json.dumps(log, ensure_ascii=False, indent=2))
        if wandb_run is not None:
            wandb_run.log(log)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(build_parser().parse_args())
