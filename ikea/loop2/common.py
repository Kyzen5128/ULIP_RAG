from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_TAXONOMY = HERE / "taxonomy_v1.json"
DEFAULT_VERSION = os.environ.get("ULIP2_IKEA_VERSION", "ikea-us-20260716-v1")
DEFAULT_ROOT = Path(os.environ.get("ULIP2_IKEA_ROOT", "/mnt/P300/data/ULIP/ULIP_RAG_2_0"))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            count += 1
    temp.replace(path)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(*values: Any) -> str:
    return " ".join(str(v or "").strip().casefold() for v in values if v is not None)


def contains_pattern(text: str, pattern: str) -> bool:
    escaped = re.escape(pattern.casefold()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text.casefold()) is not None


def snapshot_root(version: str, base_root: Path = DEFAULT_ROOT) -> Path:
    return base_root / version


def split_name_for_group(
    group: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> str:
    """Return the deterministic family split used by the IKEA dataset loader.

    Keeping this in one shared helper prevents the generated corpus and the
    multimodal samples from silently assigning a product family to different
    splits. The MD5 here is only a stable partition function, not a security
    primitive.
    """
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios cannot be negative")
    total = train_ratio + val_ratio + test_ratio
    if total <= 0 or total > 1.000001:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be in (0, 1]")
    value = int(hashlib.md5(str(group).encode("utf-8")).hexdigest(), 16) % 10000
    if value < int(train_ratio * 10000):
        return "train"
    if value < int((train_ratio + val_ratio) * 10000) or test_ratio == 0:
        return "val"
    return "test"
