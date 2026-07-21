#!/usr/bin/env python3
"""Deterministic IKEA vanilla-vs-RAG self-retrieval benchmark.

This is deliberately an *in-sample diagnostic*: each selected product's own
caption is used as the query and the same product ID is the expected result.
It is useful for detecting integration regressions, but it is not a held-out
estimate of room-query or real-user retrieval quality.

The command is read-only unless ``--output`` is supplied.  It never connects
to MongoDB and writes the optional report atomically.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


try:
    from ikea.rag_serving import (
        DEFAULT_IKEA_CHECKPOINT,
        DEFAULT_RAG_CHECKPOINT,
        DEFAULT_VECTOR_DIR,
        RAGCompatibilityError,
        RAGServingAdapter,
        load_adapter,
    )
except ModuleNotFoundError:  # Allow ``python ikea/evaluate_rag_retrieval.py``.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from ikea.rag_serving import (  # type: ignore
        DEFAULT_IKEA_CHECKPOINT,
        DEFAULT_RAG_CHECKPOINT,
        DEFAULT_VECTOR_DIR,
        RAGCompatibilityError,
        RAGServingAdapter,
        load_adapter,
    )


SCHEMA_VERSION = "ikea-rag-self-retrieval-benchmark/1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CORPUS_DIR = Path(
    "/mnt/P300/data/ULIP/4090_cheng_archive_20260715/ULIP_RAG/rag_corpus_1095"
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def select_caption_rows(
    rows: Sequence[Mapping[str, Any]], *, limit: int, seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Select evaluable rows using a stable SHA-256 seeded ordering."""

    if limit <= 0:
        raise ValueError("limit must be positive")

    eligible: List[Dict[str, Any]] = []
    excluded = {
        "missing_caption": 0,
        "missing_product_id": 0,
        "missing_category": 0,
    }
    seen_ids: set[str] = set()
    for row_index, raw in enumerate(rows):
        product_id = str(raw.get("id") or "").strip()
        category = str(raw.get("category") or "").strip()
        caption = str(raw.get("caption") or "").strip()
        if not caption:
            excluded["missing_caption"] += 1
            continue
        if not product_id:
            excluded["missing_product_id"] += 1
            continue
        if not category:
            excluded["missing_category"] += 1
            continue
        if product_id in seen_ids:
            raise ValueError(f"duplicate product id in PC metadata: {product_id}")
        seen_ids.add(product_id)
        selection_key = hashlib.sha256(
            f"{seed}\0{product_id}\0{row_index}".encode("utf-8")
        ).digest()
        eligible.append(
            {
                "row_index": row_index,
                "id": product_id,
                "category": category,
                "caption": caption,
                "_selection_key": selection_key,
            }
        )

    eligible.sort(key=lambda row: (row["_selection_key"], row["id"], row["row_index"]))
    selected = eligible[: min(limit, len(eligible))]
    for row in selected:
        row.pop("_selection_key", None)
    return selected, excluded


def _rank_of(results: Sequence[Mapping[str, Any]], expected_id: str, depth: int) -> Optional[int]:
    for position, result in enumerate(results[:depth], start=1):
        if str(result.get("id") or "") == expected_id:
            return position
    return None


def _top_k_overlap(left_ids: Sequence[str], right_ids: Sequence[str], k: int) -> float:
    """Overlap coefficient for two top-k lists, adjusted for short categories."""

    left = list(left_ids[:k])
    right = list(right_ids[:k])
    denominator = min(k, max(len(left), len(right)))
    if denominator == 0:
        return 0.0
    return len(set(left).intersection(right)) / denominator


def evaluate_adapter(
    adapter: RAGServingAdapter,
    selected: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 10,
    rag_top_k: int = 5,
    include_per_query: bool = False,
) -> Dict[str, Any]:
    """Run both modes and return conservative metrics over all attempts."""

    if top_k < 10:
        raise ValueError("top_k must be at least 10 to compute MRR@10")
    if rag_top_k <= 0:
        raise ValueError("rag_top_k must be positive")
    if not selected:
        raise ValueError("no caption-bearing metadata rows were selected")

    modes = ("vanilla", "rag")
    accumulators: Dict[str, Dict[str, Any]] = {
        mode: {
            "recall_1_hits": 0,
            "recall_5_hits": 0,
            "reciprocal_rank_10_sum": 0.0,
            "completed": 0,
            "execution_errors": 0,
            "misses_at_10": 0,
        }
        for mode in modes
    }
    overlaps: List[float] = []
    failures: List[Dict[str, Any]] = []
    per_query: List[Dict[str, Any]] = []

    for sample in selected:
        expected_id = str(sample["id"])
        category = str(sample["category"])
        query = str(sample["caption"])
        mode_rows: Dict[str, Dict[str, Any]] = {}

        for mode in modes:
            try:
                response = adapter.search(
                    query,
                    top_k=top_k,
                    category=category,
                    mode=mode,
                    rag_top_k=rag_top_k,
                )
                results = list(response.get("results") or [])
                ids = [str(result.get("id") or "") for result in results]
                rank_10 = _rank_of(results, expected_id, 10)
                rank_all = _rank_of(results, expected_id, top_k)
                accumulators[mode]["completed"] += 1
                if rank_all == 1:
                    accumulators[mode]["recall_1_hits"] += 1
                if rank_all is not None and rank_all <= 5:
                    accumulators[mode]["recall_5_hits"] += 1
                if rank_10 is not None:
                    accumulators[mode]["reciprocal_rank_10_sum"] += 1.0 / rank_10
                else:
                    accumulators[mode]["misses_at_10"] += 1
                    failures.append(
                        {
                            "product_id": expected_id,
                            "category": category,
                            "mode": mode,
                            "type": "expected_not_in_top_10",
                        }
                    )
                mode_rows[mode] = {
                    "rank_at_search_depth": rank_all,
                    "rank_at_10": rank_10,
                    "result_ids": ids,
                }
            except Exception as exc:  # Continue so failures remain in the denominator.
                accumulators[mode]["execution_errors"] += 1
                accumulators[mode]["misses_at_10"] += 1
                failures.append(
                    {
                        "product_id": expected_id,
                        "category": category,
                        "mode": mode,
                        "type": "execution_error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                mode_rows[mode] = {
                    "rank_at_search_depth": None,
                    "rank_at_10": None,
                    "result_ids": [],
                }

        overlap = _top_k_overlap(
            mode_rows["vanilla"]["result_ids"],
            mode_rows["rag"]["result_ids"],
            top_k,
        )
        overlaps.append(overlap)
        if include_per_query:
            per_query.append(
                {
                    "product_id": expected_id,
                    "row_index": int(sample["row_index"]),
                    "category": category,
                    "caption_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                    "top_k_overlap": overlap,
                    "vanilla": {
                        "rank_at_search_depth": mode_rows["vanilla"]["rank_at_search_depth"],
                        "rank_at_10": mode_rows["vanilla"]["rank_at_10"],
                    },
                    "rag": {
                        "rank_at_search_depth": mode_rows["rag"]["rank_at_search_depth"],
                        "rank_at_10": mode_rows["rag"]["rank_at_10"],
                    },
                }
            )

    denominator = len(selected)
    metrics: Dict[str, Any] = {}
    for mode in modes:
        acc = accumulators[mode]
        metrics[mode] = {
            "attempted": denominator,
            "completed": acc["completed"],
            "execution_errors": acc["execution_errors"],
            "recall_at_1": acc["recall_1_hits"] / denominator,
            "recall_at_5": acc["recall_5_hits"] / denominator,
            "mrr_at_10": acc["reciprocal_rank_10_sum"] / denominator,
            "misses_at_10": acc["misses_at_10"],
        }

    result: Dict[str, Any] = {
        "metrics": metrics,
        "comparison": {
            "top_k": top_k,
            "mean_top_k_overlap": sum(overlaps) / denominator,
            "rag_minus_vanilla": {
                "recall_at_1": metrics["rag"]["recall_at_1"]
                - metrics["vanilla"]["recall_at_1"],
                "recall_at_5": metrics["rag"]["recall_at_5"]
                - metrics["vanilla"]["recall_at_5"],
                "mrr_at_10": metrics["rag"]["mrr_at_10"]
                - metrics["vanilla"]["mrr_at_10"],
            },
        },
        "failures": {
            "count": len(failures),
            "items": failures,
        },
    }
    if include_per_query:
        result["per_query"] = per_query
    return result


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with fsync + same-directory atomic rename."""

    path = path.expanduser().resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {path.parent}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _corpus_dir_for(profile: str, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    if profile == "legacy1095":
        return LEGACY_CORPUS_DIR
    raise ValueError("--profile provenance requires --corpus-dir")


def _validate_adapter_profile(adapter: RAGServingAdapter, requested: str) -> None:
    """Confirm the adapter actually completed the requested validation policy."""

    status = adapter.status()
    actual = status.get("profile")
    if actual != requested:
        raise RAGCompatibilityError(
            f"adapter profile mismatch: requested={requested!r}, actual={actual!r}"
        )
    validation = status.get("profile_validation")
    if not isinstance(validation, Mapping):
        raise RAGCompatibilityError("adapter did not report profile validation")
    if requested == "legacy1095" and not validation.get(
        "immutable_bundle_validated"
    ):
        raise RAGCompatibilityError("legacy1095 immutable bundle was not validated")
    if requested == "provenance" and not validation.get("schema_version"):
        raise RAGCompatibilityError("provenance chain was not validated")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--top-k", type=int, default=10, help="Search depth and overlap K; must be >= 10."
    )
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument(
        "--profile", choices=("legacy1095", "provenance"), default="legacy1095"
    )
    parser.add_argument("--ikea-checkpoint", type=Path, default=DEFAULT_IKEA_CHECKPOINT)
    parser.add_argument("--rag-checkpoint", type=Path, default=DEFAULT_RAG_CHECKPOINT)
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a concrete device such as cuda:0"
    )
    parser.add_argument("--output", type=Path, help="Optional atomic JSON output path.")
    parser.add_argument("--include-per-query", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.top_k < 10:
        raise ValueError("--top-k must be at least 10")
    if args.rag_top_k <= 0:
        raise ValueError("--rag-top-k must be positive")
    corpus_dir = _corpus_dir_for(args.profile, args.corpus_dir)
    device = None if args.device == "auto" else args.device

    # Core constructors currently print diagnostics.  Preserve JSON-only
    # stdout by routing those messages to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        adapter = load_adapter(
            ikea_checkpoint=args.ikea_checkpoint,
            rag_checkpoint=args.rag_checkpoint,
            corpus_dir=corpus_dir,
            corpus_profile=args.profile,
            vector_dir=args.vector_dir,
            device=device,
            rag_top_k=args.rag_top_k,
        )
    _validate_adapter_profile(adapter, args.profile)
    selected, excluded = select_caption_rows(
        adapter.pc_meta, limit=args.limit, seed=args.seed
    )
    benchmark = evaluate_adapter(
        adapter,
        selected,
        top_k=args.top_k,
        rag_top_k=args.rag_top_k,
        include_per_query=args.include_per_query,
    )
    meta_path = args.vector_dir / "meta_pc.jsonl"
    vector_path = args.vector_dir / "vectors_pc.npy"
    adapter_status = adapter.status()
    adapter_artifacts = dict(adapter_status.get("artifacts", {}))
    adapter_summary = {
        key: value for key, value in adapter_status.items() if key != "artifacts"
    }
    adapter_artifacts.update(
        {
            "meta_pc": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
            "vectors_pc": {
                "path": str(vector_path),
                "sha256": sha256_file(vector_path),
            },
        }
    )
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "evaluation": {
            "kind": "in_sample_caption_self_retrieval",
            "is_held_out": False,
            "warning": (
                "Each query is the indexed product's own caption. These metrics are an "
                "integration diagnostic and must not be reported as held-out retrieval quality."
            ),
            "query": "meta_pc.caption",
            "expected": "same meta_pc.id",
            "filter": "exact meta_pc.category (case-insensitive equality)",
            "failure_policy": "execution errors and missing expected IDs remain misses in the denominator",
            "overlap_definition": (
                "|set(vanilla[:K]) intersection set(rag[:K])| / "
                "min(K, max(returned lengths))"
            ),
        },
        "selection": {
            "method": "SHA-256 ordering of seed, product_id, and canonical row_index",
            "seed": args.seed,
            "limit_requested": args.limit,
            "selected_count": len(selected),
            "eligible_count": len(adapter.pc_meta) - sum(excluded.values()),
            "metadata_rows": len(adapter.pc_meta),
            "excluded": excluded,
            "selected": [
                {
                    "row_index": int(row["row_index"]),
                    "id": row["id"],
                    "category": row["category"],
                    "caption_sha256": hashlib.sha256(
                        str(row["caption"]).encode("utf-8")
                    ).hexdigest(),
                }
                for row in selected
            ],
        },
        "configuration": {
            "profile_requested": args.profile,
            "device": str(adapter.device),
            "top_k": args.top_k,
            "rag_top_k": args.rag_top_k,
            "ikea_checkpoint": str(args.ikea_checkpoint),
            "rag_checkpoint": str(args.rag_checkpoint),
            "corpus_dir": str(corpus_dir),
            "vector_dir": str(args.vector_dir),
        },
        "adapter_status": adapter_summary,
        "artifacts": adapter_artifacts,
        **benchmark,
    }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
        if args.output is not None:
            atomic_write_json(args.output, report)
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        error = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        json.dump(error, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
