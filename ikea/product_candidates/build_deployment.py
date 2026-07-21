"""Review or explicitly materialize an immutable product-candidate profile.

The default mode is dry-run: it validates policies, vectors, canonical row
identity, and a read-only MongoDB snapshot, then prints the deterministic
profile identity.  It creates no output directory.  Writing requires both
``--write`` and an explicit ``--output`` path.

Example review command::

    python -m ikea.product_candidates.build_deployment \
      --source-version ikea733-v1

After integration review only::

    python -m ikea.product_candidates.build_deployment \
      --source-version ikea733-v1 \
      --write --output /mnt/P300/data/ULIP/product_candidates/deployments/ikea733-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Optional, Sequence

from .bundles import (
    BundleError,
    build_deployment_bundle,
    load_deployment_profile,
    write_deployment_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path("/mnt/P300/data/ULIP/checkpoint_last.pt")
DEFAULT_VECTOR_DIR = Path("/mnt/P300/data/ikea_data/vectors")
DEFAULT_TOKENIZER_CODE = REPO_ROOT / "core/utils/tokenizer.py"
DEFAULT_TOKENIZER_BPE = REPO_ROOT / "core/utils/bpe_simple_vocab_16e6.txt.gz"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic 733-row product-candidate profile from the "
            "canonical ObjectId join. Default: validate only, write nothing."
        )
    )
    parser.add_argument(
        "--source-version",
        required=True,
        help="Explicit stable source label; no timestamp/random value is generated.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--vectors", type=Path, default=DEFAULT_VECTOR_DIR / "vectors_pc.npy"
    )
    parser.add_argument(
        "--meta", type=Path, default=DEFAULT_VECTOR_DIR / "meta_pc.jsonl"
    )
    parser.add_argument(
        "--vector-schema", type=Path, default=DEFAULT_VECTOR_DIR / "schema.json"
    )
    parser.add_argument("--tokenizer-code", type=Path, default=DEFAULT_TOKENIZER_CODE)
    parser.add_argument("--tokenizer-bpe", type=Path, default=DEFAULT_TOKENIZER_BPE)
    parser.add_argument(
        "--tokenizer-id", default="ulip-simple-tokenizer-clip77-v1"
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "policies",
    )
    parser.add_argument("--mongo-db", default="furniture_db")
    parser.add_argument("--catalog-collection", default="ikea_product")
    parser.add_argument(
        "--dimension-collection", default="ikea_product_v2_2026q2"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Materialize the reviewed profile (never enabled by default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New destination directory; required with --write and never overwritten.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.write != (args.output is not None):
        raise SystemExit("--write and --output must be supplied together")

    try:
        build = build_deployment_bundle(
            checkpoint_path=args.checkpoint,
            vectors_path=args.vectors,
            meta_path=args.meta,
            vector_schema_path=args.vector_schema,
            tokenizer_code_path=args.tokenizer_code,
            tokenizer_bpe_path=args.tokenizer_bpe,
            source_version=args.source_version,
            tokenizer_id=args.tokenizer_id,
            policy_dir=args.policy_dir,
            mongo_db=args.mongo_db,
            catalog_collection=args.catalog_collection,
            dimension_collection=args.dimension_collection,
        )
        report = {
            "mode": "write" if args.write else "dry_run_no_files_written",
            "profile_id": build.profile["profile_id"],
            "deployment_identity_sha256": build.profile[
                "deployment_identity_sha256"
            ],
            "source_version": build.profile["source_version"],
            "provenance": _jsonable(build.profile["provenance"]),
            "inventory": _jsonable(build.summary),
            "generated_member_count": len(build.files),
            "copied_source_member_count": len(build.source_files),
            "copied_source_members": sorted(build.source_files),
        }
        if args.write:
            output = write_deployment_bundle(build, args.output)
            verified = load_deployment_profile(output)
            report["output"] = str(output)
            report["offline_verification"] = bool(all(verified.checks.values()))
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except BundleError as exc:
        print(f"bundle validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
