#!/usr/bin/env python3
"""Expand a hash-bound full-sheet review attestation into row decisions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import load_json, read_jsonl, sha256_file, write_jsonl
else:
    from .common import load_json, read_jsonl, sha256_file, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--sheet-manifest", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queue = read_jsonl(args.queue)
    sheets = read_jsonl(args.sheet_manifest)
    attestation = load_json(args.attestation)
    if attestation.get("schema_version") != "ikea-geometry-bulk-review-attestation/2.0":
        raise SystemExit("invalid bulk review attestation schema")
    if attestation.get("queue_sha256") != sha256_file(args.queue):
        raise SystemExit("review queue hash mismatch")
    if attestation.get("sheet_manifest_sha256") != sha256_file(args.sheet_manifest):
        raise SystemExit("sheet manifest hash mismatch")
    if attestation.get("reviewed_all_rows") is not True:
        raise SystemExit("reviewed_all_rows must be explicitly true")
    reviewer = str(attestation.get("reviewer") or "").strip()
    reviewer_type = str(attestation.get("reviewer_type") or "").strip()
    if not reviewer or not reviewer_type:
        raise SystemExit("reviewer and reviewer_type are required")

    queue_ids = [str(row["product_id"]) for row in queue]
    sheet_ids = [str(row["product_id"]) for row in sheets]
    if len(queue_ids) != len(set(queue_ids)) or len(sheet_ids) != len(set(sheet_ids)):
        raise SystemExit("queue and sheet manifest product IDs must be unique")
    if set(queue_ids) != set(sheet_ids):
        raise SystemExit("sheet manifest must cover the exact review queue")

    rejects = {}
    for row in attestation.get("rejects") or []:
        product_id = str(row.get("product_id") or "")
        notes = str(row.get("notes") or "").strip()
        if not product_id or product_id in rejects or not notes:
            raise SystemExit(f"invalid or duplicate reject record: {product_id!r}")
        rejects[product_id] = notes
    unknown = set(rejects) - set(queue_ids)
    if unknown:
        raise SystemExit(f"reject IDs outside queue: {sorted(unknown)}")

    accept_note = str(attestation.get("accept_note") or "").strip()
    if not accept_note:
        raise SystemExit("accept_note is required")
    reviewed = []
    for row in queue:
        product_id = str(row["product_id"])
        row["decision"] = "reject" if product_id in rejects else "accept"
        row["reviewer"] = reviewer
        row["reviewer_type"] = reviewer_type
        row["notes"] = rejects.get(product_id, accept_note)
        row["review_attestation"] = {
            "path": str(args.attestation),
            "sha256": sha256_file(args.attestation),
            "sheet_manifest": str(args.sheet_manifest),
            "sheet_manifest_sha256": sha256_file(args.sheet_manifest),
        }
        reviewed.append(row)
    write_jsonl(args.output, reviewed)
    print(json.dumps({
        "reviewed": len(reviewed),
        "accepted": len(reviewed) - len(rejects),
        "rejected": len(rejects),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
