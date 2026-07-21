#!/usr/bin/env python3
"""Bind explicit visual QA decisions to an immutable geometry review queue."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import load_json, read_jsonl, write_jsonl
else:
    from .common import load_json, read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queue = read_jsonl(args.queue)
    payload = load_json(args.decisions)
    reviewer = payload.get("reviewer")
    reviewer_type = payload.get("reviewer_type")
    decisions = payload.get("decisions") or []
    if not reviewer or not reviewer_type:
        raise SystemExit("reviewer and reviewer_type are required")

    by_id = {}
    for decision in decisions:
        product_id = str(decision.get("product_id") or "")
        if not product_id or product_id in by_id:
            raise SystemExit(f"missing or duplicate product_id: {product_id!r}")
        if decision.get("decision") not in {"accept", "reject"}:
            raise SystemExit(f"invalid decision for {product_id}")
        by_id[product_id] = decision

    queue_ids = {str(row["product_id"]) for row in queue}
    decision_ids = set(by_id)
    if queue_ids != decision_ids:
        missing = sorted(queue_ids - decision_ids)
        extra = sorted(decision_ids - queue_ids)
        raise SystemExit(f"decision coverage mismatch missing={missing} extra={extra}")

    reviewed = []
    for row in queue:
        decision = by_id[str(row["product_id"])]
        row["decision"] = decision["decision"]
        row["reviewer"] = reviewer
        row["reviewer_type"] = reviewer_type
        row["notes"] = decision.get("notes")
        reviewed.append(row)
    write_jsonl(args.output, reviewed)
    accepted = sum(row["decision"] == "accept" for row in reviewed)
    rejected = sum(row["decision"] == "reject" for row in reviewed)
    print(f"reviewed={len(reviewed)} accepted={accepted} rejected={rejected} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
