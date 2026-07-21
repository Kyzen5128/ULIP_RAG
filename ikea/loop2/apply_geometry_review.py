#!/usr/bin/env python3
"""Publish a reviewed geometry manifest after verifying every evidence hash."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ikea.loop2.common import read_jsonl, sha256_file, write_json, write_jsonl
else:
    from .common import read_jsonl, sha256_file, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.geometry_manifest)
    decisions = {row["product_id"]: row for row in read_jsonl(args.decisions)}
    for row in rows:
        decision = decisions.get(row["product_id"])
        if not decision:
            continue
        if decision.get("decision") not in {"accept", "reject", "pending"}:
            raise SystemExit(f"invalid decision for {row['product_id']}")
        if sha256_file(Path(decision["canonical_glb"])) != decision.get("canonical_glb_sha256"):
            raise SystemExit(f"canonical GLB drift for {row['product_id']}")
        if sha256_file(Path(decision["source_image"])) != decision.get("source_image_sha256"):
            raise SystemExit(f"source image drift for {row['product_id']}")
        for evidence in decision.get("render_evidence") or []:
            if sha256_file(Path(evidence["path"])) != evidence.get("sha256"):
                raise SystemExit(f"render drift for {row['product_id']}")
        if decision["decision"] in {"accept", "reject"} and not decision.get("reviewer"):
            raise SystemExit(f"reviewer is required for {row['product_id']}")
        if decision["decision"] == "accept":
            row["status"] = "ok"
        elif decision["decision"] == "reject":
            row["status"] = "rejected_visual"
        row["visual_review"] = {
            key: decision.get(key) for key in ("decision", "reviewer", "reviewer_type", "notes")
        }
        row["visual_review"]["evidence_bundle_sha256"] = sha256_file(args.decisions)
    write_jsonl(args.output, rows)
    report = {
        "schema_version": "ikea-geometry-review-application/2.0",
        "base_manifest": str(args.geometry_manifest),
        "base_manifest_sha256": sha256_file(args.geometry_manifest),
        "decisions": str(args.decisions),
        "decisions_sha256": sha256_file(args.decisions),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "status_counts": dict(sorted(Counter(row.get("status") for row in rows).items())),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output.with_suffix(".manifest.json"), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
