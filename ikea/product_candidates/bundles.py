"""Build and verify immutable product-candidate deployment profiles.

The production row identity is deliberately narrow and non-negotiable::

    vectors_pc.npy row i
      <-> meta_pc.jsonl line i
      <-> ObjectId(meta["id"])
      <-> MongoDB document _id

There is no ASIN, URL, name, cursor-order, or fuzzy fallback.  MongoDB is used
only while building a snapshot.  :func:`load_deployment_profile` is completely
offline and verifies every source/member hash before returning a ready profile.

Canonical JSON uses sorted keys, compact separators, UTF-8, no NaN, and one
trailing newline.  JSONL uses the same encoding for every row.  No wall-clock
time, random identifier, or filesystem metadata is included, so rebuilding an
unchanged explicit ``source_version`` produces identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np


EXPECTED_ROWS = 733
EXPECTED_EMBEDDING_DIM = 512
OBJECT_ID_RE = re.compile(r"^[0-9a-f]{24}$")
PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_DIR = PACKAGE_DIR / "policies"

CATEGORY_POLICY_FILE = "category_v2t36_to_ikea15_v1.json"
DIMENSION_POLICY_FILE = "dimension_axis_v1.json"
QUALITY_POLICY_FILE = "quality_overrides_v1.json"

V2T_36 = {
    "sofa", "couch", "loveseat", "armchair", "accent chair", "lounge chair",
    "chair", "stool", "bench", "ottoman", "table", "dining table",
    "coffee table", "side table", "end table", "console table", "desk",
    "nightstand", "bedside table", "bedside cabinet", "wardrobe", "closet",
    "armoire", "dresser", "cabinet", "bookshelf", "bookcase", "shelf",
    "shelving unit", "tv stand", "media console", "bed", "bed frame", "lamp",
    "floor lamp", "rug",
}

IKEA_15 = {
    "Sofa", "Storage Ottoman", "Bench", "Dining Chair", "Bar Stool",
    "Dining Table", "Wardrobe", "Bookshelf", "Sideboard", "Office Desk",
    "Vanity Table", "Bed", "Filing Cabinet", "TV Stand", "Recliner",
}

PROVENANCE_KEYS = (
    "model_checkpoint_sha256",
    "vector_bundle_sha256",
    "catalog_manifest_sha256",
    "dimension_bundle_sha256",
    "tokenizer_bundle_sha256",
    "category_policy_version",
    "dimension_axis_policy_version",
)

CHECK_KEYS = (
    "model_checkpoint",
    "vector_bundle",
    "catalog_manifest",
    "dimension_bundle",
    "tokenizer_bundle",
    "row_identity_join",
    "policy_versions",
)


class BundleError(RuntimeError):
    """Raised when a source snapshot or deployment bundle is unsafe."""


@dataclass(frozen=True)
class BundleBuild:
    """In-memory deterministic build; no filesystem output is implied."""

    files: Mapping[str, bytes]
    source_files: Mapping[str, Path]
    profile: Mapping[str, Any]
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class DeploymentProfile:
    """Verified runtime inputs returned by :func:`load_deployment_profile`."""

    profile_id: str
    deployment_profile_id: str
    provenance: Mapping[str, Any]
    checks: Mapping[str, bool]
    inventory: Mapping[str, Any]
    inventory_details: Mapping[str, Any]
    vectors: np.ndarray
    catalog_rows: tuple[Mapping[str, Any], ...]
    dimension_rows: tuple[Mapping[str, Any], ...]
    rows: tuple[Mapping[str, Any], ...]
    tokenizer_id: str
    tokenizer_code_path: Path
    tokenizer_bpe_path: Path
    model_checkpoint_path: Path
    model: Any = None
    tokenizer: Any = None


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one accepted canonical JSON representation."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BundleError(f"value is not canonical-JSON serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    encoded: list[bytes] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise BundleError(f"JSONL row {index} is not an object")
        encoded.append(canonical_json_bytes(dict(row)))
    return b"".join(encoded)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_member_descriptor(
    path: Path, relative_path: str, *, logical_name: str
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise BundleError(f"missing {logical_name}: {path}")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise BundleError(f"invalid bundle path for {logical_name}: {relative_path!r}")
    return {
        "logical_name": logical_name,
        "bundle_path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "content_sha256": sha256_file(path),
    }


def _member_descriptor(
    relative_path: str, data: bytes, *, logical_name: str, rows: Optional[int] = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "logical_name": logical_name,
        "bundle_path": relative_path,
        "bytes": len(data),
        "content_sha256": sha256_bytes(data),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"JSON root must be an object: {path}")
    return value


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid canonical JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"canonical JSON root must be an object: {path}")
    expected = canonical_json_bytes(value)
    if raw != expected:
        raise BundleError(f"non-canonical JSON bytes: {path}")
    return value, raw


def _read_canonical_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = Path(path).read_bytes()
    rows: list[dict[str, Any]] = []
    try:
        for index, line in enumerate(raw.decode("utf-8").splitlines()):
            if not line:
                raise BundleError(f"blank JSONL row {index}: {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BundleError(f"JSONL row {index} is not an object: {path}")
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid canonical JSONL {path}: {exc}") from exc
    expected = canonical_jsonl_bytes(rows)
    if raw != expected:
        raise BundleError(f"non-canonical JSONL bytes: {path}")
    return rows, raw


def _validate_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise BundleError(f"{label} must be lowercase SHA-256")
    return value


def _safe_bundle_member(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise BundleError("bundle member path must be non-empty")
    root = Path(root).resolve()
    path = (root / relative_path).resolve()
    if root != path and root not in path.parents:
        raise BundleError(f"bundle member escapes profile root: {relative_path!r}")
    return path


def _load_policy_set(policy_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    policy_dir = Path(policy_dir)
    paths = {
        "category": policy_dir / CATEGORY_POLICY_FILE,
        "dimension": policy_dir / DIMENSION_POLICY_FILE,
        "quality": policy_dir / QUALITY_POLICY_FILE,
    }
    policies = {name: _read_json(path) for name, path in paths.items()}
    _validate_policies(policies)
    canonical = {name: canonical_json_bytes(value) for name, value in policies.items()}
    return policies, canonical


def _validate_policies(policies: Mapping[str, Mapping[str, Any]]) -> None:
    category = policies["category"]
    if category.get("policy_version") != "v2t36_to_ikea15_v1":
        raise BundleError("unexpected category policy version")
    canonical = category.get("canonical_categories")
    if not isinstance(canonical, list) or set(canonical) != IKEA_15 or len(canonical) != 15:
        raise BundleError("category policy must declare the exact IKEA-15 taxonomy")
    terms = category.get("request_terms")
    if not isinstance(terms, Mapping) or set(terms) != V2T_36 or len(terms) != 36:
        raise BundleError("category policy must cover the exact V2T-36 request terms")
    valid_status = {
        "enabled",
        "mapped_blocked",
        "ambiguous_unresolved",
        "catalog_gap_unsupported",
        "controlled_unsupported",
    }
    for term, rule in terms.items():
        if not isinstance(rule, Mapping) or rule.get("status") not in valid_status:
            raise BundleError(f"invalid category rule for {term!r}")
        targets = rule.get("canonical_categories")
        if not isinstance(targets, list) or len(targets) != len(set(targets)):
            raise BundleError(f"invalid canonical category list for {term!r}")
        if not set(targets).issubset(IKEA_15):
            raise BundleError(f"unknown canonical target for {term!r}")
        if rule.get("status") in {"enabled", "mapped_blocked"} and not targets:
            raise BundleError(f"mapped term {term!r} has no target")
        if rule.get("status") not in {"enabled", "mapped_blocked"} and targets:
            raise BundleError(f"unresolved/unsupported term {term!r} must fail closed")
    normalization = category.get("normalization")
    if not isinstance(normalization, Mapping):
        raise BundleError("category normalization policy is missing")
    if set(normalization) != {
        "casefold", "strip", "internal_whitespace", "cross_tier_policy"
    } or normalization.get("casefold") is not True or normalization.get("strip") is not True or normalization.get("internal_whitespace") != "preserve":
        raise BundleError("category normalization must not collapse internal whitespace")
    tier_policy = normalization.get("cross_tier_policy")
    if not isinstance(tier_policy, Mapping) or set(tier_policy) != {
        "raw_request_term_casefold_duplicate",
        "distinct_aliases_same_canonical",
        "canonical_overlap_warning",
    } or tier_policy.get("raw_request_term_casefold_duplicate") != "reject_http_422" or tier_policy.get("distinct_aliases_same_canonical") != "primary_precedence_drop_secondary" or tier_policy.get("canonical_overlap_warning") != "CATEGORY_CANONICAL_POOL_DEDUPED":
        raise BundleError("category cross-tier duplicate policy drifted")
    if terms["chair"]["status"] != "controlled_unsupported" or terms["stool"]["status"] != "controlled_unsupported":
        raise BundleError("controlled chair/stool terms must remain unsupported")
    if terms["bed"]["status"] != "mapped_blocked" or terms["bed frame"]["status"] != "mapped_blocked":
        raise BundleError("Bed request term must be mapped-but-blocked")
    exact_enabled = {
        "sofa": ["Sofa"],
        "couch": ["Sofa"],
        "loveseat": ["Sofa"],
        "bench": ["Bench"],
        "ottoman": ["Storage Ottoman"],
        "dining table": ["Dining Table"],
        "desk": ["Office Desk"],
        "wardrobe": ["Wardrobe"],
        "closet": ["Wardrobe"],
        "armoire": ["Wardrobe"],
        "bookshelf": ["Bookshelf"],
        "bookcase": ["Bookshelf"],
        "shelf": ["Bookshelf"],
        "shelving unit": ["Bookshelf"],
        "tv stand": ["TV Stand"],
        "media console": ["TV Stand"],
    }
    actual_enabled = {
        term: rule["canonical_categories"]
        for term, rule in terms.items()
        if rule["status"] == "enabled"
    }
    if actual_enabled != exact_enabled:
        raise BundleError("enabled V2T alias mapping drifted from the reviewed exact map")
    actual_blocked = {
        term: rule["canonical_categories"]
        for term, rule in terms.items()
        if rule["status"] == "mapped_blocked"
    }
    if actual_blocked != {"bed": ["Bed"], "bed frame": ["Bed"]}:
        raise BundleError("mapped-blocked aliases must be exactly bed and bed frame")

    dimension = policies["dimension"]
    if dimension.get("policy_version") != "dimension_axis_v1":
        raise BundleError("unexpected dimension-axis policy version")
    height = dimension.get("height_policy_by_canonical_category")
    if not isinstance(height, Mapping) or set(height) != IKEA_15:
        raise BundleError("dimension policy must cover all IKEA-15 categories")
    for category_name, rule in height.items():
        if not isinstance(rule, Mapping) or rule.get("policy") not in {
            "optional", "required", "unknown"
        }:
            raise BundleError(f"invalid height policy for {category_name}")
    if height["Wardrobe"]["policy"] != "required" or height["Bookshelf"]["policy"] != "required":
        raise BundleError("Wardrobe and Bookshelf must require height")
    if height["Filing Cabinet"]["policy"] != "unknown":
        raise BundleError("unverified Filing Cabinet height must fail closed as unknown")
    behavior = dimension.get("missing_height_behavior")
    if not isinstance(behavior, Mapping) or behavior.get("required") != "fail_closed_HEIGHT_UNVERIFIED" or behavior.get("unknown") != "fail_closed_HEIGHT_UNVERIFIED":
        raise BundleError("required/unknown missing height must fail closed")

    quality = policies["quality"]
    if quality.get("policy_version") != "quality_overrides_v1":
        raise BundleError("unexpected quality policy version")
    expected = quality.get("expected_inventory")
    exact_expected_inventory = {
        "total_rows": 733,
        "bed_rows": 49,
        "sofa_rows": 50,
        "known_special_sofa_rows": 32,
        "unverified_sofa_rows": 18,
        "bar_stool_rows": 50,
        "string_parser_collapse_rows": 8,
        "paired_dict_range_rows": 7,
        "max_only_rows": 2,
    }
    if not isinstance(expected, Mapping) or dict(expected) != exact_expected_inventory:
        raise BundleError("quality policy expected inventory is missing")
    category_overrides = quality.get("category_overrides")
    if not isinstance(category_overrides, Mapping):
        raise BundleError("quality category overrides must be an object")
    if category_overrides.get("Bed", {}).get("catalog_quality_status") != "blocked":
        raise BundleError("Bed must be blocked")
    if not category_overrides.get("Sofa", {}).get("all_rows_shape_fail_closed"):
        raise BundleError("all Sofa rows must fail closed")
    if category_overrides.get("Bar Stool", {}).get("catalog_quality_status") != "degraded":
        raise BundleError("Bar Stool must be degraded")

    sofa = quality.get("sofa_special_evidence")
    if not isinstance(sofa, Mapping):
        raise BundleError("Sofa evidence must be an object")
    sofa_ids: list[str] = []
    expected_sofa_groups = {
        "sofa_with_chaise": 10,
        "sectional_corner": 10,
        "sectional_chaise_open": 11,
        "chaise_add_on_only": 1,
    }
    if set(sofa) != set(expected_sofa_groups):
        raise BundleError("unexpected Sofa evidence groups")
    for group, expected_count in expected_sofa_groups.items():
        ids = sofa[group].get("ids")
        if not isinstance(ids, list) or len(ids) != expected_count:
            raise BundleError(f"Sofa evidence count mismatch for {group}")
        sofa_ids.extend(ids)
    _validate_object_id_list(sofa_ids, "Sofa evidence", unique=True)
    if len(sofa_ids) != 32:
        raise BundleError("Sofa evidence must contain exactly 32 rows")
    if any(
        "SPECIAL_FOOTPRINT_UNMODELED" not in set(rule.get("special_flags") or [])
        for rule in sofa.values()
    ):
        raise BundleError("every known Sofa group must be special-footprint unmodeled")
    sofa_remainder = quality.get("sofa_unverified_remainder")
    if not isinstance(sofa_remainder, Mapping) or sofa_remainder.get("expected_count") != 18 or set(sofa_remainder.get("special_flags") or []) != {
        "FOOTPRINT_SHAPE_UNVERIFIED", "SPECIAL_FOOTPRINT_UNMODELED"
    }:
        raise BundleError("the 18 unreviewed Sofa rows must remain shape-unverified")

    ranges = quality.get("range_evidence")
    expected_range_groups = {
        "string_parser_collapse": 8,
        "paired_dict_range": 7,
        "max_only": 2,
    }
    if not isinstance(ranges, Mapping) or set(ranges) != set(expected_range_groups):
        raise BundleError("range evidence must contain the three reviewed classes")
    range_ids: list[str] = []
    for group, expected_count in expected_range_groups.items():
        ids = ranges[group].get("ids")
        if not isinstance(ids, list) or len(ids) != expected_count:
            raise BundleError(f"range evidence count mismatch for {group}")
        range_ids.extend(ids)
    _validate_object_id_list(range_ids, "range evidence", unique=True)
    string_rule = ranges["string_parser_collapse"]
    slash = string_rule.get("slash_multivalue_ids")
    hyphen = string_rule.get("hyphen_ids")
    if not isinstance(slash, list) or not isinstance(hyphen, list) or set(slash) | set(hyphen) != set(string_rule["ids"]) or set(slash) & set(hyphen):
        raise BundleError("string collapse slash/hyphen evidence is inconsistent")
    if len(slash) != 6 or len(hyphen) != 2:
        raise BundleError("string collapse evidence must be six slash and two hyphen rows")
    exact_range_states = {
        "string_parser_collapse": ("range_collapsed", "range_collapsed", set()),
        "paired_dict_range": ("suspect", "suspect", {"DIMENSION_RANGE_PRESENT"}),
        "max_only": ("suspect", "suspect", {"DIMENSION_MAX_ONLY"}),
    }
    for group, (status, quality_status, flags) in exact_range_states.items():
        rule = ranges[group]
        if rule.get("dimension_status") != status or rule.get("dimension_quality_status") != quality_status or rule.get("force_hard_filter_ineligible") is not True or set(rule.get("special_flags") or []) != flags:
            raise BundleError(f"reviewed range state drifted for {group}")


def _validate_object_id_list(values: Sequence[Any], label: str, *, unique: bool) -> None:
    if any(not isinstance(value, str) or not OBJECT_ID_RE.fullmatch(value) for value in values):
        raise BundleError(f"{label} contains an invalid ObjectId string")
    if unique and len(values) != len(set(values)):
        raise BundleError(f"{label} contains duplicate ObjectIds")


def _load_ordered_meta(meta_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(meta_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise BundleError(f"blank meta row at line {line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise BundleError(f"meta row {line_number} is not an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read ordered metadata {meta_path}: {exc}") from exc
    if len(rows) != EXPECTED_ROWS:
        raise BundleError(f"ordered metadata must have {EXPECTED_ROWS} rows, got {len(rows)}")
    ids = [row.get("id") for row in rows]
    _validate_object_id_list(ids, "ordered metadata", unique=True)
    categories = {row.get("category") for row in rows}
    if categories != IKEA_15:
        raise BundleError(f"ordered metadata taxonomy drifted: {sorted(categories)}")
    for index, row in enumerate(rows):
        json_path = row.get("json_path")
        if not isinstance(json_path, str) or Path(json_path).stem != row["id"]:
            raise BundleError(f"meta row {index} does not bind id to JSON filename")
        if not isinstance(row.get("category"), str) or not row["category"]:
            raise BundleError(f"meta row {index} has no category")
    return rows


def _canonical_bson_sha256(document: Mapping[str, Any]) -> str:
    try:
        from bson.json_util import CANONICAL_JSON_OPTIONS, dumps
    except ImportError as exc:  # pragma: no cover - deployment environment guard
        raise BundleError("pymongo/bson is required to hash MongoDB documents") from exc
    encoded = dumps(document, json_options=CANONICAL_JSON_OPTIONS, sort_keys=True)
    value = json.loads(encoded)
    return sha256_bytes(canonical_json_bytes(value))


def _bson_value_to_json(value: Any) -> Any:
    """Convert a selected BSON value to deterministic extended JSON."""

    try:
        from bson.json_util import CANONICAL_JSON_OPTIONS, dumps
    except ImportError as exc:  # pragma: no cover
        raise BundleError("pymongo/bson is required to convert MongoDB values") from exc
    return json.loads(dumps(value, json_options=CANONICAL_JSON_OPTIONS, sort_keys=True))


def _fetch_mongo_snapshot(
    ordered_meta: Sequence[Mapping[str, Any]],
    *,
    mongo_db: str,
    catalog_collection: str,
    dimension_collection: str,
    mongo_client: Any = None,
) -> dict[str, Any]:
    try:
        from bson import ObjectId
    except ImportError as exc:  # pragma: no cover
        raise BundleError("pymongo/bson is required for canonical ObjectId joins") from exc

    owns_client = mongo_client is None
    if owns_client:
        try:
            from ikea.mongo_conn import get_client
        except ImportError:
            from mongo_conn import get_client  # type: ignore
        mongo_client = get_client(appname="ulip-product-candidate-bundle-readonly")

    object_ids = [ObjectId(str(row["id"])) for row in ordered_meta]
    try:
        database = mongo_client[mongo_db]
        database.command("ping")
        catalog_col = database[catalog_collection]
        dimension_col = database[dimension_collection]
        # Only find/count operations are used.  Results are keyed by ObjectId;
        # Mongo cursor order is never observed as product row order.
        catalog_docs = {
            str(document["_id"]): document
            for document in catalog_col.find({"_id": {"$in": object_ids}})
        }
        dimension_docs = {
            str(document["_id"]): document
            for document in dimension_col.find({"_id": {"$in": object_ids}})
        }
        collection_counts = {
            "catalog": int(catalog_col.count_documents({})),
            "dimension": int(dimension_col.count_documents({})),
        }
    finally:
        if owns_client:
            mongo_client.close()

    if len(catalog_docs) != EXPECTED_ROWS:
        missing = [row["id"] for row in ordered_meta if row["id"] not in catalog_docs]
        raise BundleError(
            f"canonical catalog ObjectId join must be {EXPECTED_ROWS}/{EXPECTED_ROWS}; "
            f"missing={missing[:8]} count={len(missing)}"
        )
    if len(dimension_docs) > EXPECTED_ROWS:
        raise BundleError("dimension ObjectId join returned more rows than the manifest")

    for index, row in enumerate(ordered_meta):
        product_id = str(row["id"])
        catalog_category = str(catalog_docs[product_id].get("main_type") or "").strip()
        if catalog_category != row["category"]:
            raise BundleError(
                f"catalog category mismatch at row {index}: meta={row['category']!r}, "
                f"mongo={catalog_category!r}"
            )
        dimension_doc = dimension_docs.get(product_id)
        if dimension_doc is not None:
            dimension_category = str(dimension_doc.get("main_type") or "").strip()
            if dimension_category != row["category"]:
                raise BundleError(
                    f"dimension category mismatch at row {index}: meta={row['category']!r}, "
                    f"mongo={dimension_category!r}"
                )

    return {
        "catalog_docs": catalog_docs,
        "dimension_docs": dimension_docs,
        "collection_counts": collection_counts,
    }


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        return None
    return result


def _build_catalog_rows(
    ordered_meta: Sequence[Mapping[str, Any]],
    catalog_docs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, meta in enumerate(ordered_meta):
        product_id = str(meta["id"])
        document = catalog_docs[product_id]
        name = _optional_string(document.get("name"))
        if name is None:
            raise BundleError(f"catalog product {product_id} has no name")
        price_display = _optional_string(document.get("price_string"))
        if price_display is None and document.get("price") is not None:
            price_display = str(document.get("price"))
        rows.append(
            {
                "row_index": row_index,
                "product_id": product_id,
                "name": name,
                "category": str(meta["category"]),
                "caption": _optional_string(meta.get("caption")),
                "catalog_metadata": {
                    "brand": _optional_string(document.get("brand")),
                    "price_display": price_display,
                    "url": _optional_string(document.get("url")),
                    "source_type": _optional_string(document.get("type")),
                },
                "preview_asset_id": product_id,
                "catalog_source": {
                    "collection": "ikea_product",
                    "document_sha256": _canonical_bson_sha256(document),
                },
            }
        )
    return rows


def _quality_indexes(quality: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    sofa_index: dict[str, dict[str, Any]] = {}
    for group, rule in quality["sofa_special_evidence"].items():
        for product_id in rule["ids"]:
            sofa_index[product_id] = {
                "group": group,
                "special_flags": list(rule["special_flags"]),
            }

    range_index: dict[str, dict[str, Any]] = {}
    for group, rule in quality["range_evidence"].items():
        for product_id in rule["ids"]:
            flags = list(rule.get("special_flags") or [])
            if group == "string_parser_collapse":
                flags.append(
                    "DIMENSION_RANGE_COLLAPSED_SLASH"
                    if product_id in rule["slash_multivalue_ids"]
                    else "DIMENSION_RANGE_COLLAPSED_HYPHEN"
                )
            range_index[product_id] = {
                "group": group,
                "dimension_status": rule["dimension_status"],
                "dimension_quality_status": rule["dimension_quality_status"],
                "special_flags": flags,
            }
    return sofa_index, range_index


def _string_parser_range_class(raw_value: Any) -> str:
    """Re-evaluate the two lossy branches in ``parse_str_format``.

    This intentionally mirrors ``ikea/phase_a_parse_dimensions.py``: quote
    removal, ``x`` tokenization, hyphen lower-bound collapse, and the exact
    full-token ``digits/digits`` alternate-size rule.  Mixed or absent evidence
    is rejected rather than guessed.
    """

    if not isinstance(raw_value, str):
        raise BundleError("string-parser range evidence is not a string")
    cleaned = (
        raw_value.replace('"', "").replace("\u201d", "").replace("\u2033", "").strip()
    )
    saw_hyphen = False
    saw_slash_multivalue = False
    for token in (part.strip() for part in cleaned.split("x") if part.strip()):
        if "-" in token:
            saw_hyphen = True
            token = token.split("-", 1)[0].strip()
        if re.fullmatch(r"\d+/\d+", token):
            saw_slash_multivalue = True
    if saw_hyphen == saw_slash_multivalue:
        raise BundleError(
            "string-parser range evidence must identify exactly one of hyphen or slash"
        )
    return "hyphen" if saw_hyphen else "slash_multivalue"


def _dict_range_key_sets(raw_value: Any) -> tuple[set[str], set[str]]:
    if not isinstance(raw_value, Mapping):
        raise BundleError("dict range evidence is not an object")
    minimum: set[str] = set()
    maximum: set[str] = set()
    for raw_key in raw_value:
        if not isinstance(raw_key, str):
            continue
        match = re.fullmatch(r"(Min|Max)\.\s+(.+)", raw_key, flags=re.IGNORECASE)
        if match is None:
            continue
        target = minimum if match.group(1).casefold() == "min" else maximum
        target.add(match.group(2).strip().casefold())
    return minimum, maximum


def _verify_range_source_evidence(
    product_id: str,
    catalog_document: Mapping[str, Any],
    range_rule: Mapping[str, Any],
) -> list[str]:
    """Return source-derived flags, rejecting policy/source disagreement."""

    group = str(range_rule["group"])
    size_options = catalog_document.get("size_options")
    policy_flags = set(str(flag) for flag in range_rule["special_flags"])
    if group == "string_parser_collapse":
        evidence_class = _string_parser_range_class(size_options)
        computed_flag = (
            "DIMENSION_RANGE_COLLAPSED_HYPHEN"
            if evidence_class == "hyphen"
            else "DIMENSION_RANGE_COLLAPSED_SLASH"
        )
        if policy_flags != {computed_flag}:
            raise BundleError(
                f"range policy/source class mismatch for {product_id}: "
                f"computed={computed_flag}, policy={sorted(policy_flags)}"
            )
        return [computed_flag]

    minimum, maximum = _dict_range_key_sets(size_options)
    paired_bases = minimum & maximum
    if group == "paired_dict_range":
        if not paired_bases or policy_flags != {"DIMENSION_RANGE_PRESENT"}:
            raise BundleError(
                f"paired Min./Max. source evidence mismatch for {product_id}"
            )
        return ["DIMENSION_RANGE_PRESENT"]
    if group == "max_only":
        if not maximum or paired_bases or policy_flags != {"DIMENSION_MAX_ONLY"}:
            raise BundleError(
                f"Max.-only source evidence mismatch for {product_id}"
            )
        return ["DIMENSION_MAX_ONLY"]
    raise BundleError(f"unknown range evidence group for {product_id}: {group}")


def _build_dimension_rows(
    ordered_meta: Sequence[Mapping[str, Any]],
    catalog_docs: Mapping[str, Mapping[str, Any]],
    dimension_docs: Mapping[str, Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    dimension_policy = policies["dimension"]
    quality_policy = policies["quality"]
    category_overrides = quality_policy["category_overrides"]
    height_rules = dimension_policy["height_policy_by_canonical_category"]
    sofa_index, range_index = _quality_indexes(quality_policy)
    manifest_ids = {str(row["id"]) for row in ordered_meta}
    reviewed_ids = set(sofa_index) | set(range_index)
    unknown_reviewed = sorted(reviewed_ids - manifest_ids)
    if unknown_reviewed:
        raise BundleError(f"quality policy IDs are absent from ordered metadata: {unknown_reviewed}")

    rows: list[dict[str, Any]] = []
    for row_index, meta in enumerate(ordered_meta):
        product_id = str(meta["id"])
        category = str(meta["category"])
        document = dimension_docs.get(product_id)
        meta_doc = document.get("meta") if isinstance(document, Mapping) else None
        dimensions_raw = meta_doc.get("dimensions_mm") if isinstance(meta_doc, Mapping) else None
        dimensions_raw = dimensions_raw if isinstance(dimensions_raw, Mapping) else {}
        length = _positive_finite(dimensions_raw.get("length"))
        source_width = _positive_finite(dimensions_raw.get("width"))
        height = _positive_finite(dimensions_raw.get("height"))
        has_footprint = length is not None and source_width is not None
        has_three_axes = has_footprint and height is not None
        known_axis_count = sum(value is not None for value in (length, source_width, height))
        if has_three_axes:
            parse_status = "three_axes"
        elif has_footprint:
            parse_status = "two_axes"
        elif known_axis_count:
            parse_status = "partial"
        else:
            parse_status = "missing"

        wire_dimensions = None
        if has_footprint:
            wire_dimensions = {
                "unit": "mm",
                "width": max(length, source_width),
                "depth": min(length, source_width),
                "height": height,
            }
        dimension_status = "complete" if has_three_axes else "partial" if has_footprint else "missing"
        dimension_quality_status = "usable" if has_footprint else "unverified"
        footprint_kind = "rectangle"
        special_flags: set[str] = set()
        reason_codes: set[str] = set()

        category_rule = category_overrides.get(category, {})
        height_rule = height_rules[category]
        height_policy = str(height_rule["policy"])
        catalog_quality_status = category_rule.get("catalog_quality_status", "supported")
        reason_codes.update(category_rule.get("reason_codes") or [])
        height_is_eligible = height is not None or height_policy == "optional"
        hard_filter_eligible = bool(
            has_footprint
            and dimension_quality_status == "usable"
            and catalog_quality_status in {"supported", "degraded"}
            and height_is_eligible
        )
        if category_rule.get("force_hard_filter_ineligible"):
            hard_filter_eligible = False

        range_rule = range_index.get(product_id)
        if range_rule is not None:
            source_flags = _verify_range_source_evidence(
                product_id, catalog_docs[product_id], range_rule
            )
            dimension_status = str(range_rule["dimension_status"])
            dimension_quality_status = str(range_rule["dimension_quality_status"])
            special_flags.update(source_flags)
            reason_codes.update(source_flags)
            hard_filter_eligible = False

        if category == "Sofa":
            known = sofa_index.get(product_id)
            if known is None:
                special_flags.update(
                    quality_policy["sofa_unverified_remainder"]["special_flags"]
                )
                reason_codes.update(
                    quality_policy["sofa_unverified_remainder"]["reason_codes"]
                )
                shape_evidence_status = "unverified"
            else:
                special_flags.update(known["special_flags"])
                reason_codes.add("SPECIAL_FOOTPRINT_UNMODELED")
                shape_evidence_status = f"catalog_verified:{known['group']}"
            footprint_kind = "special_unmodeled"
            dimension_status = "special_unmodeled"
            dimension_quality_status = "unverified"
            hard_filter_eligible = False
        else:
            shape_evidence_status = "rectangle_assumption_unverified_front"

        if height is not None:
            height_gate_status = "verified"
        elif has_footprint and height_policy == "optional":
            height_gate_status = "footprint_only"
        else:
            height_gate_status = "unverified_fail_closed"
            if has_footprint:
                reason_codes.add("HEIGHT_UNVERIFIED")

        source_info: Optional[dict[str, Any]] = None
        if document is not None:
            parse_source = meta_doc.get("dimensions_source") if isinstance(meta_doc, Mapping) else None
            parse_source = parse_source if isinstance(parse_source, Mapping) else {}
            source_info = {
                "collection": "ikea_product_v2_2026q2",
                "document_sha256": _canonical_bson_sha256(document),
                "method": _optional_string(parse_source.get("method")),
                "source_format": _optional_string(parse_source.get("source_format")),
                "parse_confidence": _optional_string(parse_source.get("confidence")),
                "source_keys": _bson_value_to_json(parse_source.get("keys")),
            }

        rows.append(
            {
                "row_index": row_index,
                "product_id": product_id,
                "category": category,
                "dimensions": wire_dimensions,
                "source_dimensions_mm": {
                    "length": length,
                    "width": source_width,
                    "height": height,
                },
                "dimension_parse_status": parse_status,
                "dimension_status": dimension_status,
                "dimension_quality_status": dimension_quality_status,
                "height_policy": height_policy,
                "height_policy_source": str(height_rule["source"]),
                "height_gate_status": height_gate_status,
                "catalog_quality_status": catalog_quality_status,
                "hard_filter_eligible": hard_filter_eligible,
                "footprint_kind": footprint_kind,
                "shape_evidence_status": shape_evidence_status,
                "front_direction_status": "unknown_not_available",
                "allowed_yaws_deg": [0, 90, 180, 270] if hard_filter_eligible else None,
                "special_flags": sorted(special_flags),
                "reason_codes": sorted(reason_codes),
                "dimension_source": source_info,
            }
        )
    _validate_built_quality(rows, quality_policy, sofa_index, range_index)
    return rows


def _validate_built_quality(
    rows: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    sofa_index: Mapping[str, Any],
    range_index: Mapping[str, Any],
) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise BundleError("dimension profile row count drifted")
    by_category: dict[str, list[Mapping[str, Any]]] = {
        category: [row for row in rows if row["category"] == category]
        for category in IKEA_15
    }
    expected = quality["expected_inventory"]
    checks = {
        "Bed": (len(by_category["Bed"]), expected["bed_rows"]),
        "Sofa": (len(by_category["Sofa"]), expected["sofa_rows"]),
        "Bar Stool": (len(by_category["Bar Stool"]), expected["bar_stool_rows"]),
    }
    for category, (actual, wanted) in checks.items():
        if actual != wanted:
            raise BundleError(f"{category} inventory drifted: {actual} != {wanted}")
    if any(row["catalog_quality_status"] != "blocked" or row["hard_filter_eligible"] for row in by_category["Bed"]):
        raise BundleError("all Bed rows must be blocked and hard-ineligible")
    if any(row["hard_filter_eligible"] or row["footprint_kind"] != "special_unmodeled" for row in by_category["Sofa"]):
        raise BundleError("all Sofa rows must fail closed as unmodeled footprints")
    if any(
        row["dimension_status"] != "special_unmodeled"
        or row["dimension_quality_status"] != "unverified"
        or row["allowed_yaws_deg"] is not None
        for row in by_category["Sofa"]
    ):
        raise BundleError("all Sofa rows must use the effective special-unmodeled state")
    known_sofa = [row for row in by_category["Sofa"] if row["product_id"] in sofa_index]
    unknown_sofa = [row for row in by_category["Sofa"] if row["product_id"] not in sofa_index]
    if len(known_sofa) != expected["known_special_sofa_rows"] or len(unknown_sofa) != expected["unverified_sofa_rows"]:
        raise BundleError("known/unverified Sofa inventory drifted")
    if any(
        "SPECIAL_FOOTPRINT_UNMODELED" not in row["special_flags"]
        for row in known_sofa
    ):
        raise BundleError("known special Sofa rows lost their reviewed shape evidence")
    if any(
        "FOOTPRINT_SHAPE_UNVERIFIED" not in row["special_flags"]
        or row["shape_evidence_status"] != "unverified"
        for row in unknown_sofa
    ):
        raise BundleError("unreviewed Sofa rows must remain shape-unverified")
    if any(row["catalog_quality_status"] != "degraded" for row in by_category["Bar Stool"]):
        raise BundleError("all Bar Stool rows must be degraded")
    rows_by_id = {str(row["product_id"]): row for row in rows}
    if len(rows_by_id) != EXPECTED_ROWS:
        raise BundleError("dimension profile product IDs are not unique")
    for product_id, rule in range_index.items():
        row = rows_by_id[product_id]
        if row["hard_filter_eligible"]:
            raise BundleError("all reviewed range rows must be hard-ineligible")
        expected_flags = set(rule["special_flags"])
        if not expected_flags.issubset(set(row["special_flags"])):
            raise BundleError(f"range evidence flags drifted for {product_id}")
        # Sofa's stronger all-shapes fail-closed override is the effective
        # status, while its range flag remains preserved for provenance.
        if row["category"] != "Sofa" and (
            row["dimension_status"] != rule["dimension_status"]
            or row["dimension_quality_status"] != rule["dimension_quality_status"]
        ):
            raise BundleError(f"range dimension state drifted for {product_id}")
    for row in rows:
        dimensions = row.get("dimensions")
        height = dimensions.get("height") if isinstance(dimensions, Mapping) else None
        if (
            row["height_policy"] in {"required", "unknown"}
            and height is None
            and row["hard_filter_eligible"]
        ):
            raise BundleError("missing required/unknown height must fail closed")


def _ordered_ids_sha256(rows: Sequence[Mapping[str, Any]], key: str = "product_id") -> str:
    return sha256_bytes(canonical_json_bytes([str(row[key]) for row in rows]))


def _deployment_identity_material(
    *,
    source_version: str,
    provenance: Mapping[str, Any],
    quality_policy_version: str,
    policy_hashes: Mapping[str, str],
    ordered_ids_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "ulip-deployment-identity/1.0",
        "source_version": source_version,
        "provenance": dict(provenance),
        "quality_policy_version": quality_policy_version,
        "policy_hashes": {
            name: policy_hashes[name] for name in sorted(policy_hashes)
        },
        "ordered_ids_sha256": ordered_ids_sha256,
    }


def _inventory_details(
    vectors: np.ndarray,
    catalog_rows: Sequence[Mapping[str, Any]],
    dimension_rows: Sequence[Mapping[str, Any]],
    mongo_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    categories = Counter(str(row["category"]) for row in catalog_rows)
    dimension_status = Counter(str(row["dimension_status"]) for row in dimension_rows)
    quality_status = Counter(str(row["catalog_quality_status"]) for row in dimension_rows)
    return {
        "vector_shape": [int(vectors.shape[0]), int(vectors.shape[1])],
        "vector_dtype": str(vectors.dtype),
        "vector_rows": int(vectors.shape[0]),
        "catalog_rows": len(catalog_rows),
        "dimension_profile_rows": len(dimension_rows),
        "mongo_catalog_join_rows": len(mongo_snapshot["catalog_docs"]),
        "mongo_dimension_join_rows": len(mongo_snapshot["dimension_docs"]),
        "mongo_collection_counts": dict(mongo_snapshot["collection_counts"]),
        "category_counts": dict(sorted(categories.items())),
        "dimension_status_counts": dict(sorted(dimension_status.items())),
        "catalog_quality_counts": dict(sorted(quality_status.items())),
        "hard_filter_eligible_rows": sum(
            bool(row["hard_filter_eligible"]) for row in dimension_rows
        ),
        "footprint_rows": sum(row["dimensions"] is not None for row in dimension_rows),
        "three_axis_rows": sum(
            isinstance(row["dimensions"], Mapping)
            and row["dimensions"].get("height") is not None
            for row in dimension_rows
        ),
    }


def _validate_vectors(vectors: np.ndarray, ordered_meta: Sequence[Mapping[str, Any]]) -> None:
    if vectors.ndim != 2 or tuple(vectors.shape) != (EXPECTED_ROWS, EXPECTED_EMBEDDING_DIM):
        raise BundleError(
            f"production vectors must be {(EXPECTED_ROWS, EXPECTED_EMBEDDING_DIM)}, "
            f"got {tuple(vectors.shape)}"
        )
    if vectors.dtype != np.float32:
        raise BundleError(f"production vectors must be float32, got {vectors.dtype}")
    if not np.isfinite(vectors).all():
        raise BundleError("production vectors contain NaN or infinity")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if np.any(norms <= 0) or not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
        raise BundleError(
            f"production vectors are not unit normalized: min={norms.min()}, max={norms.max()}"
        )
    if len(ordered_meta) != vectors.shape[0]:
        raise BundleError("vector/meta row count mismatch")


def build_deployment_bundle(
    *,
    checkpoint_path: Path,
    vectors_path: Path,
    meta_path: Path,
    vector_schema_path: Path,
    tokenizer_code_path: Path,
    tokenizer_bpe_path: Path,
    source_version: str,
    tokenizer_id: str = "ulip-simple-tokenizer-clip77-v1",
    policy_dir: Path = DEFAULT_POLICY_DIR,
    mongo_db: str = "furniture_db",
    catalog_collection: str = "ikea_product",
    dimension_collection: str = "ikea_product_v2_2026q2",
    mongo_client: Any = None,
) -> BundleBuild:
    """Build canonical deployment bytes in memory.

    No output directory is created and no source file or MongoDB document is
    mutated.  Pass the result explicitly to :func:`write_deployment_bundle`
    only after review.
    """

    if not isinstance(source_version, str) or not PROFILE_ID_RE.fullmatch(source_version):
        raise BundleError("source_version must match [A-Za-z0-9._-]+")
    if not isinstance(tokenizer_id, str) or not tokenizer_id.strip():
        raise BundleError("tokenizer_id must be non-empty")

    policies, policy_bytes = _load_policy_set(policy_dir)
    ordered_meta = _load_ordered_meta(meta_path)
    vectors = np.load(Path(vectors_path), mmap_mode="r")
    _validate_vectors(vectors, ordered_meta)
    snapshot = _fetch_mongo_snapshot(
        ordered_meta,
        mongo_db=mongo_db,
        catalog_collection=catalog_collection,
        dimension_collection=dimension_collection,
        mongo_client=mongo_client,
    )
    catalog_rows = _build_catalog_rows(ordered_meta, snapshot["catalog_docs"])
    # Record actual collection names rather than relying on defaults embedded
    # in helper labels.
    for row in catalog_rows:
        row["catalog_source"]["collection"] = catalog_collection
    dimension_rows = _build_dimension_rows(
        ordered_meta,
        snapshot["catalog_docs"],
        snapshot["dimension_docs"],
        policies,
    )
    for row in dimension_rows:
        if row["dimension_source"] is not None:
            row["dimension_source"]["collection"] = dimension_collection

    inventory = _inventory_details(vectors, catalog_rows, dimension_rows, snapshot)
    if inventory["vector_rows"] != EXPECTED_ROWS or inventory["catalog_rows"] != EXPECTED_ROWS or inventory["dimension_profile_rows"] != EXPECTED_ROWS:
        raise BundleError("deployment inventory must preserve all 733 row identities")

    files: dict[str, bytes] = {}
    catalog_member_path = "catalog/catalog_rows.jsonl"
    dimension_member_path = "dimensions/dimension_quality_rows.jsonl"
    checkpoint_member_path = "model/checkpoint_last.pt"
    vectors_member_path = "vectors/vectors_pc.npy"
    meta_member_path = "vectors/meta_pc.jsonl"
    vector_schema_member_path = "vectors/schema.json"
    tokenizer_code_member_path = "tokenizer/tokenizer.py"
    tokenizer_bpe_member_path = "tokenizer/bpe_simple_vocab_16e6.txt.gz"
    files[catalog_member_path] = canonical_jsonl_bytes(catalog_rows)
    files[dimension_member_path] = canonical_jsonl_bytes(dimension_rows)
    source_files = {
        checkpoint_member_path: Path(checkpoint_path).expanduser().resolve(),
        vectors_member_path: Path(vectors_path).expanduser().resolve(),
        meta_member_path: Path(meta_path).expanduser().resolve(),
        vector_schema_member_path: Path(vector_schema_path).expanduser().resolve(),
        tokenizer_code_member_path: Path(tokenizer_code_path).expanduser().resolve(),
        tokenizer_bpe_member_path: Path(tokenizer_bpe_path).expanduser().resolve(),
    }

    policy_bundle_paths = {
        "category": f"policies/{CATEGORY_POLICY_FILE}",
        "dimension": f"policies/{DIMENSION_POLICY_FILE}",
        "quality": f"policies/{QUALITY_POLICY_FILE}",
    }
    for name, relative_path in policy_bundle_paths.items():
        files[relative_path] = policy_bytes[name]

    checkpoint_descriptor = _source_member_descriptor(
        source_files[checkpoint_member_path],
        checkpoint_member_path,
        logical_name="ulip_checkpoint",
    )
    vector_descriptors = {
        "vectors_pc": _source_member_descriptor(
            source_files[vectors_member_path],
            vectors_member_path,
            logical_name="vectors_pc",
        ),
        "meta_pc": _source_member_descriptor(
            source_files[meta_member_path],
            meta_member_path,
            logical_name="meta_pc",
        ),
        "schema": _source_member_descriptor(
            source_files[vector_schema_member_path],
            vector_schema_member_path,
            logical_name="vector_schema",
        ),
    }
    model_manifest = {
        "schema_version": "ulip-model-checkpoint-manifest/1.0",
        "source_version": source_version,
        "artifact": checkpoint_descriptor,
    }
    vector_manifest = {
        "schema_version": "ulip-vector-bundle-manifest/1.0",
        "source_version": source_version,
        "components": vector_descriptors,
        "shape": [EXPECTED_ROWS, EXPECTED_EMBEDDING_DIM],
        "dtype": "float32",
        "finite": True,
        "unit_normalized_atol": 1e-5,
        "row_count": EXPECTED_ROWS,
        "ordered_ids_sha256": _ordered_ids_sha256(
            [{"product_id": row["id"]} for row in ordered_meta]
        ),
    }
    catalog_manifest = {
        "schema_version": "ulip-catalog-manifest/1.0",
        "source_version": source_version,
        "member": _member_descriptor(
            catalog_member_path,
            files[catalog_member_path],
            logical_name="catalog_rows",
            rows=EXPECTED_ROWS,
        ),
        "ordered_ids_sha256": _ordered_ids_sha256(catalog_rows),
        "source": {
            "database": mongo_db,
            "collection": catalog_collection,
            "canonical_join": "ObjectId(meta_pc.id)==Mongo._id",
            "joined_rows": EXPECTED_ROWS,
        },
        "category_policy_version": policies["category"]["policy_version"],
        "category_policy_sha256": sha256_bytes(policy_bytes["category"]),
    }
    dimension_manifest = {
        "schema_version": "ulip-dimension-quality-manifest/1.0",
        "source_version": source_version,
        "member": _member_descriptor(
            dimension_member_path,
            files[dimension_member_path],
            logical_name="dimension_quality_rows",
            rows=EXPECTED_ROWS,
        ),
        "ordered_ids_sha256": _ordered_ids_sha256(dimension_rows),
        "source": {
            "database": mongo_db,
            "collection": dimension_collection,
            "canonical_join": "ObjectId(meta_pc.id)==Mongo._id",
            "joined_rows": len(snapshot["dimension_docs"]),
            "missing_rows_preserved_as_null": EXPECTED_ROWS - len(snapshot["dimension_docs"]),
        },
        "dimension_axis_policy_version": policies["dimension"]["policy_version"],
        "dimension_axis_policy_sha256": sha256_bytes(policy_bytes["dimension"]),
        "quality_policy_version": policies["quality"]["policy_version"],
        "quality_policy_sha256": sha256_bytes(policy_bytes["quality"]),
    }
    tokenizer_manifest = {
        "schema_version": "ulip-tokenizer-bundle-manifest/1.0",
        "source_version": source_version,
        "tokenizer_id": tokenizer_id,
        "context_length": 77,
        "safe_content_tokens": 75,
        "overflow_policy": "runtime_reject_no_silent_truncation",
        "members": {
            "code": _source_member_descriptor(
                source_files[tokenizer_code_member_path],
                tokenizer_code_member_path,
                logical_name="tokenizer_code",
            ),
            "bpe": _source_member_descriptor(
                source_files[tokenizer_bpe_member_path],
                tokenizer_bpe_member_path,
                logical_name="tokenizer_bpe",
            ),
        },
    }

    manifest_payloads = {
        "model_checkpoint": model_manifest,
        "vector_bundle": vector_manifest,
        "catalog_manifest": catalog_manifest,
        "dimension_bundle": dimension_manifest,
        "tokenizer_bundle": tokenizer_manifest,
    }
    manifest_paths = {
        name: f"manifests/{name}.json" for name in manifest_payloads
    }
    manifest_hashes: dict[str, str] = {}
    for name, payload in manifest_payloads.items():
        data = canonical_json_bytes(payload)
        files[manifest_paths[name]] = data
        manifest_hashes[name] = sha256_bytes(data)

    provenance = {
        "model_checkpoint_sha256": manifest_hashes["model_checkpoint"],
        "vector_bundle_sha256": manifest_hashes["vector_bundle"],
        "catalog_manifest_sha256": manifest_hashes["catalog_manifest"],
        "dimension_bundle_sha256": manifest_hashes["dimension_bundle"],
        "tokenizer_bundle_sha256": manifest_hashes["tokenizer_bundle"],
        "category_policy_version": policies["category"]["policy_version"],
        "dimension_axis_policy_version": policies["dimension"]["policy_version"],
    }
    identity_material = _deployment_identity_material(
        source_version=source_version,
        provenance=provenance,
        quality_policy_version=policies["quality"]["policy_version"],
        policy_hashes={
            name: sha256_bytes(data) for name, data in policy_bytes.items()
        },
        ordered_ids_sha256=catalog_manifest["ordered_ids_sha256"],
    )
    identity_sha256 = sha256_bytes(canonical_json_bytes(identity_material))
    profile_id = f"{source_version}-{identity_sha256[:16]}"
    profile = {
        "schema_version": "ulip-product-candidates-deployment/1.0",
        "profile_id": profile_id,
        "deployment_profile_id": profile_id,
        "source_version": source_version,
        "deployment_identity_sha256": identity_sha256,
        "tokenizer_id": tokenizer_id,
        "provenance": provenance,
        "manifests": {
            name: {"bundle_path": manifest_paths[name], "sha256": manifest_hashes[name]}
            for name in sorted(manifest_paths)
        },
        "policies": {
            "category": {
                "bundle_path": policy_bundle_paths["category"],
                "version": policies["category"]["policy_version"],
                "sha256": sha256_bytes(policy_bytes["category"]),
            },
            "dimension": {
                "bundle_path": policy_bundle_paths["dimension"],
                "version": policies["dimension"]["policy_version"],
                "sha256": sha256_bytes(policy_bytes["dimension"]),
            },
            "quality": {
                "bundle_path": policy_bundle_paths["quality"],
                "version": policies["quality"]["policy_version"],
                "sha256": sha256_bytes(policy_bytes["quality"]),
            },
        },
        "inventory": inventory,
    }
    files["deployment_profile.json"] = canonical_json_bytes(profile)
    return BundleBuild(
        files=MappingProxyType(dict(sorted(files.items()))),
        source_files=MappingProxyType(dict(sorted(source_files.items()))),
        profile=_deep_freeze(profile),
        summary=_deep_freeze(inventory),
    )


def _source_descriptors_from_build(build: BundleBuild) -> dict[str, Mapping[str, Any]]:
    def manifest(name: str) -> Mapping[str, Any]:
        relative_path = f"manifests/{name}.json"
        raw = build.files.get(relative_path)
        if raw is None:
            raise BundleError(f"build is missing {relative_path}")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BundleError(f"invalid in-memory manifest {relative_path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise BundleError(f"in-memory manifest is not an object: {relative_path}")
        return value

    model = manifest("model_checkpoint")
    vector = manifest("vector_bundle")
    tokenizer = manifest("tokenizer_bundle")
    candidates = [model.get("artifact")]
    components = vector.get("components")
    members = tokenizer.get("members")
    if not isinstance(components, Mapping) or not isinstance(members, Mapping):
        raise BundleError("in-memory source manifests are incomplete")
    candidates.extend(components.values())
    candidates.extend(members.values())

    descriptors: dict[str, Mapping[str, Any]] = {}
    for descriptor in candidates:
        if not isinstance(descriptor, Mapping):
            raise BundleError("in-memory source descriptor is invalid")
        bundle_path = descriptor.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            raise BundleError("in-memory source descriptor has no bundle_path")
        if bundle_path in descriptors:
            raise BundleError(f"duplicate source bundle path: {bundle_path}")
        descriptors[bundle_path] = descriptor
    if set(descriptors) != set(build.source_files):
        raise BundleError("source_files do not exactly match source manifest descriptors")
    return descriptors


def _copy_source_verified(
    source: Path,
    target: Path,
    descriptor: Mapping[str, Any],
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise BundleError(f"bundle source file is missing: {source}")
    expected_bytes = descriptor.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise BundleError("source descriptor bytes must be an integer")
    expected_sha = _validate_sha(
        descriptor.get("content_sha256"), "source descriptor content_sha256"
    )
    if source.stat().st_size != expected_bytes:
        raise BundleError(f"bundle source byte size drifted before copy: {source}")

    digest = hashlib.sha256()
    copied = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, target.open("xb") as output_handle:
        while chunk := input_handle.read(chunk_size):
            output_handle.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if copied != expected_bytes or digest.hexdigest() != expected_sha:
        raise BundleError(f"bundle source drifted while copying: {source}")
    if target.stat().st_size != expected_bytes or sha256_file(target) != expected_sha:
        raise BundleError(f"copied bundle member failed verification: {target}")


def write_deployment_bundle(build: BundleBuild, output_dir: Path) -> Path:
    """Atomically materialize a reviewed :class:`BundleBuild`.

    The destination must not already exist.  Source assets are streaming-copied
    into staging and checked against their build-time bytes/hash descriptors.
    Sources are never hard-linked, chmodded, or otherwise modified.
    """

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise BundleError(f"refusing to overwrite deployment profile: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        if set(build.files) & set(build.source_files):
            raise BundleError("generated and source bundle paths overlap")
        for relative_path, data in sorted(build.files.items()):
            target = _safe_bundle_member(staging, relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        descriptors = _source_descriptors_from_build(build)
        for relative_path, source in sorted(build.source_files.items()):
            target = _safe_bundle_member(staging, relative_path)
            _copy_source_verified(source, target, descriptors[relative_path])
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def _verify_member_file_descriptor(
    root: Path, descriptor: Mapping[str, Any], label: str
) -> Path:
    path = _safe_bundle_member(root, descriptor.get("bundle_path"))
    if not path.is_file():
        raise BundleError(f"{label} bundle member is missing: {path}")
    if path.stat().st_size != descriptor.get("bytes"):
        raise BundleError(f"{label} byte size drifted")
    expected = _validate_sha(descriptor.get("content_sha256"), f"{label}.content_sha256")
    if sha256_file(path) != expected:
        raise BundleError(f"{label} content hash drifted")
    return path


def _verify_member_descriptor(
    root: Path, descriptor: Mapping[str, Any], label: str
) -> tuple[Path, bytes]:
    path = _safe_bundle_member(root, descriptor.get("bundle_path"))
    if not path.is_file():
        raise BundleError(f"{label} bundle member is missing: {path}")
    data = path.read_bytes()
    if len(data) != descriptor.get("bytes"):
        raise BundleError(f"{label} bundle byte size drifted")
    expected = _validate_sha(descriptor.get("content_sha256"), f"{label}.content_sha256")
    if sha256_bytes(data) != expected:
        raise BundleError(f"{label} bundle content hash drifted")
    return path, data


def load_deployment_profile(profile_dir: Path) -> DeploymentProfile:
    """Load an immutable profile and fail closed on any drift.

    This function never connects to MongoDB.  A successful return means all
    seven readiness checks are true and the caller can safely expose them from
    ``/readyz``.
    """

    root = Path(profile_dir).expanduser().resolve()
    profile, _ = _read_canonical_json(root / "deployment_profile.json")
    if profile.get("schema_version") != "ulip-product-candidates-deployment/1.0":
        raise BundleError("unsupported deployment profile schema")
    profile_id = profile.get("deployment_profile_id")
    if not isinstance(profile_id, str) or not PROFILE_ID_RE.fullmatch(profile_id):
        raise BundleError("invalid deployment_profile_id")
    if profile.get("profile_id") != profile_id:
        raise BundleError("profile_id/deployment_profile_id mismatch")
    source_version = profile.get("source_version")
    if not isinstance(source_version, str) or not PROFILE_ID_RE.fullmatch(source_version):
        raise BundleError("invalid deployment source_version")

    provenance = profile.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(PROVENANCE_KEYS):
        raise BundleError("deployment provenance must contain exactly seven keys")
    for key in PROVENANCE_KEYS[:5]:
        _validate_sha(provenance[key], f"provenance.{key}")

    manifest_refs = profile.get("manifests")
    required_manifests = {
        "model_checkpoint", "vector_bundle", "catalog_manifest",
        "dimension_bundle", "tokenizer_bundle",
    }
    if not isinstance(manifest_refs, Mapping) or set(manifest_refs) != required_manifests:
        raise BundleError("deployment profile must reference exactly five manifests")
    manifests: dict[str, dict[str, Any]] = {}
    provenance_names = {
        "model_checkpoint": "model_checkpoint_sha256",
        "vector_bundle": "vector_bundle_sha256",
        "catalog_manifest": "catalog_manifest_sha256",
        "dimension_bundle": "dimension_bundle_sha256",
        "tokenizer_bundle": "tokenizer_bundle_sha256",
    }
    for name in sorted(required_manifests):
        ref = manifest_refs[name]
        if not isinstance(ref, Mapping):
            raise BundleError(f"manifest reference {name} is invalid")
        path = _safe_bundle_member(root, ref.get("bundle_path"))
        value, data = _read_canonical_json(path)
        expected = _validate_sha(ref.get("sha256"), f"manifest.{name}.sha256")
        actual = sha256_bytes(data)
        if actual != expected or actual != provenance[provenance_names[name]]:
            raise BundleError(f"canonical manifest hash drifted: {name}")
        if value.get("source_version") != source_version:
            raise BundleError(f"manifest source_version drifted: {name}")
        manifests[name] = value

    model_manifest = manifests["model_checkpoint"]
    if model_manifest.get("schema_version") != "ulip-model-checkpoint-manifest/1.0":
        raise BundleError("unsupported model checkpoint manifest")
    model_checkpoint_path = _verify_member_file_descriptor(
        root, model_manifest.get("artifact", {}), "model checkpoint"
    )

    vector_manifest = manifests["vector_bundle"]
    if vector_manifest.get("schema_version") != "ulip-vector-bundle-manifest/1.0":
        raise BundleError("unsupported vector bundle manifest")
    vector_components = vector_manifest.get("components")
    if not isinstance(vector_components, Mapping) or set(vector_components) != {
        "vectors_pc", "meta_pc", "schema"
    }:
        raise BundleError("vector manifest components are incomplete")
    if (
        vector_manifest.get("row_count") != EXPECTED_ROWS
        or vector_manifest.get("finite") is not True
        or vector_manifest.get("unit_normalized_atol") != 1e-5
    ):
        raise BundleError("vector manifest safety metadata drifted")
    vector_path = _verify_member_file_descriptor(
        root, vector_components["vectors_pc"], "PC vectors"
    )
    meta_path = _verify_member_file_descriptor(
        root, vector_components["meta_pc"], "PC metadata"
    )
    _verify_member_file_descriptor(root, vector_components["schema"], "vector schema")
    ordered_meta = _load_ordered_meta(meta_path)
    vectors = np.load(vector_path).astype(np.float32, copy=False)
    _validate_vectors(vectors, ordered_meta)
    if list(vectors.shape) != vector_manifest.get("shape") or vector_manifest.get("dtype") != "float32":
        raise BundleError("vector shape/dtype manifest drifted")
    meta_id_hash = _ordered_ids_sha256(
        [{"product_id": row["id"]} for row in ordered_meta]
    )
    if meta_id_hash != vector_manifest.get("ordered_ids_sha256"):
        raise BundleError("vector metadata row order drifted")

    catalog_manifest = manifests["catalog_manifest"]
    if catalog_manifest.get("schema_version") != "ulip-catalog-manifest/1.0":
        raise BundleError("unsupported catalog manifest")
    catalog_path, _ = _verify_member_descriptor(
        root, catalog_manifest.get("member", {}), "catalog rows"
    )
    catalog_rows, _ = _read_canonical_jsonl(catalog_path)
    if len(catalog_rows) != EXPECTED_ROWS or catalog_manifest["member"].get("rows") != EXPECTED_ROWS:
        raise BundleError("catalog row count drifted")

    dimension_manifest = manifests["dimension_bundle"]
    if dimension_manifest.get("schema_version") != "ulip-dimension-quality-manifest/1.0":
        raise BundleError("unsupported dimension bundle manifest")
    dimension_path, _ = _verify_member_descriptor(
        root, dimension_manifest.get("member", {}), "dimension rows"
    )
    dimension_rows, _ = _read_canonical_jsonl(dimension_path)
    if len(dimension_rows) != EXPECTED_ROWS or dimension_manifest["member"].get("rows") != EXPECTED_ROWS:
        raise BundleError("dimension row count drifted")

    meta_ids = [str(row["id"]) for row in ordered_meta]
    catalog_ids = [str(row.get("product_id")) for row in catalog_rows]
    dimension_ids = [str(row.get("product_id")) for row in dimension_rows]
    expected_indices = list(range(EXPECTED_ROWS))
    if catalog_ids != meta_ids or dimension_ids != meta_ids:
        raise BundleError("catalog/dimension/vector row identity join drifted")
    if [row.get("row_index") for row in catalog_rows] != expected_indices or [row.get("row_index") for row in dimension_rows] != expected_indices:
        raise BundleError("catalog/dimension row_index sequence drifted")
    if len(set(meta_ids)) != EXPECTED_ROWS:
        raise BundleError("duplicate product IDs in deployment profile")
    if catalog_manifest.get("ordered_ids_sha256") != meta_id_hash or dimension_manifest.get("ordered_ids_sha256") != meta_id_hash:
        raise BundleError("ordered ID hash differs across bundle manifests")

    policy_refs = profile.get("policies")
    if not isinstance(policy_refs, Mapping) or set(policy_refs) != {
        "category", "dimension", "quality"
    }:
        raise BundleError("deployment policy references are incomplete")
    policies: dict[str, dict[str, Any]] = {}
    for name, ref in policy_refs.items():
        if not isinstance(ref, Mapping):
            raise BundleError(f"policy reference {name} is invalid")
        path = _safe_bundle_member(root, ref.get("bundle_path"))
        value, data = _read_canonical_json(path)
        if sha256_bytes(data) != _validate_sha(ref.get("sha256"), f"policy.{name}.sha256"):
            raise BundleError(f"policy bytes drifted: {name}")
        if value.get("policy_version") != ref.get("version"):
            raise BundleError(f"policy version drifted: {name}")
        policies[name] = value
    _validate_policies(policies)
    if policies["category"]["policy_version"] != provenance["category_policy_version"] or policies["dimension"]["policy_version"] != provenance["dimension_axis_policy_version"]:
        raise BundleError("provenance policy versions drifted")
    if catalog_manifest.get("category_policy_version") != policies["category"]["policy_version"]:
        raise BundleError("catalog manifest category policy version drifted")
    if dimension_manifest.get("dimension_axis_policy_version") != policies["dimension"]["policy_version"] or dimension_manifest.get("quality_policy_version") != policies["quality"]["policy_version"]:
        raise BundleError("dimension manifest policy version drifted")
    if catalog_manifest.get("category_policy_sha256") != policy_refs["category"]["sha256"] or dimension_manifest.get("dimension_axis_policy_sha256") != policy_refs["dimension"]["sha256"] or dimension_manifest.get("quality_policy_sha256") != policy_refs["quality"]["sha256"]:
        raise BundleError("bundle manifests are bound to different policy bytes")
    sofa_index, range_index = _quality_indexes(policies["quality"])
    _validate_built_quality(
        dimension_rows, policies["quality"], sofa_index, range_index
    )

    identity_material = _deployment_identity_material(
        source_version=source_version,
        provenance=provenance,
        quality_policy_version=policies["quality"]["policy_version"],
        policy_hashes={
            name: str(ref["sha256"]) for name, ref in policy_refs.items()
        },
        ordered_ids_sha256=meta_id_hash,
    )
    identity_sha256 = sha256_bytes(canonical_json_bytes(identity_material))
    if identity_sha256 != _validate_sha(
        profile.get("deployment_identity_sha256"),
        "deployment_identity_sha256",
    ):
        raise BundleError("deployment identity hash drifted")
    if profile_id != f"{source_version}-{identity_sha256[:16]}":
        raise BundleError("deterministic deployment profile_id drifted")

    tokenizer_manifest = manifests["tokenizer_bundle"]
    if tokenizer_manifest.get("schema_version") != "ulip-tokenizer-bundle-manifest/1.0":
        raise BundleError("unsupported tokenizer manifest")
    if (
        tokenizer_manifest.get("context_length") != 77
        or tokenizer_manifest.get("safe_content_tokens") != 75
        or tokenizer_manifest.get("overflow_policy")
        != "runtime_reject_no_silent_truncation"
    ):
        raise BundleError("tokenizer manifest safety metadata drifted")
    tokenizer_members = tokenizer_manifest.get("members")
    if not isinstance(tokenizer_members, Mapping) or set(tokenizer_members) != {"code", "bpe"}:
        raise BundleError("tokenizer manifest members are incomplete")
    tokenizer_code_path, _ = _verify_member_descriptor(
        root, tokenizer_members["code"], "tokenizer code"
    )
    tokenizer_bpe_path, _ = _verify_member_descriptor(
        root, tokenizer_members["bpe"], "tokenizer BPE"
    )
    tokenizer_id = tokenizer_manifest.get("tokenizer_id")
    if not isinstance(tokenizer_id, str) or not tokenizer_id or tokenizer_id != profile.get("tokenizer_id"):
        raise BundleError("tokenizer_id drifted")

    profile_inventory = profile.get("inventory")
    if not isinstance(profile_inventory, Mapping):
        raise BundleError("deployment inventory is missing")
    inventory = {
        "vector_shape": [int(vectors.shape[0]), int(vectors.shape[1])],
        "vector_rows": int(vectors.shape[0]),
        "catalog_rows": len(catalog_rows),
        "dimension_profile_rows": len(dimension_rows),
    }
    for key, value in inventory.items():
        if profile_inventory.get(key) != value:
            raise BundleError(f"deployment inventory drifted: {key}")

    combined_rows: list[Mapping[str, Any]] = []
    for index, (catalog, dimension) in enumerate(zip(catalog_rows, dimension_rows)):
        for key in ("row_index", "product_id", "category"):
            if catalog.get(key) != dimension.get(key):
                raise BundleError(f"row {index} catalog/dimension {key} mismatch")
        merged = dict(catalog)
        merged.update(
            {key: value for key, value in dimension.items() if key not in {"row_index", "product_id", "category"}}
        )
        combined_rows.append(_deep_freeze(merged))

    vectors.setflags(write=False)
    checks = {key: True for key in CHECK_KEYS}
    return DeploymentProfile(
        profile_id=profile_id,
        deployment_profile_id=profile_id,
        provenance=_deep_freeze(dict(provenance)),
        checks=_deep_freeze(checks),
        inventory=_deep_freeze(inventory),
        inventory_details=_deep_freeze(dict(profile_inventory)),
        vectors=vectors,
        catalog_rows=tuple(_deep_freeze(row) for row in catalog_rows),
        dimension_rows=tuple(_deep_freeze(row) for row in dimension_rows),
        rows=tuple(combined_rows),
        tokenizer_id=tokenizer_id,
        tokenizer_code_path=tokenizer_code_path,
        tokenizer_bpe_path=tokenizer_bpe_path,
        model_checkpoint_path=model_checkpoint_path,
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


__all__ = [
    "BundleBuild",
    "BundleError",
    "DeploymentProfile",
    "build_deployment_bundle",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "load_deployment_profile",
    "sha256_bytes",
    "sha256_file",
    "write_deployment_bundle",
]
