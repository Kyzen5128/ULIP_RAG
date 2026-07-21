"""Fail-closed vanilla ULIP runtime for Product Candidates v1."""
from __future__ import annotations

import math
import os
import re
import threading
import types
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROVENANCE_KEYS = (
    "model_checkpoint_sha256",
    "vector_bundle_sha256",
    "catalog_manifest_sha256",
    "dimension_bundle_sha256",
    "tokenizer_bundle_sha256",
    "category_policy_version",
    "dimension_axis_policy_version",
)
READY_CHECK_KEYS = (
    "model_checkpoint",
    "vector_bundle",
    "catalog_manifest",
    "dimension_bundle",
    "tokenizer_bundle",
    "row_identity_join",
    "policy_versions",
)
WARNING_ALLOWLIST = {
    "CATEGORY_UNSUPPORTED",
    "CATEGORY_PARTIALLY_UNSUPPORTED",
    "CATEGORY_DATA_QUALITY_BLOCKED",
    "CATEGORY_DATA_QUALITY_DEGRADED",
    "CATEGORY_CANONICAL_POOL_DEDUPED",
}


ENABLED_CATEGORY_TERMS = {
    "sofa": "Sofa",
    "couch": "Sofa",
    "loveseat": "Sofa",
    "bench": "Bench",
    "ottoman": "Storage Ottoman",
    "dining table": "Dining Table",
    "desk": "Office Desk",
    "wardrobe": "Wardrobe",
    "closet": "Wardrobe",
    "armoire": "Wardrobe",
    "bookshelf": "Bookshelf",
    "bookcase": "Bookshelf",
    "shelf": "Bookshelf",
    "shelving unit": "Bookshelf",
    "tv stand": "TV Stand",
    "media console": "TV Stand",
}
BLOCKED_CATEGORY_TERMS = {"bed": "Bed", "bed frame": "Bed"}
UNSUPPORTED_CATEGORY_TERMS = {
    "armchair", "accent chair", "lounge chair", "chair", "stool", "table",
    "coffee table", "side table", "end table", "console table", "nightstand",
    "bedside table", "bedside cabinet", "dresser", "cabinet", "lamp",
    "floor lamp", "rug",
}
CONTROLLED_CATEGORY_TERMS = (
    set(ENABLED_CATEGORY_TERMS)
    | set(BLOCKED_CATEGORY_TERMS)
    | UNSUPPORTED_CATEGORY_TERMS
)
DEGRADED_CANONICAL_CATEGORIES = {"Sofa"}

# The 2.0 IKEA snapshot has a reviewed 24-category taxonomy.  Keep the legacy
# constants above unchanged so the immutable 733-row deployment remains fully
# reproducible; the runtime selects this table only when the deployment pins
# ``v2t36_to_ikea24_v2``.
ENABLED_CATEGORY_TERMS_V2 = {
    "sofa": "Sofa", "couch": "Sofa", "loveseat": "Sofa",
    "armchair": "Armchair", "accent chair": "Armchair",
    "lounge chair": "Armchair", "bench": "Bench", "ottoman": "Ottoman",
    "dining table": "Dining Table", "coffee table": "Coffee Table",
    "side table": "Side Table", "end table": "Side Table",
    "console table": "Console Table", "desk": "Office Desk",
    "nightstand": "Nightstand", "bedside table": "Nightstand",
    "bedside cabinet": "Nightstand", "wardrobe": "Wardrobe",
    "closet": "Wardrobe", "armoire": "Wardrobe", "dresser": "Cabinet",
    "cabinet": "Cabinet", "bookshelf": "Bookshelf", "bookcase": "Bookshelf",
    "shelf": "Bookshelf", "shelving unit": "Bookshelf",
    "tv stand": "TV Stand", "media console": "TV Stand",
    "bed": "Bed", "bed frame": "Bed",
}
BLOCKED_CATEGORY_TERMS_V2: Dict[str, str] = {}
UNSUPPORTED_CATEGORY_TERMS_V2 = {
    "chair", "stool", "table", "lamp", "floor lamp", "rug",
}
CONTROLLED_CATEGORY_TERMS_V2 = (
    set(ENABLED_CATEGORY_TERMS_V2)
    | set(BLOCKED_CATEGORY_TERMS_V2)
    | UNSUPPORTED_CATEGORY_TERMS_V2
)
DEGRADED_CANONICAL_CATEGORIES_V2 = {"Sofa"}
UNSAFE_DIMENSION_STATUSES = {
    "missing", "suspect", "range_collapsed", "special_unmodeled",
}
UNSAFE_SPECIAL_FLAGS = {
    "L_SHAPED", "VARIABLE_FOOTPRINT", "RECLINER_EXTENDED_FOOTPRINT",
    "COMPONENT_ONLY", "FOOTPRINT_SHAPE_UNVERIFIED",
}


class RequestPolicyError(ValueError):
    """The caller supplied a schema-valid but policy-invalid request."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class QueryTooLongError(RequestPolicyError):
    def __init__(self, content_token_count: int) -> None:
        super().__init__(
            "QUERY_TOKEN_LIMIT_EXCEEDED",
            "query_text exceeds the 75 content-token ULIP limit",
        )
        self.content_token_count = content_token_count


class RuntimeUnavailableError(RuntimeError):
    """An immutable asset, tokenizer, or vanilla encoder is unavailable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class TokenizedQuery:
    tokens: Any
    content_token_count: int
    token_count: int
    eot_present: bool


class ULIPQueryTokenizer:
    """Add an explicit 75/76 gate around the repository SimpleTokenizer."""

    def __init__(self, tokenizer: Any, tokenizer_id: str) -> None:
        if not isinstance(tokenizer_id, str) or not tokenizer_id.strip():
            raise ValueError("tokenizer_id must be a non-empty string")
        if not callable(getattr(tokenizer, "encode", None)) or not callable(tokenizer):
            raise TypeError("tokenizer must provide encode(text) and __call__(texts)")
        self.tokenizer = tokenizer
        self.tokenizer_id = tokenizer_id.strip()

    def _eot_token_id(self) -> int:
        value = getattr(self.tokenizer, "eot_token_id", None)
        if value is None:
            encoder = getattr(self.tokenizer, "encoder", None)
            if isinstance(encoder, Mapping):
                value = encoder.get("<|endoftext|>")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise RuntimeUnavailableError(
                "TOKENIZER_EOT_ID_MISSING",
                "the tokenizer bundle does not expose its EOT token id",
            )
        return int(value)

    def _sot_token_id(self) -> Optional[int]:
        value = getattr(self.tokenizer, "sot_token_id", None)
        if value is None:
            encoder = getattr(self.tokenizer, "encoder", None)
            if isinstance(encoder, Mapping):
                value = encoder.get("<|startoftext|>")
        if value is None:
            # Lightweight injected test tokenizers may expose only EOT.  The
            # production immutable SimpleTokenizer exposes both IDs.
            return None
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise RuntimeUnavailableError(
                "TOKENIZER_SOT_ID_INVALID", "the tokenizer SOT token id is invalid",
            )
        return int(value)

    @staticmethod
    def _token_at(tokens: Any, position: int) -> int:
        try:
            ndim = int(tokens.dim()) if callable(getattr(tokens, "dim", None)) else None
            if ndim == 2:
                return int(tokens[0, position].item())
            if ndim == 1:
                return int(tokens[position].item())
            array = np.asarray(tokens)
            if array.ndim == 2:
                return int(array[0, position])
            if array.ndim == 1:
                return int(array[position])
        except (IndexError, TypeError, ValueError, AttributeError) as exc:
            raise RuntimeUnavailableError(
                "TOKENIZER_OUTPUT_INVALID", "cannot inspect tokenizer output",
            ) from exc
        raise RuntimeUnavailableError(
            "TOKENIZER_OUTPUT_INVALID", "tokenizer output must be one or two dimensional",
        )

    def tokenize(self, query_text: str) -> TokenizedQuery:
        try:
            content_tokens = self.tokenizer.encode(query_text)
        except Exception as exc:
            raise RuntimeUnavailableError(
                "TOKENIZER_FAILED", "the vanilla ULIP tokenizer failed",
            ) from exc
        if not isinstance(content_tokens, Sequence):
            raise RuntimeUnavailableError(
                "TOKENIZER_OUTPUT_INVALID", "tokenizer.encode must return a sequence",
            )
        content_count = len(content_tokens)
        if content_count > 75:
            raise QueryTooLongError(content_count)
        token_count = content_count + 2
        try:
            try:
                tokens = self.tokenizer([query_text], context_length=77)
            except TypeError:
                tokens = self.tokenizer([query_text])
        except Exception as exc:
            raise RuntimeUnavailableError(
                "TOKENIZER_FAILED", "the vanilla ULIP tokenizer failed",
            ) from exc
        sot_token_id = self._sot_token_id()
        if sot_token_id is not None and self._token_at(tokens, 0) != sot_token_id:
            raise RuntimeUnavailableError(
                "TOKENIZER_SOT_MISSING", "the tokenizer did not retain SOT",
            )
        eot_present = self._token_at(tokens, token_count - 1) == self._eot_token_id()
        if not eot_present:
            raise RuntimeUnavailableError(
                "TOKENIZER_EOT_MISSING", "the tokenizer did not retain EOT",
            )
        return TokenizedQuery(
            tokens=tokens,
            content_token_count=content_count,
            token_count=token_count,
            eot_present=True,
        )


class VanillaULIPTextEncoder:
    """Small callable wrapper which never enables Core RAG or rewrites text."""

    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device
        base_model = getattr(model, "module", model)
        projection = getattr(base_model, "text_projection", None)
        self.embedding_dim = (
            int(projection.shape[1]) if projection is not None and len(projection.shape) == 2
            else None
        )

    def __call__(self, tokens: Any) -> np.ndarray:
        try:
            import torch
            import torch.nn.functional as functional
            from utils import utils as ulip_utils

            if not torch.is_tensor(tokens):
                tokens = torch.as_tensor(tokens, dtype=torch.long)
            if getattr(tokens, "is_sparse", False):
                tokens = tokens.to_dense()
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            tokens = tokens.to(device=self.device, dtype=torch.long).contiguous()
            with torch.no_grad():
                features = ulip_utils.get_model(self.model).encode_text(tokens)
                features = functional.normalize(features, dim=-1)
            return (
                features.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            )
        except Exception as exc:
            raise RuntimeUnavailableError(
                "ULIP_ENCODER_FAILED", "the vanilla ULIP text encoder failed",
            ) from exc


@dataclass(frozen=True)
class ProductRow:
    row_index: int
    product_id: str
    name: str
    category: str
    dimensions: Optional[Dict[str, Any]]
    dimension_status: str
    height_policy: str
    catalog_quality_status: str
    hard_filter_eligible: bool
    footprint_kind: str
    special_flags: Tuple[str, ...]
    allowed_yaws_deg: Tuple[int, ...]

    @property
    def production_eligible(self) -> bool:
        if not self.hard_filter_eligible:
            return False
        if self.category == "Sofa":
            # Product Candidates policy v1 has no positively verified rectangular
            # sofa rows.  A future bundle/policy version must explicitly change this.
            return False
        if self.catalog_quality_status not in {"supported", "degraded"}:
            return False
        if self.dimension_status in UNSAFE_DIMENSION_STATUSES:
            return False
        if self.footprint_kind == "special_unmodeled":
            return False
        if any(flag in UNSAFE_SPECIAL_FLAGS for flag in self.special_flags):
            return False
        if self.dimensions is None:
            return False
        if self.dimension_status == "partial":
            return (
                self.height_policy == "optional"
                and self.dimensions.get("height") is None
            )
        return self.dimension_status == "complete"

    def to_wire(self, *, score: float, tier: str, matched_term: str) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "matched_request_category": matched_term,
            "category_tier": tier,
            "retrieval_score": float(min(1.0, max(-1.0, score))),
            "dimensions": dict(self.dimensions) if self.dimensions is not None else None,
            "dimension_status": self.dimension_status,
            "height_policy": self.height_policy,
            "catalog_quality_status": self.catalog_quality_status,
            "hard_filter_eligible": self.hard_filter_eligible,
            "footprint_kind": self.footprint_kind,
            "special_flags": list(self.special_flags),
        }
        if self.allowed_yaws_deg and self.allowed_yaws_deg != (0, 90, 180, 270):
            row["allowed_yaws_deg"] = list(self.allowed_yaws_deg)
        return row


@dataclass(frozen=True)
class ResolvedTier:
    pools: Tuple[Tuple[str, str], ...]  # (canonical, earliest outer-trimmed term)
    unsupported_count: int
    blocked_count: int


def _finite_positive(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("%s must be a positive finite number" % field)
    return float(value)


def _mapping_value(value: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _call_or_value(value: Any) -> Any:
    return value() if callable(value) else value


def _parse_dimensions(raw: Mapping[str, Any], status: str) -> Optional[Dict[str, Any]]:
    dimensions = _mapping_value(raw, ("dimensions", "dimensions_mm"), None)
    if dimensions is None and any(key in raw for key in ("width_mm", "depth_mm")):
        dimensions = {
            "width": raw.get("width_mm"),
            "depth": raw.get("depth_mm"),
            "height": raw.get("height_mm"),
        }
    if dimensions is None:
        if status in UNSAFE_DIMENSION_STATUSES:
            return None
        raise ValueError("trusted dimensions require a dimensions object")
    if not isinstance(dimensions, Mapping):
        raise ValueError("dimensions must be an object")
    unit = dimensions.get("unit", "mm")
    if unit != "mm":
        raise ValueError("dimension profile must use mm")
    width = _finite_positive(dimensions.get("width"), "width")
    depth = _finite_positive(dimensions.get("depth"), "depth")
    height_raw = dimensions.get("height")
    height = None if height_raw is None else _finite_positive(height_raw, "height")
    if status == "complete" and height is None:
        raise ValueError("complete dimensions require height")
    if status == "partial" and height is not None:
        raise ValueError("partial dimensions require height=null")
    return {"unit": "mm", "width": width, "depth": depth, "height": height}


def _parse_product_row(raw: Mapping[str, Any], expected_index: int) -> ProductRow:
    if not isinstance(raw, Mapping):
        raise ValueError("product profile rows must be objects")
    row_index = raw.get("row_index", expected_index)
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index != expected_index:
        raise ValueError("product row identity/order mismatch")
    product_id = _mapping_value(raw, ("product_id", "id"))
    name = _mapping_value(raw, ("name", "product_name", "title"))
    category = _mapping_value(raw, ("category", "catalog_category", "canonical_category"))
    for field, value in (("product_id", product_id), ("name", name), ("category", category)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("%s must be a non-empty string" % field)
        if value != value.strip():
            raise ValueError("%s must already be trimmed in the immutable bundle" % field)
    status = raw.get("dimension_status", raw.get("dimension_parse_status", "missing"))
    if status not in {
        "complete", "partial", "missing", "suspect", "range_collapsed",
        "special_unmodeled",
    }:
        raise ValueError("invalid dimension_status")
    height_policy = raw.get("height_policy", "unknown")
    quality = raw.get("catalog_quality_status", raw.get("category_quality_status", "unknown"))
    footprint_kind = raw.get("footprint_kind", "rectangle")
    hard_eligible = raw.get("hard_filter_eligible", False)
    if height_policy not in {"optional", "required", "unknown"}:
        raise ValueError("invalid height_policy")
    if quality not in {"supported", "degraded", "blocked", "unknown"}:
        raise ValueError("invalid catalog_quality_status")
    if footprint_kind not in {
        "rectangle", "round_bounding_box", "polygon_bounding_box",
        "special_unmodeled",
    }:
        raise ValueError("invalid footprint_kind")
    if not isinstance(hard_eligible, bool):
        raise ValueError("hard_filter_eligible must be boolean")
    flags_raw = raw.get("special_flags", raw.get("reason_codes", []))
    if not isinstance(flags_raw, (list, tuple)) or any(
        not isinstance(flag, str) or not flag for flag in flags_raw
    ):
        raise ValueError("special_flags must be non-empty strings")
    if len(flags_raw) != len(set(flags_raw)):
        raise ValueError("special_flags must be unique")
    flags = tuple(sorted(flags_raw))
    yaws_raw = raw.get("allowed_yaws_deg", (0, 90, 180, 270))
    if yaws_raw is None:
        if hard_eligible:
            raise ValueError("eligible rows require allowed_yaws_deg")
        yaws: Tuple[int, ...] = ()
    else:
        if (
            not isinstance(yaws_raw, (list, tuple))
            or not yaws_raw
            or len(yaws_raw) != len(set(yaws_raw))
            or any(
                isinstance(yaw, bool)
                or not isinstance(yaw, (int, np.integer))
                or int(yaw) not in {0, 90, 180, 270}
                for yaw in yaws_raw
            )
        ):
            raise ValueError("invalid allowed_yaws_deg")
        yaws = tuple(int(yaw) for yaw in yaws_raw)
    dimensions = _parse_dimensions(raw, status)
    # Fail closed on profile cross-field corruption rather than silently fixing it.
    if status in UNSAFE_DIMENSION_STATUSES and hard_eligible:
        raise ValueError("unsafe dimensions cannot be hard-filter eligible")
    if quality in {"blocked", "unknown"} and hard_eligible:
        raise ValueError("blocked/unknown catalog rows cannot be eligible")
    if footprint_kind == "special_unmodeled" and (
        hard_eligible or status != "special_unmodeled"
    ):
        raise ValueError("special footprint policy is inconsistent")
    if status == "partial" and height_policy in {"required", "unknown"} and hard_eligible:
        raise ValueError("unverified required height cannot be eligible")
    return ProductRow(
        row_index=row_index,
        product_id=product_id,
        name=name,
        category=category,
        dimensions=dimensions,
        dimension_status=status,
        height_policy=height_policy,
        catalog_quality_status=quality,
        hard_filter_eligible=hard_eligible,
        footprint_kind=footprint_kind,
        special_flags=flags,
        allowed_yaws_deg=yaws,
    )


class ProductCandidateRuntime:
    """Immutable category-restricted vanilla ULIP retrieval runtime."""

    def __init__(
        self,
        *,
        profile_id: str,
        provenance: Mapping[str, Any],
        vectors: np.ndarray,
        rows: Sequence[Mapping[str, Any]],
        tokenizer: ULIPQueryTokenizer,
        text_encoder: Callable[[Any], np.ndarray],
        checks: Optional[Mapping[str, bool]] = None,
        inventory: Optional[Mapping[str, Any]] = None,
        integrity_revalidator: Optional[Callable[[], Any]] = None,
    ) -> None:
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("profile_id must be non-empty")
        self.profile_id = profile_id.strip()
        self.provenance = self._validate_provenance(provenance)
        if self.provenance["category_policy_version"] == "v2t36_to_ikea24_v2":
            self._enabled_category_terms = dict(ENABLED_CATEGORY_TERMS_V2)
            self._blocked_category_terms = dict(BLOCKED_CATEGORY_TERMS_V2)
            self._unsupported_category_terms = set(UNSUPPORTED_CATEGORY_TERMS_V2)
            self._controlled_category_terms = set(CONTROLLED_CATEGORY_TERMS_V2)
            self._degraded_canonical_categories = set(
                DEGRADED_CANONICAL_CATEGORIES_V2
            )
        else:
            self._enabled_category_terms = dict(ENABLED_CATEGORY_TERMS)
            self._blocked_category_terms = dict(BLOCKED_CATEGORY_TERMS)
            self._unsupported_category_terms = set(UNSUPPORTED_CATEGORY_TERMS)
            self._controlled_category_terms = set(CONTROLLED_CATEGORY_TERMS)
            self._degraded_canonical_categories = set(
                DEGRADED_CANONICAL_CATEGORIES
            )
        matrix = np.asarray(vectors)
        if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
            raise ValueError("vectors must be a non-empty rank-2 matrix")
        if matrix.shape[0] != len(rows) or not np.isfinite(matrix).all():
            raise ValueError("vector/product row identity mismatch or non-finite vectors")
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
            raise ValueError("vector bundle rows must already be L2-normalized")
        parsed_rows = tuple(_parse_product_row(row, index) for index, row in enumerate(rows))
        identifiers = [row.product_id for row in parsed_rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("product IDs must be unique")
        self.vectors = matrix
        self.rows = parsed_rows
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        encoder_dimension = getattr(text_encoder, "embedding_dim", None)
        if encoder_dimension is not None and int(encoder_dimension) != int(matrix.shape[1]):
            raise ValueError("model/vector embedding dimensions differ")
        supplied_checks = dict(
            {key: True for key in READY_CHECK_KEYS} if checks is None else checks
        )
        if set(supplied_checks) != set(READY_CHECK_KEYS) or any(
            not isinstance(value, bool) for value in supplied_checks.values()
        ):
            raise ValueError("ready checks must contain the seven fixed booleans")
        self._base_checks = {key: supplied_checks[key] for key in READY_CHECK_KEYS}
        if integrity_revalidator is not None and not callable(integrity_revalidator):
            raise TypeError("integrity_revalidator must be callable")
        self._integrity_revalidator = integrity_revalidator
        self._integrity_lock = threading.Lock()
        self._integrity_healthy = True
        self._integrity_failure: Optional[str] = None
        default_inventory = {
            "vector_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "vector_rows": int(matrix.shape[0]),
            "catalog_rows": len(parsed_rows),
            "dimension_profile_rows": len(parsed_rows),
        }
        supplied_inventory = dict(default_inventory if inventory is None else inventory)
        if set(supplied_inventory) != set(default_inventory):
            raise ValueError("inventory must contain exactly four fields")
        shape = supplied_inventory.get("vector_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in shape
        ):
            raise ValueError("inventory.vector_shape is invalid")
        for key in ("vector_rows", "catalog_rows", "dimension_profile_rows"):
            value = supplied_inventory.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError("inventory.%s must be an integer" % key)
        normalized_inventory = {
            "vector_shape": [int(shape[0]), int(shape[1])],
            "vector_rows": int(supplied_inventory["vector_rows"]),
            "catalog_rows": int(supplied_inventory["catalog_rows"]),
            "dimension_profile_rows": int(supplied_inventory["dimension_profile_rows"]),
        }
        if normalized_inventory != default_inventory:
            raise ValueError("inventory does not match loaded runtime assets")
        self.inventory = normalized_inventory
        self._category_indices: Dict[str, np.ndarray] = {}
        for canonical in set(self._enabled_category_terms.values()):
            indices = [
                index for index, row in enumerate(parsed_rows)
                if row.category.casefold() == canonical.casefold()
            ]
            self._category_indices[canonical] = np.asarray(indices, dtype=np.int64)

    @classmethod
    def from_components(
        cls,
        profile_id: str,
        provenance: Mapping[str, Any],
        vectors: np.ndarray,
        rows: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        text_encoder: Callable[[Any], np.ndarray],
        checks: Optional[Mapping[str, bool]] = None,
        inventory: Optional[Mapping[str, Any]] = None,
        tokenizer_id: Optional[str] = None,
        integrity_revalidator: Optional[Callable[[], Any]] = None,
    ) -> "ProductCandidateRuntime":
        wrapped = tokenizer
        if not isinstance(tokenizer, ULIPQueryTokenizer):
            wrapped = ULIPQueryTokenizer(
                tokenizer,
                tokenizer_id or getattr(tokenizer, "tokenizer_id", "test-ulip-tokenizer"),
            )
        return cls(
            profile_id=profile_id,
            provenance=provenance,
            vectors=vectors,
            rows=rows,
            tokenizer=wrapped,
            text_encoder=text_encoder,
            checks=checks,
            inventory=inventory,
            integrity_revalidator=integrity_revalidator,
        )

    @staticmethod
    def _validate_provenance(provenance: Mapping[str, Any]) -> Dict[str, str]:
        if not isinstance(provenance, Mapping) or set(provenance) != set(PROVENANCE_KEYS):
            raise ValueError("provenance must contain exactly seven fields")
        output: Dict[str, str] = {}
        for key in PROVENANCE_KEYS:
            value = provenance[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("provenance.%s must be non-empty" % key)
            if key.endswith("_sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("provenance.%s must be lowercase SHA-256" % key)
            output[key] = value
        if output["category_policy_version"] not in {
            "v2t36_to_ikea15_v1", "v2t36_to_ikea24_v2",
        }:
            raise ValueError("unsupported category policy version")
        if output["dimension_axis_policy_version"] not in {
            "dimension_axis_v1", "dimension_axis_ikea_snapshot_v2",
        }:
            raise ValueError("unsupported dimension-axis policy version")
        return output

    @property
    def checks(self) -> Dict[str, bool]:
        if self._integrity_healthy:
            return dict(self._base_checks)
        # A drift detected by full bundle revalidation invalidates the active
        # deployment tuple as a whole.  Do not guess which cached component is
        # still safe merely from the first verifier exception.
        return {key: False for key in READY_CHECK_KEYS}

    @property
    def ready(self) -> bool:
        return all(self.checks.values())

    @property
    def integrity_failure(self) -> Optional[str]:
        return self._integrity_failure

    def revalidate_integrity(self) -> bool:
        """Re-hash the immutable profile and latch readiness fail-closed.

        The injected validator must perform only offline bundle verification;
        it must not reload the GPU model or query MongoDB.  A later successful
        check may recover readiness because inference continues to use the
        already-loaded, immutable in-memory tuple rather than replacing it.
        """
        if self._integrity_revalidator is None:
            return self.ready
        with self._integrity_lock:
            try:
                result = self._integrity_revalidator()
                if result is False:
                    raise RuntimeError("deployment profile revalidation returned false")
            except Exception:
                self._integrity_healthy = False
                self._integrity_failure = "DEPLOYMENT_INTEGRITY_REVALIDATION_FAILED"
                return False
            self._integrity_healthy = True
            self._integrity_failure = None
            return all(self._base_checks.values())

    def readiness_payload(self) -> Dict[str, Any]:
        failures = [] if self.ready else sorted(
            "%s_FAILED" % key.upper() for key, value in self.checks.items() if not value
        )
        if self.integrity_failure is not None:
            failures = sorted(set(failures + [self.integrity_failure]))
        return {
            "schema_version": "1.0",
            "service": "ulip-product-candidates",
            "status": "ready" if self.ready else "not_ready",
            "deployment_profile_id": self.profile_id if self.ready else None,
            "endpoint_schema_version": "1.0",
            "provenance": dict(self.provenance) if self.ready else None,
            "checks": dict(self.checks),
            "inventory": dict(self.inventory),
            "failures": failures,
        }

    def _validate_request_semantics(self, wire: Mapping[str, Any]) -> None:
        if not wire["request_id"].strip():
            raise RequestPolicyError("REQUEST_ID_EMPTY", "request_id is whitespace-only")
        if not wire["query_text"].strip():
            raise RequestPolicyError("QUERY_TEXT_EMPTY", "query_text is whitespace-only")
        primary = wire["categories_primary"]
        secondary = wire["categories_secondary"]
        normalized: List[str] = []
        for raw in list(primary) + list(secondary):
            term = raw.strip().casefold()
            if not term:
                raise RequestPolicyError(
                    "CATEGORY_TERM_EMPTY", "category terms cannot be whitespace-only",
                )
            if term in normalized:
                raise RequestPolicyError(
                    "CATEGORY_TERM_DUPLICATE",
                    "category terms must be case-insensitively unique across tiers",
                )
            normalized.append(term)
            if term not in self._controlled_category_terms:
                raise RequestPolicyError(
                    "CATEGORY_TERM_UNKNOWN", "category term is outside the V2T controlled vocabulary",
                )

    def _resolve_tier(self, raw_terms: Sequence[str]) -> ResolvedTier:
        pools: Dict[str, str] = {}
        unsupported = 0
        blocked = 0
        for raw in raw_terms:
            normalized = raw.strip().casefold()
            if normalized in self._enabled_category_terms:
                canonical = self._enabled_category_terms[normalized]
                if canonical not in pools:
                    # Preserve the earliest term's spelling/case after the
                    # contract-mandated outer trim.
                    pools[canonical] = raw.strip()
            elif normalized in self._blocked_category_terms:
                blocked += 1
            else:
                unsupported += 1
        return ResolvedTier(tuple(pools.items()), unsupported, blocked)

    def _encode_query(self, tokenized: TokenizedQuery) -> np.ndarray:
        try:
            query = np.asarray(self.text_encoder(tokenized.tokens), dtype=np.float32)
        except RuntimeUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeUnavailableError(
                "ULIP_ENCODER_FAILED", "the vanilla ULIP text encoder failed",
            ) from exc
        query = np.squeeze(query)
        if query.ndim != 1 or query.shape[0] != self.vectors.shape[1]:
            raise RuntimeUnavailableError(
                "ULIP_EMBEDDING_SHAPE_INVALID", "query/vector embedding dimensions differ",
            )
        if not np.isfinite(query).all():
            raise RuntimeUnavailableError(
                "ULIP_EMBEDDING_NONFINITE", "query embedding contains non-finite values",
            )
        norm = float(np.linalg.norm(query))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeUnavailableError(
                "ULIP_EMBEDDING_NORM_INVALID", "query embedding norm is invalid",
            )
        return np.ascontiguousarray(query / norm, dtype=np.float32)

    def _retrieve_tier(
        self,
        pools: Sequence[Tuple[str, str]],
        tier: str,
        query_vector: np.ndarray,
        top_k: int,
        used_product_ids: set,
    ) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, str, int, str]] = []
        for canonical, matched_term in pools:
            indices = self._category_indices.get(canonical, np.asarray([], dtype=np.int64))
            if not len(indices):
                continue
            # Score the whole requested canonical pool before trust gating and Top-K.
            scores = self.vectors[indices] @ query_vector
            for local_index, score in enumerate(scores):
                row_index = int(indices[local_index])
                row = self.rows[row_index]
                if not row.production_eligible or row.product_id in used_product_ids:
                    continue
                scored.append((
                    float(min(1.0, max(-1.0, float(score)))),
                    row.product_id,
                    row_index,
                    matched_term,
                ))
        scored.sort(key=lambda item: (-item[0], item[1]))
        output = []
        for score, product_id, row_index, matched_term in scored[:top_k]:
            used_product_ids.add(product_id)
            output.append(self.rows[row_index].to_wire(
                score=score, tier=tier, matched_term=matched_term,
            ))
        return output

    def retrieve(self, wire: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeUnavailableError(
                "DEPLOYMENT_PROFILE_NOT_READY", "immutable deployment profile is not ready",
            )
        self._validate_request_semantics(wire)
        tokenized = self.tokenizer.tokenize(wire["query_text"])
        primary = self._resolve_tier(wire["categories_primary"])
        secondary = self._resolve_tier(wire["categories_secondary"])
        warnings = set()
        all_term_count = len(wire["categories_primary"]) + len(wire["categories_secondary"])
        unsupported_count = primary.unsupported_count + secondary.unsupported_count
        blocked_count = primary.blocked_count + secondary.blocked_count
        if unsupported_count:
            warnings.add(
                "CATEGORY_UNSUPPORTED"
                if unsupported_count == all_term_count
                else "CATEGORY_PARTIALLY_UNSUPPORTED"
            )
        if blocked_count:
            warnings.add("CATEGORY_DATA_QUALITY_BLOCKED")

        primary_pools = list(primary.pools)
        secondary_pools = list(secondary.pools)
        primary_canonical = {canonical for canonical, _ in primary_pools}
        deduped_secondary = [
            pair for pair in secondary_pools if pair[0] not in primary_canonical
        ]
        if len(deduped_secondary) != len(secondary_pools):
            warnings.add("CATEGORY_CANONICAL_POOL_DEDUPED")
        secondary_pools = deduped_secondary
        if any(
            canonical in self._degraded_canonical_categories
            for canonical, _ in primary_pools + secondary_pools
        ):
            warnings.add("CATEGORY_DATA_QUALITY_DEGRADED")

        results: List[Dict[str, Any]] = []
        if primary_pools or secondary_pools:
            query_vector = self._encode_query(tokenized)
            used: set = set()
            top_k = wire["top_k_per_tier"]
            results.extend(self._retrieve_tier(
                primary_pools, "primary", query_vector, top_k, used,
            ))
            results.extend(self._retrieve_tier(
                secondary_pools, "secondary", query_vector, top_k, used,
            ))

        warning_list = sorted(warnings)
        if any(code not in WARNING_ALLOWLIST for code in warning_list):
            raise RuntimeUnavailableError(
                "WARNING_POLICY_INVALID", "runtime produced a warning outside the allowlist",
            )
        return {
            "schema_version": "1.0",
            # Echo request_id exactly; the strict consumer requires exact identity.
            "request_id": wire["request_id"],
            "status": "ok" if results else "no_match",
            "query_metadata": {
                "tokenizer_id": self.tokenizer.tokenizer_id,
                "token_count": tokenized.token_count,
                "truncated": False,
                "eot_present": tokenized.eot_present,
            },
            "results": results,
            "warnings": warning_list,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_profile(
        cls,
        profile: Any,
        *,
        device: str = "cuda",
        integrity_revalidator: Optional[Callable[[], Any]] = None,
    ) -> "ProductCandidateRuntime":
        """Adapt a verified ``DeploymentProfile`` without depending on its class."""
        verify = _mapping_value(profile, ("verify", "assert_ready"), None)
        if callable(verify):
            result = verify()
            if result is False:
                raise ValueError("deployment profile verification failed")
        profile_id = _mapping_value(profile, ("deployment_profile_id", "profile_id", "id"))
        provenance = _mapping_value(profile, ("provenance",))
        vectors = _call_or_value(_mapping_value(
            profile, ("vectors", "pc_vectors", "vector_matrix", "load_vectors")
        ))
        rows = _call_or_value(_mapping_value(
            profile, ("product_rows", "rows", "catalog_rows", "load_product_rows")
        ))
        checks = _mapping_value(profile, ("checks", "ready_checks"), None)
        inventory = _mapping_value(profile, ("inventory",), None)
        tokenizer = _mapping_value(profile, ("tokenizer",), None)
        text_encoder = _mapping_value(profile, ("text_encoder", "encoder"), None)
        tokenizer_id = _mapping_value(
            profile, ("tokenizer_id",), "ulip-simple-tokenizer-v1",
        )
        if tokenizer is None:
            tokenizer_code_path = _mapping_value(profile, ("tokenizer_code_path",))
            tokenizer_bpe_path = _mapping_value(
                profile, ("tokenizer_bpe_path", "bpe_path")
            )
            tokenizer = ULIPQueryTokenizer(
                load_immutable_tokenizer(tokenizer_code_path, tokenizer_bpe_path),
                tokenizer_id,
            )
        if text_encoder is None:
            checkpoint = _mapping_value(
                profile,
                ("model_checkpoint_path", "checkpoint_path", "checkpoint"),
            )
            model = load_vanilla_ulip_model(str(checkpoint), device)
            text_encoder = VanillaULIPTextEncoder(model, device)
        return cls.from_components(
            profile_id=profile_id,
            provenance=provenance,
            vectors=vectors,
            rows=rows,
            tokenizer=tokenizer,
            tokenizer_id=tokenizer_id,
            text_encoder=text_encoder,
            checks=checks,
            inventory=inventory,
            integrity_revalidator=integrity_revalidator,
        )


def load_immutable_tokenizer(code_path: Any, bpe_path: Any) -> Any:
    """Execute the verified bundle member without writing pycache into the bundle."""
    code = Path(str(code_path)).expanduser().resolve()
    bpe = Path(str(bpe_path)).expanduser().resolve()
    if not code.is_file() or not bpe.is_file():
        raise RuntimeUnavailableError(
            "TOKENIZER_BUNDLE_MISSING", "tokenizer bundle members are unavailable",
        )
    try:
        source = code.read_text(encoding="utf-8")
        module = types.ModuleType("_immutable_ulip_product_tokenizer")
        module.__file__ = str(code)
        exec(compile(source, str(code), "exec"), module.__dict__)
        tokenizer_class = getattr(module, "SimpleTokenizer")
        return tokenizer_class(bpe_path=str(bpe))
    except Exception as exc:
        raise RuntimeUnavailableError(
            "TOKENIZER_INITIALIZATION_FAILED",
            "immutable vanilla ULIP tokenizer initialization failed",
        ) from exc


def load_vanilla_ulip_model(checkpoint_path: str, device: str) -> Any:
    """Load only the vanilla ULIP base; never construct a Core RAG adapter."""
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise RuntimeUnavailableError(
            "MODEL_CHECKPOINT_MISSING", "model checkpoint is unavailable",
        )
    try:
        os.environ["SPCONV_ALGO"] = "native"
        os.environ["ATTN_BACKEND"] = "xformers"
        import torch
        import models.ULIP_models as models

        # This edge05 image has the documented cuDNN incompatibility; the
        # existing verified vanilla serving path uses the same workaround.
        torch.backends.cudnn.enabled = False

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        state = {
            (key[len("module."):] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
        saved_args = checkpoint.get("args") if isinstance(checkpoint, Mapping) else None
        if isinstance(saved_args, Mapping):
            model_args = Namespace(**dict(saved_args))
        elif saved_args is not None:
            model_args = Namespace(**vars(saved_args))
        else:
            model_args = Namespace(model="ULIP_PointBERT")
        model_name = getattr(model_args, "model", "ULIP_PointBERT")
        model_args.evaluate_3d = True
        is_rag_checkpoint = model_name == "ULIP_PointBERT_RAG" or any(
            key.startswith("rag_enhancer.") for key in state
        )
        factory_name = "ULIP_PointBERT" if is_rag_checkpoint else model_name
        model = getattr(models, factory_name)(args=model_args)
        base_state = {
            key: value for key, value in state.items()
            if not key.startswith("rag_enhancer.")
        }
        model.load_state_dict(base_state, strict=True)
        model = model.to(device).eval()
        return model
    except RuntimeUnavailableError:
        raise
    except Exception as exc:
        raise RuntimeUnavailableError(
            "MODEL_INITIALIZATION_FAILED", "vanilla ULIP model initialization failed",
        ) from exc


def load_vanilla_ulip_components(
    checkpoint_path: str,
    device: str,
    tokenizer_code_path: Optional[Any] = None,
    tokenizer_bpe_path: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Compatibility helper returning the base model and a pinned tokenizer."""
    model = load_vanilla_ulip_model(checkpoint_path, device)
    if tokenizer_code_path is not None and tokenizer_bpe_path is not None:
        tokenizer = load_immutable_tokenizer(tokenizer_code_path, tokenizer_bpe_path)
    else:
        # Test/backward-compatible use only.  Production ``from_profile`` always
        # supplies the two immutable tokenizer bundle members.
        from utils.tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer()
    return model, tokenizer
