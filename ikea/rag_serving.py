"""Core-RAG text adapter for the IKEA ULIP retrieval index.

This module intentionally does not import or mutate ``app_ikea_retrieval``.
It composes the IKEA production text backbone with the trained Core
``RAGEnhancer`` only after proving that both checkpoints contain exactly the
same frozen text space.  Equal output dimensionality alone is never accepted
as compatibility evidence.

The default corpus is the 1,095-row corpus archived with the original RAG
training run.  Set ``ULIP_RAG_CORPUS_DIR`` to use another corpus/index pair.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "core"
DEFAULT_IKEA_CHECKPOINT = Path(
    os.getenv("IKEA_ULIP_CHECKPOINT", "/mnt/P300/data/ULIP/checkpoint_last.pt")
)
DEFAULT_RAG_CHECKPOINT = Path(
    os.getenv(
        "ULIP_RAG_CHECKPOINT",
        str(CORE_DIR / "outputs/RAG2_Stage1/checkpoint_best.pt"),
    )
)
_LEGACY_CORPUS = Path(
    "/mnt/P300/data/ULIP/4090_cheng_archive_20260715/ULIP_RAG/rag_corpus_1095"
)
LEGACY_1095_PROFILE: Dict[str, Any] = {
    "name": "legacy1095",
    "corpus_rows": 1095,
    "faiss_rows": 1095,
    "faiss_dimension": 384,
    "corpus_sha256": "d9bc65b3006e7b0a085025e4a53fee775197df506ff8b0e4b7404e5fa424b537",
    "index_sha256": "2847f6638b8dda42ee52dcdfc2a5d7f2aa71b650d1d266850241f294119c3f0f",
    "ikea_checkpoint_sha256": "49ac18ab10950412f016c9e023d1c7e9b34250a257d59b70b5590d74207f3e01",
    "rag_checkpoint_sha256": "c67ec093d3118f85a8498d436f50cdfa64a0cfef2e616c0786cd14595ab5f298",
    "vectors_pc_sha256": "30070b4c576bf511ac498bc1f6ddb46502bd0cf4737a455447d5e85fc707d207",
    "meta_pc_sha256": "0c05805753bb085080e75fa8db204c795f46b6f2c55754e37ae3a819bd068a94",
}
PROVENANCE_SCHEMA = "ikea-rag-training-provenance/1.0"
SUPPORTED_CORPUS_PROFILES = ("legacy1095", "provenance")
DEFAULT_RAG_CORPUS_DIR = Path(
    os.getenv(
        "ULIP_RAG_CORPUS_DIR",
        str(_LEGACY_CORPUS if _LEGACY_CORPUS.is_dir() else CORE_DIR / "rag_corpus"),
    )
)
DEFAULT_VECTOR_DIR = Path(
    os.getenv("VEC_DIR", "/mnt/P300/data/ikea_data/vectors")
)

# These parameters define the CLIP text space before and after projection.
# They were frozen in both Core RAG stages, but that fact is verified at load
# time rather than assumed from the training recipe.
EXACT_BASE_PREFIXES: Tuple[str, ...] = (
    "token_embedding.",
    "positional_embedding",
    "transformer.",
    "ln_final.",
    "text_projection",
    "visual.",
    "image_projection",
    "logit_scale",
)


class RAGCompatibilityError(RuntimeError):
    """Raised when artifacts cannot safely be composed."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_corpus_profile(
    expected: Mapping[str, Any],
    *,
    corpus_rows: int,
    faiss_rows: int,
    faiss_dimension: int,
    corpus_sha256: str,
    index_sha256: str,
) -> None:
    """Fail closed when a named immutable corpus profile has drifted."""

    actual = {
        "corpus_rows": int(corpus_rows),
        "faiss_rows": int(faiss_rows),
        "faiss_dimension": int(faiss_dimension),
        "corpus_sha256": corpus_sha256,
        "index_sha256": index_sha256,
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in actual
        if actual[key] != expected.get(key)
    }
    if mismatches:
        raise RAGCompatibilityError(
            f"corpus profile {expected.get('name', 'unknown')} drifted: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


def _checkpoint_model_name(checkpoint: Mapping[str, Any]) -> Optional[str]:
    args = checkpoint.get("args")
    if isinstance(args, Mapping):
        value = args.get("model")
    else:
        value = getattr(args, "model", None)
    return str(value) if value else None


def _require_provenance(
    checkpoint: Mapping[str, Any], label: str, expected_strategy: str
) -> Mapping[str, Any]:
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RAGCompatibilityError(f"{label} checkpoint has no provenance mapping")
    if provenance.get("schema_version") != PROVENANCE_SCHEMA:
        raise RAGCompatibilityError(
            f"{label} checkpoint has unsupported provenance schema: "
            f"{provenance.get('schema_version')!r}"
        )
    if provenance.get("pipeline") != "ikea_rag":
        raise RAGCompatibilityError(
            f"{label} checkpoint provenance pipeline must be 'ikea_rag', got "
            f"{provenance.get('pipeline')!r}"
        )
    if provenance.get("training_strategy") != expected_strategy:
        raise RAGCompatibilityError(
            f"{label} checkpoint provenance must identify {expected_strategy}, got "
            f"{provenance.get('training_strategy')!r}"
        )
    return provenance


def _validate_recorded_rag(
    provenance: Mapping[str, Any],
    label: str,
    *,
    corpus_sha256: str,
    index_sha256: str,
) -> None:
    recorded = provenance.get("rag")
    if not isinstance(recorded, Mapping):
        raise RAGCompatibilityError(f"{label} provenance has no rag mapping")
    mismatches = {}
    for key, actual in (
        ("corpus_sha256", corpus_sha256),
        ("index_sha256", index_sha256),
    ):
        if recorded.get(key) != actual:
            mismatches[key] = {"recorded": recorded.get(key), "actual": actual}
    if mismatches:
        raise RAGCompatibilityError(
            f"{label} provenance RAG artifacts drifted: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


def validate_provenance_profile(
    base_checkpoint: Mapping[str, Any],
    enhancer_checkpoint: Mapping[str, Any],
    *,
    enhancer_checkpoint_sha256: str,
    base_checkpoint_sha256: str,
    corpus_sha256: str,
    index_sha256: str,
) -> Dict[str, Any]:
    """Validate a promotable IKEA RAG checkpoint chain.

    The enhancer must always come from a provenance-bound IKEA Stage 1 run.
    A vanilla base checkpoint is allowed for Stage 1 serving.  If the selected
    base is a RAG checkpoint, it is only promotable as a provenance-bound Stage
    2 checkpoint that names this exact Stage 1 artifact and corpus/index pair.
    """

    stage1 = _require_provenance(enhancer_checkpoint, "Stage 1", "stage_1")
    _validate_recorded_rag(
        stage1,
        "Stage 1",
        corpus_sha256=corpus_sha256,
        index_sha256=index_sha256,
    )

    base_provenance = base_checkpoint.get("provenance")
    base_model_name = _checkpoint_model_name(base_checkpoint)
    rag_base = base_model_name == "ULIP_PointBERT_RAG" or isinstance(
        base_provenance, Mapping
    ) and base_provenance.get("training_strategy") in {"stage_1", "stage_2"}
    stage2_validated = False
    stage1_dataset = stage1.get("dataset")
    stage1_base = stage1.get("base_ulip_checkpoint")
    stage1_dataset_sha = (
        stage1_dataset.get("bundle_sha256")
        if isinstance(stage1_dataset, Mapping)
        else None
    )
    stage1_base_sha = (
        stage1_base.get("sha256") if isinstance(stage1_base, Mapping) else None
    )
    if not stage1_dataset_sha or not stage1_base_sha:
        raise RAGCompatibilityError(
            "Stage 1 provenance must bind dataset.bundle_sha256 and "
            "base_ulip_checkpoint.sha256"
        )
    if rag_base:
        stage2 = _require_provenance(base_checkpoint, "Base Stage 2", "stage_2")
        _validate_recorded_rag(
            stage2,
            "Base Stage 2",
            corpus_sha256=corpus_sha256,
            index_sha256=index_sha256,
        )
        recorded_stage1 = stage2.get("stage1_checkpoint")
        recorded_sha = (
            recorded_stage1.get("sha256")
            if isinstance(recorded_stage1, Mapping)
            else None
        )
        if recorded_sha != enhancer_checkpoint_sha256:
            raise RAGCompatibilityError(
                "Base Stage 2 provenance points to a different Stage 1 checkpoint: "
                f"recorded={recorded_sha!r}, actual={enhancer_checkpoint_sha256!r}"
            )
        stage2_dataset = stage2.get("dataset")
        stage2_base = stage2.get("base_ulip_checkpoint")
        stage2_dataset_sha = (
            stage2_dataset.get("bundle_sha256")
            if isinstance(stage2_dataset, Mapping)
            else None
        )
        stage2_base_sha = (
            stage2_base.get("sha256") if isinstance(stage2_base, Mapping) else None
        )
        lineage_mismatches = {}
        if stage2_dataset_sha != stage1_dataset_sha:
            lineage_mismatches["dataset.bundle_sha256"] = {
                "stage1": stage1_dataset_sha,
                "stage2": stage2_dataset_sha,
            }
        if stage2_base_sha != stage1_base_sha:
            lineage_mismatches["base_ulip_checkpoint.sha256"] = {
                "stage1": stage1_base_sha,
                "stage2": stage2_base_sha,
            }
        if lineage_mismatches:
            raise RAGCompatibilityError(
                "Stage 1/Stage 2 training lineage mismatch: "
                + json.dumps(lineage_mismatches, ensure_ascii=False)
            )
        stage2_validated = True
    elif stage1_base_sha != base_checkpoint_sha256:
        raise RAGCompatibilityError(
            "Stage 1 provenance points to a different vanilla base checkpoint: "
            f"recorded={stage1_base_sha!r}, actual={base_checkpoint_sha256!r}"
        )

    return {
        "schema_version": PROVENANCE_SCHEMA,
        "stage1_training_strategy": stage1.get("training_strategy"),
        "base_training_strategy": (
            base_provenance.get("training_strategy")
            if isinstance(base_provenance, Mapping)
            else "vanilla"
        ),
        "stage2_validated": stage2_validated,
        "dataset_bundle_sha256": stage1_dataset_sha,
        "origin_base_checkpoint_sha256": stage1_base_sha,
    }


def validate_vector_bundle(
    vector_dir: Path,
    *,
    corpus_profile: str,
    base_checkpoint_sha256: str,
    vectors: np.ndarray,
    metadata_rows: int,
    embedding_dimension: int,
) -> Dict[str, Any]:
    """Bind product vectors to the serving checkpoint and named profile."""

    vector_dir = Path(vector_dir)
    vector_path = vector_dir / "vectors_pc.npy"
    meta_path = vector_dir / "meta_pc.jsonl"
    schema_path = vector_dir / "schema.json"
    for label, path in (
        ("PC vectors", vector_path),
        ("PC metadata", meta_path),
        ("vector schema", schema_path),
    ):
        if not path.is_file():
            raise RAGCompatibilityError(f"{label} not found: {path}")

    vector_sha256 = _sha256_file(vector_path)
    meta_sha256 = _sha256_file(meta_path)
    schema_sha256 = _sha256_file(schema_path)
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RAGCompatibilityError(f"invalid vector schema: {schema_path}") from exc
    if not isinstance(schema, Mapping):
        raise RAGCompatibilityError("vector schema must be a JSON object")

    if vectors.ndim != 2 or vectors.shape[0] != metadata_rows:
        raise RAGCompatibilityError(
            "PC vector/metadata shape mismatch: "
            f"vectors={tuple(vectors.shape)}, metadata_rows={metadata_rows}"
        )
    if int(vectors.shape[1]) != int(embedding_dimension):
        raise RAGCompatibilityError(
            "PC vector/base embedding dimension mismatch: "
            f"vectors={vectors.shape[1]}, base={embedding_dimension}"
        )

    if corpus_profile == "legacy1095":
        mismatches = {}
        for key, actual in (
            ("vectors_pc_sha256", vector_sha256),
            ("meta_pc_sha256", meta_sha256),
        ):
            expected = LEGACY_1095_PROFILE[key]
            if actual != expected:
                mismatches[key] = {"expected": expected, "actual": actual}
        if mismatches:
            raise RAGCompatibilityError(
                "legacy1095 vector bundle drifted: "
                + json.dumps(mismatches, ensure_ascii=False)
            )
    elif corpus_profile == "provenance":
        errors = {}
        if schema.get("schema_version") != "ikea-ulip-vectors/2.0":
            errors["schema_version"] = {
                "expected": "ikea-ulip-vectors/2.0",
                "actual": schema.get("schema_version"),
            }
        checkpoint = schema.get("checkpoint")
        recorded_checkpoint_sha = (
            checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None
        )
        if recorded_checkpoint_sha != base_checkpoint_sha256:
            errors["checkpoint.sha256"] = {
                "expected": base_checkpoint_sha256,
                "actual": recorded_checkpoint_sha,
            }
        if schema.get("text_embedding_mode") != "vanilla":
            errors["text_embedding_mode"] = {
                "expected": "vanilla",
                "actual": schema.get("text_embedding_mode"),
            }
        modalities = schema.get("modalities")
        if not isinstance(modalities, list) or "pc" not in modalities:
            errors["modalities"] = {"expected_contains": "pc", "actual": modalities}
        dims = schema.get("dims")
        recorded_pc_dim = dims.get("pc") if isinstance(dims, Mapping) else None
        if recorded_pc_dim != int(vectors.shape[1]):
            errors["dims.pc"] = {
                "expected": int(vectors.shape[1]),
                "actual": recorded_pc_dim,
            }
        if errors:
            raise RAGCompatibilityError(
                "provenance vector bundle is incompatible: "
                + json.dumps(errors, ensure_ascii=False)
            )
    else:  # defensive: load_adapter validates this before reaching the bundle
        raise RAGCompatibilityError(f"unsupported corpus profile: {corpus_profile!r}")

    return {
        "schema_version": schema.get("schema_version", "legacy-unversioned"),
        "vectors_pc": {"path": str(vector_path), "sha256": vector_sha256},
        "meta_pc": {"path": str(meta_path), "sha256": meta_sha256},
        "schema": {"path": str(schema_path), "sha256": schema_sha256},
    }


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    checked_tensors: int
    missing_in_rag: Tuple[str, ...]
    extra_in_rag: Tuple[str, ...]
    shape_mismatches: Tuple[str, ...]
    value_mismatches: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _state_dict(checkpoint_or_state: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    raw = checkpoint_or_state.get("state_dict", checkpoint_or_state)
    if not isinstance(raw, Mapping):
        raise RAGCompatibilityError("checkpoint does not contain a state_dict mapping")
    state: Dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        normalized = key[len("module.") :] if key.startswith("module.") else key
        if torch.is_tensor(value):
            state[normalized] = value
    return state


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise RAGCompatibilityError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # torch versions before mmap support
        return torch.load(path, map_location="cpu", weights_only=False)


def _select_prefixes(
    state: Mapping[str, torch.Tensor], prefixes: Sequence[str]
) -> Dict[str, torch.Tensor]:
    return {key: value for key, value in state.items() if key.startswith(tuple(prefixes))}


def validate_exact_base_prefixes(
    ikea_checkpoint_or_state: Mapping[str, Any],
    rag_checkpoint_or_state: Mapping[str, Any],
    prefixes: Sequence[str] = EXACT_BASE_PREFIXES,
    *,
    raise_on_error: bool = True,
) -> CompatibilityReport:
    """Require exact tensor identity for the shared CLIP text space.

    Shapes and 512-dimensional outputs are insufficient: every selected
    tensor must be present, have the same shape, dtype and value.  This catches
    accidentally mixing an IKEA vector index with another CLIP/ULIP space.
    """

    ikea = _select_prefixes(_state_dict(ikea_checkpoint_or_state), prefixes)
    rag = _select_prefixes(_state_dict(rag_checkpoint_or_state), prefixes)
    if not ikea:
        raise RAGCompatibilityError("IKEA checkpoint contains no text-backbone tensors")
    absent_groups = [prefix for prefix in prefixes if not any(key.startswith(prefix) for key in ikea)]
    if absent_groups:
        raise RAGCompatibilityError(
            f"IKEA checkpoint is missing required frozen base groups: {absent_groups}"
        )

    ikea_keys, rag_keys = set(ikea), set(rag)
    missing = tuple(sorted(ikea_keys - rag_keys))
    extra = tuple(sorted(rag_keys - ikea_keys))
    shape_mismatches: List[str] = []
    value_mismatches: List[str] = []
    checked = 0
    for key in sorted(ikea_keys & rag_keys):
        left, right = ikea[key], rag[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            shape_mismatches.append(key)
            continue
        checked += 1
        # A live serving model is already on CUDA while checkpoint tensors are
        # intentionally loaded on CPU.  Device placement is not part of the
        # embedding-space identity; compare exact values on CPU.
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            value_mismatches.append(key)

    report = CompatibilityReport(
        compatible=not (missing or extra or shape_mismatches or value_mismatches),
        checked_tensors=checked,
        missing_in_rag=missing,
        extra_in_rag=extra,
        shape_mismatches=tuple(shape_mismatches),
        value_mismatches=tuple(value_mismatches),
    )
    if raise_on_error and not report.compatible:
        raise RAGCompatibilityError(
            "IKEA/Core RAG text spaces are not exactly compatible: "
            + json.dumps(report.to_dict(), ensure_ascii=False)
        )
    return report


def tokenize_preserving_eot(
    tokenizer: Any, texts: Sequence[str], context_length: int = 77
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Tokenize like CLIP while guaranteeing that the EOT token survives."""

    if isinstance(texts, str):
        texts = [texts]
    try:
        sot = tokenizer.encoder["<|startoftext|>"]
        eot = tokenizer.encoder["<|endoftext|>"]
    except (AttributeError, KeyError) as exc:
        raise RAGCompatibilityError(
            "tokenizer must expose SimpleTokenizer.encode and SOT/EOT ids"
        ) from exc

    output = torch.zeros(len(texts), context_length, dtype=torch.long)
    diagnostics: List[Dict[str, Any]] = []
    max_content = context_length - 2
    for row, text in enumerate(texts):
        content = list(tokenizer.encode(text))
        used = content[:max_content]
        tokens = [sot, *used, eot]
        output[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        diagnostics.append(
            {
                "content_token_count": len(content),
                "input_token_count": len(content) + 2,
                "used_token_count": len(tokens),
                "truncated": len(content) > max_content,
                "eot_preserved": True,
            }
        )
    return output, diagnostics


@dataclass(frozen=True)
class RetrievedDocument:
    rank: int
    row_index: int
    score: float
    text: str
    obj_id: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiagnosticRetriever:
    """Score-preserving wrapper around Core ``RAGRetriever`` internals."""

    def __init__(self, core_retriever: Any, corpus: Sequence[Mapping[str, Any]]):
        self.core = core_retriever
        self.corpus = list(corpus)
        if int(self.core.index.ntotal) != len(self.corpus):
            raise RAGCompatibilityError(
                f"FAISS/corpus row mismatch: {self.core.index.ntotal} != {len(self.corpus)}"
            )
        encoder_dim = self.core.query_encoder.get_sentence_embedding_dimension()
        if encoder_dim is not None and int(encoder_dim) != int(self.core.index.d):
            raise RAGCompatibilityError(
                f"MiniLM/FAISS dimension mismatch: {encoder_dim} != {self.core.index.d}"
            )

    def retrieve(self, queries: Sequence[str], top_k: Optional[int] = None) -> List[List[RetrievedDocument]]:
        queries = [str(query) for query in queries]
        if any(not query.strip() for query in queries):
            raise ValueError("RAG query must be non-empty")
        k = int(top_k or self.core.top_k)
        if k <= 0:
            raise ValueError("rag top_k must be positive")
        k = min(k, len(self.corpus))
        embeddings = self.core.query_encoder.encode(
            queries, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(embeddings).all() or np.any(norms <= 0):
            raise RAGCompatibilityError("query encoder produced invalid embeddings")
        embeddings /= norms
        scores, indices = self.core.index.search(embeddings, k)
        batches: List[List[RetrievedDocument]] = []
        for row_scores, row_indices in zip(scores, indices):
            docs: List[RetrievedDocument] = []
            for rank, (score, index) in enumerate(zip(row_scores, row_indices), start=1):
                if index < 0:
                    continue
                item = self.corpus[int(index)]
                docs.append(
                    RetrievedDocument(
                        rank=rank,
                        row_index=int(index),
                        score=float(score),
                        text=str(item.get("text", "")),
                        obj_id=item.get("obj_id"),
                        category=item.get("category"),
                        source=item.get("source"),
                    )
                )
            batches.append(docs)
        return batches


class RAGServingAdapter:
    """Encode IKEA queries in vanilla or Core-RAG-enhanced ULIP space."""

    def __init__(
        self,
        *,
        base_model: torch.nn.Module,
        tokenizer: Any,
        enhancer: torch.nn.Module,
        retriever: Any,
        compatibility: CompatibilityReport,
        device: str,
        pc_vectors: Optional[np.ndarray] = None,
        pc_meta: Optional[Sequence[Mapping[str, Any]]] = None,
        artifact_status: Optional[Mapping[str, Any]] = None,
    ):
        if not compatibility.compatible:
            raise RAGCompatibilityError("refusing to start with incompatible checkpoints")
        self.device = torch.device(device)
        self.base_model = base_model.to(self.device).eval()
        self.enhancer = enhancer.to(self.device).eval()  # disables trained dropout
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.compatibility = compatibility
        self.artifact_status = dict(artifact_status or {})
        self.pc_vectors = None
        self.pc_meta = list(pc_meta or [])
        if pc_vectors is not None:
            vectors = np.asarray(pc_vectors, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(self.pc_meta):
                raise RAGCompatibilityError("PC vectors and metadata rows do not align")
            expected_dim = int(self.base_model.text_projection.shape[1])
            if vectors.shape[1] != expected_dim:
                raise RAGCompatibilityError(
                    f"PC/text projection dimension mismatch: {vectors.shape[1]} != {expected_dim}"
                )
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(vectors).all() or np.any(norms <= 0):
                raise RAGCompatibilityError("PC vectors contain non-finite or zero rows")
            self.pc_vectors = vectors / norms

    @property
    def ready(self) -> bool:
        return self.compatibility.compatible

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "modes": ["vanilla", "rag"],
            "compatibility": self.compatibility.to_dict(),
            "vector_rows": 0 if self.pc_vectors is None else int(self.pc_vectors.shape[0]),
            **self.artifact_status,
        }

    def _base_features(self, tokens: torch.Tensor) -> torch.Tensor:
        model = self.base_model
        x = model.token_embedding(tokens)
        x = x + model.positional_embedding
        x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = model.ln_final(x)
        rows = torch.arange(x.shape[0], device=x.device)
        return x[rows, tokens.argmax(dim=-1)]

    @torch.no_grad()
    def encode(self, query: str, mode: str = "rag", rag_top_k: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if mode not in {"vanilla", "rag"}:
            raise ValueError("mode must be 'vanilla' or 'rag'")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        tokens, token_diag = tokenize_preserving_eot(self.tokenizer, [query])
        tokens = tokens.to(self.device)
        base = self._base_features(tokens)
        vanilla = F.normalize(base @ self.base_model.text_projection, dim=-1)
        documents: List[RetrievedDocument] = []
        embedding = vanilla
        if mode == "rag":
            documents = self.retriever.retrieve([query], top_k=rag_top_k)[0]
            if documents:
                doc_tokens, _ = tokenize_preserving_eot(
                    self.tokenizer, [doc.text for doc in documents]
                )
                doc_features = self._base_features(doc_tokens.to(self.device)).unsqueeze(0)
                padding = torch.zeros(
                    1, len(documents), dtype=torch.bool, device=self.device
                )
                fused = self.enhancer(base, doc_features, padding)
                embedding = F.normalize(fused @ self.base_model.text_projection, dim=-1)
        vector = embedding[0].detach().cpu().numpy().astype(np.float32)
        vanilla_vector = vanilla[0].detach().cpu().numpy().astype(np.float32)
        diagnostics = {
            "mode": mode,
            "tokenization": token_diag[0],
            "retrieved_documents": [doc.to_dict() for doc in documents],
            "rag_top_k_returned": len(documents),
            "rag_vs_vanilla_cosine": float(np.dot(vector, vanilla_vector)),
            "rag_vs_vanilla_l2": float(np.linalg.norm(vector - vanilla_vector)),
        }
        return vector, diagnostics

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        category: Optional[str] = None,
        mode: str = "rag",
        rag_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.pc_vectors is None:
            raise RAGCompatibilityError("search requires PC vectors and metadata")
        vector, diagnostics = self.encode(query, mode=mode, rag_top_k=rag_top_k)
        candidates = np.arange(len(self.pc_meta))
        if category:
            wanted = category.casefold()
            candidates = np.asarray(
                [
                    i
                    for i, meta in enumerate(self.pc_meta)
                    if str(meta.get("category", "")).casefold() == wanted
                ],
                dtype=np.int64,
            )
        if len(candidates) == 0:
            return {"count": 0, "results": [], "diagnostics": diagnostics}
        k = min(max(int(top_k), 1), len(candidates))
        scores = self.pc_vectors[candidates] @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        results: List[Dict[str, Any]] = []
        for rank, local_index in enumerate(order, start=1):
            row = int(candidates[local_index])
            meta = dict(self.pc_meta[row])
            results.append(
                {
                    "rank": rank,
                    "row_index": row,
                    "id": meta.get("id"),
                    "ulip_similarity": float(scores[local_index]),
                    "category": meta.get("category"),
                    "caption_en": meta.get("caption"),
                    "preview_image": meta.get("preview_image"),
                    "json_path": meta.get("json_path"),
                }
            )
        return {"count": len(results), "results": results, "diagnostics": diagnostics}


def _import_core_modules() -> Tuple[Any, Any]:
    core_path = str(CORE_DIR)
    if core_path not in sys.path:
        sys.path.insert(0, core_path)
    import models.ULIP_models as core_models  # type: ignore
    from utils.tokenizer import SimpleTokenizer  # type: ignore

    actual = Path(core_models.__file__).resolve()
    if CORE_DIR.resolve() not in actual.parents:
        raise RAGCompatibilityError(f"imported unexpected models package: {actual}")
    return core_models, SimpleTokenizer


def load_adapter(
    *,
    ikea_checkpoint: Path = DEFAULT_IKEA_CHECKPOINT,
    rag_checkpoint: Path = DEFAULT_RAG_CHECKPOINT,
    corpus_dir: Path = DEFAULT_RAG_CORPUS_DIR,
    corpus_profile: str = "legacy1095",
    vector_dir: Optional[Path] = DEFAULT_VECTOR_DIR,
    device: Optional[str] = None,
    rag_top_k: int = 5,
    base_model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[Any] = None,
) -> RAGServingAdapter:
    """Load and validate a production adapter; all mismatches fail fast."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    corpus_profile = str(corpus_profile).strip().lower()
    if corpus_profile not in SUPPORTED_CORPUS_PROFILES:
        raise RAGCompatibilityError(
            f"unsupported corpus_profile={corpus_profile!r}; "
            f"expected one of {SUPPORTED_CORPUS_PROFILES}"
        )
    ikea_checkpoint, rag_checkpoint = Path(ikea_checkpoint), Path(rag_checkpoint)
    corpus_dir = Path(corpus_dir)
    ikea_ckpt = _load_checkpoint(ikea_checkpoint)
    rag_ckpt = _load_checkpoint(rag_checkpoint)
    compatibility = validate_exact_base_prefixes(ikea_ckpt, rag_ckpt)
    core_models, tokenizer_cls = _import_core_modules()

    ikea_state = _state_dict(ikea_ckpt)
    if base_model is None:
        raw_args = copy.deepcopy(ikea_ckpt.get("args"))
        if isinstance(raw_args, Mapping):
            args = Namespace(**raw_args)
        else:
            args = raw_args
        model_name = _checkpoint_model_name(ikea_ckpt)
        if args is None or not model_name:
            raise RAGCompatibilityError("IKEA checkpoint is missing model args")
        setattr(args, "evaluate_3d", True)
        base_factory = (
            "ULIP_PointBERT" if model_name == "ULIP_PointBERT_RAG" else model_name
        )
        base_model = getattr(core_models, base_factory)(args=args)
        base_state = {
            key: value
            for key, value in ikea_state.items()
            if not key.startswith("rag_enhancer.")
        }
        base_model.load_state_dict(base_state, strict=True)
    else:
        validate_exact_base_prefixes(ikea_ckpt, base_model.state_dict())
    tokenizer = tokenizer or tokenizer_cls()

    width = int(base_model.text_projection.shape[0])
    enhancer = core_models.RAGEnhancer(embed_dim=width)
    rag_state = _state_dict(rag_ckpt)
    enhancer_state = {
        key[len("rag_enhancer.") :]: value
        for key, value in rag_state.items()
        if key.startswith("rag_enhancer.")
    }
    if not enhancer_state:
        raise RAGCompatibilityError("RAG checkpoint contains no rag_enhancer weights")
    enhancer.load_state_dict(enhancer_state, strict=True)

    corpus_path = corpus_dir / "rag_corpus.jsonl"
    if not corpus_path.is_file():
        raise RAGCompatibilityError(f"RAG corpus not found: {corpus_path}")
    with corpus_path.open("r", encoding="utf-8") as handle:
        corpus = [json.loads(line) for line in handle if line.strip()]
    core_retriever = core_models.RAGRetriever(
        corpus_dir=str(corpus_dir), top_k=rag_top_k, device=device
    )
    retriever = DiagnosticRetriever(core_retriever, corpus)

    index_path = corpus_dir / "corpus_index.faiss"
    corpus_sha256 = _sha256_file(corpus_path)
    index_sha256 = _sha256_file(index_path)
    ikea_checkpoint_sha256 = _sha256_file(ikea_checkpoint)
    rag_checkpoint_sha256 = _sha256_file(rag_checkpoint)
    profile_validation: Dict[str, Any]
    if corpus_profile == "legacy1095":
        validate_corpus_profile(
            LEGACY_1095_PROFILE,
            corpus_rows=len(corpus),
            faiss_rows=int(core_retriever.index.ntotal),
            faiss_dimension=int(core_retriever.index.d),
            corpus_sha256=corpus_sha256,
            index_sha256=index_sha256,
        )
        checkpoint_mismatches = {}
        for label, actual in (
            ("ikea_checkpoint_sha256", ikea_checkpoint_sha256),
            ("rag_checkpoint_sha256", rag_checkpoint_sha256),
        ):
            expected = LEGACY_1095_PROFILE[label]
            if actual != expected:
                checkpoint_mismatches[label] = {
                    "expected": expected,
                    "actual": actual,
                }
        if checkpoint_mismatches:
            raise RAGCompatibilityError(
                "legacy1095 checkpoint bundle drifted: "
                + json.dumps(checkpoint_mismatches, ensure_ascii=False)
            )
        profile_validation = {
            "name": "legacy1095",
            "immutable_bundle_validated": True,
        }
    else:
        profile_validation = validate_provenance_profile(
            ikea_ckpt,
            rag_ckpt,
            enhancer_checkpoint_sha256=rag_checkpoint_sha256,
            base_checkpoint_sha256=ikea_checkpoint_sha256,
            corpus_sha256=corpus_sha256,
            index_sha256=index_sha256,
        )

    if vector_dir is None:
        raise RAGCompatibilityError(
            f"corpus profile {corpus_profile!r} requires a lineage-bound vector_dir"
        )
    vector_dir = Path(vector_dir)
    vectors = np.load(vector_dir / "vectors_pc.npy").astype(np.float32)
    with (vector_dir / "meta_pc.jsonl").open("r", encoding="utf-8") as handle:
        meta = [json.loads(line) for line in handle if line.strip()]
    vector_artifacts = validate_vector_bundle(
        vector_dir,
        corpus_profile=corpus_profile,
        base_checkpoint_sha256=ikea_checkpoint_sha256,
        vectors=vectors,
        metadata_rows=len(meta),
        embedding_dimension=int(base_model.text_projection.shape[1]),
    )

    return RAGServingAdapter(
        base_model=base_model,
        tokenizer=tokenizer,
        enhancer=enhancer,
        retriever=retriever,
        compatibility=compatibility,
        device=device,
        pc_vectors=vectors,
        pc_meta=meta,
        artifact_status={
            "profile": corpus_profile,
            "profile_validation": profile_validation,
            "ikea_checkpoint": str(ikea_checkpoint),
            "rag_checkpoint": str(rag_checkpoint),
            "corpus_dir": str(corpus_dir),
            "corpus_rows": len(corpus),
            "faiss_rows": int(core_retriever.index.ntotal),
            "faiss_dimension": int(core_retriever.index.d),
            "artifacts": {
                "ikea_checkpoint": {
                    "path": str(ikea_checkpoint),
                    "sha256": ikea_checkpoint_sha256,
                },
                "rag_checkpoint": {
                    "path": str(rag_checkpoint),
                    "sha256": rag_checkpoint_sha256,
                },
                "corpus": {"path": str(corpus_path), "sha256": corpus_sha256},
                "faiss_index": {"path": str(index_path), "sha256": index_sha256},
                **vector_artifacts,
            },
        },
    )


__all__ = [
    "CompatibilityReport",
    "DiagnosticRetriever",
    "EXACT_BASE_PREFIXES",
    "LEGACY_1095_PROFILE",
    "PROVENANCE_SCHEMA",
    "RAGCompatibilityError",
    "RAGServingAdapter",
    "RetrievedDocument",
    "load_adapter",
    "tokenize_preserving_eot",
    "validate_corpus_profile",
    "validate_exact_base_prefixes",
    "validate_provenance_profile",
    "validate_vector_bundle",
]
