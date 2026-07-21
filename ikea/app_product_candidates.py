"""Production ASGI entry point for the isolated Product Candidates v1 service.

Importing this module never loads CUDA.  The immutable profile and vanilla ULIP
encoder are loaded fail-closed during FastAPI lifespan startup.
"""
from __future__ import annotations

import os
from pathlib import Path

from product_candidates.runtime import ProductCandidateRuntime
from product_candidates.service import create_app


def load_runtime_from_env() -> ProductCandidateRuntime:
    profile_value = os.environ.get("PRODUCT_CANDIDATES_PROFILE", "").strip()
    if not profile_value:
        raise RuntimeError("PRODUCT_CANDIDATES_PROFILE is required")
    profile_path = Path(profile_value).expanduser().resolve()
    if not profile_path.is_dir():
        raise RuntimeError("PRODUCT_CANDIDATES_PROFILE does not exist")
    # Imported only at startup so unit tests and health checks do not load assets.
    if (profile_path / "deployment_profile.json").is_file():
        import json

        profile_header = json.loads(
            (profile_path / "deployment_profile.json").read_text(encoding="utf-8")
        )
    else:
        raise RuntimeError("deployment_profile.json is missing")
    if profile_header.get("schema_version") == "ulip-product-candidates-deployment/2.0":
        from product_candidates.snapshot_v2 import (
            load_snapshot_deployment as load_deployment_profile,
        )
    else:
        from product_candidates.bundles import load_deployment_profile

    profile = load_deployment_profile(profile_path)
    expected_identity = (
        profile.deployment_profile_id,
        dict(profile.provenance),
        dict(profile.inventory),
        profile.tokenizer_id,
    )

    def revalidate_profile() -> bool:
        current = load_deployment_profile(profile_path)
        current_identity = (
            current.deployment_profile_id,
            dict(current.provenance),
            dict(current.inventory),
            current.tokenizer_id,
        )
        if current_identity != expected_identity:
            raise RuntimeError("deployment profile identity drifted after startup")
        return True

    device = os.environ.get("PRODUCT_CANDIDATES_DEVICE", "cuda").strip().lower()
    if device not in {"cuda", "cpu"}:
        raise RuntimeError("PRODUCT_CANDIDATES_DEVICE must be cuda or cpu")
    runtime = ProductCandidateRuntime.from_profile(
        profile,
        device=device,
        integrity_revalidator=revalidate_profile,
    )
    # Close the verification-to-model-load window before the runtime can be
    # published to the ASGI state.  Subsequent /readyz probes repeat this check.
    if not runtime.revalidate_integrity():
        raise RuntimeError("deployment profile drifted during runtime initialization")
    return runtime


app = create_app(
    runtime_loader=load_runtime_from_env,
    bearer_token=os.environ.get("PRODUCT_CANDIDATES_API_TOKEN"),
    require_strong_token=True,
    hard_timeout_s=25.0,
    max_queue=1,
)
