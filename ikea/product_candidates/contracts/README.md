# Product Candidates v1 contracts

The request and response schemas in this directory are byte-for-byte mirrors
of the 5090-owned authoritative contracts at:

```text
/home/kyzen/V2T/HANDOFF_5090_TO_3090_PLACEMENT_LOOP_2026-07-15/contracts/
```

They must not be edited independently. Verify them with `cmp` before release.
The readiness schema is 3090-owned and implements section 9 of
`HANDOFF_5090_TO_3090_PRODUCT_CANDIDATES_DECISIONS_AND_EXEC_2026-07-15.md`.

Schema SHA-256 values verified on 2026-07-15:

```text
c37bf16ea4238d7c8ac0b6391abda2cce320d3de8bcf81cba705a62695fd2e29  ulip_product_candidate_request_v1.0.schema.json
317bbe683b399724599c98ed138b85a82e417cdf081a047dbc7b137f2f3d412b  ulip_product_candidate_response_v1.0.schema.json
81c899ef0946bb31ec9e94799904f8b04bd1cc7692740fa684b54393e398cc99  ulip_product_candidates_readyz_v1.0.schema.json
```

## Important runtime rules beyond the authoritative JSON schemas

- Product Candidates v1 emits top-level `ok` or `no_match`; it does not emit
  the response schema's reserved `partial` value.
- JSON Schema `uniqueItems` is case-sensitive. The service must additionally
  trim/casefold category terms and apply the decision handoff's duplicate/tier
  policy.
- `catalog_metadata` is permitted by the authoritative response schema but v1
  omits it, so local paths or unrestricted metadata cannot leak.
- Successful tokenization has at most 75 content BPE tokens, preserves SOT/EOT,
  reports `truncated=false`, and rejects 76+ content tokens with HTTP 422.
- Ready responses require a non-null deployment profile/provenance, every
  fixed check true, and no failures. Not-ready responses require at least one
  false check and at least one uppercase machine failure code.
- `GET /readyz` returns HTTP 200 for `ready`, HTTP 503 for `not_ready`, and must
  include `Cache-Control: no-store`. HTTP status and headers are not encoded by
  JSON Schema.

## Integrity commands

From `/home/kyzen/ULIP_RAG`:

```bash
cmp --silent \
  ikea/product_candidates/contracts/ulip_product_candidate_request_v1.0.schema.json \
  /home/kyzen/V2T/HANDOFF_5090_TO_3090_PLACEMENT_LOOP_2026-07-15/contracts/ulip_product_candidate_request_v1.0.schema.json

cmp --silent \
  ikea/product_candidates/contracts/ulip_product_candidate_response_v1.0.schema.json \
  /home/kyzen/V2T/HANDOFF_5090_TO_3090_PLACEMENT_LOOP_2026-07-15/contracts/ulip_product_candidate_response_v1.0.schema.json

sha256sum ikea/product_candidates/contracts/*.schema.json
```

The three hashes identify schema files only. They are not substitutes for the
five aggregate deployment-bundle hashes required in response provenance.
