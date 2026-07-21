-- Reproducible projection of the reviewed snapshot used by the native report blocks.
-- Run from /home/kyzen/ULIP_RAG with:
-- sqlite3 ':memory:' ".read reports/p300_dataset_audit_20260716/audit_snapshot.sql"

SELECT
  json_extract(value, '$.free_gib') AS free_gib,
  json_extract(value, '$.used_ratio') AS used_ratio,
  json_extract(value, '$.baseline_items') AS baseline_items,
  json_extract(value, '$.abo_models') AS abo_models,
  json_extract(value, '$.ready_catalogs') AS ready_catalogs
FROM json_each(
  readfile('reports/p300_dataset_audit_20260716/artifact.json'),
  '$.snapshot.datasets.headline_metrics'
);

SELECT
  json_extract(value, '$.dataset') AS dataset,
  json_extract(value, '$.size_gib') AS size_gib,
  json_extract(value, '$.share_of_data') AS share_of_data
FROM json_each(
  readfile('reports/p300_dataset_audit_20260716/artifact.json'),
  '$.snapshot.datasets.top_level_sizes'
)
ORDER BY size_gib DESC;

SELECT
  json_extract(value, '$.asset') AS asset,
  json_extract(value, '$.size_gib') AS size_gib,
  json_extract(value, '$.disposition') AS disposition
FROM json_each(
  readfile('reports/p300_dataset_audit_20260716/artifact.json'),
  '$.snapshot.datasets.review_target_sizes'
)
ORDER BY size_gib DESC;

SELECT
  json_extract(value, '$.path') AS path,
  json_extract(value, '$.size_gib') AS size_gib,
  json_extract(value, '$.content') AS content,
  json_extract(value, '$.role') AS role,
  json_extract(value, '$.decision') AS decision,
  json_extract(value, '$.reason') AS reason
FROM json_each(
  readfile('reports/p300_dataset_audit_20260716/artifact.json'),
  '$.snapshot.datasets.dataset_disposition'
)
ORDER BY size_gib DESC;

SELECT
  json_extract(value, '$.priority') AS priority,
  json_extract(value, '$.dataset') AS dataset,
  json_extract(value, '$.issue') AS issue,
  json_extract(value, '$.impact') AS impact,
  json_extract(value, '$.required_action') AS required_action
FROM json_each(
  readfile('reports/p300_dataset_audit_20260716/artifact.json'),
  '$.snapshot.datasets.quality_issues'
)
ORDER BY priority ASC;

SELECT
  json_extract(value, '$.review_order') AS review_order,
  json_extract(value, '$.asset') AS asset,
  json_extract(value, '$.size_gib') AS size_gib,
  json_extract(value, '$.classification') AS classification,
  json_extract(value, '$.precondition') AS precondition
FROM json_each(
  readfile('reports/p300_dataset_audit_20260716/artifact.json'),
  '$.snapshot.datasets.storage_review'
)
ORDER BY review_order ASC;

SELECT
  json_extract(value, '$.stage_order') AS stage_order,
  json_extract(value, '$.data_asset') AS data_asset,
  json_extract(value, '$.required_fields') AS required_fields,
  json_extract(value, '$.purpose') AS purpose,
  json_extract(value, '$.current_state') AS current_state
FROM json_each(
  readfile('reports/p300_dataset_audit_20260716/artifact.json'),
  '$.snapshot.datasets.missing_data'
)
ORDER BY stage_order ASC;
