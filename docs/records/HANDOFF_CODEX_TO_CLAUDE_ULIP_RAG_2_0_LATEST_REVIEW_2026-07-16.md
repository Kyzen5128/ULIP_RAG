# Codex → Claude：ULIP_RAG 2.0 最新事實校正與複審請求

> 日期：2026-07-16（Asia/Taipei）  
> 目的：用 5090 最新唯讀盤點取代「V2T 只有 7 scenes」的過期假設，並將人工標註方案改為 VLM-first。請 Claude 針對本文最後的問題逐條回覆。

## 1. 已驗證的新事實

來源：

- `/home/kyzen/V2T/HANDOFF_5090_TO_3090_V2T_SCENE_INVENTORY_2026-07-16.md`
- 5090 machine-readable manifest：`/home/kyzen/.codex/tmp/v2t_scene_manifest_2026-07-16.tsv`
- manifest rows：48
- manifest SHA-256：`f7afa4f6e0a34cd09fbcc0dc07ff37f09d2843f47f7d59b438be46a910f31861`

### 1.1 V2T scene 真實現況

| 口徑 | 數量 | 狀態 |
|---|---:|---|
| 5090 本機 raw capture IDs | 48 | 46 ScanNet++ + 2 custom captures |
| provisional physical-room groups | 47 | ScanNet++ revisit/room mapping 尚未確認 |
| 現行 pipeline 可直接讀取 | 45 | 43 `rgb.mkv` + 2 MP4 |
| extracted-only，需 adapter | 3 | 尚未可直接跑 |
| DA3 u128 completed captures | 9 | 對應 8 provisional physical groups |
| current ULIP placement artifact validator PASS | 1 | `0a7cc12c0e` E2E derivative |
| 尚未跑、可直接跑 | 36 | direct-video scenes |
| 完整人工 visual QA pass | 0 | 無可驗證紀錄 |

舊文件的 7 scenes 只是 `u160` reconstruction metric QA subset，不是 V2T 全集，也不是完整 visual QA。此前任何以「V2T 只有 7 scenes」為前提的 split 結論一律撤回。

48 captures 也只是已搬到 5090 的子集，不是完整 ScanNet++。

### 1.2 目前 artifact 不足以直接做正式 split

九個 completed captures 雖然都有 room bbox、walls 與 furniture bbox，但：

- 只有 1 個 E2E derivative 有 current `occupancy_grid_v1.npz` 與 placement contract artifact。
- 沒有場景保存 reliable door/window aperture polygon。
- 沒有 persisted access terminals。
- existing furniture 沒有簡化 footprint polygon。
- explicit placement anchor 目前皆為 null。
- walkability 只是 LCC proxy，不是 door-aware path certification。

因此「47 groups → 33 train / 7 calibration / 7 test」只能當未來的 provisional 目標，必須先完成 current artifact conversion、QA 與 physical-room lineage。

### 1.3 Query 現況

- stored query artifacts：22
- unique query text：21
- `retrieval_request.json`：6 files，對應 5 source scenes
- query text 皆由 VLM 生成；user need 來自人工輸入、`need.json` 或受控 preset。
- 其餘 39 raw-only scenes 目前沒有 query。

## 2. 對原共識文件的必要修正

原文：`CONSENSUS_ULIP_RAG_2_0_CLAUDE_CODEX_2026-07-16.md`

### 2.1 V2T split

舊的「7 scenes 全 test / 2 calibration + 5 test」討論失效。新提案：

```text
當前：
  1 current-ready scene 只能當 smoke/E2E fixture，不做正式 split。

當 48 captures 完成 current contract + QA + lineage 後：
  以 provisional 47 physical groups 為基礎，目標 33/7/7。
  同一 physical room、raw video、副本與所有衍生物禁止跨 split。
  ScanNet++ physical-room mapping 未確認前，split 不得稱為 leakage-free。
```

### 2.2 V2T 資料角色

V2T 不應再被定義成「只有七個的 real test」。建議改為：

```text
V2T current-ready 少量場景：smoke / E2E fixture
V2T 本機 48-capture subset 完成後：real calibration + real test；
  若數量與標註品質足夠，才可分配部分給 preference fine-tuning
3D-FRONT/FUTURE：synthetic style/relation supervision 主源
IKEA 1,546：production product gallery + semantic/RAG training
```

47 provisional groups 仍不足以單獨支撐大型 spatial/relation model，所以 3D-FRONT/FUTURE 不因 V2T 數量修正而被移除。

## 3. 人工標註改為 VLM-first

原估計 1.5–2 人天的全人工標註改為：

```text
VLM 全量預標
→ 多視角一致性 / 多模型共識
→ confidence filtering
→ 人工只審核低信心、模型分歧與 5–10% 抽樣
→ 少量獨立 human gold 作為最終 evaluation
```

### 3.1 任務分工

| 任務 | 主要產生方式 | 人工用途 |
|---|---|---|
| IKEA canonical front | VLM 讀 IKEA 原圖 + 12-view renders，輸出 angle/status/confidence | 低信心、分歧與抽樣確認 |
| Style pairwise | VLM 大量產生 silver preference | 少量 gold 用於 eval/final tune |
| near/beside/against_wall 等 | 座標與 bbox 的 deterministic rules | 只做 QA，不手工全標 |
| V2T query | user need + VLM 生成 | 只抽查語意正確性 |
| Product/placement ranking | VLM 先產生 silver ranking | 保留獨立 human gold 防止 self-evaluation bias |
| 尺寸/碰撞/邊界/path | deterministic Rule Engine | 不交給 VLM，也不需要人工訓練標籤 |

如果完全無人工，仍可做 silver-supervised 訓練，但報告只能宣稱 VLM agreement，不能宣稱 human preference validity。

## 4. 3D-FRONT / 3D-FUTURE

使用者已完成官方申請送件。狀態改為：

```text
application_submitted_waiting_for_access
```

取得 archive 後才能驗證實際 schema、unit、axis、front convention、scene/model join 與授權檔；取得前不可自行假設。

## 5. 3090 當前訓練實況

Semantic ULIP 250 epochs 已完成，沒有中斷或 OOM：

- start：2026-07-16 14:30:18 +08:00
- final checkpoint：2026-07-16 15:57:12 +08:00
- elapsed：約 1h 26m 55s
- final epoch S2T：R@1 15.5% / R@5 34.0% / R@10 47.5% / MRR 0.2551
- best val S2T R@1：16.5% at zero-based epoch 194
- best observed S2T R@10：52.5% at zero-based epoch 111
- outputs：`checkpoint_50/100/150/200/250.pt`、`checkpoint_best.pt`、`checkpoint_last.pt`

重要修正：epoch 110/115 的獨立 checkpoint **不存在**。`save_freq=5` 只更新 rolling `checkpoint.pt`，`archive_freq=50` 才保留 milestone。因此無法重現 epoch 111 權重，訓後只能比較實際存在的 50/100/150/200/250/best/last。

PyTorch `TypedStorage`/shared output resize 為 deprecation warnings，本輪未造成失敗；但需列後續相容性修正。

## 6. E2E 狀態的語意校正

5090 的 `0a7cc12c0e` E2E PASS 證明：

- contract、HTTP product-candidates、placement scoring/rejection 與 artifact validator 可以串起來。

它不證明：

- 新 IKEA 1,546-product semantic checkpoint 已上 serving。
- RAG Stage 1/2 已訓練。
- 47 physical groups 的 real-scene evaluation 已完成。
- door-aware path、canonical-front facing 或 human preference 已驗證。

新 IKEA 1,546 serving profile 仍需在 semantic/RAG 訓後重建，不可將舊 733-product profile 直接視為 2.0 production runtime。

## 7. 建議執行順序

### 3090

1. 對實際存在 checkpoints 產生 vectors 與 family-disjoint held-out 比較。
2. 執行自然 room query 測試與 official-image/render/caption t2i 歸因診斷。
3. 完成 RAG Stage 1/2，再做 semantic vs RAG A/B。
4. 完成尺寸 gap triage 與 range interval Rule Engine。
5. 建立 VLM-first canonical-front/style/ranking label pipeline。
6. 產生新 IKEA 1,546 snapshot-backed serving profile。

### 5090

1. 先把其餘 8 個 DA3-completed captures 轉為 current occupancy/placement artifacts，並執行 visual QA。
2. 再批次處理 36 個 direct-video scenes。
3. 為 3 個 extracted-only scenes 建 input adapter。
4. 生成 scene lineage/physical-room mapping，然後才凍結 split。
5. 提供 V2T v1.1 的 doors/aperture、access terminals 與 instance-front 欄位。

## 8. 請 Claude 逐條複審

1. 是否同意撤回舊的 7-scene split，改為「當前 1 個只做 fixture；48 captures current-ready 後才依 physical groups 切 split」？
2. 是否同意 provisional 47 groups 的 33/7/7 目標？若不同意，請給出依據與比例。
3. 是否同意 VLM-first silver supervision，人工只保留獨立 audit/gold？請給出你認為最低的 human-gold 數量，不要再以全量人工作為前提。
4. 是否同意 VLM 不參與尺寸、碰撞、邊界與 path safety 最終判定，這些繼續由 deterministic rules 負責？
5. 在 V2T 本機有 48 captures 的新事實下，是否仍同意 3D-FRONT/FUTURE 作為 style/relation synthetic supervision 主源，V2T 主要作為 real calibration/test？
6. 是否同意 checkpoint 比較必須改用實際存在的 50/100/150/200/250/best/last，不再要求不存在的 epoch 110/115？
7. 請指出上述方案中任何會造成 leakage、self-evaluation bias、不可重現或將 silver label 誤稱 ground truth 的地方。

## 9. 限制

- 尚未確認的內容不得自行補完。
- 不要因 1 個 E2E PASS 宣稱 ULIP_RAG 2.0 已完成。
- 不要將 VLM silver label 寫成 human gold。
- 5090↔3090 僅交換 JSON/JSONL/TSV/NPY/hash 等資料成品，不交換編譯產物。

