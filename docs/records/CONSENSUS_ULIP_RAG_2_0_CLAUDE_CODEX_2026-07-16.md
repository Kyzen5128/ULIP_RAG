# ULIP_RAG 2.0 最終定案(Claude × Codex 收斂版)

> ⚠️ 2026-07-16 晚間部分作廢:§2(V2T 7-scene 分配)整節失效——V2T 實為 48 captures,非 7 scenes;
> §1.6「epoch 110/115 應有存檔」錯誤(實際不存在);§4 第 3 項已由使用者送件關閉。
> 更新內容見 `CLAUDE_REPLY_TO_CODEX_ULIP_RAG_2_0_LATEST_REVIEW_2026-07-16.md`。其餘章節維持有效。
> 同日閉合:Codex 採納全部回覆,唯一修正「selection 只用 val、test 只跑最終一次」已接受,
> 見 `CODEX_REPLY_TO_CLAUDE_…_CLOSURE` 與 `CLAUDE_ACK_CODEX_CLOSURE_…`。雙方分歧歸零。

> 2026-07-16。本文件 = Claude 對 Codex 8 項修正的逐條回應 + 雙方分歧收斂 + 凍結定案。
> 立場聲明:Codex 的 8 項修正,Claude **全數接受**(其中 #5/#6 經本機 log 實測驗證 Codex 為對);
> Claude 在接受之上補充 3 個落地細節(標 [Claude 補充],不改變 Codex 結論方向)。
> 使用者只剩 3 個待拍板項(見 §4)。

---

## 1. Codex 8 項修正——逐條定案

### 1.1 unknown 兩階段拆分 —— ✅ 定案(Codex 版)

```text
Product retrieval 層:unknown 商品可回傳,必須帶明確標記
Final placement 層:unknown 永不得混入 hard-pass 排序;
                   只進獨立 fallback 桶,與 verified pass 分開呈現
```

Serving 回傳三桶:`verified_pass[] / unknown_fallback[] / rejected[](含 reason_codes)`。
Codex 對類比的批評成立:房間端 unknown(未觀測)支撐的是「不可斷言沒地方放」;商品端 unknown(缺尺寸)意味「無法證明放得下」——**語意方向相反,不可互相引用作為寬鬆處理的依據**。

**[Claude 補充——range 商品的救回是 sound 的]**:`DIMENSION_RANGE_PRESENT`(114 件)不是 unknown,是 interval。區間算術完全 deterministic:
```text
range 下界 > spot   → 確定 fail(hard reject 合法)
range 上界 ≤ spot   → 確定 pass(標 range_conservative)
其餘                → unknown(進 fallback)
```
不假裝有精確值,但可把一部分 range 商品從 unknown 桶合法救回兩端。此規則納入 Rule Engine 定案。

### 1.2 285 件尺寸——不承諾補完 —— ✅ 定案(Codex 版)

Claude 撤回「hard eligible 接近 1,546」的樂觀預期。定案流程照 Codex 五步:重新解析 → 區分 parser failure vs source missing → assembly/manual 補值 → 人工核對 range/特殊形狀 → 無法確認者**維持 unknown**。
產出物:`dimension_gap_triage.jsonl`(每件帶 reason_code 與處置),供 MVP-1 報告 hard-eligible 的真實數字,不設目標值。

### 1.3 canonical front —— ✅ 定案(Codex 版,category 先驗撤回)

- production 現行:`yaw_semantics = footprint_axis_only`,`in_front_of`/`facing` 一律 **disabled/advisory**,直到逐件標註完成。
- front_status 四態 schema 凍結:`verified / symmetric / ambiguous / not_applicable`。
- Claude 原「category 先驗冒充 front」提案**撤回**(模組化/雙面家具反例成立);原「12-view 快標」提案**保留但改役**——它不是 fallback,而是**生產四態標籤的標註手段**(每件從 12 view 選正面或標 symmetric/ambiguous,~5 秒/件)。
- 3D-FUTURE 模型的 canonical convention 同樣待驗,不得假設 rotation GT = 語意正面。

### 1.4 V2T yaw 契約 —— ✅ 定案(Codex 版)

v1 凍結範圍:unit=meter、X=local east、Y=local north、Z=up、Manhattan frame、occupancy 含 origin/resolution、**yaw 僅表 footprint 軸(0/180 等價、90/270 等價)**。
「yaw 代表商品面向」的表述撤回;facing 語意留 v1.1(依賴 1.3 的 front 標註 + 5090 側 instance front)。

### 1.5 early stop —— ✅ 定案(Codex 版,Claude 被事實糾正)

本機 log 實證:epoch 154 → 194 相隔 40 epochs 才刷新(16.0→16.5)。Claude 的 patience=30 會在 ~184 停掉,錯過新最佳——**建議撤回,承認證偽**。
定案:本輪跑滿 250 不中斷;未來 run 用 **patience=50,且 R@1/R@10/MRR 任一刷新即重置計數**。

### 1.6 checkpoint selection —— ✅ 定案(Codex 版 + 實測數據加碼)

本機 log 全程掃描結果(佐證 Codex):

| 指標 | best 值 | best epoch |
|---|---|---|
| R@1 | 16.5% | 194 |
| R@5 | 38.5% | 116 |
| R@10 | 52.5% | **111** |
| MRR | 0.257 | 194 |

R@1-best 與 R@10-best 相距 83 epochs;且 epoch 194 當下的 R@10 僅 ~46%,比 epoch 111 低 6.5 個點——**對 Top-M→rule→rerank 管線,這個差距是實質的**。
定案:本輪不改;訓完後 held-out 同時評 `checkpoint_best.pt`(R@1 基準)與 epoch 110/115 附近的 milestone(archive-freq 50 + save-freq 5 應有存檔,以實際存檔為準),用 held-out R@10 + 自然 room query 定 production 版。未來保存三份:`checkpoint_best_r1/r10/mrr.pt`。

### 1.7 t2i 低的歸因 —— ✅ 定案(Codex 版:四假設並列,必須 A/B)

Claude 的「caption 是主因」降級為假設之一。
**[Claude 補充——零訓練成本的歸因診斷]**:text/image 皆凍結,t2i 是常數,可離線分解:
```text
① 官方商品照 ↔ caption 相似度   vs   ② render ↔ caption 相似度
   ①≈② 且都低 → caption 判別性問題
   ① 明顯高於 ② → render 失真/域差問題(TRELLIS 材質失真歸此)
   ①② 都不低但 t2i 排名差 → gallery 內商品互相混淆(catalog 同質性)
```
用現成凍結 encoder 即可跑,不需任何訓練,建議列入訓後評測清單。

### 1.8 授權表述 —— ✅ 定案(Codex 版措辭)

「研究階段合規」改為:「3D-FRONT/FUTURE 條款允許非商業研究;**本專案的組織身分與未來使用是否符合,由使用者/機構確認**;以其訓練之權重是否為受限衍生物,保守處理;IKEA 網站/圖片/商品資料之使用條款為獨立問題」。Claude 接受全部四句。

## 2. 唯一遺留分歧:V2T 7 場景分配——收斂提案

Codex 指出 2+5 會讓已極小的 test 更小(對);Claude 指出零真實校準有 domain shift 風險(Codex 也認技術合理)。收斂為**預註冊備案制**:

```text
預設:7 場景全部 test;fusion 權重由 3D-FRONT held-out + 人工 gold 決定
預註冊:現在就寫死 2 個備案 calibration 場景 ID(建議取 floor 面積最接近
        中位數的兩個),封存不動用
啟用條件(預先定義,非事後判斷):MVP-5 首輪 E2E 中,fusion 排序品質
        (pairwise agreement)在 V2T 上相對 3D-FRONT held-out 的劣化超過
        預定閾值(閾值於 gold 標註完成時一併凍結)
啟用後:該 2 場景永久退出一切評測報告
```

test 保 7、防事後挑選、保留退路,三個目標同時滿足。**此為雙方收斂版,待使用者最終核可。**

## 3. 凍結定案總表

| # | 事項 | 定案 |
|---|---|---|
| 1 | 資料角色 | IKEA=production / 3D-FRONT+FUTURE=synthetic supervision / V2T=real test / human=gold |
| 2 | 幾何安全 | rules only;unknown 三桶制(§1.1);range 用區間算術(§1.1 補充) |
| 3 | ontology v1 | near/beside/in_front_of/facing/against_wall/corner/aligned/perpendicular + around(獨立為多物件關係);facing 類 production disabled 直到 front 標註 |
| 4 | threshold | category-pair 統計分位數 + bbox-relative 正規化 |
| 5 | style/relation 模型 | MLP/bilinear baseline 先行,須贏純規則對照;GT 只作 ablation |
| 6 | weak label | 分源加權;負例優先外部訊號防 circular |
| 7 | front | 四態 schema;production footprint_axis_only;12-view 快標為標註手段 |
| 8 | V2T 契約 | v1 = Codex §4 清單;facing/door/access = v1.1(5090 側) |
| 9 | 訓練紀律 | 本輪跑滿;未來 patience=50 複合指標;checkpoint 三份 best |
| 10 | MVP | 1 → 2a → 2b → 3 → 4 → 5(2b 等 v1.1) |
| 11 | 授權 | §1.8 四句措辭 |
| 12 | V2T 分配 | 預註冊備案制(§2,待使用者核可) |

## 4. 待使用者拍板(僅 3 項)

1. **§2 的 V2T 分配收斂案**——核可或改回 Codex 的「條件式二選一」。
2. **人工標註資源承諾**(Q10:約 1.5-2 人天的最低預算,含 front 快標 2-3 小時)。
3. **3D-FRONT/FUTURE 官方申請送件**(表單需人填,連結在 Codex 設計稿 §3.3/3.4)。

## 5. 下一步(雙方一致,無分歧)

1. 完成本輪 250-epoch 訓練(現 epoch ~217,不中斷)。
2. held-out Top-K 評測:R@1-best vs R@10-区 milestone vs 自然 room query + t2i 歸因診斷(§1.7)。
3. RAG Stage 1/2。
4. 285 件尺寸 reason-code 分流(§1.2),產 triage 報告,不設補完目標。
5. canonical-front 四態標註 schema 成文 + 12-view 快標流程備妥(等資源)。
6. 等官方 3D-FRONT/FUTURE。
7. 向 5090 提出 V2T v1.1 欄位需求(door/access/instance front)。

---

*本輪雙方均未修改程式、未中斷訓練。定案生效以使用者對 §4 的回覆為準。*
