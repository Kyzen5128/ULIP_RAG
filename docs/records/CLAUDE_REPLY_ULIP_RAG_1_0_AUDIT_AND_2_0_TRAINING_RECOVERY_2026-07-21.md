# Claude:ULIP_RAG 1.0 論文 Audit 與 2.0 訓練補救方案

> 日期:2026-07-21(Asia/Taipei)
> 回覆對象:`HANDOFF_CODEX_TO_CLAUDE_REVIEW_ULIP_RAG_1_0_AND_FIX_2_0_TRAINING_2026-07-21.md`
> 範圍:全程唯讀。未改 DOCX/Mongo/vectors/checkpoint/serving。所有無 artifact 支撐的敘述標「尚未確認」。
> 論文來源:`/home/kyzen/跨模態增強模型嵌入與檢索架構_完稿.docx`(本機抽文 252,101 字元 / 769 段,英文題名 CAMERA framework)。

> **v2 修訂(2026-07-21,依 Codex 複審)**:更正兩處證據/定位錯誤——(A) §3 原稱 Table 4 五值「log 零命中」錯誤,係我上輪 grep 漏掉 `4090_cheng_archive_20260715/` archive;實查五值**均命中舊 log**,且以 classification `test_acc1/test_acc5` 形式出現、與同 log 真 retrieval metric 並存(證據反而更強)。(B) §4 原稱「1.0 主模型 = IKEA733」證據不足,撤回。另調整 Furn3D 措辭、hard-negative 定義、test 使用次數(§1/§7/§8)。以下正文已就地更正。

## 0. 一句話結論

**論文 CAMERA/ULIP_RAG 1.0 的核心賣點資料集 `Furn3D-Ecom`(30K 模型 + 160K corpus + WordNet/ConceptNet)在本機及留存 archive 中未找到、無法驗證亦無法重現(UNVERIFIED / NOT PRESENT);Table 4 的檢索數值精確等於 `整數 / 2468 × 100`(2468 = ModelNet40 test 集大小,泛物件非家具),且這些值確實存在於 4090 archive 的舊 log,但原始欄位是 ModelNet40 classification 的 `test_acc1/test_acc5`——與同一份 log 內真正的 retrieval metric(`retr/s2t_r1` ~0.0003–0.001,差三個數量級)完全不同。判定:Table 4 高度疑似把 classification accuracy 重標為 retrieval RR/NDCG,非造假指控。論文的正式 1.0 主模型 lineage 目前無法由現存 artifact 確立(IKEA733 只是唯一確認投入舊 serving 者,不等於論文完整 1.0)。2.0 的低分 root cause 是 text↔shape 對齊弱(frozen CLIP 文字塔 + 模板化短 caption),不是 point encoder,也不是 RAG 能補的。**

---

## 1. 1.0 論文逐項 audit 結論

| 論文宣稱(§抽文) | 本機實測 | 判定 |
|---|---|---|
| Furn3D-Ecom 約 3 萬件家具 3D 模型(L148) | 無 `furn3d`/`ecom` 目錄;最接近=Objaverse 家具 8,632 UIDs 或 IKEA 733 | **UNVERIFIED / NOT PRESENT**(留存 archive 未找到,無法證明歷史上是否存在,不主張「誇大」) |
| 約 16 萬筆 LLaMA-3.2-3B corpus(L148) | RAG corpus 實體 = **1,095 rows**;`build_rag_corpus.py` 用 Wikipedia+spaCy+模板,**無 LLaMA** | **NOT PRESENT**(實體僅 1,095;160K 版未找到) |
| WordNet 3.0 + ConceptNet | 全 repo 無 wordnet/conceptnet import 或 API 呼叫 | **未實作** |
| 16 家具類別 | corpus 實際 17 類 | 近似,不精確 |
| Dual BERT-base 768D 兩塔 retriever / 2-layer Cross Fusion 8-head / FAISS IVFFlat | 論文方法章有描述;但**實際投入 serving 的 IKEA 模型是 ULIP PointBERT + 凍結 SLIP,非 BERT 雙塔**(見 §4 lineage) | **論文架構 ≠ 落地架構** |
| RR = Recall Rate(§4.2.1 L424-431 定義為標準 top-k recall) | 定義本身正確 | 定義 OK,數值來源見 §3 |

論文把「設計目標」寫成「已完成成果」:`build_rag_corpus.py` 的 `--categories chair table … × 20 iterations × 5 templates`(理論可產百萬)被包裝為 Furn3D-Ecom 16 萬 corpus,實跑只落地 1,095 條。

## 2. 論文 claim ↔ 實際 artifact 對照表

| 面向 | 論文 claim | 實際 artifact(路徑) | 差異倍率 |
|---|---|---|---|
| 家具 3D 模型 | ~30,000 | Objaverse `furniture_uids` 8,632 / IKEA 733 | 3.5×–41× |
| Query descriptions | ~30,000 LLaVA | `unified_data.jsonl` **88,434 rows / 7,370 unique obj**(本機實測 88,434 行) | rows 對得上,但每 obj ~12 caption,非 30K unique |
| Corpus | ~160,000 LLaMA | `rag_corpus`(1.0 用)**1,095**;CAMERA_3D `semantic_corpus.jsonl` 88,422 | 150× / 1.8× |
| 知識源 | WordNet+ConceptNet | Wikipedia+spaCy+template | 不同源 |
| CAMERA_3D 實訓量 | (隱含全量) | `train_two_stage.py` **L515-516「只取前 1000 筆」`Subset(full_ds, range(1000))`** | 實際只訓 1,000 |
| 檢索 gallery | Furn3D-Ecom 家具 | Table 4 分母 = 2,468 = **ModelNet40 test**(泛物件) | 領域不符 |

## 3. Table 4 數值可重現性判定 —— **不可重現**

論文 §4.3 附近(抽文 L466-495)敘述:
- Parts2Words:RR@1 **12.72**→13.65、RR@5 **32.98**→35.15
- ULIP-2+RAG:RR@1 **17.50**、RR@5 **42.99**

本機算術(已驗證):

```text
314 / 2468 × 100 = 12.7228525   → 論文 12.72
337 / 2468 × 100 = 13.6547812   → 論文 13.65
814 / 2468 × 100 = 32.9821718   → 論文 32.98
432 / 2468 × 100 = 17.5040519   → 論文 17.50
1061 / 2468 × 100 = 42.9902755  → 論文 42.99
```

`wc -l modelnet40_test.txt = 2468`(本機實測)。五個值全部精確 = 某整數 / 2468 × 100。

**【v2 更正】五值並非「零命中」——它們確實存在於 4090 archive 舊 log,且以 classification 欄位出現(此為更強證據)。** 我上輪只 grep 了 `CAMERA_3D` 與 `ULIP_RAG`,漏掉 `/mnt/P300/data/ULIP/4090_cheng_archive_20260715/`。本輪全域複查(Codex 指出的精確 file:line,已本機驗證):

| 論文值 | archive 來源(欄位) |
|---|---|
| 12.7228525 | `…/ULIP_RAG/…/RAG2_Stage1/log.txt` 的 `test_acc5`;`…/projects/ULIP/…/ULIP2_Instuct_c/log.txt` |
| 13.6547812 | 同 RAG2_Stage1 log 的 `test_acc5` |
| 32.9821718 | `…/projects/ULIP/…/ULIP1/log.txt` 的 `test_acc1` |
| 17.5040519 | `…/projects/ULIP/…/ULIP2_Instuct_custom/log.txt` 的 `test_acc1` |
| 42.9902755 | `…/projects/ULIP/…/ULIP2_Instuct/log.txt` 的 `test_acc1` |

**鐵證(同一份 log、同一 epoch 並存分類與檢索,本機實讀 `ULIP2_Instuct_c/log.txt`)**:

```text
"test_acc1": 8.954619...     ← ModelNet40 classification accuracy(論文 Table 4 取這一欄)
"test_acc5": 31.969205...
"retr/s2t_r1": 0.000334...   ← 同 epoch 真正 retrieval R@1 ≈ 0.03%
"retr/s2t_r5": 0.002204...
"retr/s2t_ndcg5": 0.001232...
```

真 retrieval metric 比 classification accuracy **低約三個數量級**;論文 Table 4 的「RR@1=12.72/17.50」對得上 `test_acc*`,對不上同 log 的 `retr/*`。

**判定(更正後,更強)**:Table 4 的數值**確實出現在舊 log,但原始欄位是 ModelNet40 classification 的 `test_acc1/test_acc5`,而非同一份 log 中真正的 retrieval RR/NDCG**。因此 Table 4 **高度疑似把 classification accuracy 重新標成 retrieval 指標**。標記 `SUSPECTED_METRIC_MIXING`(不作造假指控)。另仍缺 table 生成 script / 原始 CSV / Parts2Words baseline code,故完整重現鏈仍不齊;取證途徑:向原作者索取上述三項。取得前不得引用 Table 4 任何數字,亦不得以其相對提升百分比作結論。

## 4. 1.0 真實 dataset / count / split / metrics 最終判定

論文的「單一 CAMERA 實驗」實為**三條互不相同的 lineage 被混合敘述**(4090 lineage 稽核 + 本機複驗):

**Lineage A — IKEA 733(唯一投入 serving)**
- 733 products(json/glb 各 733、ply 734,本機實測)。
- loss = 雙向 InfoNCE(pc↔text + pc↔image),**無 RAG**。
- text/image encoder **凍結**(SLIP),只訓 point encoder。
- **無 held-out test;val == train 同批 733**;論文/summary 的 `best_acc1=76.53` = 6-way retrieval on 同批 733 = **train/val 洩漏數字**。

**Lineage B — ShapeNet55 / ModelNet40 研究線 RAG**
- ShapeNet train ~52,468;ModelNet40 test **2,468**;RAG corpus **1,095**。
- Table 4 的 /2468 分母來自此線的 ModelNet40。

**Lineage C — CAMERA_3D / Objaverse prototype**
- 8,632 furniture UIDs;`unified_data.jsonl` **88,434 rows**;但實訓 **只取前 1,000**。

**最終判定(v2 更正)**:
1. **論文正式 1.0 主模型 lineage 目前無法由現存 artifact 確立。** IKEA733 **只是唯一確認投入舊 IKEA serving 的模型**,不等於論文宣稱的 Furn3D-Ecom/CAMERA 完整實驗;A/B/C 是三條不同 lineage,**不應挑其中一條重新指定為論文 1.0 主模型**(我上輪指定 A 為主模型是證據不足,撤回)。使用者先前指出「IKEA733 應該是錯的」成立——不是件數錯,是「把舊 serving snapshot 當論文完整 1.0」的定位錯誤。
2. 30K/160K **在留存 archive 未找到**(NOT PRESENT / UNVERIFIED);論文結果沒有一份對應的 materialized 全集。
3. Table 4 值存在於舊 log 但為 classification 欄位(§3 更正);完整重現鏈仍不齊。
4. 可保留:ULIP 對比學習框架、pc↔text/pc↔image 雙向 InfoNCE、frozen-CLIP-align 範式、RAG 增強的**構想**。**須作廢/重寫**:Furn3D-Ecom 30K/160K 敘述、WordNet/ConceptNet 宣稱、Table 4 及其相對提升百分比、以 76.53 代表 1.0 效能、以「733 vs 1546」或「76.53 vs 17.91」作 1.0/2.0 比較。

## 5. 2.0 成績不佳 root-cause ranking(Text→PC R@1 17.9% / R@10 63.4%)

依證據強度排序(全部本機可佐證):

1. **【最高】text↔shape 對齊弱 = frozen CLIP 文字塔 + 模板化短 caption**。2.0 沿用 ULIP 範式凍結 text/image(前輪已驗 t2i/i2t 跨 epoch 常數 0.1616/0.1289 一位不差),只訓 point branch。而 catalog caption 是模板短句(實例:`"This is a Gaming easy chair in the Armchair category, offered in gray, black. Modern grey chair, black metal, adjustable."`)。1,546 件家具彼此語義相近,短模板 caption 在凍結的 CLIP 空間中判別性不足 → text 線天花板低。
2. **【高】image 線強、text 線弱的落差本身指向 caption/文字塔,而非 point encoder**。final test:i2p/p2i R@10 = 89.6/91.0%,而 t↔pc R@10 只 59.7–63.4%。point encoder 已能對齊視覺;瓶頸在文字。
3. **【中】訓練集小 + family-disjoint 硬 split**。train 1,212、val 200、test 134,family-disjoint held-out 天生比 non-disjoint 難;200-val best R@1 16.5、134-test 10.4/17.9,倍率仍達隨機的 14–35×,是「難但真實」的數字。
4. **【中】RAG 動在同一個凍結 text 空間之上,補不了基礎 embedding**。A/B 實測 RAG 未穩定勝 vanilla(R@10 −0.045),與「先修 embedding 再談 RAG」一致。
5. **【低/待確認】lr、caption/product 數量、multi-positive 缺失**——需實驗矩陣分離,見 §7。

## 6. 3D-FUTURE / 3D-FRONT 是否足夠 —— 明確回答

**3D-FUTURE(16,563 models,家具域 ~12,902,排除 Lighting/Others)**:
- **作 shape-semantic domain pretraining:規模足夠**(12,902 >> 1,546),且 super-cat/style 100%、fine-cat 89% 覆蓋,可支撐 category/style 條件對比。
- **但直接用不夠,缺三樣、須先補**:(a) **無 NL captions**(readme 確認無)→ 必須生成(§7-Stage B);(b) **無 explicit dimensions / canonical front / clearance** → spatial 監督要靠 Gate 1-3 從幾何 derive,不在此線;(c) **官方 scene image split 有 train/test model overlap 4,750**(Codex §5 實查)→ **不可直接當 object-disjoint retrieval split**,必須自建 model-disjoint split。
- 缺什麼講白:缺的是**判別性文字**與**object-disjoint split**,不是「模型數量不夠」。加更多模型無助於解 root cause;**補 caption 品質 + 自建 split** 才是關鍵。

**3D-FRONT(6,813 JSON)**:是 relation/placement 監督來源,**不是** shape-semantic pretraining 來源;其可用量待 Gate 1 全量 audit(join 47% 失敗風險已凍結為 supervision_tier 政策)。與本次「補 semantic embedding」正交,不混用。

## 7. 可直接執行的 data build / training / evaluation stages

> 全部遵守既有凍結:selection 只用 val、test 一次性、family/model-disjoint、learned 不碰幾何安全。以下為**semantic embedding 補救**,先於任何 spatial/RAG。

### Stage A — 評測地基(先做,無 GPU)
- 凍結 held-out：沿用 family-disjoint 1,212/200/134;**新增自然語言 room-style query 小測集**(非模板)作第二軸,揭露模板 caption 的過擬合。
- 主指標:**Text→PC R@1/R@10 + MRR**(這是使用者實際痛點方向),輔以 i2p 作 sanity。
- **input**:現有 catalog + corpus;**output**:`eval_harness_v2/`(凍結指標與 query 集 hash)。promotion gate 在此預註冊(不看結果)。

### Stage B — Caption 品質提升(最高 CP,root cause #1)
- 每 product 生成 **3–5 條多樣 caption**,強制涵蓋:精確類別、材質、主色、形狀特徵、功能、風格詞;來源分層 `official_metadata / VLM_silver`(VLM 須 pin model+prompt hash)。
- QA:抽審 ~10% + VLM-human agreement 隨報告發布(沿用既有 κ<0.6 退回規則)。
- **input**:1,546 catalog + 官方圖/render;**output**:`captions_v2.jsonl`(provenance-pinned);**不覆蓋**現有 corpus,新版並存。

### Stage C — Sampling 改造(root cause #3/#5)
- **multi-positive**:同 product 的多 caption / 多 render 視為正例集(非單一 diagonal)。
- **category-balanced batch**:抑制大類(Cabinet/Sofa)主導;每 batch 類別配額。
- **hard-negative(v2 更正定義)**:**不可**把「同類別不同 product」一律當 negative——同風格餐椅對 `modern wooden dining chair` 可能都是合理 positive,盲目當負例會懲罰正確匹配。hard negative 必須是:(a) 同類別但**功能**不符、(b) 同類別但**風格/材質**明確不符、(c) 外形相近但 **category 錯誤**、(d) 經人工或可靠 metadata 確認**不應匹配**者。負例來源與評測分離(防循環),且負例判定依據須帶 provenance。
- **input**:captions_v2 + ply_8192;**output**:新 dataloader（版本化 sampling policy + hash）。

### Stage D — 文字塔可訓練性 A/B(root cause #1 的直接槓桿)
三組對照(val-only 選、test 一次性):
```text
D0 frozen text（現況 baseline，必敗對照）
D1 frozen text + trainable text projection/adapter（低風險）
D2 last-N text layers unfrozen，低 lr + retention loss（防 CLIP 遺忘；Recall 倒退即不 promotion）
```
- 加 **category auxiliary loss**(輕量分類頭,穩定語義)與 **image↔text loss** 作 ablation（目前 i↔t 凍結為常數;開放它是否幫助 text 線,需實測）。

### Stage E — Initialization 公平 A/B(Codex P0-B-6)
```text
E-init-official：官方 ULIP checkpoint
E-init-ikea：舊 IKEA checkpoint
```
同資料同超參,只換 init,比 val Text→PC。**不得**用 test 選 init。

### Stage F — 3D-FUTURE domain pretraining（root cause 規模）
- 從 12,902 家具 model 產:point cloud(FPS 8192)+ multi-view render + captions(Stage B 同法)+ **自建 model-disjoint split**（避開 overlap 4,750）。
- 兩階段:先 3D-FUTURE 大域 pretrain,再 IKEA 1,546 fine-tune（retention 保 IKEA retrieval）。
- **input**:3D-FUTURE model + model_info;**output**:`front_future_pretrain/`（版本化）。

### Stage G — RAG 重評（僅在 B–F 把 vanilla 拉起來之後）
- 重跑 A/B/C（Semantic / +Reranker / +RAG），沿用「未穩定勝就不 promotion」；RAG 不得掩蓋基礎 embedding。

## 8. 各 Stage 的 input / output / loss / checkpoint / promotion gate

| Stage | input | output | loss | ckpt 選擇 | promotion gate |
|---|---|---|---|---|---|
| A | catalog+corpus | eval_harness_v2 + 凍結指標 | — | — | 指標/門檻預註冊(不看結果) |
| B | catalog+圖 | captions_v2.jsonl | — | — | VLM-human agreement 公布;κ≥0.6 |
| C | captions_v2+ply | dataloader v2 | — | — | sampling policy hash |
| D | dataloader v2 | text-tower A/B ckpts | InfoNCE(+cat aux +i↔t ablation) | val Text→PC R@10 | 須勝 D0;D2 若 Recall 倒退不 promotion |
| E | Stage D 最佳設定 | init A/B ckpts | 同 D | val Text→PC | 公平同設定;test 不參與 |
| F | 3D-FUTURE derived | pretrain→finetune ckpt | 兩階段 InfoNCE + retention | val（IKEA + FUTURE 雙軌） | IKEA retrieval 不得倒退 |
| G | 最佳 semantic ckpt | RAG A/B | RAG stage loss | val | C 穩定勝 B 才上 |

**test 使用(v2 更正,更嚴)**:所有 Stage(A–G)與所有初始化/超參 A/B **一律只看 validation**,鎖定**單一最終方案**後,**整個計畫只跑一次 held-out test**。不是「每條線各跑一次」——那仍會多次查看 test、造成 test 汙染。test 一經使用即凍結,不得回頭調任何設定。

## 9. 尚未確認項目與取得證據的方法

| 項目 | 狀態 | 取證方法 |
|---|---|---|
| Table 4 原始 script/CSV/query-gallery 定義 | 尚未確認(repo 無) | 向原作者索取生成 script + 原始資料 + metric code |
| 論文 30K/160K 是否曾在他機存在 | 尚未確認 | 需 4090/原始環境完整盤點;本機判定為不存在 |
| Parts2Words / ULIP-2 baseline 實作與 checkpoint | 尚未確認(repo 無 code) | 索取 baseline repo + ckpt lineage |
| 2.0 每 product 實際 caption 數與生成法 | 部分確認(corpus 3,561 rows / 1,212 train product,doc_type 單一 product_semantics,短模板) | 需 catalog 生成 pipeline 文件 |
| 5090 1,546 real-scene E2E 是否完成 | 尚未確認 | repo 無 completion artifact;待 5090 回報 |
| Gate 1 是否已在他處啟動 | **本機確認未啟動**(無產物/目錄,GPU 閒置) | 已查 |

## 10. Claude 對 Codex 設計的意見

```text
NO_BLOCKING_OBJECTION
```

Codex 本輪的框架我**同意且有證據支撐**:先修 semantic embedding、RAG 不得掩蓋基礎 embedding、733 不代表完整 1.0、不以 76.53 vs 17.91 作比較、weak VLM label 不冒充 gold。以下為**非阻擋** P1 建議:

- **P1-1(root cause 定位)**:請把 §2 的比較錯誤修正**再前推一步**——問題不只是「733 vs 1546 不可比」,而是 **1.0 的 76.53 本身是 train==val 洩漏數字**(§4 Lineage A),因此 1.0 從未有過可信的 held-out retrieval 基準。修正敘述時應明講此點,否則仍會被誤讀成「1.0 曾達 76%」。
- **P1-2(root cause 優先序)**:Codex P0-B 問「Text→PC 低的原因排序」——證據明確指向 **frozen text tower + 模板 caption**(§5 #1/#2),建議把 caption 品質(Stage B)與文字塔可訓練性(Stage D)列為第一優先,**先於** multi-positive/hard-neg(Stage C)與加資料(Stage F);否則會先花力氣在非瓶頸上。
- **P1-3(Table 4 措辭)**:對外/論文修訂一律用 `UNREPRODUCIBLE / SUSPECTED_METRIC_MIXING`,不用「造假」,但也**不得保留原數字**——兩者都不誠實。
- **P1-4(前輪 P1 延續)**:Gate 7 v3 contract 的 `authoritative_revalidation_required` / `safety_status: provisional_local_only` 我方確認採納,與本輪無衝突。

---

*本輪 3090 側僅唯讀:抽 DOCX 文字、grep/wc 盤點、算術核對,未改任何 DOCX/資料/程式/服務。所有數字均本機可複現(路徑見各節)。*
