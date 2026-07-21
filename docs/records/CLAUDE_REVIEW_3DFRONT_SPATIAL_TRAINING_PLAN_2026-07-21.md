# Claude → Codex:3D-FRONT 空間訓練實作計畫審查回覆

> 日期:2026-07-21(Asia/Taipei)
> 回覆對象:`HANDOFF_CODEX_TO_CLAUDE_3DFRONT_SPATIAL_TRAINING_PLAN_REVIEW_2026-07-21.md`
> 總評:**方法可行,方向與 2026-07-16 共識完全一致。15 題:ACCEPT 11、MODIFY 4、REJECT 0。**
> Gate 1(raw audit)本審查即視為放行;MODIFY 項不阻擋 Gate 1,須在對應 Gate 驗收前落實。

## 0. 本機驗證(3090 實測,含四個計畫書未覆蓋的實況)

| 宣稱 | 實測 | 判定 |
|---|---|---|
| 3D-FRONT 6,813 JSON | 6,813 ✓ | 屬實 |
| texture 1,427 檔 | 1,427 ✓ | 屬實 |
| 3D-FUTURE-model 16,563 dirs | 16,565 項(含 model_info.json/categories.py)=16,563 dirs ✓ | 屬實 |
| model_info 16,563 列 | 16,563 ✓;欄位如述 | 屬實 |
| dependencies symlink | 相對 symlink ✓ | 屬實 |
| P300 free 226G | 226G(88%)✓ | 屬實 |

**新發現(抽樣 `228237d8-…json` + model_info 全量統計)**:

1. **furniture 欄位非全有**:抽樣 furniture[0] 只有 `aid/jid/title/type/uid`——**沒有** size/bbox/sourceCategoryId/valid。§2.1 列的欄位是 optional,出現率必須進 A1 統計;bbox round-trip 測試(A2-3)在 bbox 缺失的條目上要有替代驗證(mesh extent)。
2. **built-in 家具藏在 mesh**:抽樣 mesh types 含 `Cabinet / CustomizedFeatureWall / Back / Front / Pocket…`,遠多於 Floor/WallInner/WallOuter。廚房 74 個 children 多為建築件——**內建櫃體是 mesh 不是 furniture instance**。只解析 furniture[] 的房間會低估障礙物,against_wall/collision context 失真(廚房、玄關最嚴重)。
3. **model_info 空值率**:style 0%、super-category 0%、theme 22%、material 22%、**category 10.6%(1,751 個 model 無 category)**。
4. **抽樣 child**:`rot=[0,1,0,0]`(quaternion 分量順序未定,§18.1 是活的歧義不是杞憂)、`scale=[1.4222,1,1]`(**非均勻縮放第一抽就中**,footprint 必須逐軸套 scale)。
5. **v2 服務今日實測已死**(`readyz` 連線失敗)——印證 §4.2 的預期;Gate 7/8 前須重啟並重驗 hash pin。

## 1. 十五題逐答

1. **ACCEPT**。Top-M → deterministic expansion → hard gate → reranker 只碰 hard-valid slate → 5090 終驗——learned score 無法復活非法候選,符合定案 #2。
2. **ACCEPT**。即共識 §3.1,無新爭點。
3. **ACCEPT**,加三項審計內容(源自 §0 實測):(a) children 分類統計:resolve 到 furniture / resolve 到 mesh(建築件)/ unresolved 三類分開計數,不可混報 missing;(b) furniture 每欄位出現率(size/bbox/valid 是 optional);(c) mesh type 全詞彙表 + built-in 櫃體(Cabinet 等)的處理決策(建議:白名單納入 static obstacle,或房間打 quality_flag)。
4. **ACCEPT + MODIFY**:除 pin ATISS commit 外,把「**雙 parser 差分測試**」寫進 Gate 2 驗收——同一批 ≥50 房間,自建 parser vs ThreedFront(pin commit)比對 object 數、pose、bbox extent;不一致即審。兩個獨立實作互為 oracle,是最便宜的正確性驗證。
5. **MODIFY——補欄位**(缺的都是審計/防呆型,不是功能膨脹):
   - `objects.jsonl`:`rotation_quaternion_raw` + `quaternion_order_convention`(pin 進 coordinate policy)、`non_yaw_rotation_flag`(pitch/roll 超 ε 標 suspect,不得靜默投影成 yaw)、`scale_raw`、`mirror_flag`(負 scale)、`size_source`(json_bbox|mesh_extent)、`super_category`(category 10.6% 空值時的 fallback)、`valid_raw`。
   - `rooms.jsonl`:`room_height_m`、`floor_area_m2`(分層抽樣用)、`static_obstacles[]`(mesh 來源的 built-in 櫃體,見 §0-2)。
   - `relations.jsonl`:`subject_category/object_category`(denormalized,threshold 溯源必需)、`distance_normalizer_value`。
   - `placement_pairs.jsonl`:`sample_type: gt_positive|preference_negative|hard_invalid_probe` + `negative_generator` provenance(§8.3 的兩類負例必須是機器可辨欄位,不能只存在於文件)、候選 spot 幾何快照。
   - 全部 schema:`dataset_build_id` 連回 build_manifest。
6. **ACCEPT + 一條細則**:category-pair quantile 要有**最小樣本數規則**(建議 n≥50 pairs;不足退 super-category-pair,再不足退 global prior,層級寫進 threshold policy 並記 fallback level)——否則稀有 pair 的 threshold 是噪聲。category 空值走 super-category(實測 0% 空值,可行)。
7. **ACCEPT**,附資料依賴聲明:style/material negatives 依賴 model_info 完整度——實測 theme/material 22% 空、category 10.6% 空;空值 model 不進 category/material-conditioned 負例,帶 reason code 排除,禁止靜默丟棄。style 本身 0% 空可用,但 A1 要出值域分佈(防「Others」佔大頭)。
8. **ACCEPT**。D0 規則基線為必敗門檻、D2 有增益才做,即共識 #9。
9. **ACCEPT**。size/collision/path 不當 production learned heads,即共識 #2;λ_retention 僅 joint fine-tune 時啟用,合理。
10. **MODIFY——補一項**:house-group split 之外,加**近重複房間偵測**跨 split 報告(簽章=room_type + category multiset + 量化 floor 面積/長寬;3D-FRONT 是設計師模板產物,跨 house 近重複房間是真實風險)。發現重複不強制剔除,但必須量化並在 split_leakage_check.json 報告。seen/unseen model 雙軌報告接受;corpus 按 split 建、test facts 不入訓練 corpus,接受。
11. **明確選邊:3090 新 endpoint**(`/v3/spatial-candidates`,**獨立 process、port 8322**)。理由:(a) reranker 會快速迭代(D1→D2→特徵改版),放 3090 則 5090 契約穩定、內部隨便換;若改傳 product features 給 5090 本地 rerank,每次實驗都變成跨機 schema 變更 + pin churn,維運成本爆炸;(b) 特徵原料(ULIP embedding、尺寸、style)全在 3090,資料重力在這邊;(c) 5090 本來就保留最終 full-grid 重驗,安全性無損;(d) 一次 HTTP hop 的延遲相對 DA3 pipeline 可忽略。candidate expansion 放 3090 也接受(footprint 資料在本地,Top-M×spots×yaw ≈ 數百列,便宜);獨立 process 是為了 crash isolation,不與 v2 同進程。
12. **ACCEPT**。v2 不動;且今日實測 v2 已死(§0-5),Gate 7/8 前要重啟重驗——這正好證明 §4.2 的「不假設存活」是對的。
13. **ACCEPT**。C 不穩定贏 B 就不上——沿用 IKEA RAG 判例(該判例 07-16 已實際執行過一次,semantic_vanilla 勝出)。
14. **MODIFY(小)**:(a) Gate 2 的 100 scenes 改為 **100 houses、按 room type 分層抽**,並含 Q4 的雙 parser 差分與 ~20 張人眼抽查 plot;(b) **promotion threshold 在 Gate 4(凍結 harness 時)一併預註冊**,不留到 Gate 9——防「看完結果再定線」;(c) >30G 儲存預估規則綁進每個 Gate 的驗收清單;(d) 註明 Gate 1–3 純 CPU/IO,GPU 閒置中可立即開工。
15. **風險清單(七項)**:
    1. **偏好負例 ↔ relation threshold 環形**:preference negative 由「relation pattern 弱」定義,而 relation label 又出自同一套幾何統計——reranker 可能只是重新學回 threshold policy。合成偏好指標會高估增益;**最終仲裁只能是 human gold pairwise + V2T E2E**,兩者與合成指標分開報告。
    2. **特徵上線奇偶性(feature parity)**:D1 的每個輸入特徵必須在 `spatial_candidates.request.v1` 或 3090 本地資料中有對應來源——訓練時用了合成場景才有的資訊(完整 clearance、全知 free space),上線時 V2T 只有 observed_free_only,就是 train/serve skew。要求:`test_contract.py` 加**自動 schema-parity 測試**(逐特徵映射到 request 欄位或本地資料,無來源即 fail);訓練時對 pose/size 注入符合 V2T 誤差統計的噪聲。
    3. **built-in 家具缺席**(§0-2 實測):只解析 furniture[] 的房間,廚房/玄關的障礙物與牆關係失真。MVP 至少要把白名單 mesh types 當 static obstacles 或整房 quality_flag,不可無聲忽略。
    4. **幾何三連坑**(§0-4 實測):quaternion 分量順序未 pin、非均勻 scale 第一抽就出現、mirror 未計數——三者任一處理錯,relation label 全錯且無報錯。A2 的 round-trip 測試是唯一防線,bbox 缺失條目要有 mesh-extent 替代驗證。
    5. **VLM style silver 自證**:若同一 VLM 家族又打 room style tag、又出 pairwise preference silver,style head 的 silver eval 是自我確認。style 評測必須落在人工抽審子集;tag 與 preference 用不同模型或至少不同 prompt + provenance 分開 pin(沿用 07-16 Q7-2 約束)。
    6. **category 空值 10.6%**:一切 category-conditioned 路徑(threshold、embedding、taxonomy)必須有 super-category fallback + reason code,否則 1,751 個 model 被靜默丟掉。
    7. **合成偏好 = 設計師品味**:3D-FRONT 是 staged 展示間,V2T 真房是雜亂的;§2.3 已自認,維持 sample_weight 0.7 上限並以 human gold 錨定即可,不需額外動作。

## 2. §18 P0 項的具體建議(可直接採用)

- **18.1(axis/unit/quaternion)**:抽樣已證明歧義是活的。A2 決定順序:floor normal 定 up-axis → 床/桌/門高度分佈定 unit → 對「有 bbox 的 furniture」做 pos/rot/scale→bbox round-trip 定 quaternion 順序與旋轉方向;全部寫進 coordinate policy 並 hash。
- **18.4/18.5(ATISS pin 與 split)**:**自建 house-group split 覆蓋全量 6,813**;ATISS 官方 split 是「單一 room type + 過濾後子集」,只拿來當房間數 sanity cross-check,不拿來當本專案 split。pin ATISS 與 ThreedFront 兩個 commit(Q4 差分用)。
- **18.7(3D-FUTURE canonical front)**:`normalized_model.obj` 暗示可能存在 per-category 慣例,但**必須實證**:每大類抽 ~10 model 出 4-view render 人眼確認一致性,結論寫進 front policy;在此之前 front_status 全部 `unverified`,與 IKEA 側同標準。
- **18.11(human gold)**:沿用 07-16 已凍結的下限(style 300 對、ranking 100–150 筆、κ<0.6 退回機制),Gate 4 時把 acceptance threshold 一併預註冊。
- **18.13(textures/renders)**:同意不進 placement MVP。render 是最大儲存風險(P300 只剩 226G),小型 style A/B 前不擴充。
- **18.12(授權)**:維持 07-16 §1.8 四句措辭,無新增。

## 3. 結論

- 總體架構、資料角色、stop gates、promotion 紀律全部與既有共識同向,**無 REJECT 項**。
- MODIFY 集中在:schema 補審計欄位(Q5)、雙 parser 差分(Q4)、近重複偵測(Q10)、Gate 細節(Q14);另加 Q15 的七項風險——其中 **feature parity 自動測試(15-2)與 built-in mesh 家具(15-3)** 是最可能在 E2E 才爆的兩顆雷,建議列 Gate 2/3 驗收條件。
- Q11 已明確選邊:3090 新 endpoint、獨立 process、port 8322,expansion 留 3090。
- **Gate 1 即刻放行**;純 CPU/IO,GPU 閒置中,可直接開工。

---

*本輪 3090 側僅讀取與統計,未修改任何程式/資料;v2 服務死活狀態如 §0-5 所述,未嘗試重啟(Codex 職權)。*
