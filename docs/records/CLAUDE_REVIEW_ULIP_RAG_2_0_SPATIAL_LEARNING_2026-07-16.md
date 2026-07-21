# Claude 對 V2T × ULIP_RAG 2.0 Revised Design 的審查

> 審查者:Claude(3090 側,ULIP_RAG 1.0 全系統實測負責人)
> 日期:2026-07-16
> 依據:HANDOFF_3090_TO_CLAUDE_...2026-07-16.md + 1.0 全系統實測記錄(架構審查/smoke test 報告)+ ~/V2T 兩份交接快照
> 標記沿用:[FACT]=本機可證 / [AGREE]/[MODIFY]/[REJECT]=審查立場 / [USER]=只有使用者能定

---

## 1. 核心目標

- **是否同意新 catalog 全量重抓/重訓:[AGREE]**,且理由比文件寫的更硬:
  - [FACT] 1.0 訓練管線在 `ikea_ulip.py` 對每顆點雲做中心化+unit-sphere 正規化,訓練時再加 random scale/shift/rotate/jitter(`dataset_3d.py:63-158` 增強函式,loader 實際呼叫已實測)——**舊 512D 向量在數學上就不含公制尺度與 canonical front**,文件 §2.1 的判斷正確,我以 code 佐證。
  - [FACT] 舊 733 件的品質治理記錄(`ikea/docs/bad_3d_models_manifest.json`、3d_model_quality_audit.md)顯示該批 TRELLIS 生成模型存在需人工裁決的品質分佈——當 baseline 可以,當 2.0 正式庫確實不合格。
- **是否同意四種匹配能力:[AGREE with phasing]**——四種都該在 2.0 scope,但**不同能力的「該學 vs 該算」不同**,見 §5/§7 的核心修正。
- **建議 scope 修正(本審查最重要的一條)**:
  設計稿把「尺寸/碰撞/路徑」也列為 learned heads(Stage 4)。這與 V2T 已定案鐵律衝突(V2T 架構詳解 §3.5:「放不放得下=規則;embedding 編碼不了尺寸」——那是**實測教訓**,不是偏好)。文件雖保留 hard gate,但**花人力訓練一個「規則已經能精確算」的東西,是把最貴的資源(標註)花在最沒有增益的地方**。
  **[MODIFY] 建議:learned 資源只投在規則做不到的三件事——style、relation、human preference。幾何 validity 永遠規則主導;`spatial validity head` 降級為選配 ablation(回答「learned 能否逼近 exact gate」的研究問題),不進 production 關鍵路徑。** 這同時大幅縮小 Dataset F 的標註需求(exact geometry labels 由引擎免費產,不需人)。

## 2. 新家具來源

- **source/locale/categories:[USER]**——這是採購決策,我不裁定。但給事實供選擇:
  - [FACT] 若來源仍是 IKEA:crawler 全套現成(`ikea_api`+Playwright v3 版 `fetch_v3_products.py`、尺寸解析 `phase_a_parse_dimensions.py` 實測 2112 筆、mongo_conn 統一連線)——Phase 1 成本最低的選項。
  - [FACT] 3D-FUTURE(V2T 評測已預定要用):自帶高品質 3D mesh + style/category 標註,研究授權——**若 2.0 允許「非在售商品」進訓練庫,它能直接解決 Dataset A/B/C/E 的大半監督**,建議即使正式 catalog 另選,也把 3D-FUTURE 列為訓練輔助庫。
  - 其他(Wayfair/Amazon 等):爬蟲與授權風險需逐站評估,無現成管線。
- **expected product/variant count:[USER]**;但給容量約束:[FACT] P300 剩 470G、系統碟剩 155G;1.0 是 733 件 ≈ 31G(ply 26G 為大頭)。**線性外推 10k 件 ≈ 420G+,P300 放不下**——scope freeze 時必須連儲存規劃一起凍(加碟或降點雲精度/壓縮)。
- **3D availability**:[FACT] TRELLIS image-to-3D 在 3090 已完整打通(2026-07-11 實測:E2E 113.6s/件,含去背+分類+GLB+8192 點 PLY;dinov2/GLIBC 相容性已修)。「無官方 3D → TRELLIS 重建」路徑可行,但 [FACT] 733 件的教訓是**必須配幾何 QA gate**(manifest 三件套流程可直接沿用)。
- **crawl/license/update policy:[USER]**;同意文件 §4.1——凍結前不抓。

## 3. V2T 現況

(3090 只持交接快照,以下以快照為據,5090 側請覆核)

- **available room/video/scene count**:[FACT-快照] 7 場景 @128 幀,前半凍結。**7 場景做 test 都嫌少,做 training 是不可能的**——這是整份設計稿最大的資料現實缺口。
  **[MODIFY] 建議:Stage 4/5/6 的訓練監督主來源改為 3D-FRONT 合成場景**(自帶 room layout+家具 instance+類別+pose 的 GT,relation graph 可規則推導後抽樣人工複核),**真實 V2T 場景只做 held-out test**。這一步決定 2.0 是「等資料」還是「現在就能動」。
- **可穩定輸出欄位**:[FACT-快照] SDO 含 objects(class/size/center/size_reliable)、placeable_rects(type/walls/adjacent/nl)、wall_segments、style、scale_confidence;⑤ 契約 v2.0 含 placement/constraints/feasibility/uncertainty,驗證器 3 場景 PASS。
- **尚缺欄位**:[FACT-快照] instance yaw/front、doors/windows/swing、access terminals 未確認;all-tested-poses exporter 未建(文件 §5.6 自承)。**relation graph 若無 front/yaw,只能先做 distance/bearing 類關係(near/beside/against_wall),facing 類延後——與 Stage 5 的 modulo-180 條款一致,但要明寫進 ontology v1 的「可標 vs 不可標」清單。**
- **source paths/evidence**:`~/V2T/115MOST_V2T_架構詳解.md` §2.4-2.5、§〇;5090 的 `~/V2T_DA3` 為本體。

## 4. Style 設計意見

- **labels/data**:[AGREE] Dataset C 的三層標籤來源(site tag=weak / VLM=weak / human=gold)+ curated lifestyle 圖當 weak positive。**[MODIFY] 補一條免費監督:lifestyle/room-scene 圖中「同場景共同出現的商品」天然構成 style-compatible 弱正例**(IKEA 場景圖有商品掛連),比人工便宜一個數量級,先用它把 Stage 3 跑起來,human preference 只花在 gold eval + 最終 fine-tune。
- **model/loss**:[AGREE] 雙塔(Room/Product Style Encoder)+contrastive+preference。[MODIFY] Room Style Encoder 初版不必新訓——**先用凍結 VLM/CLIP 圖像特徵 + 輕量 projection head**,證明資料有信號再談專用 encoder(這符合 V2T「baseline 哲學」:每個學件都要先有更便宜的對照)。
- **evaluation**:[AGREE] §13.2;補:style 評測必須 room-disjoint,且報 per-style-slice(避免「都推 modern」拿高分)。

## 5. Spatial/Size 設計意見

- **labels/data**:[AGREE] Dataset B schema(known_mask/polygon/clearance/canonical front 分離,unknown 不補 0)設計正確。[FACT] 1.0 的 dimensions_mm 解析器(phase_a)可直接升級為 Dataset B 生產器的一部分。
- **model/loss**:**[MODIFY-核心]** 如 §1 所述:size/collision/path 的「判定」不學,規則做;學的只有**在全部合法候選內的排序偏好**(哪個合法位置「更好」——這規則做不到)。即:Stage 4 從「訓練 validity heads」改為「訓練 pose preference reranker(僅在 hard-valid 子集內)」。Loss 表(§9)相應刪 L_size/L_collision/L_path 或降為 auxiliary,λ 資源讓給 L_pref/L_relation/L_style。
- **hard gate**:[AGREE-無條件] §12.3 照單全收,永不讓 learned 分數翻案。

## 6. Furniture Relation 設計意見

- **ontology**:[AGREE] §5.5 清單當 v1 凍結,但拆兩層明寫:
  - 幾何可推導層(near/beside/against_wall/corner/aligned/perpendicular/clearance_conflict)——規則從 3D-FRONT/SDO 自動產,零標註;
  - 功能語意層(paired_with/functional_group/facing 語意)——需 category 先驗+人工抽核。
  v1 只承諾第一層全量+第二層高頻 pattern(nightstand↔bed、coffee_table↔sofa、TV↔sofa、chair↔table 四組)。
- **graph representation**:[AGREE] node=instance、edge=相對幾何+關係;candidate 以「插入 graph 的新 node」建模正確。[MODIFY] 初版不必 Graph Transformer——**edge-feature MLP + attention pooling 先上**,§14-9 的消融本來就要比,直接把簡單版當 default。
- **labels/data**:主來源 3D-FRONT(見 §3);V2T 真實場景只驗證。
- **evaluation**:[AGREE] §13.4;補 no-anchor 情境的 abstain 正確率(§12.2 已提,進指標)。

## 7. Training stages

- **接受**:Stage 0(inventory 先行,不動就不練)、Stage 1(multi-positive family、split by family、先凍 backbone——全部正確,[FACT] 1.0 的 InfoNCE 把同 family 變體當負例確實是已知缺陷)、Stage 2(RAG 須贏過 vanilla 才上線——[FACT] 1.0 production 本來就是 vanilla,RAG 是研究線,這條與現實一致)、Stage 3、Stage 6-8 的順序與 replay/retention 設計。
- **修改**:Stage 4 如 §5(validity→preference);Stage 5 如 §6(簡化模型+資料源改 3D-FRONT);Stage 7 的人工標註量能否成立取決於 [USER] 資源承諾,建議在 scope freeze 時就定「最低可交付版=weak label only + gold eval 人工」的降級方案。
- **不接受**:無整段否決;但 §9 的 L_collision/L_size/L_path 作為主 loss 項——降級或移除(理由同上)。

## 8. Split/Leakage/Promotion

- **split**:[AGREE] §11 全部(family-group、room-group、先 split 再衍生)。補一條:[FACT] corpus 若含商品 exact facts(§6.2 允許),**corpus 也要跟著 product split 走**——test family 的 facts 不得入訓練期 corpus,否則 RAG 通道洩題(文件 §11.1 有提,把它升為 hard rule)。
- **metrics/gates**:[AGREE] §13;建議 2.0 primary acceptance 具體化:
  1. 新 catalog semantic Recall@10 ≥ 以同資料重評的 1.0-init baseline(不得倒退);
  2. size-fit false-safe = 0(hard gate 保證,報告驗證);
  3. style human pairwise win-rate vs vanilla ≥ 60%;
  4. E2E Top-1 hard-valid rate ≥ 95%;
  5. 全部在 room-disjoint + family-disjoint 軌上報。
- **risks**:①7 場景資料現實(最大);②人工標註無承諾來源;③儲存容量(§2);④跨機 artifact 交換——[FACT] 本輪剛實證跨機二進位不可移植(4090 編的 .so 在 3090 因 GLIBC 直接炸),**5090↔3090 交換一律 data-only(JSON/npy/manifest+hash),禁止交換編譯產物與 env**,寫進交換契約。

## 9. 雙機責任與 artifact 交換

- **5090**:[AGREE] §15.2 清單 + all-tested-poses exporter + relation 標註工作流。
- **3090**:[AGREE] §15.1 清單;[FACT] 對應能力全數已實證在位:crawler/mongo/TRELLIS/訓練/向量重建/serving/corpus 生產線(Llama+vocab_build)。
- **runtime location**:[AGREE] §15.3 拆分案(scene 張量留 5090、product 特徵留 3090)。跨機契約照 `retrieval_request.json` v2.0 的成功模式再定一份 **`candidate_slate.json`**(3090→5090:Top-M product 的 p_sem/p_style/p_spatial 緊湊特徵+metadata;5090→3090:最終結果),欄位含 schema version+雙側 hash。
- **transfer path/versioning**:immutable bundle + manifest hash(文件已有);補:每個 bundle 附 `requirements_freeze` 與生成端機器標識,防止再次發生「拿別台的二進位」。

## 10. P0 blocker 與建議下一步

**P0(全部 [USER],凍結前不動工)**:
1. 新家具來源/國家站/類目/量級/授權——決定 Phase 1 一切。
2. 是否採納「3D-FRONT 當訓練監督主源、V2T 真實場景只做 test」——決定 Stage 4-6 是「現在能做」還是「等資料」。
3. 人工標註資源承諾(誰、多少小時)——決定 Stage 7 是完整版還是降級版。
4. 儲存規劃(P300 剩 470G vs 新 catalog 外推需求)。

**回答 §17-E(本輪權限)**:
- Q21:我目前的授權=review/提問,本文件即交付物;**未動任何 code、未爬任何資料**。5090 side exporter 修改不在我這台的權限範圍,需另行授權。
- Q22:**同意**——source/scope 凍結前不開始任何抓取。

**建議下一步(依序)**:
1. 使用者裁定上面 P0-1~4 → 凍結 `Data Acquisition Spec`。
2. 我方(3090)在等待期間可零成本先做:①用 3D-FUTURE 樣本驗證 Dataset B/E 的 schema 可產性(不爬新商品);②relation ontology v1 兩層清單成文;③`candidate_slate.json` 契約草案。——以上任一項動工前仍會先請示。
3. 凍結後按 Phase 1→7 走,但 Stage 4 按本審查 §5 修正版執行。

---

### 附:本審查的一句話總結

**設計方向正確、資料現實不足、學習資源錯配一處**:同意全量重抓重訓與四能力 scope;把 7 場景的現實用 3D-FRONT 補上;把「規則已經會算的幾何 validity」從學習目標降級,標註預算全數轉投 style/relation/preference——這三件事修完,這份設計就能凍結。
