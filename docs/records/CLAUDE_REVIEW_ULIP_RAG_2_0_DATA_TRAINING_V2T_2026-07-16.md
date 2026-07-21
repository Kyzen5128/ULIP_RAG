# Claude 對「ULIP_RAG 2.0 資料集選擇、訓練方法與 V2T 串接設計」的審查回覆

> 審查者:Claude(3090 側)/ 2026-07-16
> 本機驗證:文件所有 [FACT] 抽查屬實——訓練行程存活(main_ikea.py, epoch 191/250, GPU 65%)、
> bundle `/mnt/P300/data/ULIP/ULIP_RAG_2_0/ikea-us-20260716-v1/` 結構完整、
> corpus train=3,561 行、P300 剩 331G。以下按 §12 十二題逐答,末段附本機 log 的追加發現。

---

## 一、12 題逐答

### Q1 資料角色四分法 —— ✅ 同意,無保留

`IKEA=production / 3D-FRONT+FUTURE=synthetic supervision / V2T=real held-out / human=preference gold` 正是正確架構。唯一補充:3D-FUTURE 家具**可以**作為 style/relation 訓練時的「家具實體」,但其 embedding 永不進 production gallery(文件 §3.4 已寫,升為 hard rule 即可)。

### Q2 幾何安全只做 rules —— ✅ 同意(這是我上輪審查的核心修正,已被正確採納)

補一個執行細節:rules-only 的可用性取決於 metadata 完整度,而文件自報的缺口不小——
- 285 件尺寸待補、canonical front 1,546 件全未確認、clearance 全未建。
- 因此 **`unknown` 的三態語意必須貫穿到 serving 輸出**:`pass / fail / unknown` 中,unknown 不淘汰、但排序降權 + 回傳 caveat(與 V2T `coverage=observed_free_only` 的「可確認放得下、不可斷言放不下」同構)。
- 建議把「缺尺寸 285 件」列 MVP-1 的 blocker 清單:phase_a 解析器補跑 + 人工掃尾,成本低、收益直接(hard-filter eligible 從 1,261 → 接近 1,546)。

### Q3 relation ontology v1 —— ✅ 全保留,但分兩層宣告

- **幾何可靠層(v1 立即上)**:`near / beside / against_wall / corner / aligned / perpendicular`——bbox+center+室界即可推導,3D-FRONT 有 GT,零人工。
- **需要 front/yaw 層(v1 條件上)**:`in_front_of / facing`——3D-FRONT 有 rotation GT,合成訓練照學;但**推論時 IKEA 商品 canonical front 未建** → 這兩類在 production 先以「category front 先驗」(sofa/bed/bookcase 類有明確正面)+ 對稱性 fallback 執行,front 未知時降級為 orientation-agnostic,不得假裝已知。
- 補一個文件 §6 推導清單有、Q3 清單漏掉的:`around`(dining chair around table)——多實例環繞單 anchor,規則可推導,建議 v1 納入。

### Q4 relation threshold —— **category-pair 模板為主,bbox-relative 正規化為輔,絕對米數僅 fallback**

理由:`near` 對 nightstand↔bed 是 0.1–0.5m,對 sofa↔TV 是 2–4m,單一絕對閾值必死。
執行建議:**閾值不要手拍——直接從 3D-FRONT 統計各 category-pair 的距離分佈取分位數**(如 P10–P90 為 near 帶),距離以 anchor bbox 對角線正規化。這樣 threshold 本身是 data-derived、可審計、可隨資料更新,也自動解決「大小家具尺度差」問題。

### Q5 style weak supervision 四來源 —— ✅ 接受,但各給 source weight + 三個防呆

1. IKEA lifestyle co-occurrence:可用,但注意它是「搭配販售意圖」不是純風格判斷(促銷湊數是雜訊)——weight 最低的 weak positive。
2. 3D-FRONT 同場景 co-occurrence:品質較高(專業設計),但 room type 分佈偏(臥室過重)→ **按 room type 分層採樣**再產 pair。
3. 3D-FUTURE style/theme:文件標 [PENDING] 正確,schema 到手驗證前不排程。
4. 人工 gold:只花在 eval + 最終 fine-tune。
**防呆(negatives)**:同意「不能只用 category 差異」;但注意一個 circular 風險——若 hard negative 由 palette/material 聚類挖掘,而 style head 又吃 palette/material 特徵,等於用同源訊號自證。**負例挖掘優先用外部訊號(來源網站 style tag 不同、3D-FRONT 不同 style 場景)**,聚類挖掘只當補充。

### Q6 Style Head 先 MLP/bilinear —— ✅ 同意(即我上輪建議,維持)

再加一條門檻:MLP 版若在 held-out pairwise accuracy 打不贏「palette 距離 + material overlap 的純規則基線」,就先修資料再談模型。每個 learned 件都要有更便宜的對照,這是 V2T 的 baseline 哲學,2.0 應繼承。

### Q7 Relation Head 先 feature MLP —— ✅ 同意

同樣加規則對照:category-pair 模板(Q4 產物)本身就是 zero-parameter relation classifier——MLP 要贏它才有存在價值。Graph Transformer 留 ablation。

### Q8 V2T 座標契約凍結 —— **部分可凍,分 v1/v1.1**

3090 側單方不能凍,但依 V2T 快照給凍結建議:
- **v1 現在可凍**:unit=meter、z-up、Manhattan-aligned、origin=非負化後原點、yaw=從 +x 逆時針(度)、bbox=center_xy + size_xy + yaw(或 polygon)、occupancy grid 帶 origin+resolution。這些在 `transform.json` 鏈已是事實標準。
- **v1 不凍(留 v1.1 欄位佔位)**:instance front(V2T 尚無)、doors/windows/swing(快照標未確認)、access terminals。**MVP-2 的 door/path 規則依賴 v1.1**——見 Q12。

### Q9 七場景全 test 還是留 calibration —— **建議 5 test + 2 calibration(固定指定、永不輪換、不進任何評測報告)**

理由:§7 的 fusion 權重(α/β/γ/δ)與 rule threshold 若純靠合成資料定,domain shift 會直接吃掉;留 2 個真實場景做 fusion 校準是最便宜的保險。
替代案:若人工資源足以在 3D-FRONT held-out 上標 calibration set,則 7 全 test 更乾淨。兩案擇一,取決於 Q10 的資源答案。

### Q10 人工標註量 —— [USER 決定],但給最低可行預算參考

- style pairwise:500–1,000 對(每對 ~10 秒 → 2–3 小時)
- relation 抽核:300 樣本(3D-FRONT 規則推導的 QA → 1–2 小時)
- V2T gold:7 場景 × 3–5 query × top-5 候選排序 ≈ 100–175 判斷(2–3 小時)
- **IKEA canonical front 快標**:用現成 12-view render 每件挑「正面」視角,~5 秒/件 × 1,546 ≈ 2–3 小時——**這項建議加進清單**,它同時解鎖 facing relation 與 front-clearance 規則,是全表 CP 值最高的一項。
合計:一人約 1.5–2 個工作天 = MVP-4 最低門檻。

### Q11 授權 —— 研究階段合規;商業化前要處理三件事

1. 3D-FRONT/FUTURE 非商業研究授權:與 V2T 的 DA3-giant/LA(CC BY-NC)同構,V2T 早已標「論文 OK/產品要換」——2.0 沿用同一策略即可。**灰區:用其訓練出的 style/relation head 權重是否算衍生物**——保守解法=商業版用自有資料(累積的 V2T 標註+IKEA lifestyle)重訓這兩顆 head,rule engine 與 IKEA semantic 線不受影響。
2. IKEA 爬蟲資料的商用地位(網站條款)是更大的問題,[USER]/法務層級。
3. 所有 bundle 的 provenance 記錄(已在做)是未來授權切割的前提——維持。

### Q12 MVP 分階段 —— ✅ 同意,但 MVP-2 拆半 + MVP-1 補 blocker

```text
MVP-1:IKEA semantic/RAG + size/category rules
        ← blocker:285 件缺尺寸、serving profile 未建(文件已列)
MVP-2a:collision/boundary/wall 規則(V2T v1 契約即可,現在就能做)
MVP-2b:door/path 規則(依賴 V2T v1.1 欄位——doors/access,5090 側先行)
MVP-3:3D-FRONT/FUTURE style/relation pretraining(等 archive 到手+schema 驗證)
MVP-4:human preference fine-tuning(等 Q10 資源承諾)
MVP-5:V2T real-scene E2E
```

---

## 二、本機 log 追加發現(文件未提,對訓練決策有用)

以下取自 `training/semantic_ulip_v2/log.txt`(epoch 189–191 實讀):

1. **過擬合已成形,250 epochs 大概率不會再更好**:train pc↔text acc 已 99.2–99.7%、train loss 0.13–0.17,而 val R@1 自 epoch 154 起 37+ epochs 停在 15–16%。checkpoint selection 用 val best 所以無害,但**建議之後的 run 加 patience≈30 的 early stop**,可省 ~1/3 算力。
2. **text/image 凍結有 log 佐證**:t2i/i2t 指標跨 epoch 完全不變(0.1616/0.1289 一位不差)——證實只有 point branch 在動,與設計一致。順帶:**t2i R@1 僅 8%,代表 caption↔render 的先天對齊偏弱**,提升 caption 判別性(顏色/形狀/材質具體詞)是 semantic 線下一步 CP 值最高的資料改進,比加 epoch 有效。
3. **16% 的解讀**:200 件 family-disjoint val 上 R@1 16% = 32× 隨機(0.5%),R@10 46–48%;作為 Top-M→rule filter→rerank 管線的第一級召回,R@10/R@20 比 R@1 更相關——**驗收 gate 建議定在 family-disjoint R@10,而非被 1.0 的 76.5%(train=val 洩漏數字)錨定**,文件 §5 已正確聲明兩者不可比,維持。
4. lr 3e-3(沿用 1.0 from-SLIP 設定)對「from 1.0 checkpoint fine-tune」偏大,前期可能破壞較多 init 語意——**不建議中斷現訓**,列為之後 ablation(peak lr 3e-4–1e-3 對照)。

## 三、一句話總結

**設計已收斂到位:四分法資料角色、規則安全邊界、MLP 先行、防洩漏 split 全部正確;本輪 12 題我全數可答且多為同意——真正的 P0 只剩三件外部依賴:3D-FRONT/FUTURE archive 申請、人工標註資源承諾(Q10)、V2T v1.1 欄位(doors/front)。** 3090 側在等待期間的最高 CP 動作:①補 285 件尺寸;②canonical front 12-view 快標;③等現訓收斂後跑 held-out 評測 + RAG A/B。
