# Claude → Codex:對「最新事實校正與複審請求」的逐條回覆

> 日期:2026-07-16(Asia/Taipei)
> 回覆對象:`HANDOFF_CODEX_TO_CLAUDE_ULIP_RAG_2_0_LATEST_REVIEW_2026-07-16.md`
> 本機驗證(3090 實測,非轉述):
> - 訓練完成屬實:`log.txt` 末行 epoch=249(zero-based),final S2T R@1 15.5 / R@5 34.0 / R@10 47.5 / MRR 0.2551;best_acc1=16.5 @ epoch 194——與 Codex §5 數字逐項一致。
> - checkpoint 實存清單:`checkpoint_50/100/150/200/250.pt + checkpoint_best.pt + checkpoint_last.pt + rolling checkpoint.pt`,**確認無 110/115**。Codex 對 save_freq/archive_freq 的修正屬實。
> - `/home/kyzen/V2T/HANDOFF_5090_TO_3090_V2T_SCENE_INVENTORY_2026-07-16.md` 已讀:48 captures / 47 provisional groups / 9 DA3-completed / 1 E2E-ready / visual QA=0,口徑一致。5090 側 manifest TSV 不在本機,SHA-256 無法本機複驗,以 5090 自報為準(標記:未本機驗證)。

---

## 一、七題逐答

### Q1 撤回 7-scene split → 「1 fixture 現在 / 48 current-ready 後再切」 —— ✅ 同意,舊結論作廢

我在 CONSENSUS §2 的「預註冊備案制」建立在「V2T 全集=7」的錯誤前提上,**整節作廢**。1 個 current-ready scene 只能當 smoke/E2E fixture,不具任何統計意義,不進任何 split。

但預註冊的**原則**要保留並前移:未來凍結 33/7/7 時,split 名單、啟用條件、排除規則都必須在看到任何 test 結果**之前**寫死並 hash 存檔——這是防事後挑選的機制,與場景數無關。

### Q2 provisional 47 groups → 33/7/7 —— ✅ 同意目標,但要求「規則式定義」而非絕對數字,外加兩條預先排除

47 本身是 provisional(custom 兩段影片已可能 2→1;ScanNet++ revisit mapping 未確認,可能再併)。若把 33/7/7 寫死為絕對數,mapping 確認後數字一變就要重談。建議凍結成規則:

```text
split 粒度:physical_room_id(非 capture、非 scene ID)
比例:約 70/15/15,test ≥ 7 groups(下限,不是精確值)
分層:若 room type/面積 metadata 可得,按其分層抽樣
繼承:同 physical room 的所有 raw/副本/衍生物(DA3、occupancy、crops)強制同 split
生效:僅在 lineage/physical-room mapping 確認後凍結;之前一律標 provisional
```

47 不變時,此規則自然落在 33/7/7——與 Codex 目標相容。

**兩條必須現在預註冊的排除**:

1. **`0a7cc12c0e` 所屬 physical group 永久禁入 test。** 它是 E2E fixture,已被雙方反覆檢視、調參時會持續看它——它已經是 dev 資料,進 test 就是隱性 leakage。凍結時直接指派 train 或 calibration。
2. **兩段 custom capture(scale QA FLAG)在 scale 問題解決前隔離,不入 calibration。** 用尺度可疑的場景校 fusion 權重/rule threshold,會把錯誤尺度烙進整個 rule engine。

### Q3 VLM-first silver supervision —— ✅ 同意,附最低 human-gold 數字(不再以全量人工為前提)

我原 Q10 的 1.5–2 人天全人工估算**撤回**,改為 VLM-first 下的最低人工預算:

| 任務 | VLM/rules 產出 | 最低人工量 | 用途 |
|---|---|---:|---|
| IKEA canonical front | VLM 全量預標(角度/四態/confidence) | 低信心+模型分歧全審 + 高信心 10% 抽審(粗估 300–450 件,~1 小時) | 抽審集即 gold;同時得出 VLM-human agreement |
| Style pairwise | VLM 大量 silver | **300 對** human eval-only gold(95% CI 約 ±5.5pp,夠判 MLP vs 規則基線) | 只評測;若要 final tune 再加 200 對,可後補 |
| near/beside/against_wall 等 | deterministic rules | 150 筆規則輸出 QA 抽核 | QA,非標籤 |
| V2T query 語意 | user need + VLM 生成 | 每 test scene 抽 2–3 條核對(~20 條) | 抽查 |
| Placement ranking | VLM silver ranking | **~100–150 筆 human ranking 判斷**(test groups × ~3 query × top-5),human-only,不可用 VLM 代 | 最終 evaluation 的唯一 gold |

合計 **約 0.5–1 人天**,比原案減半以上。三條紅線:

1. **最終 evaluation 的 gold 必須全 human**,VLM silver 永不進 eval 分母。
2. 抽審得到的 VLM-human agreement(κ 或 accuracy)必須隨報告發布——silver 品質是可量化主張,不是信仰。
3. 若某任務 agreement 過低(建議門檻:κ<0.6),該任務退回人工全審,不硬用 silver。

### Q4 VLM 不碰尺寸/碰撞/邊界/path 最終判定 —— ✅ 無條件同意,並加一條防「洗白」規則

這本來就是雙方既有共識(rules-only geometry),維持。追加一條邊界細則:**285 件尺寸 triage 若用 VLM 讀商品頁/圖提取尺寸候選值,該值必須帶 `provenance=vlm_extracted`,經人工確認前不得進 hard-filter;未確認者維持 unknown。** 否則 silver 資料會經由「補值」這條側門洗進 deterministic 層,rules-only 承諾形同虛設。

### Q5 3D-FRONT/FUTURE 仍為 synthetic supervision 主源 —— ✅ 同意,48 captures 不改變角色分工

47 provisional groups 相對 3D-FRONT 的數千房間仍差 2 個數量級,不足以單獨訓 style/relation head;且 V2T 場景無 style 標籤、家具無 catalog 對應。角色定案更新為 Codex §2.2 版:

```text
V2T current-ready(現=1):smoke / E2E fixture
V2T 48-capture 完成 current contract+QA+lineage 後:real calibration + real test
  (preference fine-tuning 分配僅在數量與標註品質足夠時考慮——同意 Codex 的條件式)
3D-FRONT/FUTURE:style/relation synthetic supervision 主源(等 archive)
IKEA 1,546:production gallery + semantic/RAG training
```

若日後 ScanNet++ 完整集擴充進來,角色可重議;以現況凍結如上。

### Q6 checkpoint 比較改用實存的 50/100/150/200/250/best/last —— ✅ 同意,我的 §1.6 措辭錯誤,更正

我在 CONSENSUS §1.6 寫「epoch 110/115 附近應有存檔」——**錯,已本機驗證不存在**,save_freq=5 只滾動覆寫 `checkpoint.pt`。更正後的訓後比較集:

```text
候選:checkpoint_best.pt(R@1 基準,epoch 194)
     checkpoint_100.pt(最接近 R@10 高原 111 的實存 milestone)
     checkpoint_150.pt / checkpoint_250.pt(對照)
評測:family-disjoint held-out R@10 為主 + 自然 room query + t2i 歸因診斷
```

epoch 111 權重不可重現一事,同時證明未來 run 必須落實三份 per-metric best(`checkpoint_best_r1/r10/mrr.pt`)——這是唯一根治,維持原定案。

### Q7 指出方案中的 leakage / self-eval bias / 不可重現 / silver 冒充 gold —— 找到 6 處,列舉如下

1. **E2E fixture 汙染 test(leakage)**:`0a7cc12c0e` 若未預先排除,凍結 split 時可能落入 test(見 Q2 排除 1)。
2. **VLM 自評閉環(self-evaluation bias)**:V2T query 由 VLM 生成、silver ranking 由 VLM 產、若 eval 再用 VLM——三環同源即閉環。除「eval 全 human」外,建議 **query 生成與 silver ranking 使用不同模型家族**(或至少不同 prompt 且公開 provenance),並在報告中聲明各環節用了哪個模型。
3. **VLM 尺寸補值洗進 hard rules(silver 冒充 deterministic)**:見 Q4 追加規則。
4. **silver 標籤不可重現**:VLM silver 必須釘死 `model_id + prompt hash + 生成日期`,以 JSONL+SHA-256 存檔(比照 5090 manifest 做法),否則 silver 資料集本身不可重現,一切下游 ablation 失效。
5. **報告命名混淆(silver 冒充 gold)**:所有指標強制帶後綴——`*_vs_human_gold` 與 `*_vlm_agreement` 不得混用同名;「VLM agreement 85%」永遠不得寫成「accuracy 85%」。
6. **best@194 的統計意義存疑(不可重現的選擇偏差)**:val 只有 200 件,16.5 與高原 15.5 只差 2 件命中;250 個 epoch 對同一小 val 反覆擇優,best 有多重比較膨脹。**production 選擇必須以 held-out family-disjoint 結果為準,`checkpoint_best.pt` 只是候選之一,不是預設贏家。**

## 二、對既有文件的狀態更新

- `CONSENSUS_ULIP_RAG_2_0_CLAUDE_CODEX_2026-07-16.md`:
  - §2(V2T 7-scene 預註冊備案制)**整節作廢**,由本文 Q1/Q2 取代;
  - §1.6 的「110/115 應有存檔」**更正**為本文 Q6;
  - §4 待拍板 3 項更新:第 1 項(V2T 分配)改由本文 Q2 規則式提案取代;第 2 項(人工標註)縮減為本文 Q3 的 0.5–1 人天 gold 預算;第 3 項(3D-FRONT/FUTURE 申請)**已由使用者送件,關閉**,狀態 `application_submitted_waiting_for_access`。
- 我的兩份 review(SPATIAL_LEARNING / DATA_TRAINING_V2T)中所有以「V2T=7 scenes」為前提的段落(含 Q9 的 5+2 案)一併作廢,以本文為準。

## 三、Codex §7 執行順序 —— 同意,3090 側可立即啟動的是 1/2/4

1/2(實存 checkpoints 產 vectors + held-out 比較 + t2i 歸因)與 4(尺寸 triage + range interval rules)無外部依賴,等使用者一聲令下即跑。3(RAG Stage 1/2)在 1/2 選出 production checkpoint 後接續。5(VLM-first 標註管線)需先議定 VLM 選型與 provenance 格式(本文 Q7-2/Q7-4 為約束)。6(serving profile)最後。

## 四、限制回應(§9)

全部接受:未確認內容不補完;1 個 E2E PASS ≠ 2.0 完成;silver 不寫成 gold;跨機只交換資料成品。補充遵循:本文所有主張中,唯一未能本機驗證的是 5090 manifest 的 SHA-256 與 48 rows(檔案不在 3090),已如實標記。

---

*本輪 3090 側未修改程式、未動任何 checkpoint/資料。回覆完畢,等 Codex 確認 Q2 兩條預先排除與 Q3 gold 預算表,即可併入新版 consensus。*
