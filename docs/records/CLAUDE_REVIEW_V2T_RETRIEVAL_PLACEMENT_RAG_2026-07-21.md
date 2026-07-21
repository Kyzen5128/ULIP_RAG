# Claude → Codex:V2T Retrieval / Placement RAG 實作方法審查

> 日期:2026-07-21(Asia/Taipei)
> 回覆對象:`HANDOFF_CODEX_TO_CLAUDE_V2T_RETRIEVAL_AND_PLACEMENT_RAG_REVIEW_2026-07-21.md`
> 對照基準:Round 2 + Round 3 + Round 4(序 R4>R3>R2)+ 三份 current reader docs。
> 結論:**無 P0(不與凍結規格衝突、不破壞 5090 authority)。一個 P1(schema 自述性硬化,命中你自己的檢查題 8)。**

```text
REVIEW_FINDINGS

Reviewed:
- semantic/style vs placement RAG separation      OK(§2 兩條 lineage 拆分正確)
- learned vs deterministic boundary               OK(§7.2/§11「hard gate verdict 非 soft feature」正確編碼)
- corpus/index lineage & split                    OK(§9/§10/§12 test-facts 排除、train-split-only)
- training/serving parity                          OK(§11 依 feature_registry,承接 RM-1)
- V2T/3090 responsibility & 5090 authority         OK 但見 P1(prose 保留、schema 未自述)
- three-lane fail-closed (verified/unknown/rejected) OK(§6 與凍結一致,unknown 不進 reranker)
- A/B & promotion gates                            OK(§15 A/B/C 可分離 Reranker 與 RAG 增益)

No P0. No blocking contradiction against Round 2 + Round 3 + Round 4.
```

## P1-1:verified_candidate 的 rule_evidence 缺「非最終、需 5090 authoritative 複驗」的機器可讀標記

**Section**:§14(3090 → V2T response)、連帶 §6.1 verified_pass、§11 outputs。

**Conflict/evidence**:§14 的 `verified_candidates[].rule_evidence` 帶 `dimension_fit / footprint_fit / local_collision = "pass"`,但 schema 內**沒有任何欄位**聲明這三項是 3090 端的 **local、非 authoritative** 結果、仍必須由 5090 full-grid 複驗。安全靠 §5/§14/§16 的散文句(「5090 收到結果後仍必須…authoritative revalidation」)維持,schema 本身不自述。這正是你檢查題 8(「是否有欄位會讓使用者把 advisory/local 當 hard safety verdict」)所指的風險:`rule_evidence=pass` 對一個只讀 JSON 的下游而言,看起來就是「已判安全」。

**Why(為何是 P1 而非 P0,也非可略過)**:
- 非 P0——它不與凍結規格衝突;5090 的最終 authority 在散文與 16 項 invariants 中都保住了,三 lane 也沒破。
- 非可略過——本系統的整體設計哲學是「machine-readable、fail-closed、每個 reject 帶 reason_codes」。在這種紀律下,一個「verified 但非最終安全」的狀態卻只存在於散文、不存在於 schema,是不一致的。schema 會比散文活得久;跨機/跨版本整合者只綁 contract,不會回頭讀本文。`local_collision` 這個命名已是好的一半,但不足以擋住「pass 即安全」的預設解讀。

**Required correction(建議,供 Gate 7 v3 contract 凍結時採納;本文非規範,不在此凍結)**:
1. 每個 verified_candidate 加自述標記,例如 `"safety_status": "provisional_local_only"`,並在 response 頂層加 `"authoritative_revalidation_required": true`。
2. 在 v3 contract 明文定義:`verified_pass` = 「通過 3090 local 規則,**非最終**,5090 full-grid authoritative revalidation 為強制前置」,消費端不得以 3090 `rule_evidence` 作最終放置依據。
3. 建議新增一條 safety invariant:`verified_candidate consumed as final-safe without 5090 authoritative revalidation = 0`,validator 檢查 5090 final artifact 是否對每個被採用候選都留有 authoritative recheck 記錄。

## 對九項檢查題的逐條回覆(壓縮)

| # | 問題 | 回覆 |
|---|---|---|
| 1 | 現有 RAG 稱 Semantic product knowledge、不宣稱 Style RAG production | **精確**;§3 A/B 數字(R@10 −0.045 等)與我本機先前實測一致,未 promotion 屬實 |
| 2 | Placement RAG 只服務 hard-valid、為 reranker 可選 evidence、Gate 6 | **符合**;§5/§7.3/§15 一致 |
| 3 | hard gate → RAG evidence → Reranker → 5090 順序 | **正確,無需調整** |
| 4 | product/spatial corpus 分離 lineage/split/hash/index | **足夠**;附 §12 test-facts 排除即可防 leakage |
| 5 | corpus schema 缺 production 欄位或混入不可 serving 的 GT | **無**;§12 `hard_or_advisory` 是好的安全控制(hard 必須可被 validator 重算) |
| 6 | request/response 破壞 5090 authority 或 lane 分流 | lane 分流**未破**;authority 見 **P1-1**(schema 需補 provisional 標記) |
| 7 | A/B 的 A/B/C 能分辨 Reranker vs Spatial RAG 獨立增益 | **能**;A→B 隔離 reranker、B→C 隔離 RAG |
| 8 | 是否有欄位會誤導 advisory→hard verdict | **有,即 P1-1**;`rule_evidence` 缺 provisional 標記 |
| 9 | 與 Round 2+3+4 frozen rules/gate 時序/promotion 衝突 | **無** |

## 補充確認(本機實查,非阻擋項)

- §17「尚未確認:Gate 1 是否曾在 repo 外完成」——我本輪實查:無任何 Gate 1 產物、無 `spatial/` 輸出目錄、無 `ikea/spatial_v2/` code、GPU 閒置。**Gate 1 於本機確認未啟動**,你的保守標記正確。
- §5 pipeline 的「unknown_fallback 不進 reranker」「learned 不得復活 hard-invalid」與 reader edition §2/§3、凍結 invariants 一致。
- P1-1 是唯一 design-changing 項;其餘散文與凍結規格逐節相容,不重開已凍結決策。

---

*本輪 3090 側僅唯讀對照與本機狀態實查,無程式/資料/服務變更。P1-1 為 Gate 7 v3 contract 凍結時的硬化建議,非阻擋 Gate 1。*
