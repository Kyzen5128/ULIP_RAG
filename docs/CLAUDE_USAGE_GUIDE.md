# Claude Code 使用技巧指南

> 給 Kyzen 的個人使用筆記
> 更新日期：2026-03-30

---

## 1. 基本觀念

Claude Code 是**有 context window 上限**的 AI。
每次對話的文字（你說的 + 我說的 + 讀的檔案）都會占用 context，滿了之後：
- 舊的對話會被壓縮或遺忘
- 我可能開始忘記之前討論的細節
- 回答品質會下降

**解法**：定期 `/compact` + 善用記憶系統。

---

## 2. 必做習慣：定期 `/compact`

### 什麼時候要 /compact？
- 每完成一個主題或任務後
- 感覺我開始「忘記」之前說的事情
- 對話很長（超過 30~50 輪）
- 貼了大量程式碼或檔案內容後

### 怎麼用？
```
/compact
```
這個指令會把對話壓縮成摘要，騰出 context 空間，同時保留重要結論。

### 重要：/compact 之後
壓縮後我仍會記得主要結論，但**細節可能消失**。
建議在 /compact 前說：
> "幫我把這次對話的結論整理一下再 compact"

---

## 3. 記憶系統（跨對話持久記憶）

Claude Code 有一個自動記憶系統，存在：
```
/home/kyzen/.claude/projects/-home-kyzen-ULIP-RAG/memory/
```

### 記憶的種類
| 類型 | 用途 |
|------|------|
| `user_*.md` | 你是誰、你的研究背景、溝通偏好 |
| `feedback_*.md` | 你糾正過我的習慣（避免重蹈覆轍）|
| `project_*.md` | 研究進度、決策、目標 |
| `reference_*.md` | 重要資源的位置（檔案路徑、伺服器等）|

### 主動叫我記住某件事
直接說：
> "記住：再跑指令之前請詢問我要不要自動跑還是給我指令我來跑，因為有時候我需要看到輸出"
> "記住：這個專案用 Objaverse 不用 ShapeNet"

### 叫我回憶
> "你記得我們之前決定 corpus 格式是什麼嗎？"

---

## 4. 指令備忘錄

| 指令 | 功能 |
|------|------|
| `/compact` | 壓縮對話，騰出 context |
| `/clear` | 清空對話（完全重置，比 compact 更激進）|
| `/help` | 列出所有指令 |
| `/fast` | 切換快速模式（同樣模型，輸出更快）|

---

## 5. 讓我發揮最好的方式

### 給我明確的任務
壞的問法：「幫我改一下 retriever」
好的問法：「幫我在 `retriever.py` 的 `retrieve()` 函數裡加上空間過濾，條件是 W < query_w AND D < query_d」

### 我會自動讀檔案，但你可以引導我
> "先看 models/retriever.py 再回答我"

### 貼錯誤訊息時
貼完整的錯誤（traceback），不要只貼最後一行。

### 跨 session 繼續工作
新對話開始時說：
> "繼續上次的工作，幫我看一下 memory 和 RESEARCH_GUIDE.md"

---

## 6. 這個專案的 Workflow

```
開始新 session
    ↓
說「繼續上次工作」→ 我會讀 MEMORY.md + RESEARCH_GUIDE.md
    ↓
討論 / 寫程式 / 看檔案
    ↓
[每完成一個里程碑]
    ↓
/compact（壓縮，保留空間）
    ↓
繼續
```

---

## 7. 我容易犯的錯（你之前糾正過我的）

這些是你已經告訴我要避免的：

1. **擅自決定用哪個資料集** — 永遠用 Objaverse，不是 ShapeNet
2. **自己幫你跑指令** — 你說「給我指令我來跑」，我應該給指令不是自己執行
3. **output 太冗長** — 特別是 objaverse 的 verbose 輸出要壓制
4. **使用 `conda run`** — 會 buffer 輸出導致沒有即時回饋，改用 `conda activate && python`

---

## 8. 快捷參考：本專案常用指令

```bash
# 啟動 conda 環境（每次新 terminal 都要）
conda activate ulip_rag

# 訓練 Stage 1（100 epochs）
cd ~/ULIP_RAG/ulip_rag
python main.py \
  --model ULIP_PointBERT \
  --use_rag_adapter \
  --training_strategy staged_1 \
  --npoints 8192 --batch-size 64 --epochs 100 \
  --rag_corpus_dir ./rag_corpus \
  --output-dir ./outputs/baseline_stage1

# 下載 Objaverse 家具模型
python scripts/download_objaverse.py

# 查看 GPU 使用狀況
watch -n 1 nvidia-smi

# 查看下載了多少 GLB
find /mnt/P300/data/objaverse -name "*.glb" | wc -l
```

---

## 9. 如果我表現很差怎麼辦

- **重複說一樣的廢話** → `/clear` 重新開始，把任務重新描述清楚
- **不記得之前的決定** → 叫我讀 `MEMORY.md` 和 `RESEARCH_GUIDE.md`
- **答案很長但沒用** → 說「簡短回答，直接給結論」
- **出 bug 繞不出去** → 說「你陷入迴圈了，退一步重新分析問題」

---

## 10. 回報問題

如果 Claude Code 本身有 bug：
https://github.com/anthropics/claude-code/issues
