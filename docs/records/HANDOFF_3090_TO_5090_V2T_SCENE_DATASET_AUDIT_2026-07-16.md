# 3090 → 5090：V2T Scene Dataset 唯讀盤點請求

## 背景校正

3090 在舊文件中讀到的「7 場景」，實際上是已完成 128-frame 全量驗證的子集，不應當成 V2T 的全部場景數。請 5090 依當前機器上的真實檔案唯讀盤點，不要依舊 handoff 推測。

## P0：請回覆

1. V2T 目前共有多少個「不同實體房間」？請區分：
   - raw scenes
   - 已跑完 pipeline
   - 已通過 QA
   - 可立即供 ULIP_RAG 2.0 使用
2. 請提供 ordered scene manifest，至少包含：
   - `scene_id`
   - `physical_room_id`
   - `source_dataset`
   - `room_type`
   - `capture/video_id`
   - pipeline/QA status
   - 可用輸出檔案路徑
3. 同一實體房間是否有多段影片、多次重建或多個 `scene_id`？若有，split 必須依 `physical_room_id` 分組，避免洩漏。
4. 哪些場景具備以下完整輸出？請統計數量與 coverage：
   - occupancy grid
   - room geometry / walls
   - doors / windows
   - existing furniture instances + bbox/polygon
   - candidate spots
   - anchor relationships
   - walkability/access terminals
   - `retrieval_request.json`
5. 每個場景可以產生多少個真實 query？目前是已有 query，還是需要之後產生／人工撰寫？
6. 場景是來自 ScanNet++、自有影片或其他來源？請分別列出數量及使用限制。
7. 未來還會增加多少場景？是固定資料集，還是 pipeline 可批次產生更多？
8. 請 5090 依真實數量提出 train/calibration/test split 建議，並硬性保證同房間、同影片、同重建衍生物不跨 split。

## 回傳要求

- 請明確說明舊文件的 7 scenes 只是 QA subset，還是有其他定義。
- 數量若無法從當前檔案驗證，標記「尚未確認」。
- 請回傳 manifest path、row count 與 SHA-256。
- 本輪只做唯讀盤點，不要重跑 pipeline、不要修改 V2T 資料或程式。
