# IKEA Pipeline 程式碼重點詳解

---

## Step 1 — `get_data.py`：爬蟲

### 重點函式：`fetch_missing_fields(url)`

這是整個爬蟲最複雜的部分。IKEA 官網部分資料（尺寸、評分、圖片列表）需要 JavaScript 渲染才看得到，`ikea_api` 拿不到，所以用 Playwright 實際開瀏覽器去抓。

```python
async def fetch_missing_fields(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)   # 無頭 Chrome
                context = await browser.new_context(user_agent="Mozilla/5.0")
                page = await context.new_page()
                await page.goto(url, timeout=60000)
                await page.wait_for_timeout(2000)   # 等 JS 渲染完
                html = await page.content()
                await browser.close()

                soup = BeautifulSoup(html, 'html.parser')
```

**為什麼要 `wait_for_timeout(2000)`？**
IKEA 頁面是 SPA（Single Page Application），`goto()` 之後 DOM 還沒完全渲染，要等 2 秒才抓得到動態生成的尺寸資料。

---

#### 尺寸抓取（最重要的部分）

```python
size_container = soup.find("div",
    class_="pip-product-dimensions__dimensions-container")

if size_container:
    measurements = size_container.find_all("p",
        class_="pip-product-dimensions__measurement-wrapper")
    for m in measurements:
        name_span = m.find("span",
            class_="pip-product-dimensions__measurement-name")
        if name_span:
            key = name_span.get_text(strip=True).replace(":", "")
            # next_sibling 是 span 後面的文字節點
            value = name_span.next_sibling
            if value:
                value = value.strip()
            dimensions[key] = value
    result["size_options"] = dimensions
```

**HTML 結構長這樣（IKEA 頁面）：**
```html
<div class="pip-product-dimensions__dimensions-container">
  <p class="pip-product-dimensions__measurement-wrapper">
    <span class="pip-product-dimensions__measurement-name">Width:</span>
    80 cm        ← 這是 next_sibling（文字節點）
  </p>
  <p class="pip-product-dimensions__measurement-wrapper">
    <span class="pip-product-dimensions__measurement-name">Height:</span>
    75 cm
  </p>
</div>
```

結果：`dimensions = {"Width": "80 cm", "Height": "75 cm"}`

> ⚠️ **這份尺寸資料只進了 MongoDB，`build_3d.py` 建資料集時沒有讀取它**，所以 ikea_data 的 json/ 裡沒有尺寸。`custom_data` 版本才補上（用使用者手動輸入）。

---

#### 容錯設計：retry + sleep

```python
for attempt in range(max_retries):      # 最多重試 3 次
    try:
        ...
    except Exception as e:
        await asyncio.sleep(random.uniform(2, 5))  # 失敗後等 2~5 秒再重試
```

主流程也有 sleep 避免被 IKEA 封鎖：
```python
time.sleep(random.uniform(1, 3))    # 每個商品之間等 1~3 秒
time.sleep(random.uniform(5, 10))   # 每個類別之間等 5~10 秒
```

---

### 重點函式：`fetch_ikea_data(keyword)`

```python
async def fetch_ikea_data(keyword):
    # 第一次查詢 limit=1，只是為了拿到 max（總商品數）
    endpoint = search.search(keyword, limit=1)
    result = await ikea_api.run_async(endpoint)
    max_products = result["searchResultPage"]["products"]["main"]["max"]

    # 第二次才真正查全部（上限 2000）
    max_limit = min(max_products, 2000)
    endpoint = search.search(keyword, limit=max_limit)
    result = await ikea_api.run_async(endpoint)

    products = result["searchResultPage"]["products"]["main"]["items"]
    return products
```

**為什麼要查兩次？**
`ikea_api` 的 search 需要指定 `limit`，但不知道某類別有幾件商品。所以先用 `limit=1` 拿到 `max` 值，再用正確的 limit 一次拿全部。

---

## Step 2 — `build_3d.py`：AI 生成 3D

### 重點邏輯：取樣策略 `sample_top_per_category()`

```python
def sample_top_per_category(col, top_n=50, min_count=30):
    selected = []
    # 統計每個類別有幾件
    category_counts = list(col.aggregate([
        {"$group": {"_id": "$main_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]))

    for cat_doc in category_counts:
        cat = cat_doc["_id"]
        count = cat_doc["count"]

        if cat is None or count < min_count:
            # 少於 30 件的類別整個跳過（資料太少沒意義）
            continue

        docs = list(col.find(
            {"main_type": cat,
             "clip_confidence": {"$type": "double"}},  # 確保欄位存在且是數字
        ).sort("clip_confidence", -1)                  # 從高到低排
         .limit(top_n))                                # 取前 50

        # 排除已經做過的（converted_3d == 'y'），支援中斷後續跑
        docs = [d for d in docs if d.get("converted_3d") != "y"]
        selected.extend(docs)
```

**`clip_confidence` 是什麼？**
在爬蟲之後，另一個腳本用 CLIP 計算每張商品主圖與 `"A product photo of a {category}"` 的相似度，分數越高代表這張圖越「標準」、越能代表該類別。這樣篩選出來的圖品質較好，TRELLIS 生成的 3D 也會比較準。

---

### 重點：TRELLIS 生成流程

```python
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

image = Image.open(img_path).convert("RGB")
outputs = pipeline.run(image, seed=1)
```

`outputs` 包含兩個東西：

| key | 型態 | 說明 |
|-----|------|------|
| `outputs["gaussian"]` | Gaussian Splat | 代表物體的 3D 高斯雲（用來渲染） |
| `outputs["mesh"]` | Mesh | 多邊形網格（有頂點、面） |

```python
# 存 PLY：把 Gaussian Splat 的資訊全部存進去
outputs["gaussian"][0].save_ply(ply_path)
# → 這個 PLY 包含 xyz + 顏色係數 + opacity + scale + 旋轉，不是純點雲

# 存 GLB：把 Gaussian 的外觀「烘焙」到 Mesh 上
glb = postprocessing_utils.to_glb(
    outputs["gaussian"][0],
    outputs["mesh"][0],
    simplify=0.95,     # 把網格面數減少到 5%，壓縮檔案大小
    texture_size=1024  # 貼圖解析度 1024x1024
)
glb.export(glb_path)
```

**`simplify=0.95` 的意思**：把原始 mesh 的面數縮減 95%，只保留 5%。TRELLIS 生成的 raw mesh 面數很多（可能幾十萬），存成 GLB 會很大，減面後通常品質夠用但檔案小很多。

---

### 重點：MongoDB 狀態標記（容錯）

```python
col.update_one({"_id": _id}, {"$set": {"converted_3d": "n"}})  # 開始前：進行中
# ... TRELLIS 生成 ...
col.update_one({"_id": _id}, {"$set": {"converted_3d": "y"}})  # 成功：完成
# 失敗時：
col.update_one({"_id": _id}, {"$set": {"converted_3d_error": str(e)}})
```

**為什麼這樣設計？**
TRELLIS 生成一個物件要幾十秒到幾分鐘，733 件跑完要好幾個小時。如果中途當機，下次重跑時 `sample_top_per_category()` 會過濾掉 `"y"` 的項目，只繼續未完成的。`"n"` 的項目（進行中被中斷的）會被重新執行。

---

## Step 3 — `ikea_ulip.py`：PyTorch Dataset

### 重點：`_PathResolver`（路徑解析器）

這個 class 解決一個實際問題：JSON 裡的路徑是相對路徑（`./ulip_output/ply/xxx.ply`），但這個相對路徑是在 `build_3d.py` 執行當下的目錄，換到不同機器或不同工作目錄就找不到了。

```python
def resolve(self, p: str):
    # 策略 1：直接是絕對路徑就用
    if os.path.isabs(p) and os.path.isfile(p):
        return p

    # 策略 2：./ulip_output/ply/xxx.ply
    #         → 把 "ulip_output/" 之後的部分取出來
    #         → 在幾個候選根目錄下找
    if norm.startswith("ulip_output/"):
        tail = norm.split("ulip_output/", 1)[1]  # e.g. "ply/xxx.ply"
        for root in self.ulip_roots:
            cand = root / tail
            if cand.is_file():
                return str(cand)

    # 策略 3：以 repo_root 為基準拼接
    cand = self.repo_root / p
    if cand.is_file():
        return str(cand)

    # 策略 4：/data/ulip_output → /ulip_output 替換
    if "/data/ulip_output" in p:
        tail = p.split("/data/ulip_output", 1)[1].lstrip("/")
        ...
```

**為什麼要這麼複雜？**
學長在不同時期改過目錄結構，有的 JSON 寫 `./ulip_output/ply/...`，有的寫 `/home/klooom/cheng/.../data/ulip_output/ply/...`。PathResolver 用多策略容錯，讓不同格式的路徑都能找到。

---

### 重點：`_load_pointcloud()`（點雲載入與處理）

```python
def _load_pointcloud(self, path: str) -> torch.Tensor:
    # 1. 讀取 PLY（trimesh 只取 vertices，忽略顏色/opacity 等 Gaussian 欄位）
    m = trimesh.load(path, process=False)
    pts = np.asarray(m.vertices, dtype=np.float32)  # shape: [N, 3+]

    # 2. 只取 xyz（Gaussian Splat 的 vertices 可能有額外欄位）
    if pts.shape[1] > 3:
        pts = pts[:, :3]

    # 3. 正規化到單位球
    pts = ulip_pc_normalize(pts)
    # 做法：移到重心，再除以最大距離
    # centroid = pts.mean(axis=0)
    # pts -= centroid
    # pts /= pts.max(np.linalg.norm(pts, axis=1))

    N, n = pts.shape[0], self.npoints  # n = 8192

    # 4. 採樣
    if N > n:
        pts = farthest_point_sample(pts, n)
        # FPS：每次選離已選點最遠的點，確保空間均勻分布
    else:
        # 點數不足：隨機重複補齊
        rep_idx = np.random.choice(N, n - N, replace=True)
        pts = np.concatenate([pts, pts[rep_idx]], axis=0)

    # 5. 訓練增強（只在 train subset 執行）
    if self.augment and self.subset == "train":
        pts_b = pts[None, ...]                       # 加 batch dim
        pts_b = random_point_dropout(pts_b)          # 隨機丟掉部分點
        pts_b = random_scale_point_cloud(pts_b)      # 隨機縮放
        pts_b = shift_point_cloud(pts_b)             # 隨機平移
        pts_b = rotate_perturbation_point_cloud(pts_b)  # 小角度隨機旋轉
        pts_b = rotate_point_cloud(pts_b)            # 隨機旋轉
        pts = pts_b.squeeze(0)

    return torch.from_numpy(pts).float()  # [8192, 3]
```

**為什麼用 FPS（Farthest Point Sampling）而不是隨機採樣？**
Gaussian Splat 的 centers 分布不均勻，物體某些部位（有細節的地方）Gaussian 密度高。如果隨機採樣，密度高的地方會被重複採到，細節少的地方採樣不足。FPS 確保每個點都盡量均勻分布，不會過度偏向某個區域。

---

### 重點：`__getitem__()` 回傳格式

```python
def __getitem__(self, idx):
    s = self.samples[idx]

    # 文字 token（LLaVA 描述 or IKEA 商品文字）
    caption = s["caption"]   # e.g. "a modern black sofa with wooden legs"
    tokenized = torch.stack([self.tokenizer(caption)])  # [1, 77]

    # 點雲
    pc = self._load_pointcloud(s["pointcloud_path"])  # [8192, 3]

    # 圖片：從 30 張渲染圖中隨機選一張（RENDER_PICK=random）
    if self.render_pick == "random":
        img_path = random.choice(s["render_paths"])
    img = self._load_image(img_path)  # [3, 224, 224]

    return (taxonomy_id, model_id, tokenized, pc, img)
    #        str          str        Tensor     Tensor  Tensor
```

**為什麼每次隨機選渲染圖？**
同一個物件有 30 張不同角度的渲染圖。訓練時每個 epoch 都隨機選不同角度，讓模型學會從任意視角都能認出這個物件，提升泛化能力。

---

## Step 4 — `main_ikea.py`：ULIP 訓練

### 重點：`train()` 函式（一個 epoch 的訓練）

```python
def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    model.train()

    for data_iter, inputs in enumerate(train_loader):
        # 依全局 iteration 更新 learning rate（cosine schedule）
        it = iters_per_epoch * epoch + optim_iter
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[it]

        # 從 Dataset 的 5 元組取出 3 個需要的 modality
        pc    = inputs[3]   # [B, 8192, 3]
        texts = inputs[2]   # [B, 1, 77]
        image = inputs[4]   # [B, 3, 224, 224]
        inputs = [pc, texts, image]

        # 送上 GPU
        inputs = [t.cuda(args.gpu, non_blocking=True) for t in inputs]

        # 混合精度前向傳播（節省 GPU 記憶體）
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            # outputs 包含：pc_embed, text_embed, image_embed 及 logit_scale
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq   # gradient accumulation 用

        # 反向傳播（scaler 處理 FP16 的梯度縮放）
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # 限制 logit_scale 在 [0, 4.6052]
        # exp(4.6052) ≈ 100，避免 softmax 溫度過低導致梯度消失
        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
```

**ULIP 的 loss 是什麼？**
三模態對比學習（InfoNCE loss）。一個 batch 裡，同一個物件的 pc_embed、text_embed、image_embed 應該互相接近，不同物件的應該互相遠離。學習的是**跨模態的共同 embedding 空間**。

---

### 重點：`eval_object_level_retrieval_6way()`（IKEA 專用評估）

標準 ULIP 用的是 zero-shot 分類（把點雲 embed 跟各類別文字 embed 比誰最近）。但 IKEA 資料太少（733 件），zero-shot 分類意義不大，所以改用**六向檢索**評估：

```python
# 六個方向：
# S2T：點雲（Shape）找最近的文字（Text）
# T2S：文字找最近的點雲
# S2I：點雲找最近的圖片（Image）
# I2S：圖片找最近的點雲
# T2I：文字找最近的圖片
# I2T：圖片找最近的文字

def _best_ranks(query_feats, gallery_feats):
    for i in range(N):
        # 計算 query[i] 與所有 gallery 的 cosine similarity
        sims = query_feats[i] @ gallery_feats.t()
        sorted_idx = torch.argsort(sims, descending=True)

        # 「同一物件」= 同 category + 同 model_id
        oid = object_ids[i]
        pos_idxs = obj2idxs[oid]   # 同物件的所有 index（可能有多個視角）

        # 找正樣本最早出現在第幾名
        best_rank = min(rank_of(p) for p in pos_idxs)
        ranks.append(best_rank)
```

**評估指標：**

```python
def _summarize(ranks):
    r1  = (r <= 1).float().mean()   # Recall@1：正樣本排第 1 的比例
    r5  = (r <= 5).float().mean()   # Recall@5
    r10 = (r <= 10).float().mean()  # Recall@10
    mrr = (1.0 / r).mean()          # Mean Reciprocal Rank
```

`acc1 = retr/s2t_r1 * 100`，即 PC→Text 的 Recall@1，這是 checkpoint 是否最佳的判斷標準。

---

## Step 5 — `app_ikea_retrieval.py`：檢索 API

### 重點：文字搜尋（`encode_text` + `search_text`）

```python
# 啟動時載入預計算的 PC embedding（不用每次重算）
PC_VECS = np.load(".../vectors_pc.npy")          # (733, 512)
PC_VECS = PC_VECS / np.linalg.norm(PC_VECS, ...)  # 正規化

@torch.no_grad()
def encode_text(query: str) -> np.ndarray:
    # 用訓練好的 ULIP text encoder 把查詢字串轉成向量
    tok = tokenizer([query])                      # [1, 77]
    feats = model.encode_text(tok)               # [1, 512]
    feats = F.normalize(feats, dim=-1)           # 正規化
    return feats.squeeze(0).numpy()              # (512,)

def search_text(query, top_k, category):
    q = encode_text(query)                        # (512,)

    # 如果指定類別，只在該類別內搜尋
    if category:
        idx = [i for i, m in enumerate(PC_META) if m["category"] == category]
        vecs = PC_VECS[idx]
    else:
        vecs = PC_VECS

    # 點積 = cosine similarity（因為兩邊都已正規化）
    scores = vecs @ q                             # (N,)

    # argpartition 比 argsort 快：只找最大的 top_k 個，不用全部排序
    top_idx = np.argpartition(-scores, k-1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]  # 再對這 k 個排序
```

**為什麼搜文字卻用 `vectors_pc.npy`（點雲向量）？**
ULIP 訓練的目標就是讓 text_embed 和 pc_embed 在同一個空間裡。搜尋時：query（文字）→ text_embed → 與 pc_embed 做 cosine similarity → 找最像的 3D 物件。本質上是「這段文字描述最像哪個 3D 家具」。

---

### 重點：上傳生成 3D（`/post/upload`）

這個 API 是給一般使用者用的——上傳一張家具照片，系統自動生成 3D 並存入 `custom_data`。

```python
@app.post("/post/upload")
async def post_upload(
    file: UploadFile,
    length_mm: float,   # ← 使用者必須手動輸入實體尺寸
    width_mm: float,
    height_mm: float,
    ...
):
```

**完整流程：**

```
1. 存暫存檔（staging 目錄）
        ↓
2. 圖片轉 JPEG
        ↓
3. 去背 rembg.remove()
   → TRELLIS 對去背後的圖效果較好，背景雜訊少
        ↓
4. CLIP 零樣本分類
   CATEGORIES = ["Sofa", "Dining Table", "Wardrobe", ...]
   PROMPTS = ["A product photo of a Sofa", ...]
   → 自動判斷類別（不用使用者選）
        ↓
5. TRELLIS 生成 GLB + raw PLY
        ↓
6. FPS 採樣 raw PLY → 8192 點的純 xyz PLY
   → 刪除 raw PLY（節省空間）
   → 這裡的 PLY 是「真正的點雲」，只有 xyz，不是 Gaussian Splat
        ↓
7. 組 JSON（寫入使用者輸入的 dimensions_mm）
        ↓
8. 一次性 os.replace() commit 到正式目錄
   → 用 replace 而不是 copy，原子操作，避免讀到一半的檔案
```

**為什麼這裡的 PLY 是「真點雲」，但 ikea_data 的 PLY 是 Gaussian Splat？**

`ikea_data`（`build_3d.py`）直接呼叫：
```python
outputs["gaussian"][0].save_ply(ply_path)
# → 存整個 Gaussian Splat（有 opacity/scale/rotation）
```

`custom_data`（`app_ikea_retrieval.py`）多做了一步：
```python
outputs["gaussian"][0].save_ply(raw_ply_tmp)   # 先存 Gaussian Splat
pts = load_xyz(raw_ply_tmp)                     # 只取 xyz
pts_8192 = sample_xyz(pts, npoints=8192)        # FPS 採樣
save_ply_xyz(pts_8192, ply_final)               # 存純 xyz
os.remove(raw_ply_tmp)                          # 刪掉原始 Gaussian PLY
```

---

## 關鍵設計決策總結

| 設計 | 為什麼這樣做 |
|------|------------|
| CLIP 篩選 top-50 | 確保進 TRELLIS 的圖片品質夠高，clip_confidence 低的圖通常是側拍、細節圖，生成效果差 |
| FPS 而非隨機採樣 | Gaussian centers 分布不均，FPS 保證空間均勻覆蓋 |
| `logit_scale.clamp_(0, 4.6052)` | 防止溫度參數 exp 後超過 100，避免 softmax 過尖銳導致訓練不穩定 |
| Hash split 而非隨機 split | 保證每次跑的 train/val 分割一致，可重現實驗 |
| 六向檢索評估 | 733 件太少做 zero-shot 分類沒意義，六向檢索更能反映 embedding 空間品質 |
| Staging + os.replace() | 確保檔案寫入原子性，API 不會返回寫到一半的資料 |
| MongoDB converted_3d 標記 | 讓 TRELLIS 生成腳本支援中斷後續跑，不會重複處理已完成的商品 |
