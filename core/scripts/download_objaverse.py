import json
import os
import time
import signal
import multiprocessing
from tqdm import tqdm
import objaverse

DOWNLOAD_DIR = "/mnt/P300/data/objaverse"
CORPUS_PATH  = "/home/kyzen/ULIP_RAG/data/CAMERA_3D/datasets/semantic_corpus.jsonl"
BATCH_SIZE   = 10       # 每批下載幾個
TIMEOUT_SEC  = 60       # 單批最多等幾秒
PROCESSES    = 4        # 並行 process 數

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
objaverse.BASE_PATH = DOWNLOAD_DIR

# ---------- 讀取所有 obj_id ----------
ids = list({json.loads(l)['obj_id']
            for l in open(CORPUS_PATH) if l.strip()})
print(f"總共 {len(ids)} 個物件")

# ---------- 過濾已下載的 ----------
def is_downloaded(uid):
    glb_path = os.path.join(DOWNLOAD_DIR, "hf-objaverse-v1", "glbs",
                            uid[:2], f"{uid}.glb")
    return os.path.exists(glb_path)

remaining = [uid for uid in ids if not is_downloaded(uid)]
print(f"已下載：{len(ids) - len(remaining)}，待下載：{len(remaining)}")

# ---------- 下載單批（子 process，可 timeout kill）----------
def download_batch(batch):
    import sys, io
    objaverse.BASE_PATH = DOWNLOAD_DIR
    # 靜音 objaverse 的輸出
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        objaverse.load_objects(batch, download_processes=PROCESSES)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

# ---------- 主迴圈 ----------
batches = [remaining[i:i+BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
failed  = []

pbar = tqdm(total=len(remaining), desc="Downloading", unit="obj")

for i, batch in enumerate(batches):
    pbar.set_description(f"批次 {i+1}/{len(batches)} | 下載中")
    p = multiprocessing.Process(target=download_batch, args=(batch,))
    p.start()

    # 等待期間每秒更新 description 顯示已過秒數
    start = time.time()
    while p.is_alive():
        elapsed = int(time.time() - start)
        pbar.set_description(f"批次 {i+1}/{len(batches)} | {elapsed}s")
        time.sleep(1)
        if elapsed >= TIMEOUT_SEC:
            pbar.set_description(f"批次 {i+1}/{len(batches)} | TIMEOUT，跳過")
            p.terminate()
            p.join()
            failed.extend(batch)
            break

    if p.is_alive() is False and p.exitcode != 0:
        failed.extend(batch)

    done = sum(1 for uid in batch if is_downloaded(uid))
    pbar.update(done)
    pbar.set_postfix(failed=len(failed), done=sum(1 for uid in ids if is_downloaded(uid)))

pbar.close()

# ---------- 結果報告 ----------
total_done = sum(1 for uid in ids if is_downloaded(uid))
print(f"\n完成：{total_done}/{len(ids)}")
if failed:
    fail_path = os.path.join(DOWNLOAD_DIR, "failed_uids.txt")
    with open(fail_path, "w") as f:
        f.write("\n".join(failed))
    print(f"失敗 {len(failed)} 個，已記錄到 {fail_path}")
