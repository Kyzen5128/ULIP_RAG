"""
ULIP ShapeNet Triplets 下載腳本
- 逐個檔案下載，失敗自動重試
- 已完成的檔案自動跳過（比對遠端大小）
- 用 wget 下載，支持斷點續傳
"""
import os
import subprocess
import time
from huggingface_hub import HfApi

# ========== 設定 ==========
REPO_ID = "SFXX/ulip"
LOCAL_DIR = "/mnt/P300/data/ULIP"
PATTERN = "ULIP_Shapenet_Triplets/"
MAX_RETRIES = 5           # 每個檔案最多重試次數
RETRY_DELAY = 10          # 重試間隔（秒）
# ==========================

# 全域 HfApi 實例，避免重複初始化
api = HfApi()

def get_file_list():
    """從 HuggingFace 取得需要下載的檔案列表（含大小）"""
    all_items = api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)
    files = []
    for item in all_items:
        # 只要檔案，不要目錄
        if hasattr(item, 'path') and hasattr(item, 'size') and item.path.startswith(PATTERN):
            files.append((item.path, item.size))
    return sorted(files, key=lambda x: x[0])

def download_file(filepath):
    """用 wget 下載單個檔案，支持斷點續傳"""
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{filepath}"
    local_path = os.path.join(LOCAL_DIR, filepath)
    
    # 建立目錄
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # 取得 token（若需要認證）
    token = api.token
    
    # 用 wget 下載（-c 斷點續傳，--tries=1 讓外層控制重試）
    cmd = [
        "wget", "-c", "-q", "--show-progress",
        "--timeout=30",
        "--tries=1",
        "-O", local_path,
        url
    ]
    
    # 如果有 token，加上認證 header
    if token:
        cmd[1:1] = ["--header", f"Authorization: Bearer {token}"]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    print("正在取得檔案列表...")
    files = get_file_list()
    print(f"共 {len(files)} 個檔案需要下載\n")
    
    completed = []
    failed = []
    skipped = []
    
    for i, (filepath, remote_size) in enumerate(files, 1):
        local_path = os.path.join(LOCAL_DIR, filepath)
        filename = os.path.basename(filepath)
        
        # 比對遠端大小，確認是否已完整下載
        if os.path.exists(local_path):
            local_size = os.path.getsize(local_path)
            if remote_size and local_size == remote_size:
                print(f"[{i}/{len(files)}] ⏭️  已完整，跳過: {filename} ({local_size/1e9:.2f}GB)")
                skipped.append(filepath)
                continue
            elif local_size > 0:
                rs = f"{remote_size/1e9:.2f}GB" if remote_size else "未知"
                print(f"[{i}/{len(files)}] 🔄 不完整 ({local_size/1e9:.2f}/{rs})，續傳: {filename}")
        
        size_str = f"{remote_size/1e9:.2f}GB" if remote_size else "未知大小"
        print(f"[{i}/{len(files)}] ⬇️  下載中: {filename} ({size_str})")
        
        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            if download_file(filepath):
                # 驗證下載完整性
                if os.path.exists(local_path):
                    dl_size = os.path.getsize(local_path)
                    if remote_size is None or dl_size == remote_size:
                        # remote_size 未知就信任 wget，或大小吻合
                        success = True
                        print(f"  ✅ 完成! ({dl_size/1e9:.2f}GB)")
                        completed.append(filepath)
                        break
                    else:
                        print(f"  ⚠️  大小不符 (本地:{dl_size} vs 遠端:{remote_size})，重試...")
                else:
                    print(f"  ⚠️  檔案不存在，重試...")
            else:
                print(f"  ❌ 第 {attempt}/{MAX_RETRIES} 次失敗，{RETRY_DELAY}秒後重試...")
            time.sleep(RETRY_DELAY)
        
        if not success:
            print(f"  ❌ 放棄: {filename}")
            failed.append(filepath)
    
    # 總結
    print("\n" + "="*60)
    print(f"下載完成！")
    print(f"  ✅ 新下載: {len(completed)} 個")
    print(f"  ⏭️  已存在: {len(skipped)} 個")
    print(f"  ❌ 失敗:   {len(failed)} 個")
    
    if failed:
        print(f"\n失敗的檔案:")
        for f in failed:
            print(f"  - {f}")
        print(f"\n重新執行此腳本即可重試失敗的檔案")

if __name__ == "__main__":
    main()