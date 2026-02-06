from huggingface_hub import snapshot_download
from huggingface_hub import whoami
print(whoami())
snapshot_download(
    repo_id="SFXX/ulip",
    repo_type="dataset",
    local_dir="/mnt/P300",
    allow_patterns="ULIP-1/*",     # 只下載 ULIP-1 資料夾
    resume_download=True,          # 斷線重連
    tqdm_class=None                # 想看進度條就拿掉
)