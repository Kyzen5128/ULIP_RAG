from huggingface_hub import snapshot_download
from huggingface_hub import whoami
print(whoami())

# 啟用進度條顯示
snapshot_download(
    repo_id="SFXX/ulip",
    repo_type="dataset",
    local_dir="ulip_full",
    resume_download=True,
    allow_patterns=[
        # "ULIP-1/initialize_models/*",
        # "ULIP-1/modelnet40_normal_resampled/*", 
        # "ULIP-1/pretrained_models/*",
        "ULIP_Shapenet_Triplets/*"
    ],
    # 這會顯示下載進度
    token=None  # 如果需要認證就填入 token
)