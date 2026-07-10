#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
presample_ply.py — 把 IKEA 的 gaussian-splat PLY（每顆 33~67 萬點）用 GPU FPS
一次性預採樣成 8192 點的純 xyz PLY，存到 <src 同層>/ply_8192/。

為什麼需要：訓練 dataloader（ikea_ulip.py）原本在 CPU worker 內對原始大 PLY 跑
numpy FPS，每筆 20-40 秒 → 正式訓練一個 epoch 要數小時。預採樣後 N==npoints，
dataloader 直接略過 FPS，epoch 進入分鐘級。

- 只取 xyz、不 normalize（normalize 留給 dataset 在載入時做；FPS 對平移/縮放不變，
  先採樣後 normalize 與原流程等價）。
- 可中斷續跑（已存在的輸出檔會跳過）。
- 用法：
    conda activate ulip && cd /home/kyzen/ULIP_RAG/core
    PYTHONPATH=/home/kyzen/ULIP_RAG/core python ../ikea/presample_ply.py \
        --src_dir /mnt/P300/data/ikea_data/ply --npoints 8192
"""
import os, glob, argparse, time
import numpy as np
import torch
import trimesh


def load_xyz(path: str) -> np.ndarray:
    m = trimesh.load(path, process=False)
    if hasattr(m, "vertices"):
        pts = np.asarray(m.vertices, dtype=np.float32)
    elif isinstance(m, trimesh.Scene):
        buf = [np.asarray(g.vertices, dtype=np.float32)
               for g in m.geometry.values() if hasattr(g, "vertices")]
        pts = np.vstack(buf) if buf else np.zeros((0, 3), np.float32)
    else:
        pts = np.zeros((0, 3), np.float32)
    return pts[:, :3] if pts.ndim == 2 and pts.shape[1] >= 3 else np.zeros((0, 3), np.float32)


def gpu_fps(pts: np.ndarray, npoints: int, device) -> np.ndarray:
    from pointnet2_ops import pointnet2_utils as p2
    xyz = torch.from_numpy(np.ascontiguousarray(pts)).float().unsqueeze(0).to(device)
    idx = p2.furthest_point_sample(xyz, npoints)          # [1, npoints] int32
    return pts[idx[0].long().cpu().numpy()]


def save_ply_xyz(pts: np.ndarray, out_path: str):
    trimesh.Trimesh(vertices=pts, faces=None, process=False).export(
        out_path, file_type="ply", encoding="binary_little_endian")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="/mnt/P300/data/ikea_data/ply")
    ap.add_argument("--out_dir", default=None, help="預設為 <src_dir 同層>/ply_8192")
    ap.add_argument("--npoints", type=int, default=8192)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.src_dir.rstrip("/")), "ply_8192")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = sorted(glob.glob(os.path.join(args.src_dir, "*.ply")))
    print(f"src={args.src_dir} ({len(files)} ply) -> out={out_dir} | device={device}")

    done = skip = fail = 0
    t0 = time.time()
    for i, f in enumerate(files):
        out = os.path.join(out_dir, os.path.basename(f))
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            skip += 1
            continue
        try:
            pts = load_xyz(f)
            if pts.shape[0] == 0:
                print(f"[EMPTY] {f}"); fail += 1; continue
            if pts.shape[0] > args.npoints:
                pts = gpu_fps(pts, args.npoints, device)
            # N <= npoints：原樣保留（dataset 端會 pad；且 <8192 的樣本本來就被跳過）
            save_ply_xyz(pts.astype(np.float32), out)
            done += 1
        except Exception as e:
            print(f"[FAIL] {f}: {type(e).__name__}: {e}"); fail += 1
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(files)}  done={done} skip={skip} fail={fail}  "
                  f"({el:.0f}s, {el/max(done,1):.1f}s/item)")

    print(f"FINISHED  done={done} skip={skip} fail={fail}  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
