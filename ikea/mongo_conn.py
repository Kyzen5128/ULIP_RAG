# -*- coding: utf-8 -*-
"""統一 MongoDB 連線入口(2026-07-10 第二階段整理)。

背景:2026-06-05 資安事件後,mongo 重建為 127.0.0.1:27017 + --auth,
     舊的無認證 URI(mongodb://localhost:27017/)一律會被拒。
     本模組取代 ikea/ 各腳本內四散的硬編連線字串。

URI 取得順序:
  1. 環境變數 MONGO_URL 或 MONGO_URI
  2. ~/.ulip_mongo.env 內的 MONGO_URI=...(chmod 600,勿進 git)

用法:
    from mongo_conn import get_client
    col = get_client()["furniture_db"]["ikea_product"]
"""
import os
from pathlib import Path

ENV_FILE = Path.home() / ".ulip_mongo.env"


def get_mongo_uri(required: bool = True):
    """回傳 mongo 連線 URI;required=False 時找不到回傳 None 而非中止。"""
    uri = os.environ.get("MONGO_URL") or os.environ.get("MONGO_URI")
    if uri:
        return uri
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("MONGO_URI="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    if required:
        raise SystemExit(
            "找不到 MongoDB 連線資訊。請設定環境變數 MONGO_URL,"
            f"或確認 {ENV_FILE} 內有 MONGO_URI=...(參考 ~/mongo_info.md)"
        )
    return None


def get_client(**kwargs):
    """回傳已帶認證的 pymongo MongoClient(預設 5s server selection timeout)。"""
    from pymongo import MongoClient
    kwargs.setdefault("serverSelectionTimeoutMS", 5000)
    return MongoClient(get_mongo_uri(), **kwargs)
