"""
全局配置项
支持通过环境变量覆盖默认值
"""

import os
from typing import Optional


class Settings:
    # ── 模型配置 ──────────────────────────────────────────────
    MODEL_NAME: str = os.getenv(
        "MODEL_NAME", "Marqo/marqo-ecommerce-embeddings-L"
    )
    # 备选轻量模型：Marqo/marqo-ecommerce-embeddings-B

    # ── 设备配置 ──────────────────────────────────────────────
    # "cuda" / "mps" / "cpu"；留空则自动选择
    DEVICE: Optional[str] = os.getenv("DEVICE", None)

    # ── 批处理配置 ────────────────────────────────────────────
    # 单次推理批大小，根据 GPU 显存调整
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))

    # ── ChromaDB 向量存储配置 ─────────────────────────────────
    # 持久化目录；设为 ":memory:" 则使用纯内存模式（重启丢失）
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", ":memory:")
    # Collection 名称，不同业务场景可隔离
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "ecom_products")
    # 距离函数："cosine" | "l2" | "ip"（内积）
    CHROMA_DISTANCE_FN: str = os.getenv("CHROMA_DISTANCE_FN", "cosine")

    # ── 内存一级缓存（L1）配置 ────────────────────────────────
    # 在 ChromaDB 之上叠一层 in-process LRU，热点商品零 IO 命中
    # 设为 0 可禁用 L1 缓存，完全依赖 ChromaDB
    L1_CACHE_SIZE: int = int(os.getenv("L1_CACHE_SIZE", "2000"))
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", str(L1_CACHE_SIZE)))
    # L1 TTL（秒），0 = 永不过期
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))

    # ── 聚类配置 ──────────────────────────────────────────────
    # 判定"相似"的余弦相似度阈值（0~1）
    CLUSTER_SIMILARITY_THRESHOLD: float = float(
        os.getenv("CLUSTER_SIMILARITY_THRESHOLD", "0.85")
    )

    # ── 图像配置 ──────────────────────────────────────────────
    # 下载远程图片的超时秒数
    IMAGE_DOWNLOAD_TIMEOUT: int = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "10"))


settings = Settings()