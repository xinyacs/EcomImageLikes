"""
EmbeddingService - 向量化编排服务（ChromaDB 版）
负责：L1/L2 缓存查询 → 批量推理 → 向量写入 → 融合权重计算
"""

from __future__ import annotations

import asyncio
from typing import List, Dict

import numpy as np

from models.schemas import Product
from services.model_service import ModelService
from services.vector_store import VectorStore


class EmbeddingService:
    """
    向量化编排服务（无状态）

    核心流程：
      1. 批量查询 VectorStore（L1 LRU → L2 ChromaDB）
      2. 对未命中商品并发异步下载图像
      3. 批量推理得到向量
      4. 写回 VectorStore（L1 + L2 同步写入）
      5. 按 mode 融合图像/文本向量，返回归一化矩阵
    """

    @classmethod
    async def get_embeddings(
        cls,
        products: List[Product],
        mode: str = "fusion",
        image_weight: float = 0.7,
    ) -> np.ndarray:
        """
        批量获取商品的融合向量

        Args:
            products     : 商品列表
            mode         : "image" | "text" | "fusion"
            image_weight : fusion 模式下图像权重

        Returns:
            embeddings: shape (N, D)，L2 归一化
        """
        store = VectorStore.get_instance()
        n = len(products)
        img_embs: Dict[int, np.ndarray] = {}
        txt_embs: Dict[int, np.ndarray] = {}

        # ── Step 1: 批量查询 VectorStore（L1 → L2） ──────────
        pid_to_idx: Dict[str, int] = {
            p.product_id: i for i, p in enumerate(products)
        }
        cached = await store.batch_get(list(pid_to_idx.keys()))

        uncached_indices: List[int] = []
        for i, product in enumerate(products):
            hit = cached.get(product.product_id)
            if hit is not None:
                img_embs[i], txt_embs[i] = hit
            else:
                uncached_indices.append(i)

        # ── Step 2 ~ 4: 推理未命中商品并写入存储 ────────────
        if uncached_indices:
            model_svc = ModelService.get_instance()
            uncached_products = [products[i] for i in uncached_indices]

            # 并发异步下载图像，降低网络 IO 等待
            images = await asyncio.gather(
                *[model_svc.load_image(p.image_url) for p in uncached_products]
            )
            texts = [p.get_text() for p in uncached_products]

            # 批量推理（offload 到线程池）
            batch_img_embs, batch_txt_embs = await model_svc.encode_products(
                images, texts
            )

            # 并发写入 VectorStore（upsert L1 + L2）
            write_tasks = []
            for local_idx, global_idx in enumerate(uncached_indices):
                img_e = batch_img_embs[local_idx]
                txt_e = batch_txt_embs[local_idx]
                product = uncached_products[local_idx]

                img_embs[global_idx] = img_e
                txt_embs[global_idx] = txt_e

                # 将商品元数据一起存入 ChromaDB，便于后续过滤查询
                meta = {
                    "name": product.name,
                    "category": product.category or "",
                    "description": (product.description or "")[:200],
                }
                write_tasks.append(
                    store.set(product.product_id, img_e, txt_e, meta)
                )

            # 所有写入并发执行
            await asyncio.gather(*write_tasks)

        # ── Step 5: 重组矩阵并融合 ───────────────────────────
        img_matrix = np.stack([img_embs[i] for i in range(n)], axis=0)  # (N, D)
        txt_matrix = np.stack([txt_embs[i] for i in range(n)], axis=0)  # (N, D)

        return cls._fuse(img_matrix, txt_matrix, mode, image_weight)

    # ── 内部工具 ──────────────────────────────────────────────

    @staticmethod
    def _fuse(
        img_embs: np.ndarray,
        txt_embs: np.ndarray,
        mode: str,
        image_weight: float,
    ) -> np.ndarray:
        """
        按模式融合图像和文本向量，融合后 L2 重归一化

        Args:
            img_embs     : (N, D)
            txt_embs     : (N, D)
            mode         : "image" | "text" | "fusion"
            image_weight : fusion 权重（文本权重 = 1 - image_weight）

        Returns:
            fused: (N, D)，L2 归一化
        """
        if mode == "image":
            fused = img_embs
        elif mode == "text":
            fused = txt_embs
        else:  # fusion
            fused = image_weight * img_embs + (1.0 - image_weight) * txt_embs

        norms = np.linalg.norm(fused, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return fused / norms

    @staticmethod
    def cosine_similarity_matrix(
        query_embs: np.ndarray,
        candidate_embs: np.ndarray,
    ) -> np.ndarray:
        """
        余弦相似度矩阵（向量已归一化，直接点积）

        Returns:
            (Q, C) 矩阵，值域 [-1, 1]
        """
        return query_embs @ candidate_embs.T