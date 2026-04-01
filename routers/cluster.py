"""
商品聚类路由
POST /api/v1/cluster/group
  - 给定 N 个商品，使用相似度阈值进行层次聚类，
    区分出「相似商品组」和「独立商品」
"""

from fastapi import APIRouter, HTTPException
from typing import List, Set
import numpy as np

from models.schemas import (
    ClusterRequest,
    ClusterResponse,
    Product,
    ProductGroup,
)
from services.embedding_service import EmbeddingService
from core.config import settings

router = APIRouter()


@router.post("/group", response_model=ClusterResponse, summary="商品相似度聚类")
async def cluster_products(req: ClusterRequest) -> ClusterResponse:
    """
    对输入的 N 个商品进行多模态相似度聚类。

    **算法说明**（贪心连通分量）
    1. 计算所有商品两两之间的余弦相似度矩阵（对称矩阵，O(N²)）
    2. 构建邻接关系：相似度 >= threshold 则两个商品视为同簇
    3. 使用 Union-Find 合并连通分量
    4. 簇内商品数 >= min_cluster_size 归为「相似组」，否则为「独立商品」
    """
    # 确定相似度阈值
    threshold = req.threshold if req.threshold is not None else settings.CLUSTER_SIMILARITY_THRESHOLD

    try:
        # ── Step 1: 批量获取所有商品向量 ─────────────────────
        all_embs = await EmbeddingService.get_embeddings(
            req.products,
            mode=req.mode,
            image_weight=req.image_weight,
        )  # shape (N, D)

        # ── Step 2: 计算全量相似度矩阵 ───────────────────────
        # 向量已归一化，矩阵乘法等价于余弦相似度
        # sim_matrix[i, j] = cosine_similarity(product_i, product_j)
        sim_matrix: np.ndarray = all_embs @ all_embs.T  # (N, N)

        # ── Step 3: Union-Find 聚类 ───────────────────────────
        n = len(req.products)
        parent = list(range(n))

        def find(x: int) -> int:
            """路径压缩"""
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    union(i, j)

        # ── Step 4: 收集各连通分量 ────────────────────────────
        from collections import defaultdict
        components: dict[int, List[int]] = defaultdict(list)
        for idx in range(n):
            components[find(idx)].append(idx)

        # ── Step 5: 区分相似组与独立商品 ──────────────────────
        similar_groups: List[ProductGroup] = []
        unique_products: List[Product] = []

        for group_id, (root, members) in enumerate(components.items()):
            if len(members) >= req.min_cluster_size:
                # 计算簇内平均相似度（上三角均值）
                avg_sim = _compute_avg_similarity(sim_matrix, members)
                similar_groups.append(
                    ProductGroup(
                        group_id=group_id,
                        products=[req.products[i] for i in members],
                        avg_similarity=round(avg_sim, 4),
                    )
                )
            else:
                unique_products.extend([req.products[i] for i in members])

        return ClusterResponse(
            similar_groups=similar_groups,
            unique_products=unique_products,
            total_products=n,
            threshold=threshold,
            mode=req.mode,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聚类计算失败: {str(e)}")


def _compute_avg_similarity(sim_matrix: np.ndarray, indices: List[int]) -> float:
    """
    计算指定商品集合内的平均两两相似度（排除自身 sim=1.0）
    
    Args:
        sim_matrix : 全量 (N, N) 相似度矩阵
        indices    : 当前簇的商品下标列表

    Returns:
        平均相似度（若簇只有 1 个元素则返回 1.0）
    """
    if len(indices) < 2:
        return 1.0
    scores = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            scores.append(sim_matrix[indices[i], indices[j]])
    return float(np.mean(scores))
