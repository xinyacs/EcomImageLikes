"""
相似度检测路由
POST /api/v1/similarity/search
  - 给定一个 query 商品和 N 个候选商品，返回相似度最高的 top_k 个
"""

from fastapi import APIRouter, HTTPException
import numpy as np

from models.schemas import (
    SimilarityRequest,
    SimilarityResponse,
    SimilarityResult,
)
from services.embedding_service import EmbeddingService

router = APIRouter()


@router.post("/search", response_model=SimilarityResponse, summary="商品相似度检索")
async def search_similar(req: SimilarityRequest) -> SimilarityResponse:
    """
    给定一个检测对象（query）和 N 个候选商品（candidates），
    计算多模态余弦相似度后返回最相似的 top_k 个结果。

    **模式说明**
    - `image`  : 仅使用图像嵌入
    - `text`   : 仅使用文本嵌入（名称 + 类目 + 描述）
    - `fusion` : 图像与文本加权融合（推荐，默认权重 image=0.7）
    """
    try:
        # 1. 获取 query 向量（shape: (1, D)）
        query_emb = await EmbeddingService.get_embeddings(
            [req.query], mode=req.mode, image_weight=req.image_weight
        )

        # 2. 获取所有候选向量（shape: (N, D)）
        candidate_embs = await EmbeddingService.get_embeddings(
            req.candidates, mode=req.mode, image_weight=req.image_weight
        )

        # 3. 计算余弦相似度（向量已归一化，点积即余弦相似度）
        #    sim_scores shape: (N,)
        sim_scores: np.ndarray = (query_emb @ candidate_embs.T).flatten()

        # 4. 取 top_k（argsort 降序）
        top_k = min(req.top_k, len(req.candidates))
        top_indices = np.argsort(sim_scores)[::-1][:top_k]

        # 5. 构造响应
        results = [
            SimilarityResult(
                product=req.candidates[idx],
                score=float(sim_scores[idx]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(top_indices)
        ]

        return SimilarityResponse(
            query=req.query,
            results=results,
            total_candidates=len(req.candidates),
            mode=req.mode,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"相似度计算失败: {str(e)}")
