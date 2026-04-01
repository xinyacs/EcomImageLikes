"""
健康检查与缓存管理路由
GET  /health          - 服务存活检查
GET  /health/cache    - 缓存统计信息
DELETE /health/cache  - 清空缓存
"""

from fastapi import APIRouter
from services.cache_service import embedding_cache
from services.model_service import ModelService
from models.schemas import CacheStatsResponse

router = APIRouter()


@router.get("/", summary="服务健康检查")
async def health_check():
    """返回服务状态与模型设备信息"""
    try:
        svc = ModelService.get_instance()
        return {"status": "ok", "device": svc.device}
    except RuntimeError:
        return {"status": "initializing", "device": None}


@router.get("/cache", response_model=CacheStatsResponse, summary="缓存统计")
async def cache_stats():
    """返回向量缓存的命中率与条目数等统计信息"""
    svc = ModelService.get_instance()
    stats = embedding_cache.stats()
    return CacheStatsResponse(
        total_entries=stats["total_entries"],
        max_size=stats["max_size"],
        hit_rate=stats["hit_rate"],
        device=svc.device,
    )


@router.delete("/cache", summary="清空缓存")
async def clear_cache():
    """手动清空所有已缓存的向量（调试/维护用）"""
    embedding_cache.clear()
    return {"status": "ok", "message": "缓存已清空"}
