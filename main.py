"""
EcomImageChecker - 电商图像相似度检测服务
基于 Marqo 电商嵌入模型，支持多模态（图像+文本）相似度计算与聚类分析
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import similarity, cluster, health
from services.model_service import ModelService
from services.vector_store import VectorStore
from core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理：启动时加载模型，关闭时释放资源
    """
    # 启动：预加载模型到内存/GPU
    print(f"正在加载模型: {settings.MODEL_NAME}")
    await ModelService.initialize()
    print(f"模型加载完成，设备: {ModelService.get_instance().device}")
    print("正在初始化向量存储...")
    await VectorStore.initialize(ModelService._executor)
    print("向量存储初始化完成")
    yield
    # 关闭：清理资源
    print("正在释放向量存储资源...")
    VectorStore.cleanup()
    print("正在释放模型资源...")
    ModelService.cleanup()


app = FastAPI(
    title="EcomImageChecker",
    description="电商商品图像多模态相似度检测 & 聚类服务",
    version="1.0.0",
    lifespan=lifespan,
)

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, prefix="/health", tags=["健康检查"])
app.include_router(similarity.router, prefix="/api/v1/similarity", tags=["相似度检测"])
app.include_router(cluster.router, prefix="/api/v1/cluster", tags=["商品聚类"])
