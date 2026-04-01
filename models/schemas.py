"""
数据模型定义
所有 API 的请求/响应 Pydantic Schema
"""

from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field, model_validator
import hashlib
import json


# ─────────────────────────────────────────────
# 核心实体：商品（Product）
# ─────────────────────────────────────────────

class Product(BaseModel):
    """
    商品实体
    
    Fields:
        name        : 商品名称（必填），用于文本嵌入
        image_url   : 商品图片 URL 或 Base64 字符串（必填），用于图像嵌入
        description : 商品描述（可选），拼接到 name 后一起编码以增强语义
        category    : 商品类目（可选），同上
        product_id  : 商品唯一 ID（可选），留空时自动由内容哈希生成，用于缓存 key
    """

    name: str = Field(..., min_length=1, description="商品名称")
    image_url: str = Field(..., min_length=1, description="商品图片 URL 或 Base64")
    description: Optional[str] = Field(None, description="商品描述（可选）")
    category: Optional[str] = Field(None, description="商品类目（可选）")
    product_id: Optional[str] = Field(None, description="商品唯一 ID，留空则自动生成")

    @model_validator(mode="after")
    def auto_generate_id(self) -> "Product":
        """若未提供 product_id，则根据内容哈希自动生成，保证相同商品命中同一缓存槽"""
        if self.product_id:
            return self
        key_str = json.dumps(
            {
                "name": self.name,
                "image_url": self.image_url,
                "description": self.description or "",
                "category": self.category or "",
            },
            sort_keys=True,
        )
        self.product_id = hashlib.md5(key_str.encode()).hexdigest()
        return self

    def get_text(self) -> str:
        """
        拼接所有文本字段作为最终的文本嵌入输入
        拼接顺序：名称 > 类目 > 描述
        """
        parts = [self.name]
        if self.category:
            parts.append(self.category)
        if self.description:
            parts.append(self.description)
        return " | ".join(parts)


# ─────────────────────────────────────────────
# 相似度检测接口
# ─────────────────────────────────────────────

class SimilarityRequest(BaseModel):
    """
    相似度检测请求体
    
    Fields:
        query       : 检测对象（单个商品）
        candidates  : 被检测集合（N 个商品）
        top_k       : 返回相似度最高的前 K 个，默认 5
        mode        : 相似度计算模式
                      "image"  - 仅使用图像嵌入
                      "text"   - 仅使用文本嵌入
                      "fusion" - 图像+文本融合（默认，加权平均）
        image_weight: fusion 模式下图像权重（0~1），默认 0.7
    """

    query: Product
    candidates: List[Product] = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=1000)
    mode: str = Field("fusion", pattern="^(image|text|fusion)$")
    image_weight: float = Field(0.7, ge=0.0, le=1.0)


class SimilarityResult(BaseModel):
    """单条相似度结果"""
    product: Product
    score: float = Field(..., description="余弦相似度（0~1）")
    rank: int = Field(..., description="排名（从 1 开始）")


class SimilarityResponse(BaseModel):
    """相似度检测响应体"""
    query: Product
    results: List[SimilarityResult]
    total_candidates: int
    mode: str


# ─────────────────────────────────────────────
# 商品聚类接口
# ─────────────────────────────────────────────

class ClusterRequest(BaseModel):
    """
    商品聚类请求体
    
    Fields:
        products            : 待聚类商品列表（至少 2 个）
        threshold           : 相似度阈值，高于此值视为相似（0~1），默认读取全局配置
        mode                : 同 SimilarityRequest.mode
        image_weight        : 同 SimilarityRequest.image_weight
        min_cluster_size    : 同一簇最少商品数，默认 2
    """

    products: List[Product] = Field(..., min_length=2)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    mode: str = Field("fusion", pattern="^(image|text|fusion)$")
    image_weight: float = Field(0.7, ge=0.0, le=1.0)
    min_cluster_size: int = Field(2, ge=2)


class ProductGroup(BaseModel):
    """一个相似商品簇"""
    group_id: int
    products: List[Product]
    avg_similarity: float = Field(..., description="簇内平均相似度")


class ClusterResponse(BaseModel):
    """商品聚类响应体"""
    similar_groups: List[ProductGroup] = Field(..., description="相似商品分组列表")
    unique_products: List[Product] = Field(..., description="无相似品的独立商品")
    total_products: int
    threshold: float
    mode: str


# ─────────────────────────────────────────────
# 缓存管理接口
# ─────────────────────────────────────────────

class CacheStatsResponse(BaseModel):
    """缓存统计信息"""
    total_entries: int
    max_size: int
    hit_rate: float
    device: str
