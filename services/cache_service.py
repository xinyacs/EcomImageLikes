"""
EmbeddingCache - 商品向量缓存服务
基于 LRU + TTL 策略，以 product_id 为 key 缓存 (image_emb, text_emb)
线程安全，支持多并发访问
"""

from __future__ import annotations

import time
import threading
from collections import OrderedDict
from typing import Optional, Tuple
import numpy as np

from core.config import settings


class CacheEntry:
    """缓存条目，包含向量数据与过期时间戳"""

    __slots__ = ("image_emb", "text_emb", "expires_at")

    def __init__(self, image_emb: np.ndarray, text_emb: np.ndarray):
        self.image_emb = image_emb  # shape (D,)
        self.text_emb = text_emb    # shape (D,)
        self.expires_at: float = (
            time.time() + settings.CACHE_TTL if settings.CACHE_TTL > 0 else float("inf")
        )

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class EmbeddingCache:
    """
    Product 级别的向量缓存
    
    策略：
      - LRU 淘汰：超过 max_size 时移除最久未使用条目
      - TTL 过期：每次 get 时检查，惰性删除过期条目
    
    线程安全：使用 threading.RLock 保护所有读写操作
    """

    def __init__(
        self,
        max_size: int = settings.CACHE_MAX_SIZE,
    ):
        self._max_size = max_size
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # 命中率统计
        self._hits: int = 0
        self._misses: int = 0

    # ── 公开接口 ──────────────────────────────────────────────

    def get(
        self, product_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        根据 product_id 获取缓存的 (image_emb, text_emb)
        命中时移动到 OrderedDict 末尾（表示最近使用）
        
        Returns:
            (image_emb, text_emb) 或 None（未命中/已过期）
        """
        with self._lock:
            entry = self._store.get(product_id)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                # 惰性删除过期条目
                del self._store[product_id]
                self._misses += 1
                return None

            # LRU：将命中条目移到末尾
            self._store.move_to_end(product_id)
            self._hits += 1
            return entry.image_emb, entry.text_emb

    def set(
        self,
        product_id: str,
        image_emb: np.ndarray,
        text_emb: np.ndarray,
    ) -> None:
        """
        写入缓存
        若 key 已存在则更新并移到末尾；若超出容量则 LRU 淘汰最旧条目
        """
        with self._lock:
            if product_id in self._store:
                self._store.move_to_end(product_id)
            self._store[product_id] = CacheEntry(image_emb, text_emb)

            # LRU 淘汰
            while len(self._store) > self._max_size:
                oldest_key, _ = self._store.popitem(last=False)

    def delete(self, product_id: str) -> bool:
        """手动删除指定 key，返回是否存在"""
        with self._lock:
            if product_id in self._store:
                del self._store[product_id]
                return True
            return False

    def clear(self) -> None:
        """清空所有缓存并重置统计"""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    # ── 统计信息 ──────────────────────────────────────────────

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return round(self._hits / total, 4) if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "total_entries": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


# 全局单例缓存
embedding_cache = EmbeddingCache()
