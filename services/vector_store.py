"""
VectorStore - 两级向量存储
  L1：进程内 LRU（OrderedDict + TTL），热点商品零延迟命中
  L2：ChromaDB 持久化向量数据库（HNSW ANN 索引）

设计要点：
  - L1 命中 → 直接返回，不走磁盘
  - L1 未命中，L2 命中 → 回填 L1，返回
  - L2 未命中 → 返回 None，由 EmbeddingService 触发推理后写入
  - 所有公开方法均为异步，内部 Chroma IO 通过 run_in_executor 非阻塞化
  - 写入 ChromaDB 时同时存储商品 metadata，便于后续按类目等字段过滤
"""

from __future__ import annotations

import asyncio
import time
import threading
from collections import OrderedDict
from typing import Optional, Tuple, List, Dict
import numpy as np

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.config import settings


# ─────────────────────────────────────────────────────────────
# L1 内存缓存（热点 LRU + TTL）
# ─────────────────────────────────────────────────────────────

class _L1Entry:
    """L1 缓存条目"""
    __slots__ = ("image_emb", "text_emb", "expires_at")

    def __init__(self, image_emb: np.ndarray, text_emb: np.ndarray):
        self.image_emb = image_emb
        self.text_emb = text_emb
        self.expires_at: float = (
            time.monotonic() + settings.CACHE_TTL
            if settings.CACHE_TTL > 0
            else float("inf")
        )

    def is_valid(self) -> bool:
        return time.monotonic() < self.expires_at


class _L1Cache:
    """
    进程内 LRU 缓存
    线程安全，使用 RLock 保护（ChromaDB 本身也会从其他线程访问）
    """

    def __init__(self, max_size: int):
        self._max = max_size
        self._store: OrderedDict[str, _L1Entry] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._max == 0:
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None or not entry.is_valid():
                if key in self._store:
                    del self._store[key]  # 惰性删除过期条目
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return entry.image_emb, entry.text_emb

    def set(self, key: str, image_emb: np.ndarray, text_emb: np.ndarray) -> None:
        if self._max == 0:
            return
        with self._lock:
            self._store[key] = _L1Entry(image_emb, text_emb)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)


# ─────────────────────────────────────────────────────────────
# L2 ChromaDB 向量存储
# ─────────────────────────────────────────────────────────────

# ChromaDB 在向量维度中分别存 image / text 两个 collection
# 这样查询时可以独立按维度检索，也可以融合后存第三个 collection
_IMG_SUFFIX = "_img"
_TXT_SUFFIX = "_txt"


class VectorStore:
    """
    两级向量存储（单例）

    使用方式：
        await VectorStore.initialize()          # 启动时调用
        store = VectorStore.get_instance()
        embs = await store.get(product_id)      # L1 → L2 两级查询
        await store.set(product_id, img, txt, meta)
    """

    _instance: Optional["VectorStore"] = None
    _executor = None  # 与 ModelService 共享线程池，由外部注入

    def __init__(self):
        # ── 初始化 ChromaDB 客户端 ────────────────────────────
        if settings.CHROMA_PERSIST_DIR == ":memory:":
            # 纯内存模式（测试 / 临时场景）
            self._client = chromadb.EphemeralClient()
        else:
            # 持久化模式（生产推荐）
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(
                    anonymized_telemetry=False,   # 关闭遥测
                    allow_reset=True,
                ),
            )

        # 距离函数元数据（创建 collection 时指定，之后不可更改）
        _meta = {"hnsw:space": settings.CHROMA_DISTANCE_FN}

        # 分别为 image / text 向量创建两个 collection
        # get_or_create 保证重启后不会重建索引
        self._col_img = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION + _IMG_SUFFIX,
            metadata=_meta,
        )
        self._col_txt = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION + _TXT_SUFFIX,
            metadata=_meta,
        )

        # ── 初始化 L1 缓存 ────────────────────────────────────
        self._l1 = _L1Cache(max_size=settings.L1_CACHE_SIZE)

        # 全局统计（L2 命中/未命中）
        self._l2_hits = 0
        self._l2_misses = 0

    # ── 生命周期 ──────────────────────────────────────────────

    @classmethod
    async def initialize(cls, executor=None) -> None:
        """在 asyncio 事件循环中异步初始化（注入线程池）"""
        cls._executor = executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, cls._create_instance)

    @classmethod
    def _create_instance(cls) -> None:
        cls._instance = VectorStore()

    @classmethod
    def get_instance(cls) -> "VectorStore":
        if cls._instance is None:
            raise RuntimeError("VectorStore 未初始化，请先调用 initialize()")
        return cls._instance

    @classmethod
    def cleanup(cls) -> None:
        if cls._instance:
            cls._instance._l1.clear()
        cls._instance = None

    # ── 核心接口 ──────────────────────────────────────────────

    async def get(
        self, product_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        两级查询
        1. L1 LRU 内存缓存
        2. ChromaDB 持久化存储

        Returns:
            (image_emb, text_emb) 或 None（完全未命中）
        """
        # ── L1 查询 ──────────────────────────────────────────
        cached = self._l1.get(product_id)
        if cached is not None:
            return cached

        # ── L2 查询（offload 到线程，避免阻塞事件循环） ──────
        result = await asyncio.get_event_loop().run_in_executor(
            self.__class__._executor,
            self._chroma_get,
            product_id,
        )
        if result is not None:
            self._l2_hits += 1
            # 回填 L1
            self._l1.set(product_id, result[0], result[1])
        else:
            self._l2_misses += 1
        return result

    async def set(
        self,
        product_id: str,
        image_emb: np.ndarray,
        text_emb: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        写入向量到 L1 + L2

        Args:
            product_id : 商品唯一 ID（Chroma document id）
            image_emb  : 图像向量，shape (D,)
            text_emb   : 文本向量，shape (D,)
            metadata   : 可选的商品元数据（name / category 等），存入 Chroma metadata 字段
        """
        # 写 L1
        self._l1.set(product_id, image_emb, text_emb)

        # 写 L2（offload）
        await asyncio.get_event_loop().run_in_executor(
            self.__class__._executor,
            self._chroma_upsert,
            product_id,
            image_emb,
            text_emb,
            metadata or {},
        )

    async def batch_get(
        self, product_ids: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        批量查询，返回 {product_id: (image_emb, text_emb)} 的命中字典
        先批量检查 L1，再对 L1 未命中的 ID 批量查 ChromaDB（减少 IO 次数）
        """
        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        l2_ids: List[str] = []

        # L1 批量检查
        for pid in product_ids:
            hit = self._l1.get(pid)
            if hit is not None:
                result[pid] = hit
            else:
                l2_ids.append(pid)

        # L2 批量查询
        if l2_ids:
            l2_result = await asyncio.get_event_loop().run_in_executor(
                self.__class__._executor,
                self._chroma_batch_get,
                l2_ids,
            )
            for pid, embs in l2_result.items():
                self._l1.set(pid, embs[0], embs[1])  # 回填 L1
                result[pid] = embs
            self._l2_hits += len(l2_result)
            self._l2_misses += len(l2_ids) - len(l2_result)

        return result

    async def delete(self, product_id: str) -> None:
        """从 L1 + L2 删除指定商品向量"""
        self._l1.delete(product_id)
        await asyncio.get_event_loop().run_in_executor(
            self.__class__._executor,
            self._chroma_delete,
            product_id,
        )

    async def clear(self) -> None:
        """清空所有向量（L1 + L2），谨慎使用"""
        self._l1.clear()
        self._l2_hits = 0
        self._l2_misses = 0
        await asyncio.get_event_loop().run_in_executor(
            self.__class__._executor,
            self._chroma_clear,
        )

    # ── ChromaDB 同步操作（在线程池执行） ────────────────────

    def _chroma_get(
        self, product_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """从 Chroma 按 ID 精确查询两个 collection"""
        try:
            img_res = self._col_img.get(ids=[product_id], include=["embeddings"])
            txt_res = self._col_txt.get(ids=[product_id], include=["embeddings"])
            if img_res["embeddings"] and txt_res["embeddings"]:
                return (
                    np.array(img_res["embeddings"][0], dtype=np.float32),
                    np.array(txt_res["embeddings"][0], dtype=np.float32),
                )
        except Exception:
            pass
        return None

    def _chroma_batch_get(
        self, product_ids: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """批量精确查询，返回命中字典"""
        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        try:
            img_res = self._col_img.get(ids=product_ids, include=["embeddings"])
            txt_res = self._col_txt.get(ids=product_ids, include=["embeddings"])

            # Chroma 返回时保持请求顺序，但只含命中项
            img_map = dict(zip(img_res["ids"], img_res["embeddings"]))
            txt_map = dict(zip(txt_res["ids"], txt_res["embeddings"]))

            for pid in product_ids:
                if pid in img_map and pid in txt_map:
                    result[pid] = (
                        np.array(img_map[pid], dtype=np.float32),
                        np.array(txt_map[pid], dtype=np.float32),
                    )
        except Exception:
            pass
        return result

    def _chroma_upsert(
        self,
        product_id: str,
        image_emb: np.ndarray,
        text_emb: np.ndarray,
        metadata: Dict,
    ) -> None:
        """Upsert 向量到两个 Chroma collection（存在则更新，不存在则插入）"""
        self._col_img.upsert(
            ids=[product_id],
            embeddings=[image_emb.tolist()],
            metadatas=[metadata],
        )
        self._col_txt.upsert(
            ids=[product_id],
            embeddings=[text_emb.tolist()],
            metadatas=[metadata],
        )

    def _chroma_delete(self, product_id: str) -> None:
        try:
            self._col_img.delete(ids=[product_id])
            self._col_txt.delete(ids=[product_id])
        except Exception:
            pass

    def _chroma_clear(self) -> None:
        """删除并重建两个 collection（相当于 truncate）"""
        _meta = {"hnsw:space": settings.CHROMA_DISTANCE_FN}
        self._client.delete_collection(settings.CHROMA_COLLECTION + _IMG_SUFFIX)
        self._client.delete_collection(settings.CHROMA_COLLECTION + _TXT_SUFFIX)
        self._col_img = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION + _IMG_SUFFIX, metadata=_meta
        )
        self._col_txt = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION + _TXT_SUFFIX, metadata=_meta
        )

    # ── 统计信息 ──────────────────────────────────────────────

    def stats(self) -> dict:
        """返回 L1 + L2 的命中率及存储规模"""
        l1_total = self._l1.hits + self._l1.misses
        l2_total = self._l2_hits + self._l2_misses
        try:
            l2_count = self._col_img.count()
        except Exception:
            l2_count = -1

        return {
            "l1_size": self._l1.size,
            "l1_max_size": settings.L1_CACHE_SIZE,
            "l1_hits": self._l1.hits,
            "l1_misses": self._l1.misses,
            "l1_hit_rate": round(self._l1.hits / l1_total, 4) if l1_total > 0 else 0.0,
            "l2_total_vectors": l2_count,
            "l2_hits": self._l2_hits,
            "l2_misses": self._l2_misses,
            "l2_hit_rate": round(self._l2_hits / l2_total, 4) if l2_total > 0 else 0.0,
            "persist_dir": settings.CHROMA_PERSIST_DIR,
        }


# ── 全局单例（在 lifespan 中初始化后使用） ───────────────────
vector_store: VectorStore = None  # type: ignore