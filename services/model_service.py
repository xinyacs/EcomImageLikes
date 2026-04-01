"""
ModelService - 模型加载、推理与生命周期管理
单例模式，全局共享一个模型实例，支持 GPU 加速
使用 OpenCLIP 从 HuggingFace Hub 加载模型
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
import io
import base64
import os

import numpy as np
import torch
import open_clip
import httpx
from PIL import Image

from core.config import settings


os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class ModelService:
    """
    模型服务单例
    
    - 使用 ThreadPoolExecutor 将 CPU/GPU 密集型推理任务
      offload 到线程池，避免阻塞 FastAPI 的 asyncio 事件循环
    - 支持批量推理，自动分批处理超出 BATCH_SIZE 的请求
    """

    _instance: Optional["ModelService"] = None
    _executor: Optional[ThreadPoolExecutor] = None

    def __init__(self):
        self.device = self._select_device()
        self.model = None
        self.preprocess_val = None
        self.tokenizer = None
        self._load_model()

    # ── 初始化 ────────────────────────────────────────────────

    @classmethod
    async def initialize(cls) -> None:
        """异步初始化入口（在 lifespan 中调用）"""
        loop = asyncio.get_event_loop()
        cls._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="model_worker",
        )
        # 在线程池中同步加载模型，避免阻塞启动
        await loop.run_in_executor(cls._executor, cls._create_instance)

    @classmethod
    def _create_instance(cls) -> None:
        """（线程内）创建模型实例"""
        cls._instance = ModelService()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            raise RuntimeError("ModelService 尚未初始化，请先调用 initialize()")
        return cls._instance

    @classmethod
    def cleanup(cls) -> None:
        """释放资源"""
        if cls._executor:
            cls._executor.shutdown(wait=False)
        if cls._instance and cls._instance.device.startswith("cuda"):
            torch.cuda.empty_cache()
        cls._instance = None

    # ── 设备选择 ──────────────────────────────────────────────

    @staticmethod
    def _select_device() -> str:
        """
        自动选择推理设备优先级：
        指定 > CUDA > MPS（Apple Silicon）> CPU
        """
        if settings.DEVICE:
            return settings.DEVICE
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ── 模型加载 ──────────────────────────────────────────────

    def _load_model(self) -> None:
        """通过 OpenCLIP 从 HuggingFace Hub 加载模型与预处理器"""
        hf_model_name = f"hf-hub:{settings.MODEL_NAME}"
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            hf_model_name
        )
        self.tokenizer = open_clip.get_tokenizer(hf_model_name)
        self.model = self.model.to(self.device)
        self.model.eval()  # 推理模式，关闭 dropout 等

    # ── 图像加载 ──────────────────────────────────────────────

    async def load_image(self, image_source: str) -> Image.Image:
        """
        异步加载图像
        支持：
          - HTTP/HTTPS URL（使用 httpx 异步下载）
          - Base64 字符串（data:image/...;base64,... 或纯 base64）
          - 本地文件路径
        """
        if image_source.startswith("data:"):
            # Base64 Data URL
            header, data = image_source.split(",", 1)
            img_bytes = base64.b64decode(data)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if image_source.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=settings.IMAGE_DOWNLOAD_TIMEOUT) as client:
                resp = await client.get(image_source)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")

        # 本地路径
        return Image.open(image_source).convert("RGB")

    # ── 核心推理：批量嵌入计算 ────────────────────────────────

    def _encode_batch(
        self,
        images: List[Image.Image],
        texts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        （同步，在线程池中执行）批量编码图像和文本，返回归一化后的 numpy 向量

        Args:
            images: PIL Image 列表
            texts:  文本字符串列表（与 images 一一对应）

        Returns:
            image_embeddings: shape (N, D)
            text_embeddings:  shape (N, D)
        """
        all_img_embs: List[np.ndarray] = []
        all_txt_embs: List[np.ndarray] = []
        batch_size = settings.BATCH_SIZE

        autocast_ctx = (
            torch.cuda.amp.autocast()
            if self.device.startswith("cuda")
            else torch.amp.autocast(self.device, enabled=False)
        )

        for start in range(0, len(images), batch_size):
            batch_imgs = images[start: start + batch_size]
            batch_txts = texts[start: start + batch_size]

            # 预处理图像：每张独立 preprocess 后堆叠成 batch tensor
            pixel_values = torch.stack(
                [self.preprocess_val(img) for img in batch_imgs]
            ).to(self.device)

            # 分词
            tokens = self.tokenizer(batch_txts).to(self.device)

            with torch.no_grad(), autocast_ctx:
                img_emb = self.model.encode_image(pixel_values, normalize=True)
                txt_emb = self.model.encode_text(tokens, normalize=True)

            all_img_embs.append(img_emb.cpu().float().numpy())
            all_txt_embs.append(txt_emb.cpu().float().numpy())

        return (
            np.concatenate(all_img_embs, axis=0),
            np.concatenate(all_txt_embs, axis=0),
        )

    async def encode_products(
        self,
        images: List[Image.Image],
        texts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        异步封装：将同步推理 offload 到线程池，
        返回 (image_embeddings, text_embeddings)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.__class__._executor,
            self._encode_batch,
            images,
            texts,
        )
