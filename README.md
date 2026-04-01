# EcomImageChecker

电商商品图像多模态相似度检测 & 聚类服务，基于 [Marqo/marqo-ecommerce-embeddings-L](https://huggingface.co/Marqo/marqo-ecommerce-embeddings-L) 模型，支持图像、文本、融合三种相似度计算模式。

---

## 功能特性

- **相似度检索**：给定一个 query 商品，从候选集中找出最相似的 top-K 商品
- **无监督聚类**：对一批商品做层次聚类，自动识别相似商品组与独立商品，可用于剔除异类商品
- **三种计算模式**：`image`（纯图像）、`text`（纯文本）、`fusion`（加权融合）
- **两级缓存**：L1 LRU 内存缓存 + L2 ChromaDB 向量存储，热点商品零推理延迟
- **GPU 加速**：自动检测 CUDA / MPS / CPU

---

## 技术栈

| 组件 | 说明 |
|------|------|
| [FastAPI](https://fastapi.tiangolo.com/) | Web 框架 |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | 模型加载与推理 |
| `Marqo/marqo-ecommerce-embeddings-L` | 电商专用多模态嵌入模型（1024 维，SigLIP 架构） |
| [ChromaDB](https://www.trychroma.com/) | 向量持久化存储（L2 缓存） |
| [PyTorch](https://pytorch.org/) | 深度学习后端 |

---

## 项目结构

```
EcomImageLikes/
├── main.py                  # FastAPI 应用入口，生命周期管理
├── core/
│   └── config.py            # 全局配置（模型名、设备、缓存参数等）
├── models/
│   └── schemas.py           # Pydantic 数据模型（请求/响应 Schema）
├── routers/
│   ├── similarity.py        # POST /api/v1/similarity/search
│   ├── cluster.py           # POST /api/v1/cluster/group
│   └── health.py            # GET  /health
├── services/
│   ├── model_service.py     # 模型加载、批量推理（单例）
│   ├── embedding_service.py # 向量编排：缓存查询 → 推理 → 融合
│   ├── cache_service.py     # L1 LRU 内存缓存
│   ├── vector_store.py      # L2 ChromaDB 向量存储
│   └── model_service.py
└── utils/
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
uvicorn main:app --reload
```

首次启动会自动从 HuggingFace Hub 下载模型（约 2.5 GB），请保持网络畅通。

启动后访问交互式文档：

```
http://127.0.0.1:8000/docs
```

---

## API 说明

### 健康检查

```
GET /health
```

---

### 相似度检索

```
POST /api/v1/similarity/search
```

给定一个 query 商品，从候选集中返回相似度最高的 top-K 商品。

**请求体示例：**

```json
{
  "query": {
    "name": "Wooden Dining Chair",
    "image_url": "https://example.com/chair.jpg",
    "category": "Furniture/Dining Chair",
    "description": "Solid wood four-leg dining chair"
  },
  "candidates": [
    {
      "name": "Bar Stool",
      "image_url": "https://example.com/stool.jpg"
    },
    {
      "name": "Laptop Computer",
      "image_url": "https://example.com/laptop.jpg"
    }
  ],
  "top_k": 5,
  "mode": "fusion",
  "image_weight": 0.7
}
```

**响应体示例：**

```json
{
  "query": { "name": "Wooden Dining Chair", "..." : "..." },
  "results": [
    { "product": { "name": "Bar Stool", "..." : "..." }, "score": 0.6123, "rank": 1 }
  ],
  "total_candidates": 2,
  "mode": "fusion"
}
```

---

### 商品聚类

```
POST /api/v1/cluster/group
```

对一批商品进行无监督聚类，区分「相似商品组」与「独立商品」，可用于异类商品过滤。

**算法**：基于余弦相似度矩阵 + Union-Find 贪心连通分量聚类。

**请求体示例：**

```json
{
  "products": [
    {
      "name": "Wooden Dining Chair",
      "image_url": "https://example.com/chair1.jpg",
      "category": "Furniture/Dining Chair",
      "description": "Solid wood four-leg dining chair"
    },
    {
      "name": "Windsor Dining Chair",
      "image_url": "https://example.com/chair2.jpg",
      "category": "Furniture/Dining Chair",
      "description": "Solid wood Windsor style dining chair"
    },
    {
      "name": "Laptop Computer",
      "image_url": "https://example.com/laptop.jpg",
      "category": "Electronics/Computer",
      "description": "Portable laptop computer"
    }
  ],
  "threshold": 0.63,
  "mode": "image",
  "image_weight": 1.0,
  "min_cluster_size": 2
}
```

**响应体示例：**

```json
{
  "similar_groups": [
    {
      "group_id": 0,
      "products": [ { "name": "Wooden Dining Chair", "..." : "..." }, { "name": "Windsor Dining Chair", "..." : "..." } ],
      "avg_similarity": 0.6653
    }
  ],
  "unique_products": [
    { "name": "Laptop Computer", "..." : "..." }
  ],
  "total_products": 3,
  "threshold": 0.63,
  "mode": "image"
}
```

---

## 参数说明

### `mode`

| 值 | 说明 |
|----|------|
| `image` | 仅使用图像嵌入计算相似度 |
| `text` | 仅使用文本嵌入（`name \| category \| description`）计算相似度 |
| `fusion` | 图像与文本加权融合：`image_weight * img + (1 - image_weight) * txt` |

> ⚠️ `text` 模式：该模型的文本编码器基于 SigLIP（英文 tokenizer），**仅支持英文输入**。中文文本会被编码为相同向量，导致所有商品相似度为 1.0，无法区分。

### `threshold`（聚类接口）

该模型在电商缩略图上的实际相似度范围约为 **0.3 ~ 0.8**，建议参考值：

| 场景 | 推荐 threshold |
|------|---------------|
| 同款商品聚合 | 0.75 ~ 0.85 |
| 同类商品聚合（如各种椅子） | 0.62 ~ 0.70 |
| 宽松聚合 | 0.55 ~ 0.62 |

---

## 配置项

通过环境变量覆盖，或直接修改 `core/config.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_NAME` | `Marqo/marqo-ecommerce-embeddings-L` | HuggingFace 模型名 |
| `DEVICE` | 自动检测 | `cuda` / `mps` / `cpu` |
| `BATCH_SIZE` | `32` | 单次推理批大小 |
| `CHROMA_PERSIST_DIR` | `:memory:` | ChromaDB 持久化目录，`:memory:` 为纯内存模式 |
| `CHROMA_COLLECTION` | `ecom_products` | ChromaDB collection 名称 |
| `L1_CACHE_SIZE` | `2000` | L1 LRU 缓存最大条目数 |
| `CACHE_TTL` | `3600` | L1 缓存 TTL（秒），0 = 永不过期 |
| `CLUSTER_SIMILARITY_THRESHOLD` | `0.85` | 聚类默认相似度阈值 |
| `IMAGE_DOWNLOAD_TIMEOUT` | `10` | 图片下载超时秒数 |

---

## Product 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | ✅ | 商品名称，参与文本嵌入 |
| `image_url` | ✅ | 图片 URL 或 Base64 字符串，参与图像嵌入 |
| `description` | ❌ | 商品描述，拼接到 name 后增强语义 |
| `category` | ❌ | 商品类目，同上 |
| `product_id` | ❌ | 唯一 ID，留空则自动由内容 MD5 生成，用作缓存 key |

文本嵌入输入格式：`name | category | description`
