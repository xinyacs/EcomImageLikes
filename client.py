"""
测试路由 3：无监督聚类剔除无关产品
调用 POST /api/v1/cluster/group
运行前请先启动服务：uvicorn main:app --reload
"""

import requests

BASE_URL = "http://127.0.0.1:8000"
MIN_CLUSTER_SIZE = 2
CONFIGS = [
    {"mode": "image",  "threshold": 0.62, "image_weight": 1.0},
    {"mode": "image",  "threshold": 0.65, "image_weight": 1.0},
    {"mode": "fusion", "threshold": 0.62, "image_weight": 0.8},
    {"mode": "fusion", "threshold": 0.65, "image_weight": 0.8},
]

# ── 餐椅（目标类）────────────────────────────────────────────────────────
DINING_CHAIR_1 = "https://m.media-amazon.com/images/I/71Sl-qs0b7L._AC_UL320_.jpg"
DINING_CHAIR_2 = "https://m.media-amazon.com/images/I/71Q66Qq9kdL._AC_UL320_.jpg"
DINING_CHAIR_3 = "https://m.media-amazon.com/images/I/716t-xmv-5L._AC_UL320_.jpg"

# ── 形似餐椅的干扰项（硬负样本）──────────────────────────────────────────
OFFICE_CHAIR = "https://m.media-amazon.com/images/I/61XrK-bAifL._AC_UL320_.jpg"
BAR_STOOL = "https://m.media-amazon.com/images/I/71gLW2yOP9L._AC_UL320_.jpg"
ROCKING_CHAIR = "https://m.media-amazon.com/images/I/71ZlFX+MxmL._AC_UL320_.jpg"
WHEELCHAIR = "https://m.media-amazon.com/images/I/81DmFAXqnuL._AC_UL320_.jpg"

# ── 完全不相关品类 ───────────────────────────────────────────────────────
LAPTOP = "https://m.media-amazon.com/images/I/71LmPdODkcL._AC_UY218_.jpg"
SNEAKER = "https://m.media-amazon.com/images/I/51wnMZWbCWL._AC_UL320_.jpg"
TOOTHBRUSH = "https://m.media-amazon.com/images/I/61bjlO22uTL._AC_UL320_.jpg"

PRODUCTS = [
    {
        "name": "Wooden Dining Chair",
        "image_url": DINING_CHAIR_1,
        "category": "Furniture/Dining Chair",
        "description": "Solid wood four-leg dining chair for dining table and home use",
    },
    {
        "name": "Windsor Dining Chair",
        "image_url": DINING_CHAIR_2,
        "category": "Furniture/Dining Chair",
        "description": "Solid wood Windsor style dining chair for restaurant and dining room",
    },
    {
        "name": "Chippendale Dining Chair",
        "image_url": DINING_CHAIR_3,
        "category": "Furniture/Dining Chair",
        "description": "Carved back dining chair, classic restaurant furniture",
    },
    {
        "name": "Office Chair",
        "image_url": OFFICE_CHAIR,
        "category": "Furniture/Office",
        "description": "Ergonomic office chair with armrests for desk use",
    },
    {
        "name": "Bar Stool",
        "image_url": BAR_STOOL,
        "category": "Furniture/Stool",
        "description": "Tall bar stool for kitchen counter or bar area",
    },
    {
        "name": "Rocking Chair",
        "image_url": ROCKING_CHAIR,
        "category": "Furniture/Rocking Chair",
        "description": "Children rocking chair with curved base for nursery",
    },
    {
        "name": "Wheelchair",
        "image_url": WHEELCHAIR,
        "category": "Medical/Mobility",
        "description": "Sports wheelchair for mobility assistance and outdoor use",
    },
    {
        "name": "Laptop Computer",
        "image_url": LAPTOP,
        "category": "Electronics/Computer",
        "description": "Portable laptop computer for work and entertainment",
    },
    {
        "name": "Running Shoes",
        "image_url": SNEAKER,
        "category": "Footwear/Sports",
        "description": "Athletic running shoes for daily exercise and sport",
    },
    {
        "name": "Toothbrush",
        "image_url": TOOTHBRUSH,
        "category": "Personal Care/Oral",
        "description": "Electric toothbrush for daily oral hygiene cleaning",
    },
]


def build_payload() -> dict:
    return {
        "products": PRODUCTS,
        "threshold": 0.8,
        "mode": "fusion",
        "image_weight": 0.3,
        "min_cluster_size": MIN_CLUSTER_SIZE,
    }


def run_filter_irrelevant_products_unsupervised(mode: str, threshold: float, image_weight: float) -> None:
    payload = build_payload()
    payload["mode"] = mode
    payload["threshold"] = threshold
    payload["image_weight"] = image_weight
    resp = requests.post(f"{BASE_URL}/api/v1/cluster/group", json=payload)

    print("\n[run_filter_irrelevant_products_unsupervised]")
    print("status_code:", resp.status_code)

    try:
        data = resp.json()
    except Exception:
        print("response:", resp.text)
        return

    print("response:", data)

    similar_groups = data.get("similar_groups", [])
    unique_products = data.get("unique_products", [])

    kept = []
    removed = []

    largest_group = max(similar_groups, key=lambda g: len(g.get("products", [])), default=None)

    if largest_group is not None:
        kept.extend(largest_group.get("products", []))

    for group in similar_groups:
        if largest_group is not None and group.get("group_id") == largest_group.get("group_id"):
            continue
        removed.extend(group.get("products", []))

    removed.extend(unique_products)

    print("mode:", mode)
    print("image_weight:", image_weight)
    print("cluster_threshold:", threshold)
    print("min_cluster_size:", MIN_CLUSTER_SIZE)
    print("group_count:", len(similar_groups))
    print("unique_count:", len(unique_products))
    if largest_group is not None:
        print("kept_group_id:", largest_group.get("group_id"))
        print("kept_group_size:", len(largest_group.get("products", [])))
    for group in similar_groups:
        print(
            "group_summary:",
            {
                "group_id": group.get("group_id"),
                "size": len(group.get("products", [])),
                "avg_similarity": group.get("avg_similarity"),
                "names": [p.get("name") for p in group.get("products", [])],
            },
        )
    print("\n保留商品:")
    for item in kept:
        print(item)

    print("\n剔除商品:")
    for item in removed:
        print(item)


def main() -> None:
    for config in CONFIGS:
        run_filter_irrelevant_products_unsupervised(
            mode=config["mode"],
            threshold=config["threshold"],
            image_weight=config["image_weight"],
        )


if __name__ == "__main__":
    main()
