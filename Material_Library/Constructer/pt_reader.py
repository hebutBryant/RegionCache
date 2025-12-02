import torch
from typing import Dict, Any, Union
path = "/home/lipz/RegionCache/Material_Library/Constructer/cache/chunks/a_cat.pt"

# data = torch.load(path, map_location="cpu")
# print("类型:", type(data))

# # 常见 2 种情况：直接存 tensor，或者存 dict
# if isinstance(data, torch.Tensor):
#     # 直接就是 [T, L, C] 之类的
#     print("tensor shape:", data.shape)
#     hidden_all = data          # 每一步的 hidden state
# elif isinstance(data, dict):
#     print("keys:", data.keys())
#     # 如果你之前是这样存的：torch.save({"hidden": stacked}, path)
#     if "hidden" in data:
#         hidden_all = data["hidden"]
#         print("hidden shape:", hidden_all.shape)   # 例如 [T, L, C]
#     else:
#         # 其他结构就自己挑 key
#         for k, v in data.items():
#             if isinstance(v, torch.Tensor):
#                 print(f"候选 tensor key = {k}, shape = {v.shape}")
# else:
#     print("⚠️ 未知的数据结构，请打印出来看看:", data)

# data = torch.load(path, map_location="cpu")

# mapping = data.get("mapping", None)

# indices = data.get("indices", None)

# print("\n=== Mapping 内容 ===")
# print(mapping)
# print("\n类型:", type(mapping))
# print(indices)



def load_region_cache(pt_path: str) -> Dict[str, Any]:
    """
    读取 region cache pt 文件，并结构化返回内容。
    
    Args:
        pt_path (str): `.pt` 文件路径
    
    Returns:
        dict: {
            "prompt": str,
            "region_name": str,
            "mapping": dict,
            "hidden_state": torch.Tensor,   # [T, L, C]
            "indices": torch.Tensor,        # [L]
            "meta": dict (可选，用于存放额外字段)
        }
    """
    data: Union[dict, torch.Tensor] = torch.load(pt_path, map_location="cpu")

    if not isinstance(data, dict):
        raise ValueError(f"❌ 文件结构不符合预期，应该是 dict，而不是 {type(data)}")

    result = {}

    # 标准字段
    result["prompt"]       = data.get("prompt", None)
    result["region_name"]  = data.get("region_name", None)
    result["mapping"]      = data.get("mapping", None)
    result["hidden_state"] = data.get("hidden_state", None)
    result["indices"]      = data.get("indices", None)

    # 额外字段收集到 meta
    meta = {}
    for key, val in data.items():
        if key not in result:
            meta[key] = val
    result["meta"] = meta if meta else None

    return result


if __name__ == "__main__":
    region = load_region_cache(path)
    print("📌 prompt:", region["prompt"])
    print(region["hidden_state"].shape)
