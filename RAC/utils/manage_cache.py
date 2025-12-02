import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple, List


def load_region_cache(pt_path: str) -> Dict[str, Any]:
    """
    读取 region cache pt 文件，并结构化返回内容。
    """
    data = torch.load(pt_path, map_location="cpu")

    if not isinstance(data, dict):
        raise ValueError(f"❌ 文件结构不符合预期，应该是 dict，而不是 {type(data)}")

    result: Dict[str, Any] = {}

    result["prompt"]       = data.get("prompt", None)
    result["region_name"]  = data.get("region_name", None)
    result["mapping"]      = data.get("mapping", None)
    result["hidden_state"] = data.get("hidden_state", None)   # [TB, K, C]
    result["indices"]      = data.get("indices", None)        # [K]

    meta = {}
    for k, v in data.items():
        if k not in result:
            meta[k] = v
    result["meta"] = meta if meta else None

    return result


def load_region_cache_as_tensor(
    pt_path: str,
    num_layers: int = 28,   # 你的 DiT 层数
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], List[str]]:
    """
    从 .pt 中读取 region cache，并整理成:

        hidden_cache  : [T, B, K, C]
        region_indices: [K]         （全局 token 下标，所有 step/layer 共用）
        tags          : 长度为 T * B，每个 cache 对应一个 tag（按存储顺序）

    Args:
        pt_path: .pt 文件路径
        num_layers: 模型的 block / layer 数（比如 28）

    Returns:
        hidden_cache   (Tensor): [T, B, K, C]
        region_indices (LongTensor): [K]
        info (dict): 额外信息（prompt, region_name, mapping, meta, num_steps, num_layers,...）
        tags (List[str]): 每一个 cache 的 tag（顺序与 hidden_cache.reshape(T*B, K, C) 对齐）
    """
    region = load_region_cache(pt_path)

    hidden_state = region["hidden_state"]
    print("hidden_state shape in load as tensor",hidden_state.shape)   # [TB, K, C]
    indices = region["indices"]             # [K]

    if hidden_state is None or indices is None:
        raise ValueError("❌ pt 中缺少 'hidden_state' 或 'indices' 字段。")

    if not torch.is_tensor(hidden_state):
        hidden_state = torch.as_tensor(hidden_state)
    if not torch.is_tensor(indices):
        indices = torch.as_tensor(indices, dtype=torch.long)

    if hidden_state.ndim != 3:
        raise ValueError(
            f"'hidden_state' 期望形状为 [TB, K, C]，但实际为 {hidden_state.shape}"
        )

    if indices.ndim != 1:
        indices = indices.view(-1)

    TB, K, C = hidden_state.shape  # TB = T * B

    if TB % num_layers != 0:
        raise ValueError(
            f"hidden_state 的第 0 维 TB={TB} 无法被 num_layers={num_layers} 整除，"
            f"无法拆成 [T, B]。请检查保存时的层数设置。"
        )

    T = TB // num_layers
    B = num_layers

    # 🔹 核心 reshape： [TB, K, C] -> [T, B, K, C]
    print("############hidden_state###########",hidden_state.shape)
    hidden_cache = hidden_state.view(T, B, K, C)

    # 🔹 从 meta 中尽量恢复 tag 顺序
    tags: List[str] = []
    meta = region.get("meta", None)

    if isinstance(meta, dict):
        # 常见情况：meta 里直接保存了 hidden_sink
        if "hidden_sink" in meta and isinstance(meta["hidden_sink"], (list, tuple)):
            for item in meta["hidden_sink"]:
                if isinstance(item, dict):
                    tags.append(str(item.get("tag", "")))
        # 如果你另存过 tags，也顺带兼容一下
        elif "tags" in meta:
            maybe_tags = meta["tags"]
            if isinstance(maybe_tags, (list, tuple)):
                tags = [str(t) for t in maybe_tags]

    # 如果没有找到 tag，或者数量对不上 TB，就用 step/block 生成一套默认 tag
    if not tags or len(tags) != TB:
        tags = [f"step{t}_block{b}" for t in range(T) for b in range(B)]

    info = {
        "prompt": region["prompt"],
        "region_name": region["region_name"],
        "mapping": region["mapping"],
        "meta": region["meta"],
        "num_steps": T,
        "num_layers": B,
        "region_len": K,
        "hidden_dim": C,
        "tags": tags,
    }

    return hidden_cache, indices, info, tags


if __name__ == "__main__":
    path = "/home/lipz/RegionCache/Material_Library/Constructer/cache/chunks/a_cat.pt"

    hidden_cache, region_indices, info, tags = load_region_cache_as_tensor(
        path, num_layers=28
    )

    print("📌 prompt:", info["prompt"])
    print("📌 region_name:", info["region_name"])
    print("📌 num_steps:", info["num_steps"])
    print("📌 num_layers:", info["num_layers"])
    print("📌 region_len(K):", info["region_len"])
    print("📌 hidden_dim(C):", info["hidden_dim"])

    print("\n=== Tensor 形状检查 ===")
    print("hidden_cache.shape:", hidden_cache.shape)      # [T, B, K, C]
    print("region_indices.shape:", region_indices.shape)  # [K]

    # T, B, K, C = hidden_cache.shape
    # print(f"\nT={T}, B={B}, K={K}, C={C}")

    # print("\n=== 抽查 step=0, layer=0 的前几个 index ===")
    # idx0 = region_indices[:10]
    # hid0 = hidden_cache[0, 0, :10]   # [10, C]

    # print("indices 示例:", idx0.tolist())
    # print("hidden 示例 shape:", hid0.shape)

    # # 🔍 顺序打印每一个 cache 的 tag（与 hidden_state / hidden_cache 顺序对齐）
    # print("\n=== 每一个 cache 的 tag（顺序检查） ===")
    # for i, tag in enumerate(tags):
    #     print(f"[{i:04d}] {tag}")

    # # 🔍 按 step 观察 hidden state 的相似度（cosine similarity）
    # print("\n=== 每一步 hidden state 的相似度（与前一步对比） ===")
    # step_sims = []

    # # 先把每个 step 的 [B, K, C] 展平为一个向量 [B*K*C]
    # hidden_flat = hidden_cache.reshape(T, -1)  # [T, B*K*C]

    # for t in range(1, T):
    #     prev = hidden_flat[t - 1]
    #     cur  = hidden_flat[t]

    #     # cosine_similarity 输入是 [N]，需要在 dim=0 上算
    #     sim = F.cosine_similarity(prev, cur, dim=0)
    #     step_sims.append(sim.item())
    #     print(f"step {t-1} -> {t} cosine similarity: {sim.item():.6f}")

    # # 整体统计信息
    # if len(step_sims) > 0:
    #     sims_tensor = torch.tensor(step_sims)
    #     print("\n=== 相似度统计 ===")
    #     print(f"mean: {sims_tensor.mean().item():.6f}")
    #     print(f"std : {sims_tensor.std().item():.6f}")
    #     print(f"min : {sims_tensor.min().item():.6f}")
    #     print(f"max : {sims_tensor.max().item():.6f}")
