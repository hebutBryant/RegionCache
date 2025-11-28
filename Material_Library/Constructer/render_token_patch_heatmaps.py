#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_token_patch_heatmaps.py

读取 cross_attn_laststep.json，
将 [B,H,4096,Nk] 的注意力矩阵在空间维度上重排为 [64,64]，
对每个 token 生成 64x64 的热力图并保存。
"""

import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========= 配置 =========
json_path   = "/home/lipz/xDiT/xDiT/RegionSparse/Cross/cross_attn_laststep.json"
save_root   = "./attn_token_patch_maps_laststep"   # 输出根目录（每层一个文件夹）
B_EXPECT    = 2                           # 期望 batch size
H_EXPECT    = 16                          # 期望 head 数
NQ_EXPECT   = 4096                        # 期望空间 token 数（必须是平方数）
NK_EXPECT   = 9                           # 期望文本 token 数
batch_idx   = 1                           # 选第几个 batch 可视化（0 或 1）
reduce_mode = "mean"                      # "mean"（对 head 取均值）或 "head0"（取第 0 个 head）
dpi         = 300                         # 输出图片分辨率
cmap        = "viridis"                   # 颜色图（可改：inferno/magma/plasma/…）
norm_method = "minmax"                    # "minmax" 或 "none"（热力图归一化）
# =======================

def check_square(n: int) -> int:
    r = int(math.isqrt(n))
    if r * r != n:
        raise ValueError(f"Nq={n} 不是完全平方数，无法重排为方阵！")
    return r

def to_bhqn(arr: np.ndarray, b: int, h: int, nq: int, nk: int) -> np.ndarray:
    """
    规范数组为 [B, H, Nq, Nk]。支持：
      - [B*H, Nq, Nk]
      - [B, H, Nq, Nk]
      - 其他形状但元素总数匹配的情况
    """
    total = b * h * nq * nk
    if arr.size != total:
        raise ValueError(
            f"元素总数不匹配：当前 {arr.size}，期望 {total} (= {b}*{h}*{nq}*{nk})."
        )
    if arr.ndim == 3 and arr.shape == (b * h, nq, nk):
        return arr.reshape(b, h, nq, nk)
    if arr.ndim == 4 and arr.shape == (b, h, nq, nk):
        return arr
    # 兜底 reshape
    return arr.reshape(b, h, nq, nk)

def normalize_map(m: np.ndarray, how: str = "minmax") -> np.ndarray:
    if how == "minmax":
        mn, mx = m.min(), m.max()
        if mx > mn:
            return (m - mn) / (mx - mn)
        return np.zeros_like(m)
    return m

def main():
    # 1) 读取 JSON
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"未找到 JSON 文件：{json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    if not data:
        raise RuntimeError("JSON 为空，未找到任何层的数据。")

    # 2) 检查 NQ_EXPECT 是否可开平方
    side = check_square(NQ_EXPECT)  # 4096 -> 64

    # 3) 创建输出目录
    os.makedirs(save_root, exist_ok=True)
    print(f"💾 输出目录：{save_root}，每层一个子目录；每个 token 一张 64×64 热力图。")

    # 4) 遍历每一层
    for layer_name, content in tqdm(data.items(), desc="按层渲染", ncols=100):
        shape = content.get("shape")
        if shape is None:
            print(f"⚠️ 跳过（无 shape 字段）：{layer_name}")
            continue

        arr = np.array(content["data"], dtype=np.float32)
        try:
            arr = to_bhqn(arr, B_EXPECT, H_EXPECT, NQ_EXPECT, NK_EXPECT)  # -> [B,H,4096,Nk]
        except Exception as e:
            print(f"⚠️ 跳过 {layer_name}: {e}")
            continue

        if not (0 <= batch_idx < B_EXPECT):
            print(f"⚠️ batch_idx={batch_idx} 越界（B={B_EXPECT}），自动取 0")
            b = 0
        else:
            b = batch_idx

        # 选定 batch
        b_arr = arr[b]   # [H,4096,Nk]

        # head 聚合
        if reduce_mode == "mean":
            # [4096, Nk]
            attn_spatial_token = b_arr.mean(axis=0)
        elif reduce_mode == "head0":
            # [4096, Nk]
            attn_spatial_token = b_arr[0]
        else:
            raise ValueError("reduce_mode 仅支持 'mean' 或 'head0'")

        # 5) 为该层创建子目录
        subdir = os.path.join(save_root, layer_name.replace(".", "_"))
        os.makedirs(subdir, exist_ok=True)

        # 6) 遍历每个 token，生成 64×64 热力图
        for t_idx in range(min(NK_EXPECT, attn_spatial_token.shape[1])):
            vec_4096 = attn_spatial_token[:, t_idx]     # [4096]
            heat_64x64 = vec_4096.reshape(side, side)   # [64,64]
            heat_64x64 = normalize_map(heat_64x64, how=norm_method)

            plt.figure(figsize=(6, 6))
            plt.imshow(heat_64x64, origin="upper", cmap=cmap)
            plt.colorbar(label="Attention")
            plt.title(f"{layer_name}\n(b={b}, heads={reduce_mode}, token={t_idx})")
            plt.axis("off")
            plt.tight_layout()

            fn = f"heat_token{t_idx:02d}_b{b}_{reduce_mode}.png"
            save_path = os.path.join(subdir, fn)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()

    print("✅ 完成：所有层的每个 token 的 64×64 热力图已保存。")

if __name__ == "__main__":
    main()
