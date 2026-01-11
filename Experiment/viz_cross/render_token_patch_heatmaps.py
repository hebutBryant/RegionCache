#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_token_patch_heatmaps.py

- 全量 token 生成 heatmap PNG
- summary PDF 仅显示 SELECTED_TOKEN_INDICES
- summary 第一个子图为 generate.png
"""

import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# ======================= 配置 =======================
json_path = "/home/lipz/RegionCache/Experiment/viz_cross/cross_attn_laststep.json"
GENERATED_IMAGE_PATH = "/home/lipz/RegionCache/Experiment/viz_cross/generate.png"
save_root = "./attn_token_patch_maps_laststep_block10"

TARGET_BLOCK_KEYWORDS = ["block10", "blocks.10", "transformer_blocks.10"]

B_EXPECT  = 2
H_EXPECT  = 16
NQ_EXPECT = 4096
NK_EXPECT = 9

batch_idx   = 1
reduce_mode = "mean"

cmap        = "viridis"
norm_method = "minmax"

# ===== 所有 token 的文本（长度必须 = NK_EXPECT）=====
TOKENS_TEXT = ["_", "a", "cat", "paly", "with", "_", "a", "ball", "</w>"]

# ===== summary 中展示的 token index =====
SELECTED_TOKEN_INDICES = [1,2,3,4,5,6,7]
# ====================================================


def check_square(n):
    r = int(math.isqrt(n))
    if r * r != n:
        raise ValueError("NQ 不是平方数")
    return r


def to_bhqn(arr, b, h, nq, nk):
    total = b * h * nq * nk
    if arr.size != total:
        raise ValueError("attention tensor size mismatch")
    return arr.reshape(b, h, nq, nk)


def normalize_map(m):
    mn, mx = m.min(), m.max()
    return (m - mn) / (mx - mn) if mx > mn else np.zeros_like(m)


def main():
    with open(json_path, "r") as f:
        data = json.load(f)

    side = check_square(NQ_EXPECT)
    os.makedirs(save_root, exist_ok=True)

    for layer_name, content in data.items():
        if not any(k in layer_name for k in TARGET_BLOCK_KEYWORDS):
            continue

        arr = np.array(content["data"], dtype=np.float32)
        arr = to_bhqn(arr, B_EXPECT, H_EXPECT, NQ_EXPECT, NK_EXPECT)

        attn = arr[batch_idx].mean(axis=0)  # [4096, Nk]

        subdir = os.path.join(save_root, layer_name.replace(".", "_"))
        os.makedirs(subdir, exist_ok=True)

        # ======================
        # ① 全量 token → PNG
        # ======================
        heatmaps_all = []

        for t_idx in range(NK_EXPECT):
            heat = normalize_map(attn[:, t_idx].reshape(side, side))
            heatmaps_all.append(heat)

            plt.figure(figsize=(6, 6))
            plt.imshow(heat, cmap=cmap)
            plt.axis("off")
            plt.savefig(
                os.path.join(subdir, f"heat_token{t_idx:02d}.png"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

        # ======================
        # ② 选中 token → summary PDF
        # ======================
        gen_img = plt.imread(GENERATED_IMAGE_PATH)
        num_cols = len(SELECTED_TOKEN_INDICES) + 1

        fig, axes = plt.subplots(
            1, num_cols,
            figsize=(2.4 * num_cols, 2.8),
            squeeze=False
        )

        # (0) generated image
        axes[0, 0].imshow(gen_img)
        axes[0, 0].axis("off")
        axes[0, 0].text(
            0.5, -0.12,
            "Generated Image",
            transform=axes[0, 0].transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold"
)

        # (1..) selected tokens
# (1..) selected tokens
        for i, t_idx in enumerate(SELECTED_TOKEN_INDICES):
            ax = axes[0, i + 1]
            ax.imshow(heatmaps_all[t_idx], cmap=cmap)
            ax.axis("off")

            ax.text(
                0.5, -0.12,
                TOKENS_TEXT[t_idx],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold"
            )


        plt.tight_layout()
        plt.savefig(
            os.path.join(subdir, "summary_token_attention.pdf"),
            bbox_inches="tight"
        )
        plt.close()

    print("✅ 完成：PNG（全 token）+ PDF（选中 token）已生成")


if __name__ == "__main__":
    main()
