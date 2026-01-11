import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def render_patch_mask(
    pt_path: str,
    grid_hw: tuple[int, int],
    out_path: str | None = None,
    title_prefix: str = "Patch mask for:",
    value: float = 1.0,
):
    """
    从你保存的 region .pt 文件中读取 indices，并渲染为 2D patch mask。

    pt 文件期望结构：
      {
        "prompt": ...,
        "region_name": ...,
        "hidden_state": Tensor[T, K, C],
        "indices": LongTensor[K]
      }

    grid_hw:
      (H, W) patch 网格尺寸，例如 (32, 32) / (64, 64)
    """

    obj = torch.load(pt_path, map_location="cpu")

    region_name = obj.get("region_name", os.path.basename(pt_path))
    indices = obj["indices"]
    if isinstance(indices, list):
        indices = torch.tensor(indices, dtype=torch.long)
    indices = indices.long().cpu()

    H, W = grid_hw
    L_total = H * W

    # sanity check：确保 index 没越界
    max_idx = int(indices.max().item()) if indices.numel() > 0 else -1
    if max_idx >= L_total:
        raise ValueError(
            f"indices 最大值 {max_idx} 超过 grid_hw={grid_hw} 的范围 (L_total={L_total})。"
            f" 你可能需要用更大的 grid_hw。"
        )

    # 构造 mask（0/1）
    mask = np.zeros((H, W), dtype=np.float32)
    idx_np = indices.numpy()
    ys = idx_np // W
    xs = idx_np % W
    mask[ys, xs] = value

    # 画图
    plt.figure(figsize=(6.5, 6.0))
    im = plt.imshow(mask, cmap="hot", vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.title(f"{title_prefix} {region_name}", fontsize=16)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    if out_path is None:
        out_path = os.path.splitext(pt_path)[0] + "_mask.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] saved mask to: {out_path}")


if __name__ == "__main__":
    pt_path = "/home/lipz/RegionCache/Material_Library/Constructer/cache/background/background.pt"

    # TODO: 把这里改成你实际的 patch 网格大小
    # 常见候选： (32,32)、(48,48)、(64,64)...
    grid_hw = (64, 64)

    render_patch_mask(pt_path, grid_hw=grid_hw)
