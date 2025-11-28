from typing import Dict, Optional
import os
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F

from diffusers import PixArtAlphaPipeline


# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 加载你本地的 PixArt 模型
#     pipe = PixArtAlphaPipeline.from_pretrained(
#         "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS",
#         torch_dtype=torch.float16,
#     ).to(device)

#     pipe.set_progress_bar_config(disable=True)

#     # ------------- 取出 transformer 和 blocks -------------
#     transformer = pipe.transformer

#     if hasattr(transformer, "transformer_blocks"):
#         blocks = transformer.transformer_blocks
#     elif hasattr(transformer, "blocks"):
#         blocks = transformer.blocks
#     else:
#         raise RuntimeError(
#             "找不到 transformer 的 blocks 属性（既没有 transformer_blocks 也没有 blocks）。"
#         )

#     # ------------- 打印每一层 block 的构造 -------------
#     print("=== PixArt transformer blocks 构造 ===")
#     for i, blk in enumerate(blocks):
#         print(f"\n----- Block {i:02d} -----")
#         print(type(blk))
#         print(blk)

#     # ------------- 如果你还想顺便跑一次 step + 统计 hidden_state 变化，可以保留下面这段 -------------

#     hidden_states = []

#     def make_hook(layer_idx):
#         def hook(module, inputs, output):
#             # output: [B, C, H, W]
#             hidden_states.append(output.detach().float().cpu())
#         return hook

#     hooks = []
#     for i, blk in enumerate(blocks):
#         h = blk.register_forward_hook(make_hook(i))
#         hooks.append(h)

#     prompt = "a cute cat in the garden"
#     generator = torch.Generator(device=device).manual_seed(0)

#     with torch.no_grad():
#         _ = pipe(
#             prompt=prompt,
#             num_inference_steps=2,   # 单步，只为触发一次 transformer forward
#             generator=generator,
#             output_type="latent",
#         )

#     for h in hooks:
#         h.remove()

#     num_blocks = len(hidden_states)
#     print(f"\n共收集到 {num_blocks} 个 transformer block 的输出 hidden_state")

#     if num_blocks < 2:
#         return

#     eps = 1e-8
#     delta_norms = []
#     rel_delta_norms = []
#     cos_sims = []

#     for l in range(num_blocks - 1):
#         h = hidden_states[l].flatten(1)
#         h_next = hidden_states[l + 1].flatten(1)

#         delta = h_next - h

#         delta_norm = delta.norm(dim=-1).mean().item()
#         base_norm = h.norm(dim=-1).mean().item()
#         rel_delta = delta_norm / (base_norm + eps)

#         cos = F.cosine_similarity(h, h_next, dim=-1).mean().item()

#         delta_norms.append(delta_norm)
#         rel_delta_norms.append(rel_delta)
#         cos_sims.append(cos)

#     print("\n=== 相邻 transformer block hidden_state 变化统计（同一扩散步内） ===")
#     for l in range(num_blocks - 1):
#         print(
#             f"Block {l:02d} -> {l+1:02d}: "
#             f"Δ||h|| = {delta_norms[l]:.4f}, "
#             f"rel Δ = {rel_delta_norms[l]:.4f}, "
#             f"cos = {cos_sims[l]:.6f}"
#         )


# if __name__ == "__main__":
#     main()
from typing import Optional
import torch
import torch.nn.functional as F

from diffusers import PixArtAlphaPipeline


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载 PixArt 模型
    pipe = PixArtAlphaPipeline.from_pretrained(
        "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16,
    ).to(device)

    pipe.set_progress_bar_config(disable=False)

    transformer = pipe.transformer

    # 2. 取出 blocks
    if hasattr(transformer, "transformer_blocks"):
        blocks = transformer.transformer_blocks
    elif hasattr(transformer, "blocks"):
        blocks = transformer.blocks
    else:
        raise RuntimeError("找不到 transformer 的 blocks 属性（transformer_blocks / blocks 都没有）")

    num_blocks = len(blocks)
    print(f"num_blocks = {num_blocks}")

    # 想观察哪一层的 hidden_state：这里默认最后一层
    TARGET_BLOCK = num_blocks - 1
    print(f"TARGET_BLOCK = {TARGET_BLOCK}")

    # 3. 跨 step 记录该层的 hidden_state
    step_hidden_states = []  # list[Tensor]，每个元素对应一个 diffusion step 的 hidden

    def hook_fn(module, inputs, output):
        # 对于某一层来说，每一次调用就是一个 step
        step_hidden_states.append(output.detach().float().cpu())

    hook_handle = blocks[TARGET_BLOCK].register_forward_hook(hook_fn)

    # 4. 跑一次完整的多步采样
    prompt = "a cute cat in the garden"
    num_inference_steps = 25
    generator = torch.Generator(device=device).manual_seed(0)

    with torch.no_grad():
        _ = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="latent",  # 不需要 decode，省算力
        )

    hook_handle.remove()

    # 5. 检查收集到的 step 数量
    num_steps_recorded = len(step_hidden_states)
    print(f"\n收集到 {num_steps_recorded} 个 step 的 hidden_state（目标 block = {TARGET_BLOCK}）")

    if num_steps_recorded < 2:
        print("step 数太少，无法做相邻 step 的相似度统计")
        return

    # 6. 统计相邻 step 的变化
    eps = 1e-8
    delta_norms = []
    rel_delta_norms = []
    cos_sims = []
    step_norms = []

    # 每个 step 自身的 norm
    for s in range(num_steps_recorded):
        h = step_hidden_states[s].flatten(1)  # [B, -1]
        norm = h.norm(dim=-1).mean().item()
        step_norms.append(norm)

    # 相邻 step 对比
    for s in range(num_steps_recorded - 1):
        h_t = step_hidden_states[s].flatten(1)
        h_t1 = step_hidden_states[s + 1].flatten(1)

        delta = h_t1 - h_t
        delta_norm = delta.norm(dim=-1).mean().item()
        base_norm = step_norms[s]
        rel_delta = delta_norm / (base_norm + eps)
        cos = F.cosine_similarity(h_t, h_t1, dim=-1).mean().item()

        delta_norms.append(delta_norm)
        rel_delta_norms.append(rel_delta)
        cos_sims.append(cos)

    # 7. 打印结果
    print("\n=== 每步 hidden_state 的整体范数（block {}） ===".format(TARGET_BLOCK))
    for s in range(num_steps_recorded):
        print(f"Step {s:02d}: ||h|| = {step_norms[s]:.4f}")

    print(f"\n=== 相邻 diffusion step 上（block {TARGET_BLOCK}）hidden_state 的变化 ===")
    for s in range(num_steps_recorded - 1):
        print(
            f"Step {s:02d} -> {s+1:02d}: "
            f"Δ||h|| = {delta_norms[s]:.4f}, "
            f"rel Δ = {rel_delta_norms[s]:.4f}, "
            f"cos = {cos_sims[s]:.6f}"
        )


if __name__ == "__main__":
    main()
