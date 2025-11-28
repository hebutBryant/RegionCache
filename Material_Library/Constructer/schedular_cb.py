import os
import json
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn.functional as F

from diffusers import PixArtAlphaPipeline
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
# === 设备与 dtype：老老实实按模型 dtype 跑 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
# 推荐：显存够就 float32；A100/4090 支持 bfloat16 可用 bf16；尽量避免 float16
DTYPE  = torch.float16  # or torch.bfloat16
generator = torch.Generator("cuda").manual_seed(1234)
pipe = PixArtAlphaPipeline.from_pretrained(
    "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS",
    torch_dtype=DTYPE,
    generator=generator
).to(device)

# 如无需 xformers 就不要开，保证路径单一
# （若你已替换自定义 Processor，建议关闭以避免走别的 kernel）
# 不调用 pipe.enable_xformers_memory_efficient_attention()

# === 自定义 Processor：严格按当前 dtype，禁用不必要的 upcast ===
class AttnProcessorMe:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args, **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        is_cross = encoder_hidden_states is not None
        input_ndim = hidden_states.ndim

        # 不做 autocast，不强制 upcast；一切按 hidden_states.dtype
        # 可显式关闭模块内 upcast 标志（若该属性存在）
        if hasattr(attn, "upcast_attention"):
            attn.upcast_attention = False
        if hasattr(attn, "upcast_softmax"):
            attn.upcast_softmax = False

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        seq_len = (encoder_hidden_states if is_cross else hidden_states).shape[1]
        bsz     = (encoder_hidden_states if is_cross else hidden_states).shape[0]
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bsz)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if not is_cross:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        q = attn.head_to_batch_dim(query)   # [B*H, Nq, d]
        k = attn.head_to_batch_dim(key)     # [B*H, Nk, d]
        v = attn.head_to_batch_dim(value)   # [B*H, Nk, d]

        # 严格按当前 dtype 计算
        scale  = getattr(attn, "scale", q.shape[-1] ** -0.5)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask  # additive mask

        # 不 upcast softmax：按模型 dtype 来
        probs = torch.softmax(scores, dim=-1)

        # 继续前向
        hs = torch.bmm(probs, v)            # [B*H, Nq, d]
        hs = attn.batch_to_head_dim(hs)     # [B, Nq, C]

        # （如需捕获注意力得分，这里按需 append probs.detach().cpu()）

        hs = attn.to_out[0](hs)
        hs = attn.to_out[1](hs)

        if input_ndim == 4:
            hs = hs.transpose(-1, -2).reshape(b, c, h, w)

        if attn.residual_connection:
            hs = hs + residual

        # 安全兜底：防止 NaN 继续传递
        hs = torch.nan_to_num(hs)

        hs = hs / attn.rescale_output_factor
        return hs

# 替换所有 attention processor 为上述实现，确保不走 fused/flash
def replace_all_attn_processors(pipe):
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    assert root is not None
    new = {name: AttnProcessorMe() for name, _ in root.attn_processors.items()}
    root.set_attn_processor(new)

replace_all_attn_processors(pipe)
orig_step = pipe.scheduler.step  # 先保存原始方法
print("orig_step",orig_step)
def wrapped_step(*args, **kwargs):
    # scheduler 有一个 counter 可以用来判断
    step_index = getattr(pipe.scheduler, "_step_index", None)
    total_steps = len(getattr(pipe.scheduler, "timesteps", []))
    if step_index is not None and step_index + 1 == total_steps:
        print(f"🚩 当前是最后一个时间步（第 {step_index+1}/{total_steps} 步）！")
    return orig_step(*args, **kwargs)

pipe.scheduler.step = wrapped_step  # 替换

# === 正常推理 ===
result = pipe(prompt="a cat on a red chair", num_inference_steps=28, guidance_scale=4.0,generator=generator)

# === 保存图片前做安全处理，避免 invalid value encountered in cast ===
# result.images[0] 是 PIL.Image；如果你从 numpy/torch 保存，也请做下面这一步
from PIL import Image
import numpy as np

img = result.images[0]


img.save("pixart_output.png")
print("✅ 已保存：pixart_output.png")
