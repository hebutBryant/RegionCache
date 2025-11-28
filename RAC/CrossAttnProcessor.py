"""
一个自定义的AttnProcessor类。 替换原本Attention类中的Processor
hook出最后一层的注意力得分  和每一层的hidden_state

"""

from typing import Counter, Dict
import torch
from typing import Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers import PixArtAlphaPipeline
from diffusers.models.attention_processor import *
# 手算版 AttnProcessor：不调用 F.scaled_dot_product_attention
from typing import Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
import json
import os
from tqdm import tqdm 
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
)

# 可选：如果你的环境可能用到这些实现，解开注释即可一并纳入替换
try:
    from diffusers.models.attention_processor import (
        XFormersAttnProcessor,
        SlicedAttnProcessor,
        FusedAttnProcessor2_0,
        LoRAAttnProcessor,
    )
    _HAS_EXTRA = True
except Exception:
    _HAS_EXTRA = False
    XFormersAttnProcessor = SlicedAttnProcessor = FusedAttnProcessor2_0 = LoRAAttnProcessor = type("Dummy", (), {})

class AttnProcessorMe:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        is_cross = True
        

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            is_cross = False
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores2_0(is_cross,query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



def replace_attnprocessors_auto(pipe, include_others: bool = False):
    """
    将 transformer/unet 上的注意力 processor 自动替换为 AttnProcessorMe：
      - 默认替换：AttnProcessor2_0、AttnProcessor
      - include_others=True 时，也会替换 XFormers/Sliced/Fused/LoRA 等实现

    返回：restore() 闭包用于恢复、以及替换统计信息（before/after）。
    """
    # 1) 拿到根模块（PixArt: transformer；SD: unet）
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    if root is None:
        raise ValueError("❌ 这个 pipeline 没有 transformer 或 unet。")

    # 2) 统计替换前的处理器类型
    before = Counter(type(p).__name__ for p in root.attn_processors.values())

    # 3) 构建要替换的类型集合
    target_types = (AttnProcessor2_0, AttnProcessor)
    if include_others and _HAS_EXTRA:
        target_types = target_types + (
            XFormersAttnProcessor,
            SlicedAttnProcessor,
            FusedAttnProcessor2_0,
            LoRAAttnProcessor,
        )

    # 4) 逐个替换
    old = dict(root.attn_processors)  # 用于恢复
    new = {}
    replaced = 0
    total = 0
    for name, proc in root.attn_processors.items():
        total += 1
        if isinstance(proc, target_types):
            new[name] = AttnProcessorMe()
            replaced += 1
        else:
            new[name] = proc

    root.set_attn_processor(new)

    # 5) 统计替换后
    after = Counter(type(p).__name__ for p in root.attn_processors.values())
    print(f"✅ 已替换 {replaced}/{total} 个 attention processor 为 AttnProcessorMe")
    print(f"   替换前: {dict(before)}")
    print(f"   替换后: {dict(after)}")

    # 6) 恢复函数
    def restore():
        root.set_attn_processor(old)
        print("↩️ 已恢复原始 attention processors")

    return restore, {"before": dict(before), "after": dict(after)}


"""
打开可以去hook hidden_state的开关

"""

def enable_hidden_capture(pipe):
    hs_sinks = {}
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    for name, m in root.named_modules():
        if isinstance(m, Attention):
            m._xfeat_capture = True
            m._xfeat_sink = []
            hs_sinks[name] = m._xfeat_sink
    return hs_sinks