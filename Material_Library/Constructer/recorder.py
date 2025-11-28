# xattn_recorder2_0.py
from typing import Dict
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





def replace_attnprocessor2_0_with_attnprocessor(pipe):
    """
    将 PixArt (或任何 Diffusers 模型) 中 transformer/unet 的所有
    AttnProcessor2_0 层替换为 AttnProcessor。
    用于禁用 PyTorch 2.0 fused scaled_dot_product_attention，
    改为旧版手算 attention 逻辑。

    返回 restore() 函数，可恢复原始 processor。
    """
    # 取出 transformer（PixArt 里是 pipe.transformer；SD 里是 pipe.unet）
    if hasattr(pipe, "transformer"):
        tr = pipe.transformer
    elif hasattr(pipe, "unet"):
        tr = pipe.unet
    else:
        raise ValueError("❌ 这个 pipeline 没有 transformer 或 unet。")

    old = dict(tr.attn_processors)
    new = {}
    replaced, total = 0, 0

    for name, proc in tr.attn_processors.items():
        total += 1
        # 只替换 AttnProcessor2_0 实例
        if isinstance(proc, AttnProcessor2_0):
            new[name] = AttnProcessorMe()
            replaced += 1
        else:
            new[name] = proc

    tr.set_attn_processor(new)

    def restore():
        """恢复原始的 attention processor 配置"""
        tr.set_attn_processor(old)

    print(f"✅ 已替换 {replaced}/{total} 个 AttnProcessor2_0 → AttnProcessor")
    return restore

# 这一行时，程序做了这些事情：

# 遍历模型里所有的子模块；

# 找出那些是 cross-attention 层（Attention 模块并且 is_cross_attention=True）；

# 对每个这样的模块动态加上两个属性：

# _xattn_capture = True
# 👉 这是一个标志位，告诉注意力处理逻辑“我现在要捕获注意力矩阵”。

# _xattn_sink = []
# 👉 这是一个空的 list，用来存储注意力矩阵。

def enable_xattn_capture(pipe):
    """
    为 pipeline 中的所有 cross-attention 层启用注意力矩阵捕获，
    并返回 {层名: list} 的字典，推理过程中矩阵会 append 到这些 list 里。
    """
    sinks = {}
    # PixArt 用 pipe.transformer；SD-UNet 则用 pipe.unet
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    assert root is not None, "未找到 transformer / unet"

    for name, module in root.named_modules():
        if isinstance(module, Attention) and getattr(module, "is_cross_attention", False):
            module._xattn_capture = True      # 打开开关
            module._xattn_sink = []           # 绑定一个 list 作接收器
            sinks[name] = module._xattn_sink  # 暴露给外部
    return sinks

def disable_xattn_capture(pipe):
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    if root is None:
        return
    for _, module in root.named_modules():
        if isinstance(module, Attention):
            if hasattr(module, "_xattn_capture"): delattr(module, "_xattn_capture")
            if hasattr(module, "_xattn_sink"):    delattr(module, "_xattn_sink")
            if hasattr(module, "_xattn_last"):    delattr(module, "_xattn_last")


if __name__ == "__main__":
    pipe = PixArtAlphaPipeline.from_pretrained(
        "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16
    ).to("cuda")

    replace_attnprocessor2_0_with_attnprocessor(pipe)
    
    sinks = enable_xattn_capture(pipe)

    # 正常推理即可
    result = pipe(prompt="a cat on a red chair", num_inference_steps=28, guidance_scale=4.0)

    # 2) 保存图片
    img = result.images[0]   # PIL.Image
    img_path = "pixart_cat_on_red_chair.png"
    img.save(img_path)
    print(f"✅ 已保存图片：{img_path}")

    # 推理后，直接从 sinks 里取矩阵
    prompt = "a cat on a red chair"

    # 有的管线可能有 tokenizer / tokenizer_2，这里做个兜底
    tok = getattr(pipe, "tokenizer", None) or getattr(pipe, "tokenizer_2", None)
    if tok is None:
        raise RuntimeError("未在 pipe 上找到 tokenizer/tokenizer_2。")

    # 不要 padding，这样长度就是“真实 token 数”（含特殊符号）
    enc = tok(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    # 如果 tokenizer 仍然做了 padding，可用 attention_mask 求和得到真实长度
    if "attention_mask" in enc and enc["attention_mask"] is not None:
        prompt_len = int(enc["attention_mask"].sum().item())
    else:
        prompt_len = int(enc["input_ids"].shape[1])

    print(f"📝 prompt token length = {prompt_len}")

    # === 2) 只保存最后一步的 cross-attn，并在最后一维裁到 prompt_len ===
    save_json = "cross_attn_firststep.json"
    dump = {}

    print(f"💾 开始保存每层最后一步的 cross-attention 注意力矩阵（裁剪到 {prompt_len} 个 token）...")

    for layer_name, lst in tqdm(sinks.items(), desc="保存层", ncols=100):
        if not lst:
            print(f"⚠️ {layer_name}: 0 条（未捕获到注意力）")
            continue

        t = lst[0]  # 最后一个扩散步
        arr = t.detach().float().cpu().numpy()  # 形状通常是 [B*H, Nq, Nk] 或 [B, H, Nq, Nk]

        # 只保留最后一维（文字 token 维）的前 prompt_len
        if arr.shape[-1] < prompt_len:
            # 极少见：如果实际矩阵的 Nk 比 prompt_len 还短，就不裁（或给出提示）
            print(f"⚠️ {layer_name}: Nk={arr.shape[-1]} < prompt_len={prompt_len}，保持原样。")
            cropped = arr
        else:
            cropped = arr[..., :prompt_len]

        dump[layer_name] = {
            "index": len(lst) - 1,
            "shape": list(cropped.shape),
            "data": cropped.round(8).tolist(),
        }

    with open(save_json, "w") as f:
        json.dump(dump, f)

    print(f"✅ 已将最后一步（裁剪到 prompt_len）的注意力矩阵写入 JSON：{save_json}")
    print(f"📦 共保存 {len(dump)} 个 cross-attn 层的结果。")