# model_path = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
import types
from typing import Dict, Optional, List, Union
import torch
import torch.nn.functional as F

from diffusers.models.attention import Attention
from diffusers.models.attention_processor import *
from diffusers.utils import deprecate  # AttnProcessorMe 里用到
from Rac_forward import rac_forward
from ReuseAttnProcessor import ReuseAttnProcessor
from call_rewrite import rac__call__
from diffusers import PixArtAlphaPipeline
from utils.manage_cache import load_region_cache_as_tensor
PixArtAlphaPipeline.__call__ = PixArtAlphaPipeline.rac__call__
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models import PixArtTransformer2DModel
PixArtTransformer2DModel.__call__ = rac_forward
DTYPE = torch.float16
DEVICE = "cuda:1"
MODEL_PATH = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
PROMPT = "a dog on the desk"
import json
import os
import time

cache_file = "../Material_Library/Constructer/cache/region_items.json"
def get_cache_simulate(cache_path=cache_file, dtype=torch.float16, device="cpu"):
    """
    模拟加载 region cache，用于区域复用 / region-guided inference。

    返回格式:
    {
        region_name: {
            "prompt": str,
            "hidden": Tensor[K, C],
            "indices": Tensor[K]
        },
        ...
    }
    """

    # ===== 1) 检查文件是否存在 =====
    if not os.path.exists(cache_path):
        print(f"⚠️ 缓存文件不存在: {cache_path}")
        return None

    # ===== 2) 解析 JSON =====
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法解析 JSON: {e}")
        return None

    if not isinstance(data, list):
        print("❌ JSON 格式异常（期望数组列表）")
        return None

    # ===== 3) 组装结构化 cache dict =====
    cache = {}

    for entry in data:
        try:
            region_name = entry["region_name"]
            hidden_state = entry["hidden_state"]     # list[list]
            indices = entry["indices"]               # list[int]
            prompt = entry.get("prompt", "")

            # 转 tensor
            hidden_tensor = torch.tensor(hidden_state, dtype=dtype, device=device)
            index_tensor = torch.tensor(indices, dtype=torch.long, device=device)

            cache[region_name] = {
                "prompt": prompt,
                "hidden": hidden_tensor,   # [K, C]
                "indices": index_tensor    # [K]
            }

        except KeyError as e:
            print(f"⚠️ 缓存条目缺失字段: {e}，跳过该条目。")
            continue

    print(f"📦 成功加载 region cache ({len(cache)} 个区域) from: {cache_path}")
    return cache

def update_pixart_pipeline_rac(pipeline):
    transformer = pipeline.transformer
    blocks = []
    if hasattr(transformer, "layers"):
        blocks = transformer.layers
    elif hasattr(transformer, "transformer_blocks"):
        blocks = transformer.transformer_blocks

    for block in blocks:
        if hasattr(block, "attn1"):
            block.attn1.set_processor(ReuseAttnProcessor())
        # if hasattr(block, "attn2"):
        #     block.attn2.set_processor(ReuseAttnProcessor())
        if hasattr(block, "attn"):
            block.attn.set_processor(ReuseAttnProcessor())

    return pipeline



if __name__ == "__main__":
    gen = torch.Generator(device="cuda:1").manual_seed(1234)

    pipe = PixArtAlphaPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    ).to(DEVICE)
    pipe = update_pixart_pipeline_rac(pipe)
    print("pipe.__call__ 绑定方法：", pipe.__call__)
    print("底层函数对象：", pipe.__call__.__func__)

    from inspect import ismethod, isfunction
    print("是否为绑定方法:", ismethod(pipe.__call__))
    print("是否指向 rac__call__:", pipe.__call__.__func__ is rac__call__)

    cache_paths = [
        "/home/liuhy/RegionCache/Material_Library/Constructer/cache/chunks/a_cat.pt",
        "/home/liuhy/RegionCache/Material_Library/Constructer/cache/chunks/a_red_chair.pt"
    ]

    all_hidden_caches = []
    all_region_indices = []
    
    # 只需要读取第一个文件的 info 用于设置 steps 和 scale (假设所有 cache 的参数一致)
    base_info = None 

    print(f"正在加载 {len(cache_paths)} 个区域缓存...")


    for i, path in enumerate(cache_paths):
        # 加载单个区域
        h_cache, r_indices, info, _ = load_region_cache_as_tensor(path, num_layers=28)
        
        # 收集 Tensor
        all_hidden_caches.append(h_cache)
        all_region_indices.append(r_indices)
        
        if i == 0:
            base_info = info
            print(f"基准配置 (来自第一个文件): Steps={base_info.get('num_inference_steps')}, Scale={base_info.get('guidance_scale')}")

    # 2. 执行拼接 (Concatenate)
    # hidden_cache shape: [num_steps, num_layers, num_tokens, dim]
    # 需要在 dim=2 (num_tokens) 上拼接
    merged_hidden_cache = torch.cat(all_hidden_caches, dim=2)

    # region_indices shape: [num_tokens]
    # 需要在 dim=0 上拼接
    merged_region_indices = torch.cat(all_region_indices, dim=0)

    print("############## Merged Hidden Cache ####################", merged_hidden_cache.shape)
    print("############## Merged Region Indices ####################", merged_region_indices.shape)

    with torch.no_grad():
        out = pipe(
            prompt=PROMPT,
            num_inference_steps=info.get("num_inference_steps", 15),
            guidance_scale=info.get("guidance_scale", 4.0),

            # ⭐ 关键：把 cache 传给 rac__call__
            cached_hidden_states=merged_hidden_cache,       # [num_steps, num_layers, K, C] or 你定义的形状
            region_indices=merged_region_indices,
            generator=gen,   # [K]
        )
    

    image = out.images[0]
    image.save("rac_test.png")
    print("保存到 rac_test.png")

    
