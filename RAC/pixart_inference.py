# model_path = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
import types
from typing import Dict, Optional, List, Union
import torch
import torch.nn.functional as F
from collections import OrderedDict
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import *
from diffusers.utils import deprecate  # AttnProcessorMe 里用到
from Rac_forward import rac_forward
from ReuseAttnProcessor import ReuseAttnProcessor
# from call_rewrite import rac__call__
from diffusers import PixArtAlphaPipeline
from utils.manage_cache import load_region_cache_as_tensor
PixArtAlphaPipeline.__call__ = PixArtAlphaPipeline.rac__call__
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models import PixArtTransformer2DModel
PixArtTransformer2DModel.__call__ = rac_forward
DTYPE = torch.float16
DEVICE = "cuda:0"
MODEL_PATH = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
PROMPT = "a dog on a desk"
import json
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Material_Library.Database.db_manager import RegionDB

cache_file = "../Material_Library/Constructer/cache/region_items.json"

class RegionCachePool:
    def __init__(self, max_size=10, device='cpu'):
        """
        :param max_size: 最多缓存多少个 region 文件
        :param device: 'cpu' (推荐，利用大内存) 或 'cuda' (速度最快但费显存)
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.device = device
        
    def get(self, key):
        """尝试获取缓存"""
        if key in self.cache:
            # 命中缓存：移动到最新位置 (LRU)
            self.cache.move_to_end(key)
            print(f"⚡ Cache Hit: {os.path.basename(key)}")
            return self.cache[key]
        return None

    def put(self, key, data):
        """
        存入缓存
        data: tuple (h_cache, r_indices, info)
        """
        if key in self.cache:
            self.cache.move_to_end(key)
            return

        # 1. 解包数据
        h_cache, r_indices, info = data

        # 2. 转移到指定设备 
        # detach() 剥离梯度，pin_memory() 加速后续传输
        if self.device == 'cpu':
            h_cache = h_cache.detach().cpu().pin_memory()
            r_indices = r_indices.detach().cpu().pin_memory()
        else:
            h_cache = h_cache.detach().to(self.device)
            r_indices = r_indices.detach().to(self.device)
            
        # 3. 存入字典
        self.cache[key] = (h_cache, r_indices, info)

        # 4. 检查容量，溢出则删除最早的
        if len(self.cache) > self.max_size:
            popped_key, _ = self.cache.popitem(last=False)
            print(f"🗑️ Cache Full, Evicting: {os.path.basename(popped_key)}")

def get_cache_simulate(cache_path=cache_file, dtype=torch.float16, device="cpu"):
    """
    模拟加载 region cache，用于区域复用 / region-guided inference。

    返回格式:
    {
        region_name: {
            "prompt": str,W
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

def update_pixart_transformer_rac(transformer):
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
        

        return transformer

if __name__ == "__main__":
    gen = torch.Generator(device="cuda:0").manual_seed(1234)

    pipe = PixArtAlphaPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    ).to(DEVICE)
    
    update_pixart_transformer_rac(pipe.transformer)
    
    print("pipe.__call__ 绑定方法：", pipe.__call__)
    print("底层函数对象：", pipe.__call__.__func__)

    # from inspect import ismethod, isfunction
    # print("是否为绑定方法:", ismethod(pipe.__call__))
    # print("是否指向 rac__call__:", pipe.__call__.__func__ is rac__call__)

    cache_paths = [
        "/home/liuhy/RegionCache/Material_Library/Constructer/cache/chunks/a_cat.pt",
        "/home/liuhy/RegionCache/Material_Library/Constructer/cache/chunks/a_red_chair.pt"
    ]

    # 1. 初始化 DB
    db = RegionDB()
    
    # 2. 初始化带 DB 通信功能的 Pool
    REGION_POOL = RegionCachePool(max_size=10, device='cpu')

    # 3. 定义我们想要复用的“意图” 
    user_queries = [
        "a cat",       # 应该能搜到 "a cat.pt" (如果之前 prompt 是 "a cat on...")
        "red chair" # 应该能搜到 "a red chair.pt"
    ]

    all_hidden_caches = []
    all_region_indices = []
    
    # 只需要读取第一个文件的 info 用于设置 steps 和 scale (假设所有 cache 的参数一致)
    base_info = None 

    print(f"正在加载 {len(cache_paths)} 个区域缓存...")

    print("🔥正在预热缓存池 (Pre-warming Cache Pool)...")
    pre_warm_start = time.time()
    for path in cache_paths:
        # 直接检查字典 key，避免触发 get() 里的 print 和 LRU 移动
        if path not in REGION_POOL.cache:
            # 读磁盘
            h_cache, r_indices, info, _ = load_region_cache_as_tensor(path, num_layers=28)
            # 存入内存池
            REGION_POOL.put(path, (h_cache, r_indices, info))
            print(f"  └─ 已加载: {os.path.basename(path)}")
    
    pre_warm_end = time.time()

    print("✅ 预热完成，开始推理！")

    import gc
    gc.collect()

    start_load_time = time.time()  ### 计时开始 ###

    for query in user_queries:
        # A. 从 DB 搜索
        results = db.search_region(query, n_results=1)
        
        if not results:
            print(f"❌ DB 未找到与 '{query}' 相关的缓存")
            continue
            
        best_match = results[0]
        file_path = best_match['id']       # 之前存的绝对路径
        metadata = best_match['metadata']
        score = best_match['distance']
        
        print(f"🎯 Query: '{query}' -> Found: '{metadata['region_name']}' (Score: {score:.4f})")
        print(f"   Path: {file_path}")
        print(f"   Current DB Status: {metadata['pool_status']}, Pos: {metadata['virtual_pos']}")

        # B. 尝试从 Pool 获取或加载 
        cached_data = REGION_POOL.get(file_path)
        
        if cached_data is not None:
            h_cache, r_indices, info = cached_data
        else:
            # Miss -> Load from disk
            if not os.path.exists(file_path):
                print(f"⚠️ 文件丢失: {file_path}")
                continue
                
            h_cache, r_indices, info, _ = load_region_cache_as_tensor(file_path, num_layers=28)
            
            # C. Put into Pool (这里会自动触发 DB update_pool_status)
            REGION_POOL.put(file_path, (h_cache, r_indices, info))

        # 收集用于推理的数据
        all_hidden_caches.append(h_cache.to(DEVICE, non_blocking=True))
        all_region_indices.append(r_indices.to(DEVICE, non_blocking=True))
        
        if base_info is None:
            base_info = info

    # 2. 执行拼接 (Concatenate)
    # hidden_cache shape: [num_steps, num_layers, num_tokens, dim]
    # 需要在 dim=2 (num_tokens) 上拼接
    merged_hidden_cache = torch.cat(all_hidden_caches, dim=2)

    # region_indices shape: [num_tokens]
    # 需要在 dim=0 上拼接
    merged_region_indices = torch.cat(all_region_indices, dim=0)

    end_load_time = time.time()    ### <--- 结束记录时间
    load_duration = end_load_time - start_load_time

    print("############## Merged Hidden Cache ####################", merged_hidden_cache.shape)
    print("############## Merged Region Indices ####################", merged_region_indices.shape)

    my_update_steps = [0,1,2,3,4,5,6,7]

    ReuseAttnProcessor.reset_time()

    with torch.no_grad():
        out = pipe(
            prompt=PROMPT,
            num_inference_steps=info.get("num_inference_steps", 15),
            guidance_scale=info.get("guidance_scale", 4.0),
            cached_hidden_states=merged_hidden_cache,       # [num_steps, num_layers, K, C] or 你定义的形状
            region_indices=merged_region_indices,
            generator=gen,   # [K]
            update_steps=my_update_steps,
        )
    

    print(f"Total Attention Time: {ReuseAttnProcessor.get_time()} ms")
    print(f"✅ 预热完成! 耗时: {pre_warm_end - pre_warm_start:.4f}s")
    print(f"⏱️ [Cache Load Stats] 读取并合并 {len(cache_paths)} 个文件耗时: {load_duration:.4f} 秒")

    image = out.images[0]
    image.save("rac_test.png")
    print("保存到 rac_test.png")

    
