# model_path = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
import types
from typing import Dict, Optional, List, Union
import torch #type: ignore
import torch.nn.functional as F  #type: ignore
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
PROMPT = "a cat is at the right "
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

def benchmark_granular_transfer(cpu_tensor, num_steps, num_layers):
    """
    cpu_tensor: 完整的缓存数据 [Total_Steps, Layers, Batch, Tokens, Dim] 
    """
    print(f"\n🔬 Starting Granular Benchmark (Shape: {cpu_tensor.shape})")
    
    # 确保 tensor 在 CPU 且 pinned (如果之前已经 pin 过了这里没开销)
    cpu_tensor = cpu_tensor.cpu().pin_memory()
    device = torch.device("cuda:0")

    # 预热 CUDA 环境
    warmup_data = torch.randn(1024, 1024, device='cpu').pin_memory()
    for _ in range(5):
        _ = warmup_data.to(device, non_blocking=True)
    torch.cuda.synchronize()

    # ==========================================
    # 场景 A: 测量 "单 Step" (传输该 Step 下所有 Layer)
    # ==========================================
    step_times = []
    # 假设第 0 维是 Step (根据你的 merged_hidden_cache 调整)
    # 如果你的 tensor 是 [Layers, Steps, ...], 请改为 slice 第 1 维
    for s in range(num_steps):
        # 1. 切片 (Slice) - 这只是视图操作，几乎不耗时
        # 假设结构: [Step, Layer, ...]
        step_chunk = cpu_tensor[s] 
        
        # 2. 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        # 模拟传输：把这一步的所有数据搬到 GPU
        _ = step_chunk.to(device, non_blocking=True)
        end_event.record()
        
        # 3. 等待完成并记录
        end_event.synchronize()
        step_times.append(start_event.elapsed_time(end_event)) # 单位: ms

    avg_step_time = sum(step_times) / len(step_times)
    print(f"⏱️ [Per Step] Avg: {avg_step_time:.4f} ms | Min: {min(step_times):.4f} ms | Max: {max(step_times):.4f} ms")

    # ==========================================
    # 场景 B: 测量 "单 Block/Layer" (最细粒度)
    # ==========================================
    layer_times = []
    
    # 双重循环模拟完全拆解
    for s in range(num_steps):
        for l in range(num_layers):
            # 切片出单个 Layer 的数据
            # 假设结构: [Step, Layer, ...]
            block_chunk = cpu_tensor[s, l] 
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = block_chunk.to(device, non_blocking=True)
            end_event.record()
            
            end_event.synchronize()
            layer_times.append(start_event.elapsed_time(end_event))

    avg_layer_time = sum(layer_times) / len(layer_times)
    print(f"⏱️ [Per Block] Avg: {avg_layer_time:.4f} ms | Min: {min(layer_times):.4f} ms | Max: {max(layer_times):.4f} ms")

    return avg_step_time, avg_layer_time

def shift_region_content(hidden_state, indices, shift_x, shift_y, feat_h=64, feat_w=64):
    """
    对缓存的区域进行 2D 平移。
    
    Args:
        hidden_state: Tensor [Layers, 1, K, Dim] (根据你的 dim=2 拼接推断)
        indices: Tensor [K]
        shift_x: 水平平移步长 (Patch单位, 1单位约等于原图16像素)
        shift_y: 垂直平移步长
        feat_h, feat_w: 特征图尺寸 (PixArt-1024 通常是 64x64)
    
    Returns:
        valid_hidden_state, valid_indices
    """
    device = indices.device
    
    # 1. 1D index -> 2D grid coordinates
    # indices 是扁平化的，范围 0 ~ 4095
    rows = indices // feat_w
    cols = indices % feat_w
    
    # 2. Apply shift
    new_rows = rows + shift_y
    new_cols = cols + shift_x
    
    # 3. Boundary Check (生成有效掩码)
    # 必须在 0 ~ 63 之间
    mask_h = (new_rows >= 0) & (new_rows < feat_h)
    mask_w = (new_cols >= 0) & (new_cols < feat_w)
    valid_mask = mask_h & mask_w  # [K] (bool)
    
    if valid_mask.sum() == 0:
        print("⚠️ Warning: Region shifted completely out of bounds!")
        return None, None

    # 4. Filter indices and hidden states
    # 只保留移位后仍在图内的点
    final_rows = new_rows[valid_mask]
    final_cols = new_cols[valid_mask]
    
    # 5. 2D -> 1D index
    new_indices = final_rows * feat_w + final_cols
    
    # hidden_state 假设形状是 [Layers, Batch, K, Dim]，我们需要在 K (dim=2) 维度进行切片

    valid_hidden_state = hidden_state[:, :, valid_mask, :]
    
    return valid_hidden_state, new_indices

if __name__ == "__main__":
    # 1. 初始设置
    # 注意：这里定义的 seed 只是为了初始化，后面每次使用前都要重置
    SEED = 1234
    gen = torch.Generator(device="cuda:0").manual_seed(SEED)

    print("⏳ Loading Pipeline...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    
    update_pixart_transformer_rac(pipe.transformer)
    
    # -----------------------------------------------------------
    # 准备数据 (Cache Pool & DB) - 保持不变
    # -----------------------------------------------------------
    cache_paths = [
        "/home/liuhy/RegionCache/Material_Library/Constructer/cache/chunks/a_cat.pt",
        "/home/liuhy/RegionCache/Material_Library/Constructer/cache/chunks/a_red_chair.pt"
    ]
    
    db = RegionDB()
    REGION_POOL = RegionCachePool(max_size=10, device='cpu') 
    region_configs = [
        {"query": "cat",       "offset": (20, 0)},   # 把猫向右移
        #{"query": "red chair", "offset": (0, 0)}  # 把椅子向左下移
    ]

    # --- Step A: 预热磁盘缓存 (Disk IO Warm-up) ---
    for path in cache_paths:
        if path not in REGION_POOL.cache:
            h_cache, r_indices, info, _ = load_region_cache_as_tensor(path, num_layers=28)
            REGION_POOL.put(path, (h_cache, r_indices, info))
    
    # -----------------------------------------------------------
    # 组装 Tensor (模拟检索)
    # -----------------------------------------------------------
    temp_hidden_caches = []
    temp_region_indices = []
    base_info = None

    # 1. 确保之前的操作已完成，开始计时
    torch.cuda.synchronize()
    start_load_time = time.perf_counter() # 使用 perf_counter 精度更高

    for config in region_configs:
        query = config["query"]
        offset_x, offset_y = config["offset"]

        # 1. Search
        t0 = time.time()
        results = db.search_region(query, n_results=1)
        print(f"🔎 DB Search Time: {(time.time() - t0)*1000:.2f} ms")
        
        if not results: 
            print(f"❌ Not found: {query}")
            continue

        file_path = results[0]['id']
        cached_data = REGION_POOL.get(file_path)
        
        # 兜底加载
        if cached_data is None and os.path.exists(file_path):
            h_c, r_i, inf, _ = load_region_cache_as_tensor(file_path, num_layers=28)
            REGION_POOL.put(file_path, (h_c, r_i, inf))
            cached_data = (h_c, r_i, inf)
            
        if cached_data:
            h_cache, r_indices, info = cached_data
            
            # Move to GPU for calculation
            h_cache = h_cache.to(DEVICE, non_blocking=True)
            r_indices = r_indices.to(DEVICE, non_blocking=True)

            # ==========================================
            #               应用位移
            # ==========================================
            if offset_x != 0 or offset_y != 0:
                print(f"↔️ Shifting '{query}': ({offset_x}, {offset_y})")
                h_cache, r_indices = shift_region_content(
                    h_cache, r_indices, 
                    shift_x=offset_x, 
                    shift_y=offset_y,
                    feat_h=64, feat_w=64  # PixArt 1024 的特征图大小
                )
            
            # 如果移出界了返回 None，需要判断
            if h_cache is not None:
                temp_hidden_caches.append(h_cache)
                temp_region_indices.append(r_indices)
                if base_info is None: base_info = info
            else:
                print(f"⚠️ Region '{query}' dropped (out of bounds).")

    if len(temp_hidden_caches) > 0:
        merged_hidden_cache = torch.cat(temp_hidden_caches, dim=2)
        merged_region_indices = torch.cat(temp_region_indices, dim=0)
    else:
        # 处理没有任何缓存的情况
        print("⚠️ No valid cache to merge.")
        sys.exit(0)

    # 2. 等待传输和拼接完成，停止计时
    torch.cuda.synchronize()
    end_load_time = time.perf_counter()
    
    load_duration_ms = (end_load_time - start_load_time) * 1000
    benchmark_granular_transfer(merged_hidden_cache, num_steps=15, num_layers=28)

    # -----------------------------------------------------------
    # Step B: GPU 模型预热 (GPU Warm-up)
    # -----------------------------------------------------------
    print("\n🔥 [Warm-up] GPU Kernel Compilation...")
    
    # 使用一个临时的 Generator 进行预热，或者预热后立刻重置主 Generator
    # 这里我们直接用 pipe 跑，但跑完必须重置状态
    warmup_steps = [0, 1] 
    torch.cuda.synchronize()
    
    try:
        with torch.no_grad():
            # 使用一个无关的临时种子进行预热，避免影响主 gen 的状态
            temp_gen = torch.Generator(device=DEVICE).manual_seed(9999)
            
            _ = pipe(
                prompt=PROMPT,
                num_inference_steps=2, 
                guidance_scale=4.0,
                cached_hidden_states=merged_hidden_cache,
                region_indices=merged_region_indices,
                generator=temp_gen, # 使用临时生成器
                update_steps=warmup_steps,
                output_type="latent",
            )
    except Exception as e:
        print(f"Warmup warning: {e}")

    torch.cuda.synchronize()
    print("✅ GPU Warm-up Done.")
    
    # 1. 显存清理
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2.重置随机种子
    gen.manual_seed(SEED) 
    print(f"🔄 Generator Reset to seed {SEED}")

    # 3. 重置 Attention Processor 的内部状态
    ReuseAttnProcessor.reset_time()
    
    # 如果 ReuseAttnProcessor 还有其他 reset 方法（比如 reset_cache, reset_counter），请在这里调用！
    if hasattr(ReuseAttnProcessor, 'reset_state'):
        ReuseAttnProcessor.reset_state()
        print("🔄 ReuseAttnProcessor State Reset")
    elif hasattr(ReuseAttnProcessor, 'reset'):
         ReuseAttnProcessor.reset()

    # -----------------------------------------------------------
    # 正式测量 (Benchmark)
    # -----------------------------------------------------------
    print("\n🚀 Starting Benchmark...")
    
    my_update_steps = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    num_inference_steps = base_info.get("num_inference_steps", 15)

    torch.cuda.synchronize()
    start_infer_time = time.time()

    with torch.no_grad():
        out = pipe(
            prompt=PROMPT,
            num_inference_steps=num_inference_steps,
            guidance_scale=base_info.get("guidance_scale", 4.0),
            cached_hidden_states=merged_hidden_cache,
            region_indices=merged_region_indices,
            generator=gen, # 这里传入的是刚刚重置过的 gen
            update_steps=my_update_steps
        )

    end_infer_time = time.time()

    # -----------------------------------------------------------
    # 结果输出
    # -----------------------------------------------------------
    print("\n📊 Benchmark Results:")
    print(f"📂 Cache Load & Assembly Time (RAM->VRAM): {load_duration_ms:.4f} ms")
    print(f"🖥️  Total Attention Time: {ReuseAttnProcessor.get_time()} ms")
    print(f"✅ 推理完成! 耗时: {end_infer_time - start_infer_time:.4f}s")

    image = out.images[0]
    image.save("rac_test.png")
    print("保存到 rac_test.png")
    