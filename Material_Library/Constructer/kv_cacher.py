import os
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
from diffusers import PixArtAlphaPipeline
from diffusers.models.attention_processor import Attention,AttnProcessor, AttnProcessor2_0

# ========= 基本配置 =========
MODEL_PATH = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
PROMPT = "a cat on a red chair"
NUM_STEPS = 10          # 你可以改成 28
GUIDANCE = 4.0
DEVICE = "cuda:1"
DTYPE  = torch.float16
SEED   = 1234


# ========= 基本配置 =========
MODEL_PATH = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
PROMPT = "a cat on a red chair"
NUM_STEPS = 10          # 你可以改成 28
GUIDANCE = 4.0
DEVICE = "cuda:1"       # 按你说的用 cuda:1
DTYPE  = torch.float16
SEED   = 1234


# ========= 1. 自定义 Processor：在 self-attn 上记录 K/V =========
class AttnProcessorKVRecord(AttnProcessor2_0):
    """
    基于 diffusers 的 AttnProcessor2_0，增加：
    - 在 self-attention(attn1) 上记录每个 timestep 的 K/V
    - 写入 attn._kv_sink["k"] / ["v"]（list，每个元素 shape=[L, C]）
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        residual = hidden_states
        is_cross = encoder_hidden_states is not None
        input_ndim = hidden_states.ndim

        # ======== 以下基本复制自 AttnProcessor2_0 ========
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

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

        key   = attn.to_k(encoder_hidden_states)  # [B, L, C]
        value = attn.to_v(encoder_hidden_states)  # [B, L, C]

        # ======== A) 在 self-attn 上记录 K/V（未拆 head，shape 更直观） ========
        if getattr(attn, "_kv_capture", False) and not is_cross:
            with torch.no_grad():
                k_log = key
                v_log = value

                # 假设 batch 是 [uncond, cond] 拼一起，丢掉 uncond
                if k_log.shape[0] % 2 == 0:
                    B = k_log.shape[0]
                    k_log = k_log[B // 2:]  # 只保留 cond half
                    v_log = v_log[B // 2:]

                # [1, L, C] -> [L, C]
                if k_log.shape[0] == 1:
                    k_log = k_log.squeeze(0)
                    v_log = v_log.squeeze(0)

                k_log = k_log.detach().to(torch.float16).cpu()
                v_log = v_log.detach().to(torch.float16).cpu()

                sink = getattr(attn, "_kv_sink", None)
                if isinstance(sink, dict):
                    sink.setdefault("k", []).append(k_log)
                    sink.setdefault("v", []).append(v_log)

        # ======== B) 标准 attention 计算继续 ========
        query = attn.head_to_batch_dim(query)
        key_heads = attn.head_to_batch_dim(key)
        value_heads = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key_heads, attention_mask)
        hidden_states = torch.bmm(attention_probs, value_heads)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ========= 2. 给所有 Attention 挂上 KV sink，并替换 Processor =========
def enable_kv_capture(pipe) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """
    - 替换 transformer 中所有 Attention 的 processor 为 AttnProcessorKVRecord
    - 给每个 Attention 挂上 _kv_sink（dict: {"k": [], "v": []}）
    - 返回 kv_sinks: {层名: {"k": list, "v": list}}
    """
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    assert root is not None, "未找到 transformer / unet"

    # 1) 替换所有 attn_processors
    new_procs = {}
    for name, proc in root.attn_processors.items():
        new_procs[name] = AttnProcessorKVRecord()
    root.set_attn_processor(new_procs)

    # 2) 给每个 Attention 模块挂 sink
    kv_sinks: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    for name, m in root.named_modules():
        if isinstance(m, Attention):
            m._kv_capture = True
            m._kv_sink = {"k": [], "v": []}
            kv_sinks[name] = m._kv_sink

    print(f"✅ 已为 {len(kv_sinks)} 个 Attention 层启用 KV 捕获（self-attn 时记录）")
    return kv_sinks


# ========= 3. 计算相邻 timestep 间的 K/V 相似度 =========
def compute_kv_similarity_adjacent(k_list: List[torch.Tensor], v_list: List[torch.Tensor]):
    """
    k_list / v_list: list of [L, C]，长度 = T
    返回:
        sims_k: [T-1]，第 i 项 = step i vs step i+1 的 cos sim
        sims_v: [T-1]
    """
    assert len(k_list) == len(v_list)
    T = len(k_list)
    if T < 2:
        return [], []

    sims_k = []
    sims_v = []

    for t in range(T - 1):
        k1 = k_list[t].reshape(-1).float()
        k2 = k_list[t + 1].reshape(-1).float()
        v1 = v_list[t].reshape(-1).float()
        v2 = v_list[t + 1].reshape(-1).float()

        sim_k = F.cosine_similarity(k1, k2, dim=0).item()
        sim_v = F.cosine_similarity(v1, v2, dim=0).item()

        sims_k.append(sim_k)
        sims_v.append(sim_v)

    return sims_k, sims_v


# ========= 4. 计算「与最后一步」的 K/V 相似度 =========
def compute_similarity_to_last(k_list: List[torch.Tensor], v_list: List[torch.Tensor]):
    """
    k_list / v_list: list of [L, C]，长度 = T
    返回:
        sims_k: [T]，第 i 项 = step i vs step(T-1) 的 cos sim
        sims_v: [T]
    """
    assert len(k_list) == len(v_list)
    T = len(k_list)
    if T == 0:
        return [], []

    k_last = k_list[-1].reshape(-1).float()
    v_last = v_list[-1].reshape(-1).float()

    sims_k = []
    sims_v = []

    for t in range(T):
        k_t = k_list[t].reshape(-1).float()
        v_t = v_list[t].reshape(-1).float()

        sim_k = F.cosine_similarity(k_t, k_last, dim=0).item()
        sim_v = F.cosine_similarity(v_t, v_last, dim=0).item()

        sims_k.append(sim_k)
        sims_v.append(sim_v)

    return sims_k, sims_v


# ========= 5. 主流程 =========
def main():
    # 1) 加载 PixArt 模型
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    # 2) 启用 KV 捕获（替换所有 Attention 的 processor）
    kv_sinks = enable_kv_capture(pipe)

    # 打印所有 Attention 层
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    print("=== 所有 Attention 层 ===")
    for name, module in root.named_modules():
        if isinstance(module, Attention):
            print(name, "| is_cross_attention =", module.is_cross_attention)

    # 3) 固定随机种子
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)

    # 4) 正常推理（期间会自动记录 self-attn 的 K/V）
    out = pipe(
        prompt=PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
    )

    # 5) 对每个 Attention 层计算 K/V 相似度
    print("\n=== 每个 Attention 层的 KV 相似度（只看 self-attn） ===")
    for name, sink in kv_sinks.items():
        k_list = sink["k"]
        v_list = sink["v"]

        # cross-attn 在 is_cross=True 时不会记录，这里会是空的
        if len(k_list) == 0:
            continue

        T = len(k_list)
        sims_k_adj, sims_v_adj = compute_kv_similarity_adjacent(k_list, v_list)
        sims_k_last, sims_v_last = compute_similarity_to_last(k_list, v_list)

        print(f"\n层: {name}")
        print(f"  记录到 {T} 个 timestep 的 KV")

        # 相邻步之间
        print(f"  相邻 step 的 K 相似度: {[f'{x:.4f}' for x in sims_k_adj]}")
        print(f"  相邻 step 的 V 相似度: {[f'{x:.4f}' for x in sims_v_adj]}")

        # 每步 vs 最后一步
        print(f"  与最后一步对比的 K 相似度: {[f'{x:.4f}' for x in sims_k_last]}")
        print(f"  与最后一步对比的 V 相似度: {[f'{x:.4f}' for x in sims_v_last]}")

    # 6) 保存图像方便 sanity check
    os.makedirs("./kv_debug", exist_ok=True)
    out.images[0].save("./kv_debug/pixart_output.png")
    print("\n[save] 生成图像已保存到 ./kv_debug/pixart_output.png")


if __name__ == "__main__":
    main()