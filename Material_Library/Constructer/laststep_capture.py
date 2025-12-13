import types
from typing import Dict, Optional, List, Union
import torch
import torch.nn.functional as F

from diffusers.models.attention import Attention
from diffusers import PixArtAlphaPipeline
from diffusers.models.attention_processor import *
from diffusers.utils import deprecate  # AttnProcessorMe 里用到

import json
import sys
import os
from tqdm import tqdm
import re
import spacy
import math
import matplotlib.pyplot as plt
nlp = spacy.load("en_core_web_sm")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Database.db_manager import RegionDB

class AttnProcessorMe:
    r"""
    自定义 attention processor:
    - 保持原有 attention 计算不变
    - 额外记录 hidden_states，用于分析
    - 只保留 token 维度（第 2 维），不保存 CFG 的 unconditional 半部分
    """

    def __init__(
        self,
        hidden_sink: Optional[list] = None,
        move_to_cpu: bool = True,
        tag: str = "",
        drop_cfg_uncond: bool = True,
        reduce_to_tokens: bool = True,
    ):
        """
        hidden_sink: 外部传进来的 list，用来收集所有记录
        move_to_cpu: 记录时把数据挪到 CPU，减轻显存占用
        tag: 标记这个 processor 对应的层名
        drop_cfg_uncond: True 时，假定 batch 是 [uncond, cond] 拼在一起，只保留 cond 半部分
        reduce_to_tokens: True 时，把 [B, L, C] 压到只和 L 有关（比如 [L]）
        """
        self.hidden_sink = hidden_sink
        self.move_to_cpu = move_to_cpu
        self.tag = tag
        self.drop_cfg_uncond = drop_cfg_uncond
        self.reduce_to_tokens = reduce_to_tokens

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
        # ======== 原始逻辑 ========
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = (
                "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise "
                "an error in the future. `scale` should directly be passed while calling the underlying pipeline "
                "component i.e., via `cross_attention_kwargs`."
            )
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

        attention_probs = attn.get_attention_scores2_0(is_cross=is_cross, query=query, key=key, attention_mask=attention_mask)

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

        # ======== 额外：记录 hidden_states ========
        # 注意：只在 hs_for_log 上动，不动真正参与计算的 hidden_states
        if self.hidden_sink is not None:
            with torch.no_grad():
                hs_for_log = hidden_states

                # 统一成 [B, L, C] 这种形状再处理
                if hs_for_log.ndim == 4:
                    B, C, Hh, Ww = hs_for_log.shape
                    hs_for_log = hs_for_log.view(B, C, Hh * Ww).transpose(1, 2)  # [B, L, C]

                # 1) 丢掉 CFG 的 unconditional 半部分
                if self.drop_cfg_uncond and hs_for_log.shape[0] % 2 == 0:
                    B = hs_for_log.shape[0]
                    hs_for_log = hs_for_log[B // 2 :]   # 只保留后半部分（通常是 cond）
                    # 现在通常是 [1, L, C]

                # 2) 不再对 channel 做平均，保留完整 (L, C)
                #    如果 batch 只剩 1，就 squeeze 掉 batch 维
                if hs_for_log.shape[0] == 1:
                    hs_for_log = hs_for_log.squeeze(0)   # [L, C]
                # 否则你也可以选择对 batch 做平均：
                # hs_for_log = hs_for_log.mean(dim=0)   # [L, C]

                # 3) 挪到 CPU + 转 fp16
                if self.move_to_cpu:
                    hs_for_log = hs_for_log.detach().cpu().to(torch.float16)
                else:
                    hs_for_log = hs_for_log.detach().to(torch.float16)

                # 4) 只记录 attn1（self-attention）的 hidden
                if ".attn1" in getattr(self, "tag", ""):
                    self.hidden_sink.append(
                        {
                            "tag": self.tag,
                            "shape": tuple(hs_for_log.shape),  # 现在应为 (L, C) = (4096, 1152)
                            "hidden": hs_for_log,
                        }
                    )
        # print("######################hidden_states##################",hidden_states.shape)

        return hidden_states



# =========================================================
# 1) 目标层名解析：支持 "transformer_blocks.10.attn2"
# =========================================================
def _norm_name(s: str) -> str:
    return s.replace(".", "_").lower()


def resolve_layer_name(pipe, user_layer_name: str) -> str:
    """
    把用户层名（下划线或点号风格）解析成真实的 transformer.named_modules() 层名。
    """
    want = _norm_name(user_layer_name)
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    assert root is not None, "未找到 transformer/unet"

    # 完全匹配优先
    for name, m in root.named_modules():
        if isinstance(m, Attention) and _norm_name(name) == want:
            return name

    # 兜底：常见 'blocks.10' + 'attn2' 的近似匹配
    candidates = []
    for name, m in root.named_modules():
        if isinstance(m, Attention):
            n = _norm_name(name)
            if "attn2" in n and any(k in n for k in ["blocks_10", "blocks.10", "block_10", "block.10", "_10_"]):
                candidates.append(name)
    if candidates:
        return sorted(candidates)[-1]

    raise RuntimeError(f"未找到 Attention 层：{user_layer_name}")


# =========================================================
# 2) 给目标层挂 sink 与标记（只此一层会写入）
# =========================================================
def enable_target_capture(pipe, target_layer: str):
    """
    只给目标 Attention 层挂 sink 与标记字段：
      _cap_is_target_layer = True
      _cap_sink            = []  # 每次覆盖，最终只保留最后一步
      _cap_cur_step
      _cap_cur_timestep
    返回 (module, sink)
    """
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    assert root is not None

    mod = None
    for name, m in root.named_modules():
        if isinstance(m, Attention) and name == target_layer:
            m._cap_is_target_layer = True
            m._cap_sink = []  # 这里会始终只保存一条（最后一步）
            m._cap_cur_step = -1
            m._cap_cur_timestep = -1
            mod = m
            break

    if mod is None:
        raise RuntimeError(f"目标层未启用 sink（找不到层）：{target_layer}")

    return mod, mod._cap_sink


# =========================================================
# 3) 在 transformer 前向“开始前”写入当前 step / timestep
#    不再判断是否最后一步，最后一步通过“覆盖策略”自然得到
# =========================================================
def install_laststep_flag_pre_hook(pipe, target_module: Attention):
    """
    注册一个 forward_pre_hook 在 transformer/unet 上：
    每次前向开始前，根据传入的 timestep：
      - 计算当前 step 索引（用调用次数计数）
      - 记录当前 timestep 数值
      - 把这些标志写入“目标层”对象（target_module）

    我们不再用“是否最后一步”做过滤，
    而是在保存时每次覆盖，推理结束后自然得到最后一步的注意力。
    """
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    assert root is not None

    # 用 pipe 上的计数器记录当前是第几次调用
    pipe._cap_step_idx = 0

    def _pre_hook(module, args):
        # PixArtAlphaTransformer forward: (..., timestep, ...)
        # 如果你确定是第 3 个参数是 timestep，则用 2；否则需要根据实际情况调整
        timestep = args[2] if len(args) >= 3 else None
        if timestep is None:
            return

        # 当前 timestep 的实际值（int）
        cur_t = int(timestep.item() if hasattr(timestep, "item") else timestep)

        ts = getattr(pipe.scheduler, "timesteps", None)
        if ts is not None:
            # 用计数器作为 step 索引，比 (ts == timestep) 稳定
            idx = getattr(pipe, "_cap_step_idx", 0)
            pipe._cap_step_idx = idx + 1
        else:
            idx = -1

        # 只给目标层写标志
        target_module._cap_cur_step = idx
        target_module._cap_cur_timestep = cur_t
        # 不再设置 _cap_is_last_step / _cap_only_last

        # 如需调试：
        # print(f"[hook] step={idx}, t={cur_t}")

    return root.register_forward_pre_hook(_pre_hook)


# =========================================================
# 4) 新的 get_attention_scores2_0：
#    每次都写入 sink，但先 clear，保证推理结束后只保留最后一步
# =========================================================
def new_get_attention_scores2_0(
    self,  # <-- 绑定到 Attention 实例的方法
    is_cross: bool,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算注意力概率，并在满足条件时将其写入“目标层”的 sink：
      - 必须是 cross-attn（is_cross=True）
      - 必须被标记为目标层（_cap_is_target_layer=True）

    不再依赖“最后一步”标志，而是每次覆盖 sink 中内容，
    这样推理结束后 sink 中自然就是最后一步的注意力。
    """

    # === 原始注意力计算（与 diffusers 逻辑一致）===
    dtype = query.dtype
    if getattr(self, "upcast_attention", False):
        query = query.float()
        key = key.float()

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0],
            query.shape[1],
            key.shape[1],
            dtype=query.dtype,
            device=query.device,
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=getattr(self, "scale", query.shape[-1] ** -0.5),
    )
    del baddbmm_input

    if getattr(self, "upcast_softmax", False):
        attention_scores = attention_scores.float()

    attention_probs = attention_scores.softmax(dim=-1)
    del attention_scores

    attention_probs = attention_probs.to(dtype)

    # === 仅对“目标层 & cross-attn”写入 sink（每次覆盖上一条） ===
    if is_cross and bool(getattr(self, "_cap_is_target_layer", False)):
        sink = getattr(self, "_cap_sink", None)
        if isinstance(sink, list):
            step_idx = int(getattr(self, "_cap_cur_step", -1))
            t_val = int(getattr(self, "_cap_cur_timestep", -1))
            tensor = attention_probs.detach().to(torch.float16).cpu()

            # 关键：每次清空再写 → 推理结束时只保留最后一步
            sink.clear()
            sink.append(
                {
                    "step": step_idx,
                    "timestep": t_val,
                    "attn_probs": tensor,  # 形状通常 [B*H, Nq, Nk]
                }
            )

            # 如需调试：
            # print(f"[save] step={step_idx}, t={t_val}, shape={tuple(tensor.shape)}")

    return attention_probs


# =========================================================
# 5) 把新方法“打补丁”到 Attention（全局或仅目标层）
# =========================================================
def patch_attention_get_scores(pipe, target_layer: Optional[str] = None):
    """
    将 new_get_attention_scores2_0 绑定为 Attention.get_attention_scores2_0。
    - target_layer=None: 给 transformer/unet 内所有 Attention 模块 patch
    - target_layer=str : 只给该层 patch
    """
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    assert root is not None

    for name, m in root.named_modules():
        if isinstance(m, Attention):
            if (target_layer is None) or (name == target_layer):
                # 把函数绑定为实例方法
                m.get_attention_scores2_0 = types.MethodType(new_get_attention_scores2_0, m)


# =========================================================
# 6) 将 AttnProcessor2_0 替换为我们自定义的 AttnProcessorMe
# =========================================================
def replace_attnprocessor2_0_with_attnprocessor(
    pipe,
    move_to_cpu: bool = True,
):
    """
    将 PixArt (或任何 Diffusers 模型) 中 transformer/unet 的所有
    AttnProcessor2_0 层替换为 AttnProcessorMe。

    - 在整个推理流程中记录 full hidden_states 到一个 shared hidden_sink 列表。
    - 不做区域裁剪：即 region_indices=None，缓存整个 hidden_states。

    参数:
        pipe: Diffusers 的 pipeline（包含 transformer 或 unet）
        move_to_cpu: 记录时是否把 hidden_states 挪到 CPU，减少 GPU 显存占用。

    返回:
        restore: 调用后恢复原始 processor 配置
        hidden_sink: list，里面是若干 dict，记录了全流程的 hidden_states
                     每条记录大概长这样：
                     {
                        "step": int,
                        "timestep": int,
                        "tag": str,           # 层名，如 "transformer_blocks.10.attn2"
                        "is_cross": bool,
                        "shape": tuple,
                        "hidden": torch.Tensor (fp16, 在 CPU 或 GPU)
                     }
    """
    # 1) 找到有 attn_processors 的模块（PixArt 是 transformer，SD 之类是 unet）
    if hasattr(pipe, "transformer"):
        tr = pipe.transformer
    elif hasattr(pipe, "unet"):
        tr = pipe.unet
    else:
        raise ValueError("❌ 这个 pipeline 没有 transformer 或 unet。")

    # 全流程的 hidden_state 都会塞到这个 list 里
    hidden_sink: list = []

    # 备份原来的 processors
    old = dict(tr.attn_processors)
    new = {}
    replaced, total = 0, 0

    for name, proc in tr.attn_processors.items():
        total += 1

        # 只替换 AttnProcessor2_0 实例
        if isinstance(proc, AttnProcessor2_0):
            new[name] = AttnProcessorMe(
                hidden_sink=hidden_sink,
                # 不传 region_indices => 默认 None => 全缓存
                move_to_cpu=move_to_cpu,
                tag=name,  # 标记来自哪一层，后处理更方便
            )
            replaced += 1
        else:
            new[name] = proc

    # 应用新的 processor 配置
    tr.set_attn_processor(new)

    def restore():
        """恢复原始的 attention processor 配置"""
        tr.set_attn_processor(old)

    print(f"✅ 已替换 {replaced}/{total} 个 AttnProcessor2_0 → AttnProcessorMe（full hidden_state cache）")
    return restore, hidden_sink

def map_chunks_to_token_indices(
    prompt,
    chunks,
    tokenizer,
    include_space_tokens: bool = False
):
    """
    将 spaCy 切分得到的短语映射到 tokenizer token 序列的索引列表。
    返回 {chunk_text: [tok_idx0, tok_idx1, ...]}。
    - 默认忽略纯空格 token（形如 '▁'），可用 include_space_tokens=True 保留。
    - chunks 可以是字符串列表，也可以是 spaCy 的 Span 对象迭代器。
    """
    import re

    # 1) token 化，拿到原始 token 列表（含 </s>）
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding=False, truncation=False)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    # 替换显示：把 '▁' 显示为空格，但保留 token 边界
    token_texts_for_join = [t.replace("▁", " ") for t in tokens]

    # 2) 构造“字符位置 -> token 索引”映射
    char_to_token = {}
    pos = 0
    for i, t in enumerate(token_texts_for_join):
        for _ in t:
            char_to_token[pos] = i
            pos += 1
    # 注意：这里拼接出的字符串长度 >= 原 prompt，
    # 但我们后面用 prompt 的字符位置查映射，通常对齐够用。

    # 3) 遍历每个 chunk，找到它在 prompt 中的字符范围 → 收集跨度内所有 token 索引
    def _is_space_token(tok: str) -> bool:
        # 仅由 '▁' 或空白构成，视为“空格 token”
        return tok.strip("▁").strip() == ""

    mapping = {}
    for ch in chunks:
        # 兼容 spaCy Span
        if hasattr(ch, "text"):
            ch = ch.text
        chunk = ch.strip()
        if not chunk:
            continue

        m = re.search(re.escape(chunk), prompt, flags=re.IGNORECASE)
        if not m:
            continue
        start_char, end_char = m.span()  # [start, end)

        # 收集跨度内的 token 索引（去重且保持顺序）
        seen = set()
        idx_list = []
        for p in range(start_char, end_char):
            tidx = char_to_token.get(p)
            if tidx is None:
                continue
            if (not include_space_tokens) and _is_space_token(tokens[tidx]):
                continue
            if tidx not in seen:
                seen.add(tidx)
                idx_list.append(tidx)

        if idx_list:
            mapping[chunk] = idx_list

    return mapping

def hash_tensor(t):
    return torch.sum(t.float() * 1e6).item()
def find_strict_duplicates(tensor):
    hash_table = {}
    duplicates = []

    for i in range(tensor.shape[0]):
        h = hash_tensor(tensor[i])
        
        if h in hash_table:
            # 二次验证（避免 hash 碰撞）
            if torch.equal(tensor[i], tensor[hash_table[h]]):
                duplicates.append((hash_table[h], i))
        else:
            hash_table[h] = i

    return duplicates


def construct_region_item(
    hidden_sink: List[dict],
    attn_scores: torch.Tensor,
    mapping: Dict[str, List[int]],
    score_ratio: float = 0.5,
    min_patches: int = 32,
    target_tag_substr: str = ".attn1",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    根据注意力得分，为每个语义 chunk 选出“高分区域”的 patch 索引，
    然后在 hidden_sink 中对整个推理流程里所有符合 target_tag_substr 的 hidden
    在这些 patch 位置上取值，并在时间维度上堆叠。

    返回：
        chunk_hidden:
            dict: chunk -> Tensor[T, K_i, C]
                T = 该层在整个推理流程中被记录的次数（时间步数）
                K_i = 该 chunk 选中的 patch 数
        chunk_patch_indices:
            dict: chunk -> LongTensor[K_i]
                patch 的一维索引 (0 ~ L_img-1)
    """

    # ---------- 1. 规范 attn_scores ----------
    if attn_scores.dim() == 3:
        # [H, L_img, L_text] -> [L_img, L_text]
        attn_mean = attn_scores.mean(dim=0)
    elif attn_scores.dim() == 2:
        attn_mean = attn_scores
    elif attn_scores.dim() == 1:
        attn_mean = attn_scores          # [L_img]
    else:
        raise ValueError(f"attn_scores 维度不支持: {attn_scores.shape}")

    # 记录 L_img / L_text
    if attn_mean.dim() == 2:
        L_img_attn, L_text = attn_mean.shape
    else:
        L_text = None
        L_img_attn = attn_mean.shape[0]

    # ---------- 2. 从 hidden_sink 中挑出“这一层在整个推理过程的所有 hidden” ----------
    # 条件：
    #   - tag 含 target_tag_substr（比如 ".attn1"）
    #   - 空间维长度与 attn_scores 的 L_img 匹配
    hidden_seq: List[torch.Tensor] = []

    for item in hidden_sink:
        tag = str(item.get("tag", ""))
        if target_tag_substr not in tag:
            continue

        hid = item["hidden"]  # [L_img, C] 或 [B, L_img, C]

        if hid.dim() == 3:
            # [B, L_img, C] -> 只取第 0 个 batch
            hid_use = hid[0]
        elif hid.dim() == 2:
            hid_use = hid
        else:
            # 不支持的维度，跳过
            continue

        L_img_hidden, C = hid_use.shape
        if L_img_hidden != L_img_attn:
            # 分辨率与当前 attn 不一致，跳过（属于其他层）
            continue

        hidden_seq.append(hid_use)

    if not hidden_seq:
        raise ValueError(
            f"在 hidden_sink 中找不到 tag 含 '{target_tag_substr}' 且 L_img={L_img_attn} 的条目"
        )

    # ---------- 3. 针对每个 chunk 根据 attn_mean 选 patch 索引 ----------
    chunk_hidden: Dict[str, torch.Tensor] = {}
    chunk_patch_indices: Dict[str, torch.Tensor] = {}

    device = attn_mean.device

    for chunk, token_ids in mapping.items():
        # 3.1 计算该 chunk 的每个 patch 的 score
        if attn_mean.dim() == 1:
            # 已经是一维，每个 patch 一个 score
            patch_scores = attn_mean.clone()   # [L_img]
        else:
            # 根据 mapping 取该 chunk 对应的 token 列，再在 token 维平均
            valid_token_ids = [t for t in token_ids if 0 <= t < L_text]
            if not valid_token_ids:
                continue

            token_idx = torch.tensor(valid_token_ids, dtype=torch.long, device=device)
            # [L_img, len(token_ids)] -> [L_img]
            patch_scores = attn_mean[:, token_idx].mean(dim=-1)

        # 3.2 根据 score_ratio 计算阈值
        max_score = patch_scores.max()
        threshold = max_score * float(score_ratio)

        high_mask = patch_scores >= threshold       # [L_img]
        num_high = int(high_mask.sum().item())

        if num_high == 0:
            # 极端情况：所有 score 很接近 0，fallback 到 top-k
            K = min(min_patches, L_img_attn)
            _, top_idx = torch.topk(patch_scores, k=K, dim=0)
            selected_idx = top_idx
        elif num_high < min_patches:
            # 高分区域太少，补到 min_patches
            K = min(min_patches, L_img_attn)
            _, top_idx = torch.topk(patch_scores, k=K, dim=0)
            selected_idx = top_idx
        else:
            # 正常情况：取所有 >= 阈值的 patch
            selected_idx = high_mask.nonzero(as_tuple=False).squeeze(1)  # [K_i]

        # ---------- 4. 对整个推理流程的 hidden，在这些 idx 上取值并堆叠 ----------
        per_step_hidden = []
        for hid_use in hidden_seq:
            # 确保索引在同一设备
            idx_on_dev = selected_idx.to(hid_use.device)
            h_chunk = hid_use[idx_on_dev]     # [K_i, C]
            per_step_hidden.append(h_chunk)

        hidden_all_steps = torch.stack(per_step_hidden,dim=0)
        print("hidden_all_steps shape in construct",hidden_all_steps.shape)
        dups = find_strict_duplicates(hidden_all_steps)

        print("检测结果:", dups if dups else "无重复")

        chunk_hidden[chunk] = hidden_all_steps.detach().cpu()
        chunk_patch_indices[chunk] = selected_idx.detach().cpu().long()

    return chunk_hidden, chunk_patch_indices




# =========================================================
# 7) 使用示例：PixArtAlpha 上捕捉目标层最后一步 cross-attn
#            + 使用整个推理流程的 .attn1 hidden
# =========================================================
if __name__ == "__main__":
    prompt = "a cat on a red chair"
    pipe = PixArtAlphaPipeline.from_pretrained(
        "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16,
    ).to("cuda:2")

    # 1) 替换所有 AttnProcessor2_0，用自定义版本记录 hidden_sink
    restore, hidden_sink = replace_attnprocessor2_0_with_attnprocessor(
        pipe,
        move_to_cpu=True,   # 建议 True，不然 full cache 很吃显存
    )

    # 2) 选择一个 cross-attn 层，用来捕获 attention（这里只 hook 这一个）
    USER_LAYER_NAME = "transformer_blocks.10.attn2"
    real_name = resolve_layer_name(pipe, USER_LAYER_NAME)
    print("[target cross-attn layer]", real_name)

    target_module, attn_sink = enable_target_capture(pipe, real_name)
    hook = install_laststep_flag_pre_hook(pipe, target_module)
    patch_attention_get_scores(pipe, target_layer=real_name)

    # 3) 正常推理
    gen = torch.Generator(device="cuda").manual_seed(1234)
    out = pipe(
        prompt=prompt,
        num_inference_steps=15,
        guidance_scale=4.0,
        generator=gen,
    )

    for i in range(5):
        print(i, hidden_sink[i]["tag"])

    # 4) 读取 cross-attn（只保留了最后一步）
    assert attn_sink, "❌ 没捕到注意力：检查层名 / 是否 cross-attn / 是否正确 patch"
    last = attn_sink[-1]
    print(
        "step:",
        last["step"],
        "timestep:",
        last["timestep"],
        "attn_probs shape:",
        tuple(last["attn_probs"].shape),
    )

    hook.remove()
    print("共记录 hidden entries（所有层/所有步）:", len(hidden_sink))

    # ------------- 文本解析 & token mapping -------------
    doc = nlp(prompt)
    tokenizer = getattr(pipe, "tokenizer", None) or getattr(pipe, "tokenizer_2", None)
    if tokenizer is None:
        raise RuntimeError("❌ 没有在 pipeline 中找到 tokenizer。")
    
    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False
    )

    input_ids = encoding["input_ids"][0].tolist()
    chunks = [chunk.text for chunk in doc.noun_chunks]
    mapping = map_chunks_to_token_indices(prompt=prompt, chunks=chunks, tokenizer=tokenizer)
    print("mapping:", mapping)

    # 看一下 hidden_sink 的一个例子
    item = hidden_sink[0]
    print("hidden_sink[0].hidden shape:", item["hidden"].shape)

    # ------------- 可视化 cross-attn heatmap（最后一步） -------------
    attn = last["attn_probs"]        # 形状类似: (heads * batch, HW, T_text) 或 (heads, HW, T_text)
    print("raw attn shape:", attn.shape)

    # 平均 head → [HW, T_text]
    attn_mean = attn.mean(dim=0)     # [HW, T_text]
    hw, n_tokens = attn_mean.shape
    H = W = int(math.sqrt(hw))
    assert H * W == hw, f"HW({hw}) 不能被 sqrt 整除，无法 reshape 成正方形，当前 sqrt={math.sqrt(hw)}"

    # 渲染你关心的 chunk 的注意力热力图
    target_chunks = ["a cat", "a red chair"]

    fig, axes = plt.subplots(1, len(target_chunks), figsize=(6 * len(target_chunks), 6))
    if len(target_chunks) == 1:
        axes = [axes]

    for ax, chunk in zip(axes, target_chunks):
        if chunk not in mapping:
            print(f"⚠️ chunk '{chunk}' 不在 mapping 里，跳过")
            continue

        token_ids = mapping[chunk]  # 比如 [1, 2] 或 [5, 6, 7]
        chunk_attn = attn_mean[:, token_ids].mean(dim=-1)   # [HW]

        chunk_map = chunk_attn.reshape(H, W)
        chunk_map = chunk_map.detach().cpu().numpy()

        im = ax.imshow(chunk_map, cmap="viridis")
        ax.set_title(f"Cross-Attn for: '{chunk}'")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_path = "./cross_attn_chunks.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✔️ cross-attn 热力图已保存到: {save_path}")

    # ------------- 基于最后一步 cross-attn + 全流程 .attn1 hidden 构造 region cache -------------
    # 这里 attn_mean 是最后一步 cross-attn 的 [L_img, L_text]
    print("len hidden_sink:", len(hidden_sink))

    chunk_hidden, chunk_patch_indices = construct_region_item(
        hidden_sink=hidden_sink,   # 全过程的 self-attn(.attn1) hidden
        attn_scores=attn_mean,     # 只用最后一步 cross-attn 来选 patch
        mapping=mapping,
        score_ratio=0.2,
        min_patches=64,
        target_tag_substr=".attn1",
    )

    

    # 打印每个 chunk 的 shape：
    #   hidden: [T, K_i, C]    indices: [K_i]
    for chunk, h in chunk_hidden.items():
        idx = chunk_patch_indices[chunk]
        print(f"\n=== chunk: {chunk} ===")
        print("  chunk 文本长度:", len(chunk))
        print("  hidden shape (T, K_i, C):", h.shape)
        print("  选中 patch 个数 K_i:", idx.shape)

    # ------------- 给某个 chunk 画 patch mask（用 indices） -------------
    chunk_name = "a cat"
    if chunk_name not in chunk_patch_indices:
        raise KeyError(
            f"chunk '{chunk_name}' 不在 chunk_patch_indices 里，可用 keys={list(chunk_patch_indices.keys())}"
        )

    indices = chunk_patch_indices[chunk_name]  # [K]
    L_img = attn_mean.shape[0]                 # e.g. 4096

    H = W = int(math.sqrt(L_img))
    assert H * W == L_img, f"L_img={L_img} 不是正方形网格，可能需要 reshape 到非方阵格式"

    mask_1d = torch.zeros(L_img, dtype=torch.float32)
    mask_1d[indices] = 1.0
    mask_2d = mask_1d.view(H, W)
    print("patch mask 2D shape:", mask_2d.shape)

    save_path = f"patch_mask_{chunk_name.replace(' ', '_')}.png"
    plt.figure(figsize=(6, 6))
    plt.title(f"Patch mask for: {chunk_name}")
    plt.imshow(mask_2d.numpy(), cmap="hot")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"✔️ patch mask 已保存为: {save_path}")

    # 初始化 DB
    db = RegionDB()

    # 这里的 hidden_tensor 形状是 [T, K_i, C]，indices 是 [K_i]
    os.makedirs("./cache/chunks", exist_ok=True)
    for region_name, h in chunk_hidden.items():
        # 获取 indices 并转为 list 方便存储或仅作记录
        idx_tensor = chunk_patch_indices[region_name]
        idx_list = idx_tensor.tolist()

        save_obj = {
            "prompt": prompt,
            "region_name": region_name,
            "hidden_state": h,
            "indices": chunk_patch_indices[region_name],
        }

        file_name = f"{region_name.replace(' ', '_')}.pt"
        abs_path = os.path.abspath(f"./cache/chunks/{file_name}")

        torch.save(save_obj, abs_path)

        db.add_region(
            prompt=prompt,                  # 原始 Prompt，如 "a cat on a red chair"
            region_name=region_name,        # Chunk 名，如 "a cat"
            file_path=abs_path,             # 物理路径作为 ID
            indices_list=idx_list           # 可选：存入 metadata
        )

    print(f"✔ 已分别保存 {len(chunk_hidden)} 个 region 文件到 ./cache/chunks/")