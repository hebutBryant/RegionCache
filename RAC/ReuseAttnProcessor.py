import os
import sys
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Material_Library.Constructer.pt_reader import load_region_cache

import torch
import torch.nn as nn
from typing import Optional
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0
from typing import Sequence, Union, Optional



def prepare_attn_mask(
    region_indices: Union[Sequence[int], Sequence[Sequence[int]], torch.Tensor],
    num_tokens: Optional[int] = None,
    expand_batch_head: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    构造一个适用于 scaled_dot_product_attention 的 attn_mask（bool 型）：

    语义（与 PyTorch 一致）：
      - attn_mask == True  → 允许注意
      - attn_mask == False → 不允许注意（会被 mask 掉）

    行为：
      - active token（非缓存位置）作为 query：可以关注所有 token（整行全 True）
      - cached token（region_indices 中的那些位置）作为 query：
            只能关注“自己”这个位置
            -> 这一行只有对角 (i, i) 为 True，其余为 False

    Args:
        region_indices:
            - 比如 [1, 2, 5, 6]，表示要复用的 token 位置
            - 或 [[1,2],[5,6]]，多个区域，会自动 flatten + unique
        num_tokens:
            - 整个序列长度 L（推荐显式传入，比如 L = hidden_states.shape[1]）
            - 如果为 None，则用 max(region_indices)+1 作为 L 的下界
        expand_batch_head:
            - True: 返回 [1, 1, L, L]，方便直接喂给 SDPA / diffusers 的 prepare_attention_mask
            - False: 返回 [L, L]
        device:
            - mask 放到哪个 device 上；默认留在 CPU，后面自己 .to(device) 也行

    Returns:
        attn_mask: bool tensor, True 表示“允许注意”
            - [L, L] or [1, 1, L, L]
    """
    # 1) 处理 region_indices -> 1D LongTensor, 去重
    if isinstance(region_indices, torch.Tensor):
        idx = region_indices.view(-1).long()
    else:
        flat = []
        for x in region_indices:
            if isinstance(x, (list, tuple)):
                flat.extend(list(x))
            else:
                flat.append(x)
        if len(flat) == 0:
            idx = torch.empty(0, dtype=torch.long)
        else:
            idx = torch.as_tensor(flat, dtype=torch.long)

    idx = torch.unique(idx)

    # 2) 确定序列长度 L
    if idx.numel() == 0:
        # 没有缓存区域 => L 只能来自 num_tokens，整块全 True（所有位置都可注意）
        assert num_tokens is not None, "region_indices 为空时必须显式指定 num_tokens"
        L = int(num_tokens)
        attn = torch.ones(L, L, dtype=torch.bool, device=device)  # 全 True = 全可见
    else:
        max_pos = int(idx.max().item())
        if num_tokens is None:
            L = max_pos + 1
        else:
            L = int(num_tokens)
            assert max_pos < L, f"索引 {max_pos} 超出 num_tokens={L}"

        # 默认全 True：所有 query 都能看所有 key
        attn = torch.ones(L, L, dtype=torch.bool, device=device)

        # 对 cached 行做约束：先整行 False，再把对角设回 True
        attn[idx, :] = False      # 该行全部不允许注意
        attn[idx, idx] = True     # 该行只允许自注意

    if expand_batch_head:
        attn = attn.unsqueeze(0).unsqueeze(0)  # [L,L] -> [1,1,L,L]

    return attn

def validate_mask(mask: torch.Tensor, region):
    """
    检查 bool attn_mask 是否满足：
      - region 行：只有对角为 True，其他为 False
      - 非 region 行：全 True
    """
    # 支持传 [1,1,L,L] 或 [L,L]
    mat = mask
    if mat.dim() == 4:
        mat = mat.squeeze(0).squeeze(0)
    elif mat.dim() != 2:
        raise ValueError(f"mask 维度必须是 2 或 4, 当前是: {mat.shape}")

    L = mat.shape[0]
    assert mat.shape[0] == mat.shape[1], "mask 必须是方阵 [L, L]"

    # 统一 region 类型
    if isinstance(region, torch.Tensor):
        reg_idx = region.view(-1).long().tolist()
    else:
        reg_idx = list(region)
    reg_idx = sorted(set(int(i) for i in reg_idx))

    # 检查合法范围
    for t in reg_idx:
        assert 0 <= t < L, f"region index {t} 越界 (L={L})"

    all_idx = set(range(L))
    region_set = set(reg_idx)
    non_region = sorted(all_idx - region_set)

    # 1) 检查 region 行：对角 True，其他 False
    for t in reg_idx:
        row = mat[t]  # [L]
        # 对角必须 True
        if row[t].item() is not True:
            raise AssertionError(
                f"[❌] 冻结 token {t} 的对角元素不是 True，而是 {row[t].item()}"
            )
        # 非对角必须全 False
        off_diag = torch.ones(L, dtype=torch.bool, device=mat.device)
        off_diag[t] = False
        if row[off_diag].any():  # 只要有 True 就不对
            raise AssertionError(
                f"[❌] 冻结 token {t} 的非对角元素存在 True → {row.int().tolist()}"
            )

    # 2) 检查非 region 行：整行必须全 True
    for t in non_region:
        row = mat[t]
        if not row.all():
            raise AssertionError(
                f"[❌] 非冻结 token {t} 的行不是全 True → {row.int().tolist()}"
            )

    print("[✔] mask 验证通过：冻结 token 行为仅自注意，其余 token 行全可见。")
    return True


def _flatten_region_indices(
    region_indices: Union[Sequence[int], Sequence[Sequence[int]], torch.Tensor]
) -> torch.LongTensor:
    """
    和 prepare_attn_mask 里一样的 index 展平逻辑，单独拿出来给覆盖 hidden_states 用。
    """
    if isinstance(region_indices, torch.Tensor):
        idx = region_indices.view(-1).long()
    else:
        flat = []
        for x in region_indices:
            if isinstance(x, (list, tuple)):
                flat.extend(list(x))
            else:
                flat.append(x)
        if len(flat) == 0:
            return torch.empty(0, dtype=torch.long)
        idx = torch.as_tensor(flat, dtype=torch.long)

    # 复用时重复没意义，unique 一下
    idx = torch.unique(idx)
    return idx


from packaging import version
from typing import Any, Dict, Optional, Union
import inspect
import warnings
def deprecate(*args, take_from: Optional[Union[Dict, Any]] = None, standard_warn=True, stacklevel=2):
    from .. import __version__

    deprecated_kwargs = take_from
    values = ()
    if not isinstance(args[0], tuple):
        args = (args,)

    for attribute, version_name, message in args:
        if version.parse(version.parse(__version__).base_version) >= version.parse(version_name):
            raise ValueError(
                f"The deprecation tuple {(attribute, version_name, message)} should be removed since diffusers'"
                f" version {__version__} is >= {version_name}"
            )

        warning = None
        if isinstance(deprecated_kwargs, dict) and attribute in deprecated_kwargs:
            values += (deprecated_kwargs.pop(attribute),)
            warning = f"The `{attribute}` argument is deprecated and will be removed in version {version_name}."
        elif hasattr(deprecated_kwargs, attribute):
            values += (getattr(deprecated_kwargs, attribute),)
            warning = f"The `{attribute}` attribute is deprecated and will be removed in version {version_name}."
        elif deprecated_kwargs is None:
            warning = f"`{attribute}` is deprecated and will be removed in version {version_name}."

        if warning is not None:
            warning = warning + " " if standard_warn else ""
            warnings.warn(warning + message, FutureWarning, stacklevel=stacklevel)

    if isinstance(deprecated_kwargs, dict) and len(deprecated_kwargs) > 0:
        call_frame = inspect.getouterframes(inspect.currentframe())[1]
        filename = call_frame.filename
        line_number = call_frame.lineno
        function = call_frame.function
        key, value = next(iter(deprecated_kwargs.items()))
        raise TypeError(f"{function} in {filename} line {line_number - 1} got an unexpected keyword argument `{key}`")

    if len(values) == 0:
        return
    elif len(values) == 1:
        return values[0]
    return values




class ReuseAttnProcessor:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "RegionCacheAttnProcessor2_0 requires PyTorch 2.0, "
                "please upgrade PyTorch to 2.0 or later."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        step_cache: Optional[torch.Tensor] = None,
        region_indices: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # ====== 1) 处理废弃的 scale 参数（可能从 cross_attention_kwargs 里传进来） ======
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = (
                "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will "
                "raise an error in the future. `scale` should directly be passed while calling the underlying "
                "pipeline component i.e., via `cross_attention_kwargs`."
            )
            deprecate("scale", "1.0.0", deprecation_message)
            # 不用的话就弹掉，避免后面继续往下传
            kwargs.pop("scale", None)
        if step_cache is None:
            step_cache = kwargs.pop("block_cache", None)
        
        # 统一变量名
        cached_hidden_states = step_cache

        # 只有当 cache 和 indices 都有值时，才启用复用逻辑
        is_reusing = (cached_hidden_states is not None) and (region_indices is not None)

        if encoder_hidden_states is None:
            attn_type = "SELF-ATTENTION"
        else:
            attn_type = "CROSS-ATTENTION"

        print(f"\n===== [{attn_type}] =====")
        print(f"[Attention] Received ReuseAttnProcessor keys: {list(kwargs.keys())}")
        # ====== 2) 从 kwargs 中取出 region cache 相关参数 ======
        # 标准键名：region_indices + cached_hidden_states
        ###
        if(region_indices is not None):
            print("######################region_indices in ReuseAttnProcessor#########################",region_indices.shape)
        if(cached_hidden_states is not None):
            print("######################cached_hidden_states in ReuseAttnProcessor#########################",cached_hidden_states.shape)


        # 其余 kwargs（如果有）可以继续往下传给别的逻辑，
        # 目前这个 Processor 自己不用再管 kwargs 了

        # ====== 3) 原有逻辑开始 ======
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None and encoder_hidden_states is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # (3) group_norm
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            if is_reusing:
                # self-attention 情况：构造一个全 True 的 attention_mask
                B, N, _ = hidden_states.shape
                print("###########sequence_length#############",sequence_length)
                attention_mask = prepare_attn_mask(region_indices=region_indices, num_tokens=sequence_length,device="cuda:1")
                print(f"Attention mask shape: {attention_mask.shape}")
                # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.expand(batch_size, attn.heads, attention_mask.size(-2), attention_mask.size(-1))

            else:
                if attention_mask is not None:
                    if attention_mask.shape[-1] != query.shape[1]:
                        target_length = query.shape[1]
                        attention_mask = F.pad(attention_mask, (0, target_length - attention_mask.shape[-1]))
                        attention_mask = attention_mask.repeat(batch_size, 1, 1, 1)

            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # [B, L, H*D] -> [B, H, L, D]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # SDPA
        if attention_mask is None:
            print("Attention mask: None")
        else:
            print(f"Attention mask shape: {attention_mask.shape}")


        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # [B,H,L,D] -> [B,L,H*D]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 输出投影 + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # reshape 回 4D（如果一开始是 4D）
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # residual + rescale
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor


        # (4) ⭐ 在这里做 region cache 覆盖，再去算 Q/K/V ⭐
        # 这里 region cache 的形状为 [region_len, hidden_dim]
        # 框架中的 hidden_states 形状为 [batch_size, token_num, hidden_dim]
        if is_reusing:
            if hidden_states.ndim != 3:
                raise NotImplementedError(
                    "Region cache 目前只支持 [batch, seq_len, dim] 形状的 hidden_states；"
                    "如果你在 UNet 2D attn 上使用，请在 flatten 之后再做 cache。"
                )

            # 规范 index
            if isinstance(region_indices, torch.Tensor):
                idx = region_indices.view(-1).long().to(hidden_states.device)
            else:
                idx = torch.as_tensor(list(region_indices), dtype=torch.long, device=hidden_states.device)

            B, L, C = hidden_states.shape

            if idx.numel() > 0:
                max_pos = int(idx.max().item())
                if max_pos >= L:
                    raise ValueError(f"region_indices 最大值 {max_pos} 超过当前序列长度 L={L}")

            # ====== 情况 1：cache 已经是 [B, L, C]，和 hidden_states 完全对齐（原逻辑） ======
            if cached_hidden_states.dim() == 3 and cached_hidden_states.shape == hidden_states.shape:
                if idx.numel() > 0:
                    hidden_states[:, idx, :] = cached_hidden_states[:, idx, :]

            # ====== 情况 2：region cache 是 [K, C]，没有 batch 维（你现在的设计） ======
            elif cached_hidden_states.dim() == 2:
                K, C_cache = cached_hidden_states.shape
                if C_cache != C:
                    raise ValueError(
                        f"cached_hidden_states 隐维不匹配: C_cache={C_cache}, C={C}."
                    )
                if K != idx.numel():
                    raise ValueError(
                        f"region_cache 的长度 K={K} 与 region_indices 中的 token 数 {idx.numel()} 不一致。"
                    )

                # 扩展到 [B, K, C]，对每个 batch 复用同一份 region cache
                cache_expanded = cached_hidden_states.to(hidden_states.device, hidden_states.dtype)
                cache_expanded = cache_expanded.unsqueeze(0).expand(B, -1, -1)  # [B, K, C]

                # 把每个 batch 对应的这些 token 都替换掉
                if idx.numel() > 0:
                    hidden_states[:, idx, :] = cache_expanded

            else:
                raise ValueError(
                    f"不支持的 cached_hidden_states.shape={cached_hidden_states.shape}；"
                    "期望 [B, L, C] 或 [K, C]。"
                )

        # 这里你还根据 region_indices 重新构造了一个 attention_mask

        return hidden_states

if __name__ == "__main__":
    path = "/home/lipz/RegionCache/Material_Library/Constructer/cache/chunks/a_cat.pt"
    num_tokens = 4096
    region_cache = load_region_cache(path)  # 设定要复用(hidden冻结)的位置
    region = region_cache.get("indices", None)

    # region 可能是 tensor / list 都行
    mask = prepare_attn_mask(region_indices=region, num_tokens=num_tokens)

    print("=== 输入区域索引 ===")
    print(region)

    print("\n=== 生成掩码 shape ===")
    print(mask.shape)  # 期望: [1, 1, L, L]

    print("\n=== 生成掩码矩阵（去掉 batch/head 维度）===")
    print(mask.squeeze(0).squeeze(0).int())  # 以 0/1 输出方便阅读（1=True，0=False）

    print("\n=== 验证掩码是否合法 ===")
    ok = validate_mask(mask=mask, region=region)
    print("validate_mask 返回:", ok)
