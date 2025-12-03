from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput


def rac_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Dict[str, torch.Tensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    *,
    # ⭐ Region-Aware Cache 相关新增参数（keyword-only，避免破坏原有调用）
    region_indices: Optional[torch.Tensor] = None,   # [K]，patch/token 索引
    step_cache: Optional[torch.Tensor] = None,       # [num_blocks, K, D]，按 block 存的 cache
):
    """
    RAC (Region-Aware Cache) 版本的 PixArtTransformer2DModel.forward。

    主要区别：
    - 接收全局 step_cache: [num_blocks, region_len(K), hidden_dim(D)]
    - 在每个 transformer block 调用时，将对应的 block_cache 和 region_indices
      塞到 cross_attention_kwargs 中，供自定义的 RegionCacheAttnProcessor2_0 使用。

    其它行为（时间嵌入、caption_projection、patch/unpatch 等）与原始 forward 保持一致。
    """

    if self.use_additional_conditions and added_cond_kwargs is None:
        raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")

    # -------- 0. 预处理 attention_mask / encoder_attention_mask（保持和原 forward 一致） --------
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # -------- 1. Input & patch embed（与原 forward 完全一致） --------
    batch_size = hidden_states.shape[0]
    height, width = (
        hidden_states.shape[-2] // self.config.patch_size,
        hidden_states.shape[-1] // self.config.patch_size,
    )

    # pos_embed 内部做 patchify + 位置编码，输出 [B, L, D]
    hidden_states = self.pos_embed(hidden_states)  # [B, L, D]
    model_dim = hidden_states.shape[-1]

    # -------- 2. timestep / 条件嵌入（与原 forward 完全一致） --------
    timestep, embedded_timestep = self.adaln_single(
        timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
    )

    if self.caption_projection is not None:
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

    # -------- 3. Region Cache 形状检查（在拿到 model_dim 之后再检查） --------
    if step_cache is not None:
        if step_cache.dim() != 3:
            raise ValueError(
                f"[RAC] step_cache 期望维度为 3 [num_blocks, K, D]，但得到 {step_cache.shape}"
            )

        num_blocks, region_len, cache_dim = step_cache.shape
        if cache_dim != model_dim:
            raise ValueError(
                f"[RAC] step_cache hidden_dim (D={cache_dim}) 与模型 dim (D={model_dim}) 不一致。"
            )
        if num_blocks != len(self.transformer_blocks):
            raise ValueError(
                f"[RAC] step_cache 中的 block 数({num_blocks}) 与 transformer_blocks 数量({len(self.transformer_blocks)}) 不一致。"
            )

        step_cache = step_cache.to(hidden_states.device)

    if region_indices is not None:
        if region_indices.dim() != 1:
            raise ValueError(
                f"[RAC] region_indices 期望为一维 [K]，但得到 {region_indices.shape}"
            )
        region_indices = region_indices.to(hidden_states.device)
        if step_cache is not None and region_indices.shape[0] != step_cache.shape[1]:
            raise ValueError(
                f"[RAC] region_indices 长度 K={region_indices.shape[0]} "
                f"与 step_cache 中 region_len={step_cache.shape[1]} 不一致。"
            )

    # 基础 cross_attention_kwargs 复制一份，后面每个 block 再做 per-block 覆盖


    # -------- 4. Transformer Blocks（这里插入 per-block region cache） --------
    for block_idx, block in enumerate(self.transformer_blocks):

        
        # 为当前 block 构造 cross_attention_kwargs
        block_ca_kwargs = {} # 拷贝一份，避免在循环里互相污染

        # 只要传进来的有，就塞到 kwargs 里，具体使用逻辑由 RegionCacheAttnProcessor2_0 决定
        if region_indices is not None:
            block_ca_kwargs["region_indices"] = region_indices
        if step_cache is not None:
            # 当前 block 的 cache: [K, D]
            block_ca_kwargs["block_cache"] = step_cache[block_idx]
        print(f"[Attention] Received block_ca_kwargs keys: {list(block_ca_kwargs.keys())}")
        ###
        if(step_cache is not None and region_indices is not None):
            print(f"[Attention] block {block_idx} block_cache shape: {block_ca_kwargs['block_cache'].shape}, region_indices shape: {block_ca_kwargs['region_indices'].shape}")

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                block_ca_kwargs,  # ✅ 改成 per-block kwargs
                None,
            )
        else:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=block_ca_kwargs,  # ✅ 改成 per-block kwargs
                class_labels=None,
            )

    # -------- 5. Output（与原 forward 完全一致） --------
    shift, scale = (
        self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
    ).chunk(2, dim=1)

    hidden_states = self.norm_out(hidden_states)
    # Modulation
    hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states.squeeze(1)

    # unpatchify
    hidden_states = hidden_states.reshape(
        shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
    )

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
