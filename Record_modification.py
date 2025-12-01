"""
这个文件记录我修改diffuers框架的哪个地方，为之后在third party中
在本地环境进行修改
"""


# 1. get_attention_scores2_0   
"""
这个函数用在了Atttion类中， 为了替换在AttnProcessor 中get_attention_scores函数

这个函数通过向 sink里加入 aatention score去帮助我们去抠图

"""


def get_attention_scores2_0(
        self, is_cross:bool,query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        print("###################use get_attention_scores2_0################")
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
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
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)
        if is_cross and getattr(self, "_xattn_capture", False):
            sink = getattr(self, "_xattn_sink", None)
            if sink is not None:
                sink.append(attention_probs.detach().cpu())  # 只“拿到矩阵”——不做任何额外信息
            else:
                # 若外部没提供容器，就存到最近一次结果里
                self._xattn_last = attention_probs.detach().cpu()

        return attention_probs



# 2.  rac__call__ 函数
"""
这个函数的目的是去替换 pipeline中原始的__call__函数
在rac_call中吧我们整个流程所需要的 cache传进去  
这个要写在 pipeline_pixart中 class PixArtAlphaPipeline(DiffusionPipeline):下面

"""


def rac__call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 120,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        RAC 版 PixArt/Lumina pipeline 的 __call__：
        在原始实现基础上，支持通过 **kwargs 传入 region cache 相关参数，并在内部构造 cross_attention_kwargs
        传给 transformer → AttentionProcessor。
        """

        # 旧参数的弃用提示
        if "mask_feature" in kwargs:
            deprecation_message = (
                "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect "
                "the end results. It will be removed in a future version."
            )
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)
        print("################################Use rac call########################")
        # ========= 1. 从 kwargs 中解析 cross_attention_kwargs & region cache 参数 =========
        # 如果外部已经传了 cross_attention_kwargs，就先拿出来
        cross_attention_kwargs = kwargs.pop("cross_attention_kwargs", None)
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        else:
            # 拷贝一份，避免原 dict 被原地修改
            cross_attention_kwargs = dict(cross_attention_kwargs)

        # 我们自定义的 region cache 参数，从 **kwargs 里拿
        region_indices = kwargs.pop("region_indices", None)
        cached_hidden_states = kwargs.pop("cached_hidden_states", None)

        # 塞进 cross_attention_kwargs，之后会传到 transformer → AttnProcessor
        if region_indices is not None:
            cross_attention_kwargs["region_indices"] = region_indices
        if cached_hidden_states is not None:
            cross_attention_kwargs["cached_hidden_states"] = cached_hidden_states
        # ========= 解析 kwargs 结束，下面保持原始逻辑不变，只在 transformer 调用处多传一个参数 =========

        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        if use_resolution_binning:
            if self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_256_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # guidance_scale > 1.0 才启用 CFG
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            # 你的 debug 打印可以保留或去掉
            # print("do_classifier_free_guidance")
            # print("prompt_embeds.shape:", prompt_embeds.shape)
            # print("prompt_attention_mask.shape:", prompt_attention_mask.shape)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        # print(f"timesteps.shape: {timesteps.shape}")
        # print(f"num_inference_steps: {num_inference_steps}")

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # print(f"latent_channels: {latent_channels}")
        # print(f"latents.shape: {latents.shape}")

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs: Dict[str, Optional[torch.Tensor]] = {"resolution": None, "aspect_ratio": None}
        if self.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * num_images_per_prompt, 1)
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)

            if do_classifier_free_guidance:
                resolution = torch.cat([resolution, resolution], dim=0)
                aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # CFG 时复制 latents
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    is_mps = latent_model_input.device.type == "mps"
                    is_npu = latent_model_input.device.type == "npu"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if (is_mps or is_npu) else torch.float64
                    else:
                        dtype = torch.int32 if (is_mps or is_npu) else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                # ====== 关键修改：在调用 transformer 时传入 cross_attention_kwargs ======
                """
                开始调用 多步预测噪声

                """
                print(f"现在处于{i}扩散步")

                step_cache = cached_hidden_states[i]
                print("#######################step_cache in rac call###############################",step_cache.shape)
                print("#######################region_indices in rac call###############################",region_indices.shape)   


                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,  # ⭐ 这里把我们组好的 dict 传进去
                    return_dict=False,
                    step_cache = step_cache,
                    region_indices = region_indices,
                )[0]
                # ===============================================================

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                # compute previous image: x_t -> x_t-1
                if num_inference_steps == 1:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[1]
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # callback & 进度条
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 8. 解码 & 后处理
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)





# 3. rac_forward
"""
这个函数可以定义在 函数外部的Rac_forward.py文件中
其实也可以定义在内部， 用于 Transformer2DModel中的forward函数替换

"""

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
        print("block_ca_kwargs keys:",region_indices.shape,step_cache[block_idx].shape)

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



#3. Attention类中的 forward函数
""""
在Attention中的forward 原本有一个功能

if len(unused_kwargs) > 0:
        logger.warning(
            f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
        )
    cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

去观察 **kwargs是否有无关的参数， 这个会造成我们新加的两个参数被去掉，所以我重新定义了一个变量去保存   


"""

def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **cross_attention_kwargs,
) -> torch.Tensor:
    r"""
    The forward method of the `Attention` class.

    Args:
        hidden_states (`torch.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`torch.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`torch.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.

    Returns:
        `torch.Tensor`: The output of the attention layer.
    """
    # The `Attention` class can call different attention processors / attention functions
    # here we simply pass along all tensors to the selected processor class
    # For standard processors that are defined here, `**cross_attention_kwargs` is empty
    if cross_attention_kwargs:   # 只有当真的有 RAC 参数时才打印
        print(f"[Attention] Received cross_attention_kwargs keys: {list(cross_attention_kwargs.keys())}")
    saved_cross_attention_kwargs = cross_attention_kwargs


    attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
    quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
    unused_kwargs = [
        k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
    ]
    if len(unused_kwargs) > 0:
        logger.warning(
            f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
        )
    cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}



    return self.processor(
        self,
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        **saved_cross_attention_kwargs,
    )