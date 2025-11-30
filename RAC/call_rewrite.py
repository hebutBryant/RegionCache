
import torch
from torch import nn
from typing import Callable, Dict, List, Optional, Tuple, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


@torch.no_grad()
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