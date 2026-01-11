# latent_stepper.py
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import PixArtAlphaPipeline
"""
功能:
逐步保存 latents
"""

# -----------------------
# Helpers
# -----------------------
@torch.no_grad()
def encode_prompt_full(pipe: PixArtAlphaPipeline, prompt: str, do_cfg: bool):
    """
    Return the official 4-tuple from diffusers:
    (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask)
    This keeps shapes perfectly aligned with the pipeline's expectation.
    """
    prompt_embeds, prompt_mask, neg_embeds, neg_mask = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=do_cfg,
        negative_prompt="",            # PixArt-Alpha default uncond
        num_images_per_prompt=1,
        clean_caption=True,
        max_sequence_length=120,
    )
    return prompt_embeds, prompt_mask, neg_embeds, neg_mask


def build_cfg_batch(
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_embeds: Optional[torch.Tensor],
    guidance: float,
):
    """
    Build latent/text batches for CFG.
    If guidance <= 1.0 -> single path (cond only).
    Else -> two paths (uncond + cond), doubling the batch along dim 0.
    Return: (latent_in, embeds_in, use_cfg)
    """
    if guidance is None or guidance <= 1.0:
        return latents, prompt_embeds, False

    if (negative_embeds is None) or (negative_embeds.dim() != prompt_embeds.dim()):
        print("[WARN] negative_embeds missing/mismatched. Using zeros_like(prompt) as unconditional.")
        negative_embeds = torch.zeros_like(prompt_embeds)

    # shape align fallback (rarely needed if you use encode_prompt_full)
    if negative_embeds.shape != prompt_embeds.shape:
        try:
            nb, ns, nc = negative_embeds.shape
            pb, ps, pc = prompt_embeds.shape
            if ns == ps and nc == pc and nb == 1 and pb > 1:
                negative_embeds = negative_embeds.repeat(pb, 1, 1)
            elif nb == pb and nc == pc and ns == 1 and ps > 1:
                negative_embeds = negative_embeds.repeat(1, ps, 1)
            if negative_embeds.shape != prompt_embeds.shape:
                negative_embeds = torch.zeros_like(prompt_embeds)
        except Exception:
            negative_embeds = torch.zeros_like(prompt_embeds)

    embeds_in = torch.cat([negative_embeds, prompt_embeds], dim=0)  # [2B, S, C]
    latent_in = latents.repeat(2, 1, 1, 1)                           # [2B, C, H, W]
    return latent_in, embeds_in, True


def apply_cfg_general(
    out: torch.Tensor,
    guidance: float,
    use_cfg: bool,
    latents: torch.Tensor,
) -> torch.Tensor:
    """
    Generic CFG merge that tolerates different model output layouts.
    Acceptable layouts (before pre-processing):
      - (B, C, H, W)
      - (2B, C, H, W)
      - (B, 2C, H, W)
      - (2B, 2C, H, W)  <-- learned-sigma + CFG (we'll reduce channels first)
    """
    B, C, H, W = latents.shape

    # Preprocess: reduce learned sigma if channel-doubled
    # Case: (2B, 2C, H, W) or (B, 2C, H, W)
    if out.dim() == 4 and out.shape[1] == 2 * C:
        out, _sigma = out.chunk(2, dim=1)  # keep the prediction branch -> (..., C, H, W)

    if (not use_cfg) or (guidance is None) or (guidance <= 1.0):
        # No CFG mixing; prefer cond branch if dual-batch/dual-channel still present
        if out.shape == (B, C, H, W):
            return out
        elif out.shape == (2 * B, C, H, W):
            return out[B:]                 # cond half
        else:
            raise ValueError(
                f"[apply_cfg_general] unexpected out shape {tuple(out.shape)}; "
                f"expected (B,C,H,W) or (2B,C,H,W) with B={B},C={C}."
            )

    # CFG mixing paths
    if out.shape == (2 * B, C, H, W):
        uncond, cond = out[:B], out[B:]
        return uncond + guidance * (cond - uncond)
    elif out.shape == (B, C, H, W):
        # No duplication happened; still valid (some custom graphs)
        return out
    else:
        raise ValueError(
            f"[apply_cfg_general] unexpected out shape {tuple(out.shape)}; "
            f"expected (2B,C,H,W) or (B,C,H,W) with B={B},C={C}."
        )


def make_added_cond_kwargs(height: int, width: int, batch: int, device: str):
    """
    Some PixArt branches require resolution + aspect_ratio micro-conditions.
    """
    H, W = height, width
    res = torch.tensor([H, W], device=device, dtype=torch.float32).expand(batch, 2)   # [B, 2]
    ar  = torch.tensor([H / W], device=device, dtype=torch.float32).expand(batch)     # [B]
    return {"resolution": res, "aspect_ratio": ar}


# -----------------------
# Config + Runner
# -----------------------
@dataclass
class RunnerCfg:
    model_id: str
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    steps: int = 15
    guidance: float = 4.0        # <=1.0 turns CFG off
    height: int = 1024
    width: int = 1024
    out_dir: str = "./latent_steps_position"
    save_every: int = 1          # save latent every N steps
    save_final_image: bool = True
    final_image_name: str = "final.png"


class LatentStepper:
    """
    - Accept external latents (BCHW)
    - Continue denoising from a given scheduler index `start_index`
    - Save intermediate latents and decode final image with the pipeline VAE
    """
    def __init__(self, cfg: RunnerCfg):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

        self.pipe = PixArtAlphaPipeline.from_pretrained(cfg.model_id, torch_dtype=cfg.dtype).to(cfg.device)
        self.pipe.scheduler.set_timesteps(cfg.steps, device=cfg.device)

        # Record in_channels for consistency checks
        self.latent_channels = getattr(self.pipe.transformer.config, "in_channels", 4)

    @torch.no_grad()
    def run(
        self,
        latents: torch.Tensor,
        prompt: str,
        start_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[str]]:
        """
        latents: [B,C,H,W] current latent (must match transformer in_channels/resolution)
        prompt:  text condition
        start_index: begin from scheduler.timesteps[start_index] (0 means first step)
        Return: (final_latents, final_image_path or None)
        """
        assert latents.ndim == 4, f"latents must be BCHW, got {tuple(latents.shape)}"
        assert latents.shape[1] == self.latent_channels, \
            f"latent channels {latents.shape[1]} != transformer in_channels {self.latent_channels}"
        assert str(latents.device) == self.cfg.device or latents.device == torch.device(self.cfg.device), \
            f"latents must be on device {self.cfg.device}, got {latents.device}"
        assert latents.dtype == self.cfg.dtype, f"latents dtype {latents.dtype} != {self.cfg.dtype}"

        timesteps = self.pipe.scheduler.timesteps  # [t_0, t_1, ..., t_{N-1}] (usually descending)
        assert 0 <= start_index < len(timesteps), f"start_index out of range: {start_index}/{len(timesteps)}"

        # Some schedulers support this helper. If absent, this is a no-op for you; comment if needed.
        if hasattr(self.pipe.scheduler, "set_begin_index"):
            self.pipe.scheduler.set_begin_index(start_index)

        # Encode text + masks using the official quartet
        do_cfg = (self.cfg.guidance > 1.0)
        prompt_embeds, prompt_mask, neg_embeds, neg_mask = encode_prompt_full(self.pipe, prompt, do_cfg)

        for local_i, t in enumerate(timesteps[start_index:]):
            # Build two/single paths (latents + text)
            latent_in, embeds_in, use_cfg = build_cfg_batch(latents, prompt_embeds, neg_embeds, self.cfg.guidance)
            if use_cfg:
                mask_in = torch.cat([neg_mask, prompt_mask], dim=0)   # [2B, S]
            else:
                mask_in = prompt_mask                                  # [B, S]

            # Scheduler-scale the latent input to expected magnitude
            latent_in = self.pipe.scheduler.scale_model_input(latent_in, t)

            # Prepare per-step inputs
            t_in = torch.tensor([t], device=self.cfg.device, dtype=torch.long).expand(latent_in.shape[0])
            added = make_added_cond_kwargs(self.cfg.height, self.cfg.width, latent_in.shape[0], self.cfg.device)

            # Forward pass through PixArt transformer
            out = self.pipe.transformer(
                hidden_states=latent_in,
                timestep=t_in,
                encoder_hidden_states=embeds_in,
                encoder_attention_mask=mask_in,
                added_cond_kwargs=added,
                return_dict=True,
            ).sample  # could be (2B, 2C, H, W) when learned sigma is enabled

            # Merge learned-sigma channels if present; then apply CFG
            C = latents.shape[1]
            if out.dim() == 4 and out.shape[1] == 2 * C:
                out, _sigma = out.chunk(2, dim=1)  # -> (2B, C, H, W)

            noise_pred = apply_cfg_general(out, self.cfg.guidance, use_cfg, latents)

            # Shape sanity
            assert noise_pred.shape == latents.shape, \
                f"noise_pred {tuple(noise_pred.shape)} must match latents {tuple(latents.shape)}"

            # Scheduler update: x_t -> x_{t-1}
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # Save this step's latents
            global_step = start_index + local_i
            if (global_step % self.cfg.save_every) == 0:
                torch.save(latents.detach().cpu(), os.path.join(self.cfg.out_dir, f"latent_{global_step:03d}.pt"))

        # Decode final image
        img_path = None
        if self.cfg.save_final_image:
            # Note: PixArt pipelines usually come with a VAE; use its scaling factor
            x = latents / self.pipe.vae.config.scaling_factor
            img = self.pipe.vae.decode(x).sample
            pil = self.pipe.image_processor.postprocess(img, output_type="pil")[0]
            img_path = os.path.join(self.cfg.out_dir, self.cfg.final_image_name)
            pil.save(img_path)

        return latents, img_path


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    cfg = RunnerCfg(
        model_id="/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS",
        device="cuda:1",
        dtype=torch.float16,
        steps=28,
        guidance=4.0,
        height=1024,
        width=1024,
        out_dir="./latent_steps_position",
        save_every=1,
        save_final_image=True,
        final_image_name="final.png",
    )

    stepper = LatentStepper(cfg)

    # Prepare an initial latent using the pipeline helper (ensures shape/dtype/device match)
    B = 1
    C = getattr(stepper.pipe.transformer.config, "in_channels", 4)
    latents0 = stepper.pipe.prepare_latents(
        batch_size=B,
        num_channels_latents=C,
        height=cfg.height,
        width=cfg.width,
        dtype=cfg.dtype,
        device=cfg.device,
        generator=torch.Generator(device=cfg.device).manual_seed(1234),
    )

    # Run from step 0 (or set start_index>0 to resume mid-chain)
    final_latents, final_img_path = stepper.run(
        latents=latents0,
        prompt="A majestic castle as the main subject, with a cat sitting in the lower right corner of the scene.",
        start_index=0,
    )

    print("Final latents shape:", tuple(final_latents.shape))
    if final_img_path:
        print("Saved image:", final_img_path)
