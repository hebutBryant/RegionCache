# 复用latents （nsdi24）
# latent_reuse.py
import os
import re
from typing import Dict, List, Tuple, Optional
from xian import RunnerCfg, LatentStepper


import torch

# Assumes LatentStepper and RunnerCfg are already defined/imported from your file
# from latent_stepper import LatentStepper, RunnerCfg


class LatentBank:
    """
    Indexes and loads saved latent checkpoints from a directory.
    Expected filename pattern: 'latent_{step:03d}.pt' produced by LatentStepper.
    """

    _PAT = re.compile(r"^latent_(\d{3})\.pt$")

    def __init__(self, directory: str, map_location: Optional[str] = None):
        self.directory = directory
        self.map_location = map_location  # e.g., "cuda:0" or "cpu"
        self._index: Dict[int, str] = {}
        self.refresh()

    def refresh(self) -> None:
        """Rescan the directory and rebuild the index."""
        self._index.clear()
        if not os.path.isdir(self.directory):
            return
        for name in os.listdir(self.directory):
            m = self._PAT.match(name)
            if m:
                step = int(m.group(1))
                self._index[step] = os.path.join(self.directory, name)

    def steps(self) -> List[int]:
        """Return sorted list of available steps (global_step)."""
        return sorted(self._index.keys())

    def has_step(self, step: int) -> bool:
        return step in self._index

    def latest(self) -> Optional[int]:
        """Return the greatest step number, or None if empty."""
        s = self.steps()
        return s[-1] if s else None

    def load(self, step: int, dtype: Optional[torch.dtype] = None, device: Optional[str] = None) -> torch.Tensor:
        """
        Load a latent tensor saved at given step.
        Args:
            step: global step integer that matches the filename latent_{step:03d}.pt
            dtype/device: optional override (will .to(...))
        """
        if step not in self._index:
            raise FileNotFoundError(f"No latent file for step={step} in {self.directory}")
        path = self._index[step]
        t = torch.load(path, map_location=self.map_location or device or "cpu")
        if dtype is not None or device is not None:
            t = t.to(device=device if device is not None else t.device,
                     dtype=dtype if dtype is not None else t.dtype)
        return t

    def load_latest(self, dtype: Optional[torch.dtype] = None, device: Optional[str] = None) -> Tuple[int, torch.Tensor]:
        step = self.latest()
        if step is None:
            raise FileNotFoundError(f"No latent_*.pt found under: {self.directory}")
        return step, self.load(step, dtype=dtype, device=device)


class LatentReuser:
    """
    Cross-prompt resumer (跨 prompt 复用):
    - Pick an intermediate latent from LatentBank
    - Resume denoising with a new prompt/guidance from that step onward
    """

    def __init__(self, stepper: "LatentStepper", bank: LatentBank):
        self.stepper = stepper
        self.bank = bank

    def resume_from_step(
        self,
        step: int,
        new_prompt: str,
        guidance: Optional[float] = None,
        save_final_image: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[str]]:
        """
        Resume denoising from a specific saved 'global_step' using a NEW prompt.

        Args:
            step: global step index that the latent file name carries (e.g., 12 for 'latent_012.pt').
                  Resume will start at the NEXT scheduler step (i.e., start_index = step + 1),
                  because the stored latent_XXX.pt is already x_{t_{step+1}} after that update.
            new_prompt: the new text prompt to condition the remaining steps
            guidance: optional override for CFG scale
            save_final_image: optional override for saving final image

        Returns:
            (final_latents, final_image_path or None)
        """
        # 1) Load the saved latent at given step and move to correct device/dtype
        latents = self.bank.load(
            step,
            dtype=self.stepper.cfg.dtype,
            device=self.stepper.cfg.device,
        )

        # 2) Adjust runtime options if the caller wants to override guidance/save flag
        old_guidance = self.stepper.cfg.guidance
        old_save_flag = self.stepper.cfg.save_final_image
        if guidance is not None:
            self.stepper.cfg.guidance = guidance
        if save_final_image is not None:
            self.stepper.cfg.save_final_image = save_final_image

        try:
            # 3) Resume from NEXT step
            #    If you saved 'latent_{k}.pt' after finishing scheduler step k,
            #    then you should continue at start_index = k + 1.
            start_index = step + 1

            # Guard: if we try to resume past the final step, just decode and return.
            max_idx = len(self.stepper.pipe.scheduler.timesteps) - 1
            if start_index > max_idx:
                # Direct decode the latent as final image
                x = latents / self.stepper.pipe.vae.config.scaling_factor
                img = self.stepper.pipe.vae.decode(x).sample
                pil = self.stepper.pipe.image_processor.postprocess(img, output_type="pil")[0]
                out_path = os.path.join(self.stepper.cfg.out_dir, self.stepper.cfg.final_image_name)
                pil.save(out_path)
                return latents, out_path

            # 4) Run the remaining denoising with the new prompt
            final_latents, final_img_path = self.stepper.run(
                latents=latents,
                prompt=new_prompt,
                start_index=start_index,
            )
        finally:
            # Restore the original runtime options
            self.stepper.cfg.guidance = old_guidance
            self.stepper.cfg.save_final_image = old_save_flag

        return final_latents, final_img_path

    def resume_from_latest(
        self,
        new_prompt: str,
        guidance: Optional[float] = None,
        save_final_image: Optional[bool] = None,
    ) -> Tuple[int, torch.Tensor, Optional[str]]:
        """
        Convenience wrapper: load the latest saved latent and resume with a new prompt.
        Returns: (loaded_step, final_latents, final_image_path or None)
        """
        step, lat = self.bank.load_latest(dtype=self.stepper.cfg.dtype, device=self.stepper.cfg.device)

        # Store back to disk? Not needed here—LatentStepper.run will keep saving subsequent steps.
        # We reuse resume_from_step logic but avoid double-loading.
        # To avoid re-loading, temporarily monkey-patch a tiny 'run_from_loaded' path:
        # For clarity and simplicity, just call resume_from_step.
        final_latents, final_img_path = self.resume_from_step(
            step=step,
            new_prompt=new_prompt,
            guidance=guidance,
            save_final_image=save_final_image,
        )
        return step, final_latents, final_img_path



def main():
    # === 1. 基础配置 ===
    cfg = RunnerCfg(
        model_id="/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS",
        device="cuda:2",
        dtype=torch.float16,
        steps=28,
        guidance=4.0,
        height=1024,
        width=1024,
        out_dir="./latent_steps_position",   # 存有之前 latent 的目录
        save_every=9999,          # 不保存中间 latent
        save_final_image=True,    # 仅保存最终图片
        final_image_name="cat_reuse.png",
    )

    # === 2. 初始化 Stepper 与 LatentBank ===
    stepper = LatentStepper(cfg)
    bank = LatentBank(cfg.out_dir, map_location=cfg.device)

    # === 3. 初始化复用器 ===
    reuser = LatentReuser(stepper, bank)

    # === 4. 选择复用步 (可手动修改) ===
    # 例如复用上次“狼”生成到第 12 步保存的 latent_012.pt
    reuse_step = 10
    new_prompt = "a cat"

    # === 5. 调用复用，继续去噪但不保存中间 latent ===
    final_latents, final_img_path = reuser.resume_from_step(
        step=reuse_step,
        new_prompt=new_prompt,
        guidance=5.0,             # 可选调整 CFG 强度
        save_final_image=True,    # 只输出最后图片
    )

    print(f"✅ Reused from step {reuse_step}")
    print(f"Final latents shape: {tuple(final_latents.shape)}")
    print(f"Saved image: {final_img_path}")


if __name__ == "__main__":
    main()