import os
import time
import json
import types
import torch
from typing import Optional, Dict, List, Tuple

from diffusers import PixArtAlphaPipeline
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0
from tgate import TgatePixArtAlphaLoader
from PIL import Image


# ======================================================
#                    配置区域
# ======================================================
DTYPE = torch.float16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_ID = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
JSON_FILE = "/home/liuhy/test.json"
OUTPUT_DIR = "/home/liuhy/tgate_multi_results_auto_mask"

# TGATE 参数
GATE_STEP = 10
SP_INTERVAL = 5
FI_INTERVAL = 1
WARM_UP = 2
NUM_INFERENCE_STEPS = 15
GUIDANCE_SCALE = 4.0
SEED = 42

# Auto-mask 参数
MASK_THRESHOLD = 0.35      # soft mask -> binary 的阈值（可调）
MASK_DILATE = 1            # 简单膨胀次数（0=不膨胀，1/2=更宽松）
CAPTURE_USE_LAST = True    # 用最后一次捕获（True）或平均所有（False）

torch.manual_seed(SEED)
if DEVICE.startswith("cuda"):
    torch.cuda.manual_seed_all(SEED)


# ======================================================
#   1) 手算版 AttnProcessor + Cross-Attn Capture
# ======================================================
class AttnProcessorMe:
    """
    手算 attention，且在 cross-attention 时把 attention_probs 存入 attn._xattn_sink
    """

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

        # 注意：你已有 get_attention_scores2_0(is_cross, ...) 逻辑
        attention_probs = attn.get_attention_scores2_0(is_cross, query, key, attention_mask)

        # ✅ 关键补丁：捕获 cross-attention
        if is_cross and getattr(attn, "_xattn_capture", False) and hasattr(attn, "_xattn_sink"):
            # attention_probs: [B*H, Q, K]
            attn._xattn_sink.append(attention_probs.detach())

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def replace_attnprocessor2_0_with_attnprocessor(pipe):
    """
    将 transformer/unet 的 AttnProcessor2_0 替换为 AttnProcessorMe
    返回 restore() 用于恢复
    """
    if hasattr(pipe, "transformer"):
        tr = pipe.transformer
    elif hasattr(pipe, "unet"):
        tr = pipe.unet
    else:
        raise ValueError("pipeline 没有 transformer 或 unet")

    old = dict(tr.attn_processors)
    new = {}
    replaced, total = 0, 0

    for name, proc in tr.attn_processors.items():
        total += 1
        if isinstance(proc, AttnProcessor2_0):
            new[name] = AttnProcessorMe()
            replaced += 1
        else:
            new[name] = proc

    tr.set_attn_processor(new)

    def restore():
        tr.set_attn_processor(old)

    print(f"✅ 已替换 {replaced}/{total} 个 AttnProcessor2_0 → AttnProcessorMe")
    return restore


def enable_xattn_capture(pipe) -> Dict[str, list]:
    sinks = {}
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    assert root is not None, "未找到 transformer / unet"

    for name, module in root.named_modules():
        if isinstance(module, Attention) and getattr(module, "is_cross_attention", False):
            module._xattn_capture = True
            module._xattn_sink = []
            sinks[name] = module._xattn_sink
    return sinks


def disable_xattn_capture(pipe):
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    if root is None:
        return
    for _, module in root.named_modules():
        if isinstance(module, Attention):
            if hasattr(module, "_xattn_capture"):
                delattr(module, "_xattn_capture")
            if hasattr(module, "_xattn_sink"):
                delattr(module, "_xattn_sink")


# ======================================================
#   2) Mask 后处理：token diff -> token indices -> mask
# ======================================================
def parse_prompts(item):
    """
    返回排序后的提示词列表：
    [('original', ...), ('modified', ...), ('modified2', ...)]
    """
    prompts = []
    if "original" in item:
        prompts.append(("original", item["original"]))

    mods = []
    for k, v in item.items():
        if k.startswith("modified"):
            suffix = k.replace("modified", "")
            idx = 1 if suffix == "" else int(suffix)
            mods.append((idx, k, v))
    mods.sort(key=lambda x: x[0])

    for _, k, v in mods:
        prompts.append((k, v))
    return prompts


def get_edit_token_indices(tokenizer, original_prompt: str, modified_prompt: str) -> List[int]:
    """
    最简单的 token-diff：modified 里出现但 original 没出现的 token 视为变化语义
    返回这些 token 在 modified 的 token 序列中的 index
    """
    orig_tokens = tokenizer.tokenize(original_prompt)
    mod_tokens = tokenizer.tokenize(modified_prompt)

    edit_tokens = list({t for t in mod_tokens if t not in orig_tokens})
    if len(edit_tokens) == 0:
        # 没检测到变化，返回空列表（后面会 fallback 到全图编辑或小 mask）
        return []

    edit_indices = [i for i, t in enumerate(mod_tokens) if t in edit_tokens]
    return edit_indices


def _dilate_mask(mask: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """
    mask: [1,1,H,W] in {0,1} or [0,1]
    简单 maxpool 膨胀
    """
    if iters <= 0:
        return mask
    for _ in range(iters):
        mask = torch.nn.functional.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return mask


def build_prompt_aligned_mask(
    xattn_sinks: Dict[str, list],
    edit_token_indices: List[int],
    latent_h: int,
    latent_w: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    返回 binary mask: [1,1,H,W]  1=允许编辑(重算)  0=复用/冻结
    """
    layer_maps = []

    # 如果没有 edit token，直接返回全 1（等价全图编辑）
    if len(edit_token_indices) == 0:
        return torch.ones((1, 1, latent_h, latent_w), device=device, dtype=dtype)

    for _, attn_list in xattn_sinks.items():
        if len(attn_list) == 0:
            continue

        if CAPTURE_USE_LAST:
            attn = attn_list[-1]  # [B*H, Q, K]
        else:
            # 平均所有捕获（更稳但更慢、更占内存）
            attn = torch.stack(attn_list, dim=0).mean(dim=0)

        # attn: [B*H, Q, K]
        BH, Q, K = attn.shape
        attn = attn.view(-1, Q, K)  # [H, Q, K]  (B=1 时成立)

        # 取 edit token
        attn_edit = attn[:, :, edit_token_indices]  # [H, Q, |T|]
        attn_edit = attn_edit.mean(dim=0).mean(dim=-1)  # [Q]

        layer_maps.append(attn_edit)

    # 如果没捕获到任何层，fallback 全 1
    if len(layer_maps) == 0:
        return torch.ones((1, 1, latent_h, latent_w), device=device, dtype=dtype)

    mask = torch.stack(layer_maps, dim=0).mean(dim=0)  # [Q]
    mask = mask.view(latent_h, latent_w)

    # normalize -> [0,1]
    mask = mask - mask.min()
    mask = mask / (mask.max() + 1e-6)

    # threshold -> binary
    mask = (mask >= MASK_THRESHOLD).to(dtype=dtype)
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # dilate
    mask = _dilate_mask(mask, iters=MASK_DILATE)

    return mask


# ======================================================
#   3) Scheduler.step Hook：record / freeze
# ======================================================
class LatentController:
    """
    mode='record': 记录每一步 prev_sample 到 current_latents
    mode='freeze': 使用 cached_latents + mask 融合
                  mask=1 -> 用新 prev_sample (允许编辑)
                  mask=0 -> 用 ref latent (冻结复用)
    """
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.orig_step = scheduler.step
        self.mode = "record"

        self.mask = None                 # [1,1,H,W]
        self.cached_latents = []
        self.current_latents = []
        self.step_count = 0

    def hook(self):
        self.scheduler.step = types.MethodType(self.patched_step, self.scheduler)

    def unhook(self):
        self.scheduler.step = self.orig_step

    def set_mode(self, mode, cached_latents=None, mask=None):
        self.mode = mode
        self.step_count = 0
        if mode == "record":
            self.current_latents = []
        elif mode == "freeze":
            if cached_latents is None or mask is None:
                raise ValueError("Freeze mode requires cached_latents and mask")
            self.cached_latents = cached_latents
            self.mask = mask

    def patched_step(self, scheduler, model_output, timestep, sample, **kwargs):
        out = self.orig_step(model_output, timestep, sample, **kwargs)

        if isinstance(out, tuple):
            prev_sample = out[0]
        else:
            prev_sample = out.prev_sample

        if self.mode == "record":
            self.current_latents.append(prev_sample.detach().clone())

        elif self.mode == "freeze":
            if self.step_count < len(self.cached_latents):
                ref = self.cached_latents[self.step_count].to(prev_sample.device)
                mask = self.mask.to(prev_sample.device).to(prev_sample.dtype)
                prev_sample = mask * prev_sample + (1.0 - mask) * ref

        self.step_count += 1

        if isinstance(out, tuple):
            return (prev_sample,) + out[1:]
        else:
            return out.__class__(prev_sample=prev_sample, pred_original_sample=out.pred_original_sample)


# ======================================================
#                    主流程
# ======================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {JSON_FILE}")

    print(f"Loading model from {MODEL_ID}...")
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    # 包装 TGATE
    pipe = TgatePixArtAlphaLoader(pipe)

    # scheduler hook：record/freeze
    controller = LatentController(pipe.scheduler)
    controller.hook()

    # 为 auto-mask：替换 attn processor（只做一次，避免每个样本反复替换）
    restore_attn = replace_attnprocessor2_0_with_attnprocessor(pipe)

    for item_idx, item in enumerate(data):
        print(f"\n=== Processing Item {item_idx} ===")
        prompt_sequence = parse_prompts(item)

        cached_latents = None
        prev_prompt_text = None

        for step_idx, (p_type, prompt_text) in enumerate(prompt_sequence):
            print(f"  > [{p_type}] {prompt_text[:80]}")

            # -------------------------
            # 1) original：record latents
            # -------------------------
            if step_idx == 0:
                controller.set_mode("record")
                prev_prompt_text = prompt_text

                start_t = time.time()
                result = pipe.tgate(
                    prompt=prompt_text,
                    gate_step=GATE_STEP,
                    sp_interval=SP_INTERVAL,
                    fi_interval=FI_INTERVAL,
                    warm_up=WARM_UP,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                )
                image = result.images[0]
                cost = time.time() - start_t

                save_name = f"{item_idx:03d}_{step_idx}_{p_type}.png"
                save_path = os.path.join(OUTPUT_DIR, save_name)
                image.save(save_path)
                print(f"    Saved: {save_path} ({cost:.2f}s)")

                cached_latents = controller.current_latents
                print(f"    Cached {len(cached_latents)} latent steps for next turn.")
                continue

            # -------------------------
            # 2) modified：auto mask + freeze
            # -------------------------
            assert cached_latents is not None and prev_prompt_text is not None

            # 2.1 计算 edit token indices（基于上一轮 prompt）
            edit_token_indices = get_edit_token_indices(
                tokenizer=pipe.tokenizer,
                original_prompt=prev_prompt_text,
                modified_prompt=prompt_text,
            )
            print(f"    Edit token indices: {edit_token_indices[:12]}{'...' if len(edit_token_indices)>12 else ''}")

            # 2.2 开启 cross-attn capture
            xattn_sinks = enable_xattn_capture(pipe)

            # 2.3 先跑一次推理（同一次推理里：既捕获 attention，也执行 freeze）
            # mask 需要 latent 分辨率
            ref_h, ref_w = cached_latents[0].shape[-2:]

            # 先用一个临时 mask=全1 让它正常走（因为我们要先捕获 attention）
            tmp_mask = torch.ones((1, 1, ref_h, ref_w), device=DEVICE, dtype=DTYPE)
            controller.set_mode("freeze", cached_latents=cached_latents, mask=tmp_mask)

            start_t = time.time()
            result = pipe.tgate(
                prompt=prompt_text,
                gate_step=GATE_STEP,
                sp_interval=SP_INTERVAL,
                fi_interval=FI_INTERVAL,
                warm_up=WARM_UP,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            )
            cost_probe = time.time() - start_t

            # 2.4 根据捕获的 attention 构建真正 mask
            mask = build_prompt_aligned_mask(
                xattn_sinks=xattn_sinks,
                edit_token_indices=edit_token_indices,
                latent_h=ref_h,
                latent_w=ref_w,
                device=DEVICE,
                dtype=DTYPE,
            )

            disable_xattn_capture(pipe)

            # 2.5 用真实 mask 再跑一次：这次才是“RegionCache/TGate + auto-mask freeze”
            controller.set_mode("freeze", cached_latents=cached_latents, mask=mask)

            start_t = time.time()
            result2 = pipe.tgate(
                prompt=prompt_text,
                gate_step=GATE_STEP,
                sp_interval=SP_INTERVAL,
                fi_interval=FI_INTERVAL,
                warm_up=WARM_UP,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            )
            image2 = result2.images[0]
            cost = time.time() - start_t

            save_name = f"{item_idx:03d}_{step_idx}_{p_type}.png"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            image2.save(save_path)
            print(f"    Saved: {save_path} (probe {cost_probe:.2f}s + final {cost:.2f}s)")

            # 更新“上一轮 prompt”（用于 modified2、modified3）
            prev_prompt_text = prompt_text

            # 可选：也更新 cached_latents（如果你希望逐轮把当前结果变成新的 origin）
            # cached_latents = controller.current_latents  # ❌ 注意：freeze 模式下 current_latents 不会更新
            # 更正确做法：如果你想滚动更新 cache，需要在 freeze 里也记录（可再加一个 record_in_freeze 开关）

    controller.unhook()
    restore_attn()
    print("\nAll done!")


if __name__ == "__main__":
    main()
