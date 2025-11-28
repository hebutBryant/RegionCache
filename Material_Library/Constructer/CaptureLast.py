import torch, json, os
import torch.nn.functional as F
from typing import Optional
from diffusers import PixArtAlphaPipeline
from diffusers.models.attention import Attention
import re
import spacy
nlp = spacy.load("en_core_web_sm")

# ========= 配置 =========
MODEL_PATH = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
PROMPT = "a cat on a red chair"
NUM_STEPS = 3
GUIDANCE = 4.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16
SEED   = 1234

# 你想抓的层名（两种写法都支持）
USER_LAYER_NAME = "transformer_blocks.10.attn2"  # or "transformer.blocks.10.attn2"

SAVE_PT_ATTn  = "attn_laststep_block10_attn2.pt"
SAVE_NPZ_ATTn = "attn_laststep_block10_attn2.npz"
SAVE_PT_HS    = "self_attn_hidden.pt"   # 新增：保存 self-attn 全流程 hidden state


"""
在推理之前 通过在transformer 中的Attention加入sink    
两个sink:
  - 一个 sink 获取 attn score（指定 cross-attn 层，最后一步）
  - 一个 sink 获取 self-attention 的 hidden state（所有层、所有步）
"""


# ========= 自定义 Processor：保存目标 cross-attn scores + 所有 self-attn hidden =========
class AttnProcessorCaptureLast:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args, **kwargs,
    ):
        residual = hidden_states
        is_cross = encoder_hidden_states is not None
        input_ndim = hidden_states.ndim

        # 关闭内部 upcast，严格按当前 dtype
        if hasattr(attn, "upcast_attention"): attn.upcast_attention = False
        if hasattr(attn, "upcast_softmax"):   attn.upcast_softmax   = False

        # === 标准 attention 计算 ===
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        seq_len = (encoder_hidden_states if is_cross else hidden_states).shape[1]
        bsz     = (encoder_hidden_states if is_cross else hidden_states).shape[0]
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bsz)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if not is_cross:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        q = attn.head_to_batch_dim(query)   # [B*H, Nq, d]
        k = attn.head_to_batch_dim(key)     # [B*H, Nk, d]
        v = attn.head_to_batch_dim(value)   # [B*H, Nk, d]

        scale  = getattr(attn, "scale", q.shape[-1] ** -0.5)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = torch.softmax(scores, dim=-1)  # [B*H, Nq, Nk]

        # ===== A) 只在“最后一步” & “指定 cross-attn 层”时保存 attn scores =====
        if is_cross and getattr(attn, "_cap_is_target_layer", False) and getattr(attn, "_cap_is_last_step", False):
            print("###########进入创建attn score sink############")
            sink = getattr(attn, "_cap_sink", None)
            if isinstance(sink, list):
                sink.clear()  # 只留最后一步
                sink.append(probs.detach().to(torch.float16).cpu())

        # === 输出 hidden ===
        hs = torch.bmm(probs, v)
        hs = attn.batch_to_head_dim(hs)
        hs = attn.to_out[0](hs)
        hs = attn.to_out[1](hs)

        if input_ndim == 4:
            hs = hs.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hs = hs + residual

        hs = torch.nan_to_num(hs)
        hs = hs / attn.rescale_output_factor

        # ===== B) 对所有 self-attention 层，记录整个推理流程的 hidden state =====
        #   条件：
        #   - m._hs_capture = True（由 enable_hidden_capture_all_blocks 设置）
        #   - 不是 cross-attn（即 self-attn）
        if getattr(attn, "_hs_capture", False) and not is_cross:
            with torch.no_grad():
                hs_for_log = hs

                # 统一成 [B, L, C]
                if hs_for_log.ndim == 4:
                    B, C, Hh, Ww = hs_for_log.shape
                    hs_for_log = hs_for_log.view(B, C, Hh * Ww).transpose(1, 2)  # [B, L, C]

                # 丢掉 CFG 的 unconditional 半部分（假定 [uncond, cond] 拼 batch）
                if hs_for_log.shape[0] % 2 == 0:
                    B = hs_for_log.shape[0]
                    hs_for_log = hs_for_log[B // 2:]  # 只保留后半部分（cond）

                # 现在通常 [1, L, C] → squeeze 成 [L, C]
                if hs_for_log.shape[0] == 1:
                    hs_for_log = hs_for_log.squeeze(0)  # [L, C]

                hs_for_log = hs_for_log.detach().to(torch.float16).cpu()

                sink_hs = getattr(attn, "_hs_sink", None)
                print("AttnProcessor capture last len(sink_hs)",len(sink_hs))
                if isinstance(sink_hs, list):
                    sink_hs.append(hs_for_log)  # 每一步 append 一次

        return hs


# ========= 工具函数 =========
def _norm(s: str) -> str:
    return s.replace(".", "_").replace("/", "_").lower()

def resolve_layer_name(pipe, user_name: str) -> str:
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    target_norm = _norm(user_name)
    found = []
    for name, m in root.named_modules():
        if isinstance(m, Attention):
            if _norm(name) == target_norm:
                return name
            # 兜底：包含 “blocks_10” 且 “attn2”
            if "attn2" in _norm(name) and any(
                k in _norm(name) for k in ["blocks_10", "block_10", "blocks.10", "block.10", "10"]
            ):
                found.append(name)
    if found:
        return sorted(found)[-1]
    raise RuntimeError(f"未找到目标 Attention 层：{user_name}")

def replace_all_attn_processors(pipe):
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    new = {n: AttnProcessorCaptureLast() for n, _ in root.attn_processors.items()}
    root.set_attn_processor(new)

def enable_target_capture(pipe, target_layer: str):
    """给目标 cross-attn 层挂 sink 与标记（用于保存最后一步 attn scores）"""
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    target_sink = None
    for name, m in root.named_modules():
        if isinstance(m, Attention):
            m._cap_is_target_layer = (name == target_layer)
            if m._cap_is_target_layer:
                m._cap_sink = []
                target_sink = m._cap_sink
    if target_sink is None:
        raise RuntimeError("目标层未启用 sink")
    return target_sink

def make_laststep_hook(pipe):
    """
    包装 scheduler.step：
    - 当 timestep == timesteps[-1] 时，把 _cap_is_last_step=True 广播到所有 Attention
    """
    orig = pipe.scheduler.step
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))

    def wrapped(model_output, timestep, sample, *args, **kwargs):
        cur_t  = int(timestep.item() if hasattr(timestep, "item") else timestep)
        last_t = pipe.scheduler.timesteps[-1]
        last_t = int(last_t.item() if hasattr(last_t, "item") else last_t)
        is_last = (cur_t == last_t)

        # 广播标记
        for _, m in root.named_modules():
            if isinstance(m, Attention):
                m._cap_is_last_step = is_last

        return orig(model_output, timestep, sample, *args, **kwargs)

    pipe.scheduler.step = wrapped

def get_target_attn_module(pipe, user_name: str) -> Attention:
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    assert root is not None
    def _norm(s): return s.replace(".", "_").lower()
    want = _norm(user_name)
    target = None
    for name, m in root.named_modules():
        if isinstance(m, Attention) and _norm(name) == want:
            target = m
            break
    if target is None:
        raise RuntimeError(f"未找到 Attention 层：{user_name}")
    return target

def enable_hidden_capture_all_blocks(pipe, only_cross: bool = False):
    """
    给 transformer 里的每个 Attention 模块挂上 hidden_state 的 sink。
    返回 {层名: list}，每步都会往对应 list 里 append 一条记录。

    only_cross=True 时只给 cross-attn (attn2) 挂，节省内存。
    我们目前要抓 self-attn，因此在 main 里调用时用 only_cross=False。
    """
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    assert root is not None, "未找到 transformer / unet"

    sinks = {}
    for name, m in root.named_modules():
        if isinstance(m, Attention):
            if only_cross and getattr(m, "is_cross_attention", None) is not True:
                continue
            m._hs_capture = True
            m._hs_sink = []
            sinks[name] = m._hs_sink
    print(f"✅ hidden_state 捕获已启用：{len(sinks)} 个 Attention 层")
    return sinks


# ========= 主流程 =========
def main():
    # 加载
    pipe = PixArtAlphaPipeline.from_pretrained(MODEL_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    # 替换所有 Attention 的 Processor
    replace_all_attn_processors(pipe)
    root = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    assert root is not None, "当前模型不包含 transformer/unet"

    print("=== 所有 Attention 层 ===")
    for name, module in root.named_modules():
        if isinstance(module, Attention):
            print(name, " | is_cross_attention =", module.is_cross_attention)

    # 启用 self-attention hidden 捕获（只需 once）
    # only_cross=False 表示 self + cross 都挂 sink，但 Processor 里只在 self-attn 上写入 _hs_sink
    hs_sinks = enable_hidden_capture_all_blocks(pipe, only_cross=False)

    # 解析目标 cross-attn 层名并启用该层的 attn_scores 捕获
    target_layer = resolve_layer_name(pipe, USER_LAYER_NAME)
    print(f"[target] 使用层：{target_layer}")
    attn_sink = enable_target_capture(pipe, target_layer)

    # 挂 scheduler 的“最后一步”判定
    make_laststep_hook(pipe)

    # 固定种子在 __call__ 里传入（注意：不要传给 from_pretrained）
    gen = torch.Generator(DEVICE).manual_seed(SEED)

    # 正常推理
    out = pipe(prompt=PROMPT, num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE, generator=gen)

    # ====== 1) cross-attn 最后一步 attention 保存 ======
    # if not attn_sink:
    #     raise RuntimeError("❌ 没有捕获到最后一步的注意力（请检查层名/是否 cross-attn/是否确实走到了最后一步）。")

    attn_last = attn_sink[-1]  # Tensor [B*H, Nq, Nk] on CPU, float16
    print(f"[ok] 捕获到注意力：{tuple(attn_last.shape)}，dtype={attn_last.dtype}")

    torch.save({"layer": target_layer, "prompt": PROMPT, "attn": attn_last}, SAVE_PT_ATTn)
    print(f"[save] torch: {os.path.abspath(SAVE_PT_ATTn)}")

    try:
        import numpy as np
        np.savez_compressed(SAVE_NPZ_ATTn, attn=attn_last.numpy())
        print(f"[save] npz : {os.path.abspath(SAVE_NPZ_ATTn)}")
    except Exception as e:
        print(f"[warn] npz 保存失败：{e}")

    # 也保存图片
    out.images[0].save("pixart_output.png")
    print("[save] image: pixart_output.png")

    # ====== 2) 保存整条 self-attention 流程的 hidden state ======
    # hs_sinks: {层名: [step0_tensor[L,C], step1_tensor[L,C], ...]}
    packed = {}
    for name, lst in hs_sinks.items():
        if len(lst) == 0:
            continue
        # 形状: [num_steps, L, C]
        try:
            stacked = torch.stack(lst, dim=0)   # 每一步同一层 shape 一致，可以 stack
        except Exception as e:
            print(f"[warn] 层 {name} stack 失败: {e}")
            continue
        packed[name] = stacked   # [T, L, C]

    torch.save(
        {
            "prompt": PROMPT,
            "num_steps": NUM_STEPS,
            "hidden": packed,   # dict: layer_name -> [T, L, C]
        },
        SAVE_PT_HS,
    )
    print(f"[save] self-attn hidden 全流程: {os.path.abspath(SAVE_PT_HS)}")
    print(f"共 {len(packed)} 个 attention 层记录了 hidden state")

if __name__ == "__main__":
    main()
