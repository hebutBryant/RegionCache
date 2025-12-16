import types
import time
import torch
from diffusers import PixArtAlphaPipeline

# ===============================================
#                 ⚙️ 配置参数
# ===============================================
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
PROMPT = "a dog on a desk, high quality"

# N=4：每4步做一次 full forward，其他3步直接复用缓存输出（不跑 transformer）
N = 3

NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.0
SEED = 42
OUT_PATH = "pixart_deepcache_N_fast.png"

torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ===============================================
#              🔧 工具：解析/重建输出
# ===============================================
def extract_noise_pred(out):
    """稳健取出 noise_pred tensor"""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        if len(out) == 0:
            raise TypeError("Empty tuple/list returned from transformer.")
        return out[0]
    if hasattr(out, "sample"):
        return out.sample
    raise TypeError(f"Unsupported transformer output type: {type(out)}")


def detach_if_tensor(x):
    return x.detach() if isinstance(x, torch.Tensor) else x


def detach_tree(obj):
    """把 tuple/list 里的 tensor 都 detach 一下，避免意外引用计算图"""
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    if isinstance(obj, tuple):
        return tuple(detach_tree(x) for x in obj)
    if isinstance(obj, list):
        return [detach_tree(x) for x in obj]
    # ModelOutput / dict-like: 不做深拷贝，避免破坏结构；只在需要时处理 sample
    return obj


def build_cached_output(template, eps):
    """
    用 cached template 重建 transformer.forward 的返回值，使 pipeline 兼容：
    - Tensor: 直接返回 eps
    - tuple/list: 把第一个元素替换成 eps，其余保持 template 的 tail
    - ModelOutput(sample=...): 修改 sample 字段返回
    """
    if template["kind"] == "tensor":
        return eps

    if template["kind"] == "tuple":
        tail = template["tail"]
        return (eps,) + tail

    if template["kind"] == "list":
        tail = template["tail"]
        return [eps] + tail

    if template["kind"] == "modeloutput":
        out_obj = template["obj"]
        out_obj.sample = eps
        return out_obj

    raise TypeError(f"Unknown template kind: {template['kind']}")


def make_template_from_out(out):
    """
    full forward 时缓存返回结构模板，供 reuse 时构造返回值。
    """
    if isinstance(out, torch.Tensor):
        return {"kind": "tensor"}

    if isinstance(out, tuple):
        # 缓存除第一个元素以外的 tail（并 detach，避免图引用）
        tail = detach_tree(out[1:])
        return {"kind": "tuple", "tail": tail}

    if isinstance(out, list):
        tail = detach_tree(out[1:])
        return {"kind": "list", "tail": tail}

    if hasattr(out, "sample"):
        # 直接缓存对象本身（它通常是轻量 dataclass）
        # reuse 时只改 out.sample
        return {"kind": "modeloutput", "obj": out}

    raise TypeError(f"Unsupported transformer output type: {type(out)}")


# ===============================================
#          🧠 DeepCache (真正加速：跳过 forward)
# ===============================================
class StepCache:
    """
    真正加速版：
      - full step: 调 orig_forward，拿到 eps，并缓存 eps + 返回结构模板
      - reuse step: 不调 orig_forward，直接返回缓存 eps（按模板包装）
    """
    def __init__(self, N: int):
        self.N = int(N)
        self.step_idx = 0

        self.cache_eps = None          # Tensor (B,C,H,W)
        self.template = None           # dict describing return structure

    def should_full(self):
        return (self.step_idx % self.N) == 0 or (self.cache_eps is None) or (self.template is None)

    def advance(self):
        self.step_idx += 1


def enable_step_cache_on_transformer(pipe: PixArtAlphaPipeline, cache: StepCache):
    transformer = pipe.transformer
    orig_forward = transformer.forward

    def patched_forward(self, *args, **kwargs):
        # ===== reuse step：直接返回缓存，不跑 transformer =====
        if not cache.should_full():
            eps = cache.cache_eps
            out = build_cached_output(cache.template, eps)
            cache.advance()
            return out

        # ===== full step：正常 forward，并更新缓存 =====
        out = orig_forward(*args, **kwargs)
        eps = extract_noise_pred(out)

        # 缓存 eps + 模板（都 detach，省显存且避免引用计算图）
        cache.cache_eps = eps.detach()
        cache.template = make_template_from_out(out)

        cache.advance()
        return build_cached_output(cache.template, cache.cache_eps)

    transformer.forward = types.MethodType(patched_forward, transformer)


# ===============================================
#                 🚀 主流程
# ===============================================
def main():
    pipe = PixArtAlphaPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    pipe.set_progress_bar_config(disable=False)

    cache = StepCache(N=N)
    enable_step_cache_on_transformer(pipe, cache)

    start = time.time()
    image = pipe(
        prompt=PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    ).images[0]
    end = time.time()

    print(f"[DeepCache FAST] N={N}, steps={NUM_INFERENCE_STEPS}, time={end-start:.3f}s")
    image.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
