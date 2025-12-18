import os
import torch
from diffusers import DiffusionPipeline

# ============ 配置区 ============
MODEL_DIR = "/home/liuhy/DiT_XL_2_512_model"
OUT_PATH = "dit_out.png"

# DiT(class-conditional) 需要类别ID：int 或 list[int]
# 例如 ImageNet 的类别 id (0~999)，这里随便写个示例
CLASS_ID = [934]

# 采样参数
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 4.0   # DiT 有的实现会用到，有的会忽略；留着不坏
SEED = 1234

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dtype：优先 bfloat16（如果你的 GPU 支持），否则 float16，再否则 float32
if DEVICE == "cuda":
    # bfloat16 在部分GPU可用（A100/4090/部分30系也可能支持）
    # 不确定的话，用 float16 最稳
    DTYPE = torch.float16
else:
    DTYPE = torch.float32
# ===============================


def main():
    print(f"[INFO] Loading model from: {MODEL_DIR}")
    print(f"[INFO] Device: {DEVICE}, dtype: {DTYPE}")

    # 建议：不要混用 device_map="cuda" + 手动 .to("cuda")
    # 如果你是单卡推理，最简单稳妥：加载后 .to(device)
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
    )

    pipe = pipe.to(DEVICE)

    # 一些小优化（可选）
    # pipe.enable_attention_slicing()
    # pipe.enable_vae_slicing()  # 如果 pipeline 里有 VAE 才有意义

    # 随机种子
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    # ======== 关键：传 class_labels（必须是数字，不是字符串）========
    # 支持 int 或 list[int]，这里用 int
    print(f"[INFO] Running inference with class_labels={CLASS_ID}")

    with torch.inference_mode():
        out = pipe(
            class_labels=CLASS_ID,
            generator=generator,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        )

    image = out.images[0]
    image.save(OUT_PATH)
    print(f"[DONE] Saved image to: {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
