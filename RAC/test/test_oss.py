import os
import io
import torch
from diffusers import PixArtAlphaPipeline
from PIL import Image

# ===============================
#        ☁️ 阿里云 OSS 配置
# ===============================
import os
import oss2

auth = oss2.Auth(
    os.getenv("OSS_ACCESS_KEY_ID"),
    os.getenv("OSS_ACCESS_KEY_SECRET")
)

bucket = oss2.Bucket(
    auth,
    "oss-cn-beijing.aliyuncs.com",
    "dit-experiment"
)



try:
    info = bucket.get_bucket_info()
    print("✅ 可以访问 OSS")
    print("Bucket 名称:", info.name)
    print("地域:", info.location)
except oss2.exceptions.OssError as e:
    print("❌ 访问 OSS 失败")
    print(e)

# ===============================
#        ⚙️ 推理配置
# ===============================
# DTYPE = torch.float16
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL_ID = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
# PROMPT = "a dog on a desk, high quality"

# NUM_INFERENCE_STEPS = 20
# GUIDANCE_SCALE = 7.0
# SEED = 42
# STEP = 0   # 你可以在循环中动态修改

# # ===============================
# #        🚀 加载模型
# # ===============================
# pipe = PixArtAlphaPipeline.from_pretrained(
#     MODEL_ID,
#     torch_dtype=DTYPE
# ).to(DEVICE)

# pipe.set_progress_bar_config(disable=True)

# generator = torch.Generator(device=DEVICE).manual_seed(SEED)

# # ===============================
# #        🎨 生成图片
# # ===============================
# with torch.no_grad():
#     image: Image.Image = pipe(
#         PROMPT,
#         num_inference_steps=NUM_INFERENCE_STEPS,
#         guidance_scale=GUIDANCE_SCALE,
#         generator=generator
#     ).images[0]

# # ===============================
# #        ☁️ 上传到 OSS（test/ 目录）
# # ===============================
# buf = io.BytesIO()
# image.save(buf, format="PNG")
# buf.seek(0)

# oss_path = f"test/pixart_step_{STEP:06d}_seed_{SEED}.png"

# bucket.put_object(oss_path, buf)

# print(f"✅ 上传成功：oss://{OSS_BUCKET_NAME}/{oss_path}")