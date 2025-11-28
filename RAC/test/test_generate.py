import torch
from diffusers import PixArtAlphaPipeline
from PIL import Image

# 选择 GPU 或 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
pipe = PixArtAlphaPipeline.from_pretrained(
    "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS",
    torch_dtype=torch.float16,
).to(device)

# 提示词
prompt = "There are two lamps above the table and a television below it."

# 生成图片
with torch.no_grad():
    image = pipe(prompt, num_inference_steps=20, guidance_scale=4.5).images[0]

# 保存到本地
save_path = "generated_image.png"
image.save(save_path)

print(f"图片已保存到: {save_path}")
