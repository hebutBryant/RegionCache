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

prompt ="""A sepia-toned classic Mercedes-Benz 2.5L streamlined racing car gleams under soft light, its sleek, aerodynamic body and wire-spoke wheels exuding 1950s motorsport elegance. Displayed at an outdoor vintage event, it’s surrounded by other historic racers and onlookers. The iconic three-pointed star adorns its curved hood. A plaque nearby confirms its identity, capturing automotive history in timeless, polished detail.
"""
turn1 = "Zoom in closely on the front and side of the main silver race car"

# prompt = prompt+turn1

# 提示词prompt = "There are two lamps above the table and a television below it."
# 

# 生成图片
with torch.no_grad():
    image = pipe(prompt, num_inference_steps=20, guidance_scale=4.5).images[0]

# 保存到本地
save_path = "generated_image.png"
image.save(save_path)

print(f"图片已保存到: {save_path}")
