import torch
from diffusers import PixArtAlphaPipeline
from PIL import Image
import subprocess

# ======================
# 设备
# ======================
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# ======================
# 加载模型
# ======================

pipe = PixArtAlphaPipeline.from_pretrained(
    "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS",
    torch_dtype=torch.float16,
).to(device)

# prompt = """A sepia-toned classic Mercedes-Benz 2.5L streamlined racing car gleams under soft light, its sleek, aerodynamic body and wire-spoke wheels exuding 1950s motorsport elegance. Displayed at an outdoor vintage event, it’s surrounded by other historic racers and onlookers. The iconic three-pointed star adorns its curved hood. A plaque nearby confirms its identity, capturing automotive history in timeless, polished detail.
# """

prompt = "a dog on a chair"
# ======================
# 生成图片
# ======================

torch.cuda.synchronize()

starter = torch.cuda.Event(enable_timing=True)
ender   = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    starter.record()          # 一定在 with 内
    image = pipe(
        prompt,
        num_inference_steps=15,
        guidance_scale=4.5
    ).images[0]
    ender.record()            # 一定在 with 内

torch.cuda.synchronize()      # 必须有

elapsed_ms = starter.elapsed_time(ender)
print(f"Generation time: {elapsed_ms:.2f} ms")
save_path = "generated_image.png"
image.save(save_path)
print(f"图片已保存到: {save_path}")

# ======================
# 发送到远程 ECS（scp）
# ======================
# remote_user = "root"
# remote_ip = "123.56.83.186"
# remote_path = "/mnt/data/test"
# password = "258919@Lpz"

# scp_cmd = [
#     "sshpass", "-p", password,
#     "scp",
#     "-o", "StrictHostKeyChecking=no",
#     save_path,
#     f"{remote_user}@{remote_ip}:{remote_path}/"
# ]

# subprocess.run(scp_cmd, check=True)
# print("✅ 图片已成功发送到远程服务器")

