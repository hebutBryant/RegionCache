import os
import torch
from diffusers import PixArtAlphaPipeline
from PIL import Image
import torchvision.transforms as T

# 选择 GPU 或 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# =======================
# 1. 加载 PixArt 模型
# =======================
pipe = PixArtAlphaPipeline.from_pretrained(
    "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS",
    torch_dtype=torch.float16,
).to(device)
pipe.set_progress_bar_config(disable=True)

vae = pipe.vae
scheduler = pipe.scheduler

# =======================
# 2. 配置：输入图像 & 时间步 & 输出目录
# =======================
input_image_path = "/home/lipz/RegionCache/Material_Library/Constructer/pixart_output.png"  # 你要处理的原始完整图片
output_dir = "noised_steps_partial"           # 保存加噪结果的文件夹
os.makedirs(output_dir, exist_ok=True)

# 时间步列表（示例）
num_train_timesteps = scheduler.config.num_train_timesteps
print(f"Scheduler num_train_timesteps = {num_train_timesteps}")

timesteps = [
    0,
    int(num_train_timesteps * 0.25),
    int(num_train_timesteps * 0.5),
    int(num_train_timesteps * 0.75),
    num_train_timesteps - 1,
]

print("将对以下 timesteps 加噪(局部):", timesteps)

# =======================
# 3. 读取并预处理输入图像 → latent x0
# =======================
image = Image.open(input_image_path).convert("RGB")

# PixArt-XL-2-1024-MS 是 1024 分辨率，你可以按需改
image = image.resize((1024, 1024), Image.LANCZOS)

to_tensor = T.ToTensor()
image_tensor = to_tensor(image).unsqueeze(0).to(device)  # [1, 3, H, W], 0..1
image_tensor = 2.0 * image_tensor - 1.0                  # 映射到 [-1, 1]

with torch.no_grad():
    # VAE 编码到 latent 空间
    latents_x0 = vae.encode(image_tensor.to(dtype=torch.float16)).latent_dist.sample()
    # 按 LDM 约定放大
    latents_x0 = latents_x0 * vae.config.scaling_factor   # [1, C, H_lat, W_lat]

# =======================
# 3.5 在 latent 空间中定义一个“加噪区域”
# =======================
_, _, H_lat, W_lat = latents_x0.shape
print(f"latent shape: H={H_lat}, W={W_lat}")

# 示例：定义 latent 中央区域为加噪区域（你可以自行修改）
h_start = H_lat // 4
h_end   = H_lat * 3 // 4
w_start = W_lat // 4
w_end   = W_lat * 3 // 4

# 创建 mask: 1 表示“要加噪的区域”，0 表示保持原样
mask = torch.zeros_like(latents_x0, dtype=torch.float16, device=device)
mask[:, :, h_start:h_end, w_start:w_end] = 1.0  # [1, C, H_lat, W_lat]

# 如果你想要 soft 边缘，可以在这里对 mask 做模糊，这里先用硬的 0/1 mask

# =======================
# 4. 对每个时间步 t：只对指定区域加噪 → 解码 → 保存
# =======================
for t in timesteps:
    t_tensor = torch.tensor([t], device=device, dtype=torch.long)

    # 生成高斯噪声（可以固定 seed 保证可复现）
    noise = torch.randn_like(latents_x0)

    with torch.no_grad():
        # 整图的“带噪版本”
        latents_noised_full = scheduler.add_noise(latents_x0, noise, t_tensor)

        # 只在 mask=1 的区域使用带噪版本，其余区域保持原 x0
        latents_xt = latents_x0 * (1.0 - mask) + latents_noised_full * mask

        # 解码回图像空间：记得除回 scaling_factor
        decoded = vae.decode(
            (latents_xt / vae.config.scaling_factor).to(dtype=torch.float16)
        ).sample

    # 映射回 [0, 1]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.detach().cpu()

    # 转成 PIL
    img_np = decoded[0].permute(1, 2, 0).numpy()  # [H, W, C]
    img_uint8 = (img_np * 255).round().astype("uint8")
    img_pil = Image.fromarray(img_uint8)

    # 保存
    save_name = f"partial_noised_t{t}.png"
    save_path = os.path.join(output_dir, save_name)
    img_pil.save(save_path)
    print(f"保存: {save_path}")

print("所有时间步【局部加噪】图像已保存完毕。")
