import torch
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.io import read_image
from tqdm import tqdm

def gpu_batch_ssim(folder_base, folder_acc):
    # 1. 确保 Metric 初始状态为空
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    ssim_metric.reset()
    
    filenames = os.listdir(folder_acc)
    count = 0
    
    print("正在使用 GPU 计算 SSIM...")
    
    with torch.no_grad():
        for fname in tqdm(filenames):
            path_base = os.path.join(folder_base, fname)
            path_acc = os.path.join(folder_acc, fname)
            
            if not os.path.exists(path_base):
                continue
            
            # 2. 读取图片并归一化
            try:
                img_base = read_image(path_base).float() / 255.0
                img_acc = read_image(path_acc).float() / 255.0
            except Exception as e:
                print(f"读取图片出错 {fname}: {e}")
                continue

            # 3. 强制只取前3个通道 (C, H, W) -> 兼容 RGBA 和 RGB
            if img_base.shape[0] == 4: img_base = img_base[:3, :, :]
            if img_acc.shape[0] == 4: img_acc = img_acc[:3, :, :]

            # 简单的形状检查
            if img_base.shape != img_acc.shape:
                print(f"跳过尺寸不匹配图片: {fname} {img_base.shape} vs {img_acc.shape}")
                continue
            
            # 增加 Batch 维度 -> (1, C, H, W) 并移至 GPU
            img_base = img_base.unsqueeze(0).cuda()
            img_acc = img_acc.unsqueeze(0).cuda()
            
            # 4. update() 会更新内部状态，同时也返回当前的 batch 分数
            ssim_metric.update(img_acc, img_base)
            count += 1
            
    # 5. 最后一次性计算平均值
    if count > 0:
        final_avg = ssim_metric.compute().item()
        print(f"处理图片数量: {count}")
        print(f"GPU 平均 SSIM: {final_avg:.4f}")
    else:
        print("未找到匹配图片或目录为空。")

if __name__ == "__main__":
    base_dir = "results/original_sdxl"
    acc_dir = "results/deepcache_sdxl"
    gpu_batch_ssim(base_dir, acc_dir)