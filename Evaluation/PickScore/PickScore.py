import torch
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.io import read_image
from tqdm import tqdm

def gpu_batch_ssim(folder_base, folder_acc):
    # 1. 检查文件夹是否存在
    if not os.path.exists(folder_base) or not os.path.exists(folder_acc):
        print("文件夹路径不存在")
        return

    # 2. 在函数内部初始化 Metric，确保每次调用都是全新的状态
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    filenames = os.listdir(folder_acc)
    # 过滤非图片文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    filenames = [f for f in filenames if f.lower().endswith(image_extensions)]
    
    print(f"正在使用 {device} 计算 SSIM，共找到 {len(filenames)} 张图片...")
    
    
    total_ssim = 0.0
    count = 0
    
    with torch.no_grad():
        for fname in tqdm(filenames):
            path_base = os.path.join(folder_base, fname)
            path_acc = os.path.join(folder_acc, fname)
            
            # 确保基准文件夹里也有对应的图
            if not os.path.exists(path_base):
                continue
            
            try:
                # 读取并转为 float32 [0, 1]
                img_base = read_image(path_base).float() / 255.0
                img_acc = read_image(path_acc).float() / 255.0

                # 检查尺寸是否一致 (SDXL生成有时会有微小尺寸差异，会导致报错)
                if img_base.shape != img_acc.shape:
                    print(f"跳过 {fname}: 尺寸不匹配 {img_base.shape} vs {img_acc.shape}")
                    continue
                
                # 增加 Batch 维度 -> (1, C, H, W) 并移至 GPU
                img_base = img_base.unsqueeze(0).to(device)
                img_acc = img_acc.unsqueeze(0).to(device)
                
                # 计算当前 batch (单张) 的分数
                batch_score = ssim_metric(img_acc, img_base)
                
                total_ssim += batch_score.item()
                count += 1
                
            except Exception as e:
                print(f"处理 {fname} 时出错: {e}")
                continue

    final_avg = total_ssim / count if count > 0 else 0
    print(f"处理完成，有效图片数量: {count}")
    print(f"平均 SSIM: {final_avg:.4f}")
    
    return final_avg

if __name__ == "__main__":
    base_dir = "results/original_sdxl"
    acc_dir = "results/deepcache_sdxl"
    gpu_batch_ssim(base_dir, acc_dir)