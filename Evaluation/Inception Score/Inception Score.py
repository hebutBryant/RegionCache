import torch_fidelity

def calc_is_using_library(path_to_images):
    print(f"正在计算 {path_to_images} 的 Inception Score...")
    
    # input1 是图片路径
    # cuda=True 使用 GPU 加速
    # isc=True 表示计算 Inception Score
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=path_to_images, 
        cuda=True, 
        isc=True, 
        fid=False, 
        verbose=False
    )
    
    return metrics_dict['inception_score_mean'], metrics_dict['inception_score_std']

# 使用示例
if __name__ == "__main__":
    # 确保文件夹里有足够多的图片（通常建议 > 10,000 张以获得稳定结果）
    mean, std = calc_is_using_library("./generated_images")
    print(f"Inception Score: {mean:.4f} ± {std:.4f}")