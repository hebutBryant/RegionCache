import os
import torch
import ImageReward as RM
from PIL import Image
import pandas as pd
from tqdm import tqdm

output_csv = "ir_evaluation_results_optimized.csv"
use_fp16 = True  # 开关：是否开启半精度加速

data_list = [
    {"image_path": "dataset/cat.jpg", "prompt": "a painting of a cute cat", "id": "001"},
    {"image_path": "dataset/cyberpunk.png", "prompt": "futuristic city, cyberpunk style", "id": "002"}
]

def main():
    # 1. 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  运行设备: {device}")

    # 2. 加载模型
    print("⏳ 正在加载 ImageReward 模型...")
    try:
        # download_root 可以指定模型下载/缓存的路径，防止C盘爆满
        model = RM.load("ImageReward-v1.0", device=device)
        
        # 开启评估模式
        model.eval()
        
        if use_fp16 and device == "cuda":
            print("⚡ 已开启 FP16 半精度加速")
            model = model.half()
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    results = []
    
    print(f"▶️ 开始处理 {len(data_list)} 组数据...")
    
    for item in tqdm(data_list, desc="Scoring"):
        img_path = item["image_path"]
        prompt = item["prompt"]
        img_id = item.get("id", "N/A")
        
        record = {
            "id": img_id,
            "image_path": img_path,
            "prompt": prompt,
            "score": None,
            "status": "fail",
            "error_msg": ""
        }

        if not os.path.exists(img_path):
            record["error_msg"] = "File not found"
            results.append(record)
            continue

        try:
            # 加载并转换图片
            img = Image.open(img_path).convert("RGB")
            
            # 如果开启了半精度，使用 autocast 更加安全
            with torch.no_grad():
                # 注意：ImageReward 的 score 函数内部可能对输入有特定要求
                # 如果报错，可以移除上面的 .half()，仅保留 no_grad
                score = model.score(prompt, img)
            
            record["score"] = score
            record["status"] = "success"
            
        except Exception as e:
            record["error_msg"] = str(e)
            # 如果是因为显存溢出 (OOM)，尝试清理
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            
        results.append(record)

    # 3. 结果统计与保存
    if results:
        df = pd.DataFrame(results)
        valid_df = df[df["status"] == "success"]
        
        avg_score = valid_df["score"].mean() if not valid_df.empty else 0
        
        print(f"\n✅ 处理完成！")
        print(f"📊 成功: {len(valid_df)} / 总数: {len(results)}")
        print(f"📈 平均分: {avg_score:.4f}")
        
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"💾 结果保存至: {output_csv}")
    else:
        print("⚠️ 结果为空。")

if __name__ == "__main__":
    main()