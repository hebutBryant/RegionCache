import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = "/home/hdd/lpz/dataset/results_content_memory_part2"


def pixel_similarity(img1, img2, threshold=10):
    # img1, img2: uint8 numpy array, shape (H, W, 3)
    assert img1.shape == img2.shape
    
    diff = np.max(np.abs(img1.astype(np.int16) - img2.astype(np.int16)), axis=2)
    mask = diff > threshold
    diff_ratio = mask.mean()
    similarity = (1 - diff_ratio) * 100
    return similarity


img1_path = "/home/hdd/lpz/dataset/results_content_memory_part2/00120_00008_000082993/origin_0.png"
img2_path = "/home/hdd/lpz/dataset/results_content_memory_part2/00120_00008_000082993/result_0.png"

img1 = cv2.imread(img1_path)   # BGR, uint8
img2 = cv2.imread(img2_path)

assert img1 is not None and img2 is not None, "Image loading failed"

print(pixel_similarity(img1, img2))

import matplotlib.pyplot as plt
import numpy as np

# 原始数据（百分比）
imgEdit = [88.76, 87.10, 87.85, 86.02]
pico    = [71.92, 70.41, 71.08, 69.55]

# 归一化到 [0, 1]
imgEdit = np.array(imgEdit) / 100.0
pico    = np.array(pico) / 100.0

# 编辑轮数：从第 2 轮开始
turns = [2, 3, 4, 5]

plt.figure(figsize=(6, 4))
plt.plot(turns, imgEdit, marker='o', label='ImgEdit')
plt.plot(turns, pico, marker='o', label='PicoBanana')

plt.xlabel("Editing Turn")
plt.ylabel("Pixel Similarity")
plt.ylim(0.0, 1.0)
plt.xticks(turns)
plt.legend()
plt.grid(True)

plt.tight_layout()

# 保存为 PDF（论文直接可用）
plt.savefig("multi_turn_pixel_similarity.pdf", format="pdf", bbox_inches="tight")

plt.show()




# def load_json(path):
#     with open(path, "r") as f:
#         return json.load(f)

# def compute_redundancy_from_bbox(resolution, bbox):
#     H = resolution["height"]
#     W = resolution["width"]

#     img_area = H * W

#     x1, y1, x2, y2 = bbox
#     x1 = max(0, x1)
#     y1 = max(0, y1)
#     x2 = min(W, x2)
#     y2 = min(H, y2)

#     edit_area = max(0, x2 - x1) * max(0, y2 - y1)

#     redundancy = 1.0 - (edit_area / img_area)
#     return redundancy

# # ================= 主流程 =================

# round_redundancy = defaultdict(list)

# for sample in os.listdir(ROOT):
#     sample_dir = os.path.join(ROOT, sample)
#     if not os.path.isdir(sample_dir):
#         continue

#     json_path = os.path.join(sample_dir, "result.json")
#     if not os.path.exists(json_path):
#         continue

#     data = load_json(json_path)

#     resolution = data["resolution"]

#     # -------- Round 1 --------
#     if "edit_obj1" in data:
#         bbox1 = data["edit_obj1"]["bbox"]
#         r1 = compute_redundancy_from_bbox(resolution, bbox1)
#         round_redundancy[1].append(r1)

#     # -------- Round 2 --------
#     if "edit_obj2" in data:
#         bbox2 = data["edit_obj2"]["bbox"]
#         r2 = compute_redundancy_from_bbox(resolution, bbox2)
#         round_redundancy[2].append(r2)

# # ================= 画图 =================

# rounds = sorted(round_redundancy.keys())
# avg_redundancy = [np.mean(round_redundancy[r]) for r in rounds]

# plt.figure(figsize=(6, 4))
# plt.plot(rounds, avg_redundancy, marker="o")
# plt.xlabel("Edit Round")
# plt.ylabel("Average Redundancy (Non-edit Area Ratio)")
# plt.title("Average Redundancy per Editing Round (BBox-based)")
# plt.grid(True)
# plt.ylim(0, 1)
# plt.yticks(np.linspace(0, 1, 6))
# plt.tight_layout()
# plt.savefig("average_redundancy_per_round.pdf")
# plt.close()