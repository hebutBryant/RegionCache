import matplotlib.pyplot as plt
import numpy as np

datasets = [
    "PICO-Banana",
    "ImgEdit",
    "MagicBrush",
    "SEED-Data",
    "AnyEdit"
]

# 示例数值（0–1 区间）
similarity = [0.856, 0.894, 0.8105, 0.927, 0.8451]

x = np.arange(len(datasets))

plt.figure(figsize=(5.4, 3.6))  # 论文单栏友好

bars = plt.bar(
    x,
    similarity,
    width=0.42,
    color=["C0", "C0", "C0", "C0", "C0", "C0"]
)

# y 轴从 0 开始
plt.ylim(0.0, 1.01)

plt.ylabel("Similarity Score")
plt.title("Conceptual Similarity of Multi-turn Image Editing Datasets")

plt.xticks(x, datasets, rotation=20)

# ✅ 在柱子顶部标注“百分数”
for bar, val in zip(bars, similarity):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.015,
        f"{val * 100:.2f}%",   # ← 关键修改：百分数
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("dataset_similarity.pdf", bbox_inches="tight")
plt.close()
