import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 0) 输出目录
# =====================================================
output_dir = "figs_final_with_values"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# 1) 数据（ms）
# =====================================================
steps = [15, 20, 25]
resolutions = ["512x512", "1024x1024", "2048x2048"]

# RegionCache
# RegionCache (self-attention / total)
rc_attn = {
    "512x512":   [110.14, 153.20, 183.27],
    "1024x1024": [489.46, 657.89, 794.17],
    "2048x2048": [1237.49, 1542.63, 1853.23],
}

rc_total = {
    "512x512":   [458.85, 647.57, 761.80],
    "1024x1024": [1869.49, 2339.68, 2778.37],
    "2048x2048": [4204.18, 5283.54, 6358.61],
}

# PixArt baseline (self-attention / total)
pixart_attn = {
    "512x512":   [308.71, 412.04, 515.49],
    "1024x1024": [2276.01, 3033.77, 3795.39],
    "2048x2048": [7388.95, 9855.77, 12319.53],
}

pixart_total = {
    "512x512":   [657.42, 844.46, 1031.79],
    "1024x1024": [3439.09, 4498.61, 5563.30],
    "2048x2048": [9988.90, 13095.86, 16198.84],
}

# =====================================================
# 工具函数：给柱子加数值
# =====================================================
def add_bar_labels(bars, fmt="{:.2f}", offset=0.01):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset * height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9
        )

# =====================================================
# Fig.1 Attention ratio vs steps (结论图)
# =====================================================
plt.figure(figsize=(6.5, 4))
x = np.arange(len(steps))
bar_width = 0.22

for i, r in enumerate(resolutions):
    ratios = np.array(pixart_attn[r]) / np.array(pixart_total[r]) * 100
    bars = plt.bar(x + i * bar_width, ratios, bar_width, label=r)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(x + bar_width, steps)
plt.xlabel("Inference steps")
plt.ylabel("Self-Attention ratio (%)")

# 80%+，给足空间
plt.ylim(0, 90)

plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig1_pixart_attn_ratio_vs_steps.pdf"))
plt.close()
# =====================================================
# Fig.2–4 Attention time comparison (PixArt vs RC)
# =====================================================
def plot_attn_compare(resolution):
    plt.figure(figsize=(6.5, 4))
    methods = ["PixArt", "RegionCache"]
    x = np.arange(len(methods))
    bar_width = 0.22
    offsets = [-bar_width, 0, bar_width]

    pix_vals = np.array(pixart_attn[resolution]) / 1000.0
    rc_vals = np.array(rc_attn[resolution]) / 1000.0

    for i, step in enumerate(steps):
        values = [pix_vals[i], rc_vals[i]]
        bars = plt.bar(
            x + offsets[i],
            values,
            bar_width,
            label=f"{step} steps"
        )
        add_bar_labels(bars, fmt="{:.2f}")

    plt.xticks(x, methods)
    plt.xlabel("Method")
    plt.ylabel("Self-Attention Time (s)")
    plt.title(f"Self-Attention Time Comparison ({resolution})")
    plt.legend(title="Inference steps")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"fig_attn_compare_{resolution}.pdf")
    )
    plt.close()

# plot_attn_compare("512x512")
# plot_attn_compare("1024x1024")
# plot_attn_compare("2048x2048")




def plot_attn_compare_by_steps(
    resolution,
    pixart_attn,
    rc_attn,
    steps,
    output_dir="figs_final",
):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6.5, 4))

    x = np.arange(len(steps))
    bar_width = 0.32

    # 转成秒
    pix_vals = np.array(pixart_attn[resolution]) / 1000.0
    rc_vals = np.array(rc_attn[resolution]) / 1000.0

    # PixArt（左） —— C0
    bars_pix = plt.bar(
        x - bar_width / 2,
        pix_vals,
        bar_width,
        color="C0",
        edgecolor="black",
        linewidth=0.8,
        label="PixArt"
    )

    # RegionCache（右） —— C1
    bars_rc = plt.bar(
        x + bar_width / 2,
        rc_vals,
        bar_width,
        color="C1",
        edgecolor="black",
        linewidth=0.8,
        label="RegionCache"
    )

    # ===== 数值标注（含加速比）=====
    for i in range(len(steps)):
        # PixArt：只标时间
        h_pix = pix_vals[i]
        plt.text(
            bars_pix[i].get_x() + bar_width / 2,
            h_pix,
            f"{h_pix:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

        # RegionCache：时间 + speedup
        h_rc = rc_vals[i]
        speedup = h_pix / h_rc

        plt.text(
            bars_rc[i].get_x() + bar_width / 2,
            h_rc,
            f"{h_rc:.2f}\n({speedup:.2f}×)",
            ha="center",
            va="bottom",
            fontsize=9,
            linespacing=1.1
        )

    plt.xticks(x, [f"{s} steps" for s in steps])
    plt.xlabel("Inference steps")
    plt.ylabel("Self-Attention Time (s)")
    plt.title(f"Self-Attention Time Comparison ({resolution})")
    plt.legend()

    # 给顶部留空间，防止顶头
    plt.ylim(0, max(pix_vals.max(), rc_vals.max()) * 1.20)

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        f"fig_attn_compare_by_steps_{resolution}_with_speedup.pdf"
    )
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {save_path}")



steps = [15, 20, 25]

for resolution in ["512x512", "1024x1024", "2048x2048"]:
    plot_attn_compare_by_steps(
        resolution=resolution,
        pixart_attn=pixart_attn,
        rc_attn=rc_attn,
        steps=steps,
        output_dir="figs_final_paper",
    )



# output_dir = "figs_final_paper"
# os.makedirs(output_dir, exist_ok=True)

# steps = [15, 20, 25]
# resolutions = ["512x512", "1024x1024", "2048x2048"]

# # 计算 speedup
# speedup = {s: [] for s in steps}

# for i, s in enumerate(steps):
#     for r in resolutions:
#         sp = pixart_attn[r][i] / rc_attn[r][i]
#         speedup[s].append(sp)

# # =============================
# # Plot
# # =============================
# plt.figure(figsize=(6.5, 4))

# x = np.arange(len(resolutions))

# for s in steps:
#     plt.plot(
#         x,
#         speedup[s],
#         marker="o",
#         linewidth=2,
#         label=f"{s} steps"
#     )
#     for xi, yi in zip(x, speedup[s]):
#         plt.text(
#             xi,
#             yi,
#             f"{yi:.1f}×",
#             ha="center",
#             va="bottom",
#             fontsize=9
#         )

# plt.xticks(x, resolutions)
# plt.xlabel("Image Resolution")
# plt.ylabel("Self-Attention Speedup (×)")
# plt.title("Speedup vs Image Resolution")
# plt.legend(title="Inference steps")

# plt.ylim(0, max(max(v) for v in speedup.values()) * 1.15)
# plt.tight_layout()

# save_path = os.path.join(output_dir, "fig_speedup_vs_resolution.pdf")
# plt.savefig(save_path, format="pdf", bbox_inches="tight")
# plt.close()

# print(f"Saved figure to: {save_path}")



output_dir = "figs_final_paper"
os.makedirs(output_dir, exist_ok=True)

steps = [15, 20, 25]
resolutions = ["512x512", "1024x1024", "2048x2048"]

# 计算 speedup
speedup = {r: [] for r in resolutions}
for r in resolutions:
    for i in range(len(steps)):
        sp = pixart_attn[r][i] / rc_attn[r][i]
        speedup[r].append(sp)

# ============================
# Plot
# ============================
plt.figure(figsize=(6.5, 4))

x = np.arange(len(resolutions))
bar_width = 0.22
offsets = [-bar_width, 0, bar_width]

for i, step in enumerate(steps):
    values = [speedup[r][i] for r in resolutions]
    bars = plt.bar(
        x + offsets[i],
        values,
        bar_width,
        label=f"{step} steps"
    )
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}×",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(x, resolutions)
plt.xlabel("Image Resolution")
plt.ylabel("Self-Attention Speedup (×)")
plt.title("Self-Attention Speedup across Image Resolutions")
plt.legend(title="Inference steps")

plt.ylim(0, max(max(v) for v in speedup.values()) * 1.15)
plt.tight_layout()

save_path = os.path.join(output_dir, "fig_speedup_vs_resolution_bar.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.close()

print(f"Saved figure to: {save_path}")


steps = [15, 20, 25]
resolutions = ["512x512", "1024x1024", "2048x2048"]

# 计算 total speedup
speedup_total = {r: [] for r in resolutions}
for r in resolutions:
    for i in range(len(steps)):
        sp = pixart_total[r][i] / rc_total[r][i]
        speedup_total[r].append(sp)

# ============================
# Plot
# ============================
plt.figure(figsize=(6.5, 4))

x = np.arange(len(resolutions))
bar_width = 0.22
offsets = [-bar_width, 0, bar_width]

for i, step in enumerate(steps):
    values = [speedup_total[r][i] for r in resolutions]
    bars = plt.bar(
        x + offsets[i],
        values,
        bar_width,
        label=f"{step} steps"
    )
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.2f}×",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(x, resolutions)
plt.xlabel("Image Resolution")
plt.ylabel("Total Latency Speedup (×)")
plt.title("End-to-End Speedup across Image Resolutions")
plt.legend(title="Inference steps")

plt.ylim(0, max(max(v) for v in speedup_total.values()) * 1.15)
plt.tight_layout()

save_path = os.path.join(
    output_dir,
    "fig_total_speedup_vs_resolution_bar.pdf"
)
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.close()

print(f"Saved figure to: {save_path}")


def plot_total_time_with_speedup(
    resolution,
    pixart_total,
    rc_total,
    steps,
    output_dir="figs_final_paper",
):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6.5, 4))

    x = np.arange(len(steps))
    bar_width = 0.32

    # 转成秒
    pix_vals = np.array(pixart_total[resolution]) / 1000.0
    rc_vals = np.array(rc_total[resolution]) / 1000.0

    # PixArt —— C0
    bars_pix = plt.bar(
        x - bar_width / 2,
        pix_vals,
        bar_width,
        color="C0",
        edgecolor="black",
        linewidth=0.8,
        label="PixArt"
    )

    # RegionCache —— C1
    bars_rc = plt.bar(
        x + bar_width / 2,
        rc_vals,
        bar_width,
        color="C1",
        edgecolor="black",
        linewidth=0.8,
        label="RegionCache"
    )

    # ===== 数值标注 =====
    for i in range(len(steps)):
        # PixArt：只标时间
        h_pix = pix_vals[i]
        plt.text(
            bars_pix[i].get_x() + bar_width / 2,
            h_pix,
            f"{h_pix:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9
        )

        # RegionCache：时间 + speedup
        h_rc = rc_vals[i]
        speedup = pix_vals[i] / rc_vals[i]

        plt.text(
            bars_rc[i].get_x() + bar_width / 2,
            h_rc,
            f"{h_rc:.2f}s\n({speedup:.2f}×)",
            ha="center",
            va="bottom",
            fontsize=9,
            linespacing=1.1
        )

    plt.xticks(x, [f"{s} steps" for s in steps])
    plt.xlabel("Inference steps")
    plt.ylabel("Total Latency (s)")
    plt.title(f"End-to-End Latency Comparison ({resolution})")
    plt.legend()

    plt.ylim(0, max(pix_vals.max(), rc_vals.max()) * 1.20)
    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        f"fig_total_time_with_speedup_{resolution}.pdf"
    )
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {save_path}")


for resolution in ["512x512", "1024x1024", "2048x2048"]:
    plot_total_time_with_speedup(
        resolution=resolution,
        pixart_total=pixart_total,
        rc_total=rc_total,
        steps=steps,
        output_dir="figs_final_paper",
    )



def plot_attn_on_ax(
    ax,
    resolution,
    pixart_attn,
    rc_attn,
    steps,
):
    x = np.arange(len(steps))
    bar_width = 0.32

    pix_vals = np.array(pixart_attn[resolution]) / 1000.0
    rc_vals = np.array(rc_attn[resolution]) / 1000.0

    bars_pix = ax.bar(
        x - bar_width / 2,
        pix_vals,
        bar_width,
        color="C0",
        edgecolor="black",
        linewidth=0.8,
        label="PixArt"
    )

    bars_rc = ax.bar(
        x + bar_width / 2,
        rc_vals,
        bar_width,
        color="C1",
        edgecolor="black",
        linewidth=0.8,
        label="RegionCache"
    )

    for i in range(len(steps)):
        h_pix = pix_vals[i]
        h_rc = rc_vals[i]
        speedup = h_pix / h_rc

        ax.text(
            bars_pix[i].get_x() + bar_width / 2,
            h_pix,
            f"{h_pix:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

        ax.text(
            bars_rc[i].get_x() + bar_width / 2,
            h_rc,
            f"{h_rc:.2f}\n({speedup:.2f}×)",
            ha="center",
            va="bottom",
            fontsize=8,
            linespacing=1.0
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in steps])
    ax.set_ylim(0, max(pix_vals.max(), rc_vals.max()) * 1.25)
    ax.set_title(f"{resolution}", fontsize=10)


def plot_total_on_ax(
    ax,
    resolution,
    pixart_total,
    rc_total,
    steps,
):
    x = np.arange(len(steps))
    bar_width = 0.32

    pix_vals = np.array(pixart_total[resolution]) / 1000.0
    rc_vals = np.array(rc_total[resolution]) / 1000.0

    bars_pix = ax.bar(
        x - bar_width / 2,
        pix_vals,
        bar_width,
        color="C0",
        edgecolor="black",
        linewidth=0.8
    )

    bars_rc = ax.bar(
        x + bar_width / 2,
        rc_vals,
        bar_width,
        color="C1",
        edgecolor="black",
        linewidth=0.8
    )

    for i in range(len(steps)):
        h_pix = pix_vals[i]
        h_rc = rc_vals[i]
        speedup = h_pix / h_rc

        ax.text(
            bars_pix[i].get_x() + bar_width / 2,
            h_pix,
            f"{h_pix:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

        ax.text(
            bars_rc[i].get_x() + bar_width / 2,
            h_rc,
            f"{h_rc:.2f}\n({speedup:.2f}×)",
            ha="center",
            va="bottom",
            fontsize=8,
            linespacing=1.0
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in steps])
    ax.set_ylim(0, max(pix_vals.max(), rc_vals.max()) * 1.25)


fig, axes = plt.subplots(
    2, 3,
    figsize=(14, 6),
    sharex=False
)

resolutions = ["512x512", "1024x1024", "2048x2048"]
steps = [15, 20, 25]

# 第一行：Attention
for col, res in enumerate(resolutions):
    plot_attn_on_ax(
        ax=axes[0, col],
        resolution=res,
        pixart_attn=pixart_attn,
        rc_attn=rc_attn,
        steps=steps,
    )

# 第二行：Total
for col, res in enumerate(resolutions):
    plot_total_on_ax(
        ax=axes[1, col],
        resolution=res,
        pixart_total=pixart_total,
        rc_total=rc_total,
        steps=steps,
    )

# 行标签（左侧）
axes[0, 0].set_ylabel("Self-Attention Time (s)")
axes[1, 0].set_ylabel("Total Latency (s)")

# x 轴标签（底行）
for ax in axes[1, :]:
    ax.set_xlabel("Inference Steps")

# 全局 legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig(
    "figs_final_paper/fig_latency_attn_total_all_in_one.pdf",
    format="pdf",
    bbox_inches="tight"
)
plt.close()


steps = [15, 20, 25]
x = np.arange(len(steps))
bar_width = 0.22

attn_ratio = {
    "512x512":  [47.1, 47.8, 48.7],
    "1024x1024":[66.2, 67.5, 68.3],
    "2048x2048":[82.5, 83.4, 84.0],
}

# =============================
# 画图
# =============================
plt.figure(figsize=(6.5, 4))

bars_512 = plt.bar(
    x - bar_width,
    attn_ratio["512x512"],
    bar_width,
    color="C0",
    edgecolor="black",
    linewidth=0.8,
    label="512×512"
)

bars_1024 = plt.bar(
    x,
    attn_ratio["1024x1024"],
    bar_width,
    color="C1",
    edgecolor="black",
    linewidth=0.8,
    label="1024×1024"
)

bars_2048 = plt.bar(
    x + bar_width,
    attn_ratio["2048x2048"],
    bar_width,
    color="C2",
    edgecolor="black",
    linewidth=0.8,
    label="2048×2048"
)

# =============================
# 数值标注
# =============================
def add_labels(bars):
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

add_labels(bars_512)
add_labels(bars_1024)
add_labels(bars_2048)

# =============================
# 坐标 & 样式
# =============================
plt.xticks(x, steps)
plt.xlabel("Inference steps")
plt.ylabel("Self-Attention ratio (%)")
plt.ylim(0, 90)

plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False
)

plt.tight_layout()

# =============================
# 保存
# =============================
os.makedirs("Picture", exist_ok=True)
plt.savefig(
    "figs_final_paper/fig1_attn_ratio_vs_steps.pdf",
    format="pdf",
    bbox_inches="tight"
)
plt.close()