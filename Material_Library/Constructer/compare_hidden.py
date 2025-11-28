import numpy as np
import torch
import torch.nn.functional as F
import math
import json
import os
import matplotlib.pyplot as plt
from laststep_capture import *

import os
import math
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from diffusers import PixArtAlphaPipeline
# 这里假设 Attention, AttnProcessor2_0 等已经在前面 import 过


# =========================================================
# 工具 1：根据 prompt + region_id 抓取该区域的 hidden state
# =========================================================
def get_region_hidden_for_prompt(
    pipe,
    prompt: str,
    region_id: str = "a cat",              # 这里就是你说的 id，对应 noun_chunk 文本
    user_layer_name: str = "transformer_blocks.10.attn2",
    num_inference_steps: int = 20,
    guidance_scale: float = 4.0,
    seed: int = 1234,
    score_ratio: float = 0.2,
    min_patches: int = 64,
    target_tag_substr: str = ".attn1",
    device: str = "cuda:0",
):
    """
    对单个 prompt 进行一次推理，在指定层上：
      1) 记录最后一步 cross-attn
      2) 利用 mapping + cross-attn 得分，构造 region（比如 'a cat'）的 patch indices
      3) 返回该 region 对应的 hidden_state (K, C) 以及 pooled 向量 (C)

    要求：以下函数/对象已定义：
        - replace_attnprocessor2_0_with_attnprocessor
        - resolve_layer_name
        - enable_target_capture
        - install_laststep_flag_pre_hook
        - patch_attention_get_scores
        - map_chunks_to_token_indices
        - construct_region_item
        - 全局的 nlp（spacy）已初始化
    """

    # ========= 0. 替换 AttnProcessor，开启 hidden_sink =========
    restore, hidden_sink = replace_attnprocessor2_0_with_attnprocessor(
        pipe,
        move_to_cpu=True,
    )

    # ========= 1. 定位 cross-attn 层并打补丁 =========
    real_name = resolve_layer_name(pipe, user_layer_name)
    print(f"[get_region_hidden_for_prompt] target layer = {real_name}")

    target_module, attn_sink = enable_target_capture(pipe, real_name)
    hook = install_laststep_flag_pre_hook(pipe, target_module)
    patch_attention_get_scores(pipe, target_layer=real_name)

    # ========= 2. 正常采样 =========
    gen = torch.Generator(device=device).manual_seed(seed)
    _ = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=gen,
    )

    # ========= 3. 拿到 cross-attn 的最后一步 =========
    assert attn_sink, "❌ 没有捕获到 cross-attn，检查层名 / 补丁逻辑"
    last = attn_sink[-1]
    attn = last["attn_probs"]          # [B*H, HW, T_text]
    print("[get_region_hidden_for_prompt] raw attn shape:", attn.shape)

    print("[get_region_hidden_for_prompt] hidden entries:", len(hidden_sink))

    # ========= 4. 文本解析 + chunk→token mapping =========
    global nlp  # 假设你在外面已经 nlp = spacy.load(...)
    doc = nlp(prompt)

    tokenizer = getattr(pipe, "tokenizer", None) or getattr(pipe, "tokenizer_2", None)
    if tokenizer is None:
        raise RuntimeError("❌ 没有在 pipeline 中找到 tokenizer。")

    chunks = [chunk.text for chunk in doc.noun_chunks]
    mapping = map_chunks_to_token_indices(prompt=prompt, chunks=chunks, tokenizer=tokenizer)
    print("[get_region_hidden_for_prompt] mapping:", mapping)

    if region_id not in mapping:
        raise ValueError(f"❌ region_id={region_id!r} 不在 mapping 里，当前 chunks={list(mapping.keys())}")

    # ========= 5. 计算 head-averaged cross-attn: [L_img, L_text] =========
    attn_mean = attn.mean(dim=0)   # [HW, T_text]

    # ========= 6. 构造 region 的 hidden + indices =========
    chunk_hidden, chunk_patch_indices = construct_region_item(
        hidden_sink=hidden_sink,
        attn_scores=attn_mean,
        mapping=mapping,
        score_ratio=score_ratio,
        min_patches=min_patches,
        target_tag_substr=target_tag_substr,
    )

    if region_id not in chunk_hidden:
        raise ValueError(
            f"❌ region_id={region_id!r} 没有在 chunk_hidden 里，"
            f"当前可用={list(chunk_hidden.keys())}，请检查 mapping / score_ratio"
        )

    region_hidden = chunk_hidden[region_id]         # [K, C] on CPU
    region_indices = chunk_patch_indices[region_id] # [K]

    # ========= 7. pooled feature（区域语义向量） =========
    pooled_feat = region_hidden.float().mean(dim=0) # [C]

    # ========= 8. 清理 / 恢复 =========
    hook.remove()
    restore()   # 恢复原始的 AttentionProcessor

    extras = {
        "mapping": mapping,
        "attn_mean": attn_mean,      # [L_img, L_text]（在 GPU 上）
        "hidden_sink": hidden_sink,  # 如果后面还要分析轨迹/别的区域，可以用
        "real_layer_name": real_name,
    }

    print(
        f"[get_region_hidden_for_prompt] region_id={region_id!r}, "
        f"K={region_hidden.shape[0]}, C={region_hidden.shape[1]}"
    )
    return region_hidden, region_indices, pooled_feat, extras


# =========================================================
# 工具 2：两个 region hidden 的相似度指标
# =========================================================
def region_mean_cosine(h1: torch.Tensor, h2: torch.Tensor) -> float:
    """
    h1: [K1, C]
    h2: [K2, C]
    对每个 region 沿 patch 维度做 mean pooling，得到 [C]，然后算 cosine。
    用来衡量“整体语义”是否一致。
    """
    f1 = h1.mean(dim=0)   # [C]
    f2 = h2.mean(dim=0)   # [C]
    return F.cosine_similarity(f1, f2, dim=0).item()


def chamfer_like_cosine(h1: torch.Tensor, h2: torch.Tensor) -> float:
    """
    用 cosine 做一个 Chamfer-style patch set 相似度：
      - 对 h1 中每个 patch，在 h2 中找最相似的 patch（cos 最大），取平均
      - 对 h2 中每个 patch，在 h1 中找最相似的 patch，取平均
      - 返回两者的平均值，范围大致在 [-1, 1]

    反映的是两个 patch 分布在特征空间中的“互相覆盖程度”。
    """
    h1_n = F.normalize(h1.float(), dim=1)   # [K1, C]
    h2_n = F.normalize(h2.float(), dim=1)   # [K2, C]

    # cos 矩阵: [K1, K2]
    cos_mat = torch.matmul(h1_n, h2_n.T)

    # 每个 h1 patch 在 h2 中的最大相似度
    sim1 = cos_mat.max(dim=1).values.mean().item()
    # 每个 h2 patch 在 h1 中的最大相似度
    sim2 = cos_mat.max(dim=0).values.mean().item()

    return 0.5 * (sim1 + sim2)


# =========================================================
# 工具 3：对比两个 prompt 的同一 region 语义相似度
# =========================================================
def compare_region_semantic_similarity(
    pipe,
    prompt1: str,
    prompt2: str,
    region_id: str = "a cat",
    user_layer_name: str = "transformer_blocks.10.attn2",
    num_inference_steps: int = 20,
    guidance_scale: float = 4.0,
    seed: int = 1234,
    device: str = "cuda:0",
):
    """
    对两个 prompt：
      - 分别在推理过程中抽取同一个 region_id 的 hidden state（K×C）
      - 计算：
          1) mean pooling 后的 cosine 相似度（整体语义）
          2) Chamfer-like patch-set 相似度（patch 分布级）

    返回一个 dict，包含所有中间结果（方便后续画图/做 t-SNE）。
    """

    print("\n========== [Prompt 1] ==========")
    h1, idx1, f1, extra1 = get_region_hidden_for_prompt(
        pipe=pipe,
        prompt=prompt1,
        region_id=region_id,
        user_layer_name=user_layer_name,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        device=device,
    )

    print("\n========== [Prompt 2] ==========")
    h2, idx2, f2, extra2 = get_region_hidden_for_prompt(
        pipe=pipe,
        prompt=prompt2,
        region_id=region_id,
        user_layer_name=user_layer_name,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        device=device,
    )

    # -------- 1) mean pooling 语义相似度 --------
    sim_mean = region_mean_cosine(h1, h2)

    # -------- 2) Chamfer-like patch 集合相似度 --------
    sim_chamfer = chamfer_like_cosine(h1, h2)

    print("\n=========== 对比结果 ===========")
    print(f"Prompt 1: {prompt1!r}")
    print(f"Prompt 2: {prompt2!r}")
    print(f"Region  : {region_id!r}")
    print(f"Mean-Pooled Cosine Similarity   : {sim_mean:.4f}")
    print(f"Chamfer-like Patch-Set Similarity: {sim_chamfer:.4f}")
    print("================================")

    result = {
        "prompt1": prompt1,
        "prompt2": prompt2,
        "region_id": region_id,
        "user_layer_name": user_layer_name,
        "cosine_mean": sim_mean,
        "cosine_chamfer": sim_chamfer,
        "h1": h1,         # [K1, C]
        "h2": h2,         # [K2, C]
        "idx1": idx1,     # [K1]
        "idx2": idx2,     # [K2]
        "extra1": extra1,
        "extra2": extra2,
    }
    return result

def extract_layer_hidden_trajectory_from_extras(
    extra: dict,
    target_tag_substr: str = "transformer_blocks.10.attn1",
):
    """
    从 compare_region_semantic_similarity 返回的 extra 里，
    提取某个 self-attn 层在整个推理过程中的 hidden 轨迹。

    extra 里包含:
        - extra["hidden_sink"]: AttnProcessorMe 记录的所有条目
          每条大概是 {"tag": str, "hidden": Tensor[L, C], ...}

    返回:
        hs_list: list[Tensor[L, C]]，按调用顺序排列（近似对应 step 顺序）
    """
    hidden_sink = extra["hidden_sink"]
    records = [
        item for item in hidden_sink
        if target_tag_substr in str(item.get("tag", ""))
    ]
    if not records:
        raise ValueError(
            f"在 hidden_sink 中找不到 tag 含 '{target_tag_substr}' 的记录。\n"
            f"可用 tag 示例: {[item.get('tag','') for item in hidden_sink[:10]]}"
        )

    hs_list = []
    for item in records:
        hs = item["hidden"]
        if not isinstance(hs, torch.Tensor):
            hs = torch.tensor(hs)
        hs_list.append(hs)  # [L, C]

    print(
        f"[extract_layer_hidden_trajectory_from_extras] layer='{target_tag_substr}', "
        f"steps={len(hs_list)}, shape={tuple(hs_list[0].shape)}"
    )
    return hs_list

def plot_region_trajectory_similarity_two_prompts(
    res: dict,
    self_attn_layer_substr: str = "transformer_blocks.10.attn1",
    save_path: str = "./cache/cat_region_trajectory_sim.png",
    smooth: bool = False,   # 如果你希望折线更平滑，可设 True
):
    """
    绘制跨 prompt 同一语义区域的扩散轨迹相似度曲线。
    Y 轴固定为 [0,1]，越接近1表示语义越一致。
    """

    idx1 = res["idx1"]
    idx2 = res["idx2"]
    extra1 = res["extra1"]
    extra2 = res["extra2"]

    # ---- 1) 提取 self-attn 隐空间轨迹 ----
    hs_list1 = extract_layer_hidden_trajectory_from_extras(
        extra1, target_tag_substr=self_attn_layer_substr
    )
    hs_list2 = extract_layer_hidden_trajectory_from_extras(
        extra2, target_tag_substr=self_attn_layer_substr
    )

    T = min(len(hs_list1), len(hs_list2))
    print(f"[轨迹对齐] hs1={len(hs_list1)} hs2={len(hs_list2)} → 使用 T={T}")

    sims = []
    for t in range(T):
        hs1 = hs_list1[t]  # [L, C]
        hs2 = hs_list2[t]

        f1_t = hs1[idx1].mean(dim=0)
        f2_t = hs2[idx2].mean(dim=0)

        sim = F.cosine_similarity(f1_t.float(), f2_t.float(), dim=0).item()
        sims.append(sim)

    # ---- 2) 可选 smoothing（指数加权移动平均）----
    if smooth:
        smoothed = []
        alpha = 0.4  # 越大越贴近真实值，越小越平滑
        for i, v in enumerate(sims):
            if i == 0:
                smoothed.append(v)
            else:
                smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
        sims_to_plot = smoothed
    else:
        sims_to_plot = sims

    # ---- 3) 画图 ----
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    steps = list(range(T))
    plt.figure(figsize=(6, 4))
    plt.plot(steps, sims_to_plot, marker="o", linewidth=2)

    # 🔥 你要求的固定 Y 轴范围
    plt.ylim(0, 1)

    plt.xticks(steps)  # 让 X 轴显示整步
    plt.xlabel("Diffusion Step")
    plt.ylabel("Cosine Similarity of Region Embeddings")
    plt.title(
        f"Semantic Trajectory Alignment of '{res['region_id']}'\n"
        f"P1: {res['prompt1']}\nP2: {res['prompt2']}"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ 跨 prompt 区域轨迹相似折线已保存 → {save_path}")
    return {"steps": steps, "sims": sims}


def plot_region_two_prompt_raw_trajectory(
    res: dict,
    self_attn_layer_substr: str = "transformer_blocks.10.attn1",
    save_path: str = "./cache/cat_region_raw_dynamics.png",
    smooth: bool = False,
    scaling: str = "norm",  # 可选: "norm", "pca"
):
    """
    绘制两个 prompt 语义区域随 diffusion step 演变的轨迹（而不是相似度）。

    - 蓝色：prompt1 的 "a cat"
    - 红色：prompt2 的 "a cat"

    可选 embedding scalar 化方式：
        "norm" → 直接用 ||f_t|| 表示语义能量
        "pca"  → 使用 PCA 将轨迹 embed 到一维（更反映方向性）

    目的：
        看两条曲线是否形状一致（即轨迹有相同规律）。
    """

    from sklearn.decomposition import PCA

    idx1 = res["idx1"]
    idx2 = res["idx2"]
    extra1 = res["extra1"]
    extra2 = res["extra2"]

    # ---- 取轨迹 ----
    hs_list1 = extract_layer_hidden_trajectory_from_extras(extra1, self_attn_layer_substr)
    hs_list2 = extract_layer_hidden_trajectory_from_extras(extra2, self_attn_layer_substr)

    T = min(len(hs_list1), len(hs_list2))

    f1_list, f2_list = [], []

    for t in range(T):
        h1 = hs_list1[t][idx1].mean(dim=0)  # [C]
        h2 = hs_list2[t][idx2].mean(dim=0)
        f1_list.append(h1.float().cpu().numpy())
        f2_list.append(h2.float().cpu().numpy())

    f1 = np.stack(f1_list)  # [T, C]
    f2 = np.stack(f2_list)  # [T, C]

    # ---- 标量化 ----
    if scaling == "norm":
        y1 = np.linalg.norm(f1, axis=1)
        y2 = np.linalg.norm(f2, axis=1)

    elif scaling == "pca":
        concat = np.concatenate([f1, f2], axis=0)
        pca = PCA(n_components=1)
        y = pca.fit_transform(concat).flatten()
        y1, y2 = y[:T], y[T:]
    else:
        raise ValueError("scaling must be 'norm' or 'pca'")

    # ---- smoothing ----
    if smooth:
        def smooth_curve(v):
            out = []
            alpha = 0.4
            for i, val in enumerate(v):
                out.append(val if i == 0 else alpha * val + (1 - alpha) * out[-1])
            return out
        y1, y2 = smooth_curve(y1), smooth_curve(y2)

    steps = list(range(T))

    # ---- plot ----
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(steps, y1, "-o", label=f"Prompt1: {res['prompt1']}", linewidth=2)
    plt.plot(steps, y2, "-o", label=f"Prompt2: {res['prompt2']}", linewidth=2)
    
    plt.xlabel("Diffusion step")
    plt.ylabel("Semantic Feature Magnitude" if scaling=="norm" else "PCA trajectory value")
    plt.title(f"Semantic Evolution of '{res['region_id']}' Across Prompts")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📄 已保存轨迹对比图 → {save_path}")

    return {"steps": steps, "y1": y1, "y2": y2}






def compute_chunk_influence_from_cross_attn(attn_probs, mapping, chunk_name: str):
    """
    从 cross-attn 概率中，计算某个文本 chunk（比如 'a cat'）
    对所有图像 patch 的影响向量 I（长度 HW）。

    参数：
        attn_probs: Tensor, 形状 [heads*batch, HW, T_text] 或 [H, HW, T_text]
        mapping:    文本 chunk 到 token id 的映射 dict，比如 {"a cat": [3,4], ...}
        chunk_name: 要分析的 chunk 名，比如 "a cat"

    返回：
        influence: Tensor[HW]，第 j 个元素是 patch j 对该 chunk 的注意力总和。
    """
    # 1) 取平均，去掉 heads 维度
    if attn_probs.dim() == 3:
        attn_mean = attn_probs.mean(dim=0)  # [HW, T_text]
    else:
        raise ValueError(f"attn_probs 期望是 3 维 [H, HW, T_text]，实际是 {attn_probs.shape}")

    HW, T_text = attn_mean.shape

    if chunk_name not in mapping:
        raise ValueError(f"chunk '{chunk_name}' 不在 mapping 里，现有 keys={list(mapping.keys())}")

    token_ids = mapping[chunk_name]
    # 过滤非法 token 索引
    token_ids = [i for i in token_ids if 0 <= i < T_text]
    if not token_ids:
        raise ValueError(f"chunk '{chunk_name}' 没有有效 token id，原始 token_ids={mapping[chunk_name]}")

    token_idx = torch.tensor(token_ids, dtype=torch.long, device=attn_mean.device)

    # 2) I_j = sum_{i in tokens(chunk)} A_{j,i}
    influence = attn_mean[:, token_idx].sum(dim=1)  # [HW]

    return influence


def plot_influence_compare_two_prompts(
    influence_A: torch.Tensor,
    influence_B: torch.Tensor,
    H: int,
    W: int,
    prompt_A: str,
    prompt_B: str,
    chunk_name: str,
    save_path: str = "./cache/cat_chunk_influence_compare.png",
):
    """
    可视化两个 prompt 下，同一个文本 chunk（如 'a cat'）
    对所有 patch 的影响分布对比。

    influence_A / B: [HW]
    H, W:            patch 网格尺寸（比如 64×64）
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    vA = influence_A.view(H, W).detach().cpu().numpy()
    vB = influence_B.view(H, W).detach().cpu().numpy()

    # 简单归一化到 [0,1]，便于视觉对比
    def normalize(x):
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        return x

    vA_n = normalize(vA)
    vB_n = normalize(vB)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title(f"Influence of '{chunk_name}'\n{prompt_A}")
    plt.imshow(vA_n, cmap="hot")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.title(f"Influence of '{chunk_name}'\n{prompt_B}")
    plt.imshow(vB_n, cmap="hot")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 已保存 cat 影响对比图: {save_path}")


def run_prompt_and_get_cross_attn_and_mapping(
    pipe,
    prompt: str,
    user_layer_name: str = "transformer_blocks.10.attn2",
    device: str = "cuda:1",
):
    """
    跑一次 PixArt 推理，返回：
      - cross-attn 最后一步 attn_probs（指定层，比如 blocks.10.attn2）
      - hidden_sink（如果你后面还想用）
      - noun chunk -> token_ids 的 mapping
    """
    # 1) 替换 AttnProcessor2_0，记录 hidden_sink
    restore, hidden_sink = replace_attnprocessor2_0_with_attnprocessor(
        pipe,
        move_to_cpu=True,
    )

    # 2) 定位目标 cross-attn 层
    real_name = resolve_layer_name(pipe, user_layer_name)
    target_module, attn_sink = enable_target_capture(pipe, real_name)
    hook = install_laststep_flag_pre_hook(pipe, target_module)
    patch_attention_get_scores(pipe, target_layer=real_name)

    # 3) 正常推理
    gen = torch.Generator(device=device).manual_seed(1234)
    _ = pipe(
        prompt=prompt,
        num_inference_steps=15,
        guidance_scale=4.0,
        generator=gen,
    )

    hook.remove()

    assert attn_sink, "❌ 没捕到 cross-attn，检查层名/patch 是否正确"
    last = attn_sink[-1]
    attn_probs = last["attn_probs"]  # [heads*batch, HW, T_text]

    # 4) 文本分析 & mapping
    doc = nlp(prompt)
    tokenizer = getattr(pipe, "tokenizer", None) or getattr(pipe, "tokenizer_2", None)
    if tokenizer is None:
        raise RuntimeError("❌ pipeline 中没有 tokenizer。")

    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False
    )
    chunks = [chunk.text for chunk in doc.noun_chunks]
    mapping = map_chunks_to_token_indices(prompt=prompt, chunks=chunks, tokenizer=tokenizer)

    return {
        "prompt": prompt,
        "attn_probs": attn_probs,
        "hidden_sink": hidden_sink,
        "mapping": mapping,
    }




# =========================================================
# 示例 main：对比“红椅子猫”和“沙滩猫”的 'a cat' 区域
# =========================================================
if __name__ == "__main__":
    # 这里假设你已经在外面初始化了 spacy：
    # import spacy
    # nlp = spacy.load("en_core_web_sm")

    DEVICE = "cuda:1"

    pipe = PixArtAlphaPipeline.from_pretrained(
        "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16,
    ).to(DEVICE)

    prompt1 = "a cat on a red chair"
    prompt2 = "a cat sitting on a sandy beach"

    # 1) 先跑一遍，拿到两个 prompt 的 cat region hidden + 语义相似度
    res = compare_region_semantic_similarity(
        pipe=pipe,
        prompt1=prompt1,
        prompt2=prompt2,
        region_id="a cat",                  # noun_chunk 文本要能被 mapping 到
        user_layer_name="transformer_blocks.10.attn2",
        num_inference_steps=20,
        guidance_scale=4.0,
        seed=1234,
        device=DEVICE,
    )

    # 2) 如果你想做 t-SNE / PCA，可直接从 res["h1"], res["h2"] 拿到所有 patch 级 hidden：
    h1 = res["h1"]  # [K1, C]
    h2 = res["h2"]  # [K2, C]

    # 3) 画“跨 prompt 的 cat 区域轨迹相似度”折线图
    traj_res = plot_region_trajectory_similarity_two_prompts(
        res,
        self_attn_layer_substr="transformer_blocks.27.attn1",
        save_path="./cache/cat_region_trajectory_sim.png",
    )

    print("trajectory steps:", traj_res["steps"])
    print("trajectory sims :", traj_res["sims"])

    traj_curve = plot_region_two_prompt_raw_trajectory(
        res,
        self_attn_layer_substr="transformer_blocks.27.attn1",
        save_path="./cache/cat_region_raw_trajectory.png",
        smooth=True,
        scaling="pca",   # "norm" 也可，论文推荐 PCA
    )
    print("\n📌 Result files generated:")
    print("   ├─ ./cache/cat_region_alignment_similarity.png   (similarity curve)")
    print("   └─ ./cache/cat_region_raw_trajectory.png       (trajectory comparison)")
    print("-------------------------------------------------------------")
    print("🎉 Experiment complete.\n")

    print("\n------------------------")
    print("▶ Experiment 3: Influence Similarity (a cat → other patches)")
    print("------------------------")

    # 提取上一次 compare_region_semantic_similarity 运行时保存的 cross-attn & mapping
    # 注意：res 是 compare_region_semantic_similarity 返回的
    print(res)

    attn_self_P1 = res["extra1"][-1]["attn"] if "attn" in res["extra1"][-1] else None
    attn_self_P2 = res["extra2"][-1]["attn"] if "attn" in res["extra2"][-1] else None

    # 如果 compare_region_semantic_similarity 未保存 cross-attn，我们重新 hook 两次
    if attn_self_P1 is None or attn_self_P2 is None:
        print("⚠ res 中没有 cross-attention，需要额外运行 hook 获取")
        out_A = run_prompt_and_get_cross_attn_and_mapping(pipe, prompt1, device=DEVICE)
        out_B = run_prompt_and_get_cross_attn_and_mapping(pipe, prompt2, device=DEVICE)

        attn_self_P1 = out_A["attn_probs"]
        attn_self_P2 = out_B["attn_probs"]
        mapping_P1 = out_A["mapping"]
        mapping_P2 = out_B["mapping"]
    else:
        # 如果 res 里已经包含 mapping 和 attn，我们调用统一接口从 res 结构中获取
        mapping_P1 = res.get("mapping1", None)
        mapping_P2 = res.get("mapping2", None)

    # 如果 mapping 仍为空，从 tokenizer 重新构造
    if mapping_P1 is None or mapping_P2 is None:
        doc1 = nlp(prompt1)
        doc2 = nlp(prompt2)
        tok = pipe.tokenizer if hasattr(pipe, "tokenizer") else pipe.tokenizer_2

        mapping_P1 = map_chunks_to_token_indices(prompt1, [c.text for c in doc1.noun_chunks], tok)
        mapping_P2 = map_chunks_to_token_indices(prompt2, [c.text for c in doc2.noun_chunks], tok)

    chunk = "a cat"
    assert chunk in mapping_P1 and chunk in mapping_P2, f"chunk '{chunk}' not found in mapping!"

    # -------- 计算影响向量 --------
    influence_P1 = compute_chunk_influence_from_cross_attn(attn_self_P1, mapping_P1, chunk)
    influence_P2 = compute_chunk_influence_from_cross_attn(attn_self_P2, mapping_P2, chunk)

    # -------- 数值一致性指标 --------
    sim = F.cosine_similarity(influence_P1.float(), influence_P2.float(), dim=0).item()
    print(f"📌 Influence Cosine Similarity (a cat → all patches): {sim:.4f}")

    # -------- 可视化 --------
    L = influence_P1.shape[0]
    H = W = int(math.sqrt(L))
    assert H * W == L, f"L={L} 不能 reshape 成方阵"

    save_path = "./cache/cat_influence_compare.png"
    plot_influence_compare_two_prompts(
        influence_P1,
        influence_P2,
        H=H,
        W=W,
        prompt_A=prompt1,
        prompt_B=prompt2,
        chunk_name=chunk,
        save_path=save_path
    )

    print(f"🎯 Influence visualization saved → {save_path}")
    print("=== Experiment 3 Complete ===\n")