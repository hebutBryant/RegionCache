import re
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("A cat on the table in a dark room, cinematic lighting.")

for chunk in doc.noun_chunks:
    print(chunk.text)

import torch
from diffusers import PixArtAlphaPipeline
# model_path = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"

MODEL_PATH = "/home/lipz/xDiT/xDiT/cfs/dit/PixArt-XL-2-1024-MS"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = PixArtAlphaPipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
tokenizer = getattr(pipe, "tokenizer", None) or getattr(pipe, "tokenizer_2", None)
if tokenizer is None:
    raise RuntimeError("❌ 没有在 pipeline 中找到 tokenizer。")

# prompt = "A cat on the table in a dark room, cinematic lighting."
prompt = "A cat on the table in a dark room, cinematic lighting."

# 编码
encoding = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=True,
    padding=False,
    truncation=False
)

input_ids = encoding["input_ids"][0].tolist()

# 解码为 token 序列
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# 打印结果
print("📝 Token 序列：")
for i, tok in enumerate(tokens):
    print(f"{i:>2d}: {tok}")

print(f"\n共 {len(tokens)} 个 token。")


def map_chunks_to_token_indices(
    prompt,
    chunks,
    tokenizer,
    include_space_tokens: bool = False
):
    """
    将 spaCy 切分得到的短语映射到 tokenizer token 序列的索引列表。
    返回 {chunk_text: [tok_idx0, tok_idx1, ...]}。
    - 默认忽略纯空格 token（形如 '▁'），可用 include_space_tokens=True 保留。
    - chunks 可以是字符串列表，也可以是 spaCy 的 Span 对象迭代器。
    """
    import re

    # 1) token 化，拿到原始 token 列表（含 </s>）
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding=False, truncation=False)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    # 替换显示：把 '▁' 显示为空格，但保留 token 边界
    token_texts_for_join = [t.replace("▁", " ") for t in tokens]

    # 2) 构造“字符位置 -> token 索引”映射
    char_to_token = {}
    pos = 0
    for i, t in enumerate(token_texts_for_join):
        for _ in t:
            char_to_token[pos] = i
            pos += 1
    # 注意：这里拼接出的字符串长度 >= 原 prompt，
    # 但我们后面用 prompt 的字符位置查映射，通常对齐够用。

    # 3) 遍历每个 chunk，找到它在 prompt 中的字符范围 → 收集跨度内所有 token 索引
    def _is_space_token(tok: str) -> bool:
        # 仅由 '▁' 或空白构成，视为“空格 token”
        return tok.strip("▁").strip() == ""

    mapping = {}
    for ch in chunks:
        # 兼容 spaCy Span
        if hasattr(ch, "text"):
            ch = ch.text
        chunk = ch.strip()
        if not chunk:
            continue

        m = re.search(re.escape(chunk), prompt, flags=re.IGNORECASE)
        if not m:
            continue
        start_char, end_char = m.span()  # [start, end)

        # 收集跨度内的 token 索引（去重且保持顺序）
        seen = set()
        idx_list = []
        for p in range(start_char, end_char):
            tidx = char_to_token.get(p)
            if tidx is None:
                continue
            if (not include_space_tokens) and _is_space_token(tokens[tidx]):
                continue
            if tidx not in seen:
                seen.add(tidx)
                idx_list.append(tidx)

        if idx_list:
            mapping[chunk] = idx_list

    return mapping



chunks = [chunk.text for chunk in doc.noun_chunks]
mapping = map_chunks_to_token_indices(prompt=prompt, chunks=chunks, tokenizer=tokenizer)
print(mapping)





def load_pipeline(model_path: str = MODEL_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 建议：CUDA 用 float16，CPU 用 float32
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = PixArtAlphaPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype
    )

    pipe.to(device)

    # 可选：显存优化（有 xformers 再开）
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"enable_xformers_memory_efficient_attention failed: {e}")

    return pipe, device


def generate_image(
    pipe,
    device,
    prompt: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 4.5,
    seed: int | None = 42,
    height: int = 1024,
    width: int = 1024,
    output_path: str = "pixart_output.png"
):
    # 固定随机种子，方便复现
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # 不再手动 autocast，让 pipeline 自己按 dtype 处理
    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

    image = out.images[0]  # 已经是 PIL.Image
    image.save(output_path)
    print(f"Saved image to {output_path}")
    return image


# if __name__ == "__main__":
#     prompt = "A cat on the table in a dark room, cinematic lighting."
#     pipe, device = load_pipeline()
#     generate_image(pipe, device, prompt)
