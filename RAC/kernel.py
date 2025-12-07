import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['NUM_ACTIVE', 'N_CTX', 'HEAD_DIM'], # 加入 HEAD_DIM 作为 key
)
@triton.jit
def _sparse_query_flash_attn_kernel(
    # Pointers
    Q, K, V, Out, Indices,
    # Strides
    stride_qb, stride_qn, stride_qh, stride_qk,
    stride_kb, stride_kn, stride_kh, stride_kk,
    stride_vb, stride_vn, stride_vh, stride_vk,
    stride_ob, stride_on, stride_oh, stride_ok,
    # Shapes
    N_CTX,      
    NUM_ACTIVE, 
    H,
    HEAD_DIM,   # 真实的 Head Dim (例如 72)
    # Block constants
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr, # 必须是 2 的幂次 (例如 128)
    # Scale
    sm_scale
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    off_b = pid_bh // H
    off_h = pid_bh % H

    # --- 1. Indices Masking ---
    offs_m_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m_idx < NUM_ACTIVE
    real_q_row_idxs = tl.load(Indices + off_b * NUM_ACTIVE + offs_m_idx, mask=mask_m, other=0)

    # --- 2. Dim Masking ---
    # offs_d 的范围是 0~127 (BLOCK_DMODEL)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # mask_d: 只有 0~71 是 True, 72~127 是 False
    mask_d = offs_d < HEAD_DIM

    # --- 3. Load Q ---
    q_ptrs = Q + (off_b * stride_qb + real_q_row_idxs[:, None] * stride_qn + \
                  off_h * stride_qh + offs_d[None, :] * stride_qk)
    
    # 必须同时应用 mask_m (行有效) 和 mask_d (列有效)
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # --- 4. Init Accumulators ---
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # --- 5. Loop K/V ---
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX

        k_ptrs = K + (off_b * stride_kb + offs_n[None, :] * stride_kn + \
                      off_h * stride_kh + offs_d[:, None] * stride_kk)
        v_ptrs = V + (off_b * stride_vb + offs_n[:, None] * stride_vn + \
                      off_h * stride_vh + offs_d[None, :] * stride_vk)
        
        # Load K, V 也要加上 mask_d
        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        qk = tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        m_i_new = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i_new, m_i)
        alpha = tl.math.exp(m_i - m_i_new)
        beta = tl.math.exp(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None] + tl.dot(beta.to(tl.float16), v)
        l_i = l_i * alpha + tl.sum(beta, 1)
        m_i = m_i_new

    # --- 6. Finalize ---
    acc = acc / l_i[:, None]

    # --- 7. Store Output ---
    out_ptrs = Out + (off_b * stride_ob + real_q_row_idxs[:, None] * stride_on + \
                      off_h * stride_oh + offs_d[None, :] * stride_ok)

    # 写回时同样需要双重 mask，防止写坏 padding 区域
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_d[None, :])

def sparse_query_attention(q, k, v, indices, out=None, sm_scale=1.0):
    BATCH, N_CTX, H, D = q.shape
    _, Num_Active = indices.shape
    
    if out is None:
        out = torch.zeros_like(q)
    else:
        assert out.is_contiguous()

    # 计算向上取整的 Power of 2
    padded_d_model = triton.next_power_of_2(D)

    grid = lambda META: (triton.cdiv(Num_Active, META['BLOCK_M']), BATCH * H)
    
    _sparse_query_flash_attn_kernel[grid](
        q, k, v, out, indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N_CTX, Num_Active, H,
        HEAD_DIM=D,             # 传入真实的 D (72) 用于 Mask
        BLOCK_DMODEL=padded_d_model, # 传入 128 用于 tl.arange
        sm_scale=sm_scale
    )
    return out

if __name__ == "__main__":
    pass