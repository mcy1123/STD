"""Fused Triton GQA attention kernel for single-token decode (q_len=1).

Drop-in replacement for F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
in the sparse draft path.  Specialised for q_len=1, GQA, is_causal=False.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_single_query_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    kv_len: int,
    head_dim: int,
    n_kv_heads: int,
    n_q_per_kv: int,
    inv_scale: tl.float32,  # 1/sqrt(head_dim), computed in Python
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per query head."""
    pid = tl.program_id(0)
    kv_head = pid // n_q_per_kv
    q_local = pid % n_q_per_kv
    q_head = kv_head * n_q_per_kv + q_local

    d_offs = tl.arange(0, BLOCK_D)
    q = tl.load(Q_ptr + q_head * head_dim + d_offs, mask=d_offs < head_dim, other=0.0)

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)

    K_stride = kv_len * head_dim
    V_stride = kv_len * head_dim

    for block_start in range(0, kv_len, BLOCK_KV):
        kv_offs = block_start + tl.arange(0, BLOCK_KV)
        valid_kv = kv_offs < kv_len

        # Load K tile [BLOCK_KV, BLOCK_D]
        k = tl.load(
            K_ptr + kv_head * K_stride + kv_offs[:, None] * head_dim + d_offs[None, :],
            mask=valid_kv[:, None] & (d_offs[None, :] < head_dim),
            other=0.0,
        )

        # scores = (Q @ K^T) / sqrt(d)
        scores = tl.sum(q[None, :] * k, axis=1) * inv_scale
        scores = tl.where(valid_kv, scores, float("-inf"))

        # Online softmax update
        m_curr = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha
        l_i = l_i * alpha

        p = tl.exp(scores - m_new)
        l_curr = tl.sum(p, axis=0)
        l_i = l_i + l_curr
        m_i = m_new

        # Load V tile [BLOCK_KV, BLOCK_D]
        v = tl.load(
            V_ptr + kv_head * V_stride + kv_offs[:, None] * head_dim + d_offs[None, :],
            mask=valid_kv[:, None] & (d_offs[None, :] < head_dim),
            other=0.0,
        )

        acc = acc + tl.sum(p[:, None] * v, axis=0)

    acc = acc / l_i
    tl.store(Out_ptr + q_head * head_dim + d_offs, acc, mask=d_offs < head_dim)


def fused_gqa_attention(
    query: torch.Tensor,   # [1, n_q_heads, 1, head_dim]
    key: torch.Tensor,     # [1, n_kv_heads, kv_len, head_dim]
    value: torch.Tensor,   # [1, n_kv_heads, kv_len, head_dim]
) -> torch.Tensor:
    """Drop-in for F.scaled_dot_product_attention(q, k, v, enable_gqa=True)."""
    assert query.shape[2] == 1, f"Expected q_len=1, got {query.shape}"
    n_q_heads = query.shape[1]
    n_kv_heads = key.shape[1]
    kv_len = key.shape[2]
    head_dim = query.shape[3]
    assert n_q_heads % n_kv_heads == 0

    out = torch.empty(1, n_q_heads, 1, head_dim, dtype=query.dtype, device=query.device)

    BLOCK_KV = 128
    BLOCK_D = max(16, min(128, triton.next_power_of_2(head_dim)))
    inv_scale = float(1.0 / math.sqrt(head_dim))

    _gqa_single_query_kernel[(n_q_heads,)](
        query, key, value, out,
        kv_len=kv_len,
        head_dim=head_dim,
        n_kv_heads=n_kv_heads,
        n_q_per_kv=n_q_heads // n_kv_heads,
        inv_scale=inv_scale,
        BLOCK_KV=BLOCK_KV,
        BLOCK_D=BLOCK_D,
    )
    return out


# ---- test ----
if __name__ == "__main__":
    B, Nq, Nkv, L, D = 1, 28, 4, 2600, 128
    q = torch.randn(B, Nq, 1, D, dtype=torch.float16, device="cuda:0")
    k = torch.randn(B, Nkv, L, D, dtype=torch.float16, device="cuda:0")
    v = torch.randn(B, Nkv, L, D, dtype=torch.float16, device="cuda:0")

    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=False, enable_gqa=True
    )
    tri = fused_gqa_attention(q, k, v)

    diff_max = (ref.float() - tri.float()).abs().max().item()
    ref_norm = ref.float().norm().item()
    is_nan = torch.isnan(tri).any().item()
    print(f"Max abs diff: {diff_max:.6f}  ref_norm: {ref_norm:.3f}  NaN: {is_nan}")

    if not is_nan and diff_max < 0.5:
        print("OK")
    else:
        print("FAIL")

    import time
    W, Bn = 30, 200

    for _ in range(W):
        fused_gqa_attention(q, k, v)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(Bn):
        fused_gqa_attention(q, k, v)
    torch.cuda.synchronize()
    t_ms = (time.time() - t0) / Bn * 1000

    for _ in range(W):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(Bn):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
    torch.cuda.synchronize()
    s_ms = (time.time() - t0) / Bn * 1000

    print(f"Triton: {t_ms:.4f} ms  |  SDPA: {s_ms:.4f} ms  |  ratio: {s_ms/t_ms:.2f}x")
