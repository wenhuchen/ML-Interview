import triton
import triton.language as tl
import torch
import time

@triton.jit
def fused_bias_ln(X, BIAS, GAMMA, BETA, Y,
                  N: tl.constexpr,          #  <-- add tl.constexpr
                  eps: tl.constexpr):       #      (eps is already constexpr)
    pid  = tl.program_id(axis=0)               # row id
    offs = pid * N + tl.arange(0, N)           # offsets for that row

    # --- bias add -----------------------------------------------------
    x = tl.load(X + offs) + tl.load(BIAS + tl.arange(0, N))

    # --- mean / var ---------------------------------------------------
    mean = tl.sum(x, axis=0) / N
    var  = tl.sum(x * x, axis=0) / N - mean * mean
    inv_std = tl.rsqrt(var + eps)

    # --- affine transform --------------------------------------------
    y = (x - mean) * inv_std * tl.load(GAMMA + tl.arange(0, N)) \
        + tl.load(BETA + tl.arange(0, N))
    tl.store(Y + offs, y)

def run_once(B=256, N=4096, dtype=torch.float16, eps=1e-5):
    x     = torch.randn(B, N, device='cuda', dtype=dtype)
    bias  = torch.randn(N,    device='cuda', dtype=dtype)
    gamma = torch.ones (N,    device='cuda', dtype=dtype)
    beta  = torch.zeros(N,    device='cuda', dtype=dtype)
    y_out = torch.empty_like(x)

    # compile + launch (the first call triggers JIT compilation)
    fused_bias_ln[(B,)](
        x, bias, gamma, beta, y_out,
        N, eps,
        num_warps=4, num_stages=2
    )

    # correctness check
    y_ref = torch.nn.functional.layer_norm(x + bias, (N,), gamma, beta, eps)
    print("max diff", (y_out - y_ref).abs().max().item())

    # speed benchmark (second call avoids compile time)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        fused_bias_ln[(B,)](x, bias, gamma, beta, y_out, N, eps,
                            num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    print(f"Triton fused: {(time.time()-t0)*1e3/100:.3f} ms")

if __name__ == "__main__":
    run_once()
