import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

if _TRITON_AVAILABLE:
    # BLOCK is number of feature elements handled per program (tuneable)
    @triton.jit
    def _elwise_mul_kernel(X1_ptr, X2_ptr, OUT_ptr, N_rows, N_cols,
                           stride_row_x1, stride_col_x1,
                           stride_row_x2, stride_col_x2,
                           stride_row_out, stride_col_out,
                           BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK)
        col_idx = col_offsets
        mask = col_idx < N_cols

        # compute element addresses (assume dense contiguous rows)
        off_x1 = row * stride_row_x1 + col_idx * stride_col_x1
        off_x2 = row * stride_row_x2 + col_idx * stride_col_x2
        off_out = row * stride_row_out + col_idx * stride_col_out

        a = tl.load(X1_ptr + off_x1, mask=mask, other=0.0)
        b = tl.load(X2_ptr + off_x2, mask=mask, other=0.0)
        tl.store(OUT_ptr + off_out, a * b, mask=mask)

def multiply_and_pack(x1_space: torch.Tensor, x3_space: torch.Tensor, c: float, block=1024):
    """
    Compute x_space = x1_space * x3_space (elementwise) using Triton if available,
    then compute the time coordinate sqrt(sum(x_space**2) + c) and return concatenated tensor
      out = cat([x_time, x_space], dim=-1)
    Arguments:
      x1_space, x3_space : (N, D) tensors (must be same shape)
      c : scalar curvature (float)
      block : Triton BLOCK size
    Returns:
      out : (N, D+1) tensor
    """
    assert x1_space.shape == x3_space.shape
    N, D = x1_space.shape
    # ensure contiguous for kernel
    x1c = x1_space.contiguous()
    x3c = x3_space.contiguous()
    out_space = torch.empty_like(x1c)

    if _TRITON_AVAILABLE and x1c.is_cuda:
        # call triton kernel
        grid = (N,)
        # strides in *elements* (not bytes)
        stride_row_x1 = x1c.stride(0)
        stride_col_x1 = x1c.stride(1)
        stride_row_x3 = x3c.stride(0)
        stride_col_x3 = x3c.stride(1)
        stride_row_out = out_space.stride(0)
        stride_col_out = out_space.stride(1)
        _elwise_mul_kernel[grid](
            x1c, x3c, out_space, N, D,
            stride_row_x1, stride_col_x1,
            stride_row_x3, stride_col_x3,
            stride_row_out, stride_col_out,
            BLOCK=block
        )
    else:
        # fallback to PyTorch elementwise multiply
        out_space = x1c * x3c

    # compute time coordinate in PyTorch (vectorized, efficient)
    x_time = (out_space.square().sum(dim=-1, keepdim=True) + float(c)).clamp_min(1e-8).sqrt()
    out = torch.cat([x_time, out_space], dim=-1)
    return out