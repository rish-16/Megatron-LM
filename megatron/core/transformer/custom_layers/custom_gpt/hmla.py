import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.hypercore.nn import nn
from megatron.core.hypercore.nn.linear import LorentzLinear
from megatron.core.hypercore.nn.conv import LorentzRMSNorm
import torch
import torch.nn.functional as F
import torch.distributed as dist

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().contiguous().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

class LorentzMLA(MegatronModule):
    """
    Hyperbolic Multi-Headed Attention Layer (HMLA).

    Attributes:
        manifold (Lorentz): the embedding manifold of the vectors
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, manifold, args):
        #TODO change from args to parameters
        super().__init__()
        self.manifold = manifold
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads #// world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = LorentzLinear(self.manifold, self.dim, self.n_heads * (self.qk_head_dim - 1)) # Linear operation on the time-like dimensions
        else:
            self.wq_a = LorentzLinear(self.manifold, self.dim, self.q_lora_rank - 1)
            self.q_norm = LorentzRMSNorm(self.manifold, self.q_lora_rank)
            self.wq_b = LorentzLinear(self.manifold, self.q_lora_rank + 1, self.n_heads * (self.qk_head_dim - 1))
        self.wkv_a = LorentzLinear( self.manifold, self.dim, self.kv_lora_rank + self.qk_rope_head_dim - 1)
        self.kv_norm = LorentzRMSNorm(self.manifold, self.kv_lora_rank)
        self.wkv_b = LorentzLinear(self.manifold, self.kv_lora_rank + 1, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim - 1))
        self.wo = LorentzLinear(manifold, self.n_heads * self.v_head_dim, self.dim - 1)
        # self.softmax_scale = self.qk_head_dim ** -0.5
        self.softmax_scale = nn.Parameter(torch.tensor([math.sqrt(self.n_local_heads * self.qk_head_dim)]))
        self.bias = nn.Parameter(torch.zeros(()))
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

    def project(self, x):
        x_space = x
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.c) ** 0.5
        x = torch.cat([x_time, x_space], dim=-1)
        return x
    
    def shape_mask(self, mask, batch_size: int, num_heads: int, seq_len: int):
        if mask is None:
            return None
        if mask.dim() == 4:
            if mask.shape == (batch_size, num_heads, seq_len, seq_len):
                return mask
            else:
                raise ValueError(f"Mask with 4 dims must be [B, H, N, N]")

        if mask.dim() == 2:
            m, n = mask.shape
            # [N, N]
            if (m == seq_len) and (n == seq_len):
                mask = mask.unsqueeze(0).unsqueeze(0)
            else: # [B, N]
                mask = mask.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, N]
            return mask

        if mask.dim() == 3:
            b, m, n = mask.shape
            # [B, N, N]
            if (m == seq_len) and (n == seq_len):
                mask = mask.unsqueeze(1)  # [B, 1, N, N]
                return mask

            # [B, 1, N]
            if (m == 1) and (n == seq_len):
                mask = mask.squeeze(1)
                mask = mask.unsqueeze(1).unsqueeze(2)
                return mask

            # [B, H, N]
            if (m == num_heads) and (n == seq_len):
                mask = mask.unsqueeze(2)
                return mask

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], attn_impl='naive', past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, embed_dim = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x, return_space=True)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x, return_space=True), space_only=True), return_space=True)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim - 1) #space-like
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim - 1], dim=-1) #space-like

        q_pe = apply_rotary_emb(q_pe, freqs_cis) #space-like
        kv = self.wkv_a(x, return_space=True)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim - 1], dim=-1) #space-like
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis) #space-like

        q = torch.cat([q_nope, q_pe], dim=-1) #space-like
        kv = self.wkv_b(self.kv_norm(kv, space_only=True), return_space=True) #space-like
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim - 1)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim - 1], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)

        q_h = q.transpose(1, 2)  
        k_h = k.transpose(1, 2)  
        v_h = v.transpose(1, 2)

        if past_key_value is not None:
            pk, pv = past_key_value                                  
            k_h = torch.cat([pk, k_h], dim=2)                       
            v_h = torch.cat([pv, v_h], dim=2)

        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        scores = scores / self.softmax_scale + self.bias

        if mask is not None:
            mask = self.shape_mask(mask, bsz, self.n_local_heads, seqlen)
            scores = scores.masked_fill(mask, -1e18)

        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        vs = self.project(v_h)
        x = self.manifold.lorentzian_centroid(vs, scores).transpose(1, 2) #[B, S, H, N]
        x = self.wo(x.flatten(2))
        return x