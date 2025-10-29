from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.hypercore.nn import nn
from typing import Tuple, Optional, Literal

import torch
import torch.nn.functional as F
import torch.distributed as dist
import math
from megatron.core.hypercore.manifolds import Lorentz
import numpy as np
from megatron.core.hypercore.models.lorentz_feedforward import LorentzFeedForward
from functools import reduce
from operator import mul

global world_size, rank
world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0
from deepspeed.runtime.zero.partition_parameters import GatheredParameters

def _calc_fans(t, fan_in=None, fan_out=None):
    if fan_in is not None and fan_out is not None:
        return fan_in, fan_out
    dim = t.dim()
    if dim < 2:
        raise ValueError("Need fan_in/fan_out for 1D tensors (bias or vectors).")
    if dim == 2:  
        fan_out, fan_in = t.size(0), t.size(1)
    else:       
        receptive = reduce(mul, t.shape[2:], 1)
        fan_in  = t.size(1) * receptive
        fan_out = t.size(0) * receptive
    return fan_in, fan_out

def _xavier_uniform_(t: torch.Tensor, gain: float = 1.0,
                           fan_in: int | None = None, fan_out: int | None = None):
    fi, fo = _calc_fans(t, fan_in, fan_out)
    bound = gain * math.sqrt(6.0 / (fi + fo))
    with torch.no_grad():
        return (2 * torch.rand_like(t, dtype=t.dtype, device=t.device) - 1) * bound
        # return t.uniform_(-bound, bound)


class Gate(MegatronModule):
    """Gate module for expert routing."""
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
        train (bool): If true, return the relevant information for load balancing
    """
    def __init__(self, manifold, args):
        """
        Initializes the Gate module.

        Args:
            args: Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim - 1
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.bias_update_spd = args.bias_update_speed
        self.weight = torch.nn.Parameter(torch.empty(args.n_routed_experts, self.dim))
        self.bias = torch.nn.Parameter(torch.zeros(args.n_routed_experts)) 
        self.reset_parameters()   
        self.manifold = manifold
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=1.0) 
        torch.nn.init.zeros_(self.bias)
    def project(self, x):
        x_time = ((x ** 2).sum(dim=-1, keepdims=True) + self.manifold.c) ** 0.5
        x = torch.cat([x_time, x], dim=-1)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        weights = self.project(self.weight)
        scores = 2 * self.manifold.c + 2 * self.manifold.cinner(x, weights)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        if self.training:
            return weights.type_as(x), indices, original_scores
        else:
            return weights.type_as(x), indices

    @torch.no_grad()
    def update_bias(self, indices: torch.Tensor, is_counts=False):
        """
        indices : (num_tokens_in_step, topk) – expert ids routed this step.
        """
        if indices.numel() == 0:
            return
        if not is_counts:
            util = torch.bincount(indices.flatten(),
                                minlength=self.bias.numel()).float().to(self.bias)
            util  = util / util.sum()                 
            mean  = util.mean()
            self.bias += self.bias_update_spd * (mean - util)
        else:
            total = indices.sum().clamp_min(1)
            util = indices / total
            mean = util.mean()
            with torch.no_grad():
                self.bias += self.bias_update_spd * (mean - util)

class LorentzExpert(MegatronModule):
    """Lorentz Expert implementation for MoE."""
    """
    Expert layer for Lorentz Mixture-of-Experts (MoE) models.

    Attributes:
        manifold (Lorentz): input manifold
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
        expert_manifold (Lorentz): manifold of the expert
    """
    def __init__(self, manifold, dim: int, inter_dim: int, expert_manifold=None):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.c = manifold.c
        self.manifold = manifold
        if expert_manifold is not None:
            self.expert_manifold = expert_manifold
        else:
            self.expert_manifold = manifold
        self.w1 = nn.LorentzLinear(self.expert_manifold, dim, inter_dim - 1)
        self.w2 = nn.LorentzLinear(self.expert_manifold, inter_dim, dim - 1, manifold_out=self.manifold)
        self.w3 = nn.LorentzLinear(self.expert_manifold, dim, inter_dim - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        x = x * (self.expert_manifold.c / self.c).sqrt() # transfer inputs to expert manifold
        x1_time = F.silu(self.w1(x, return_space=True))
        x3_time = self.w3(x, return_space=True)
        x_space = x1_time * x3_time
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.expert_manifold.c).clamp_min(1e-8).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        x = self.w2(x)
        return x
    
class LorentzMoE(MegatronModule):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        manifold (Lorentz): Input manifold
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
        train (bool): If true, return the relevant information for load balancing
    """
    def __init__(self, manifold: Lorentz, args):
        """
        Initializes the MoE module.

        Args:
            args: Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.manifold = manifold
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts 
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = 0 #rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(manifold, args)
        self.curvature_list = np.linspace(0.1, 2.0, self.n_routed_experts).tolist()
        self.n_shared_experts = args.n_shared_experts
        self.expert_manifolds = [Lorentz(c=(self.curvature_list[i]), learnable=False) for i in range(self.n_routed_experts)]
        self.experts = torch.nn.ModuleList([LorentzExpert(self.manifold, args.dim, args.mice_inter_dim, self.expert_manifolds[i]) for i in range(self.n_routed_experts)])
        self.shared_experts = LorentzFeedForward(self.manifold, args.dim, args.n_shared_experts * args.mice_inter_dim)

    def project(self, x):
        x_space = x
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.c) ** 0.5
        x = torch.cat([x_time, x_space], dim=-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices, scores = self.gate(x)
        y = self.project(torch.zeros_like(x[..., 1:]))
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        ave = z + y
        denom = (-self.manifold.l_inner(ave, ave, dim=-1, keep_dim=True)).abs().clamp_min(1e-8).sqrt()
        out = self.manifold.c.sqrt() * ave / denom
        if self.training:
            return out.view(shape), indices, scores
        else:
            return out.view(shape)

# class LorentzMoE(torch.nn.Module):
#     """
#     Mixture-of-Experts (MoE) module.

#     Attributes:
#         manifold (Lorentz): Input manifold
#         dim (int): Dimensionality of input features.
#         n_routed_experts (int): Total number of experts in the model.
#         n_local_experts (int): Number of experts handled locally in distributed systems.
#         n_activated_experts (int): Number of experts activated for each input.
#         gate (nn.Module): Gating mechanism to route inputs to experts.
#         experts (nn.ModuleList): List of expert modules.
#         shared_experts (nn.Module): Shared experts applied to all inputs.
#         train (bool): If true, return the relevant information for load balancing
#     """
#     def __init__(self, manifold: Lorentz, args):
#         """
#         Initializes the MoE module.

#         Args:
#             args: Model arguments containing MoE parameters.
#         """
#         super().__init__()
#         self.dim = args.dim
#         self.manifold = manifold
#         assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
#         self.n_routed_experts = args.n_routed_experts
#         self.n_local_experts = args.n_routed_experts // world_size
#         self.n_activated_experts = args.n_activated_experts
#         self.experts_start_idx = rank * self.n_local_experts
#         self.experts_end_idx = self.experts_start_idx + self.n_local_experts
#         self.gate = Gate(manifold, args)
#         self.curvature_list = np.linspace(0.1, 2.0, self.n_routed_experts).tolist()
#         self.n_shared_experts = args.n_shared_experts
#         self.expert_manifolds = [Lorentz(c=(self.curvature_list[i]), learnable=False) for i in range(self.n_routed_experts)]
#         # self.experts = torch.nn.ModuleList([LorentzExpert(self.manifold, args.dim, args.mice_inter_dim, self.expert_manifolds[i]) for i in range(self.n_routed_experts)])
#         self.experts = torch.nn.ModuleList([
#             LorentzExpert(manifold, args.dim, args.mice_inter_dim,
#                           self.expert_manifolds[g])
#             for g in range(self.local_start, self.local_end)
#         ])
#         self.shared_experts = LorentzFeedForward(self.manifold, args.dim, args.n_shared_experts * args.mice_inter_dim)

#     def project(self, x):
#         x_space = x
#         x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.c) ** 0.5
#         x = torch.cat([x_time, x_space], dim=-1)
#         return x

#     def _owner_and_local_eid(self, global_eid: torch.Tensor):
#         nle = self.n_local_experts
#         dest_rank = (global_eid // nle)
#         local_eid = (global_eid % nle)
#         return dest_rank, local_eid

#     def _flat_token_ids(self, N: int, device):
#         return torch.arange(N, device=device, dtype=torch.long)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for the MoE module.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             torch.Tensor: Output tensor after expert routing and computation.
#         """
#         shape = x.size()
#         x = x.view(-1, self.dim)
#         N, D = x.size()
#         device = x.device
#         weights, indices, scores = self.gate(x)
#         k = indices.size(1)

#         dest_rank, local_eid = self._owner_and_local_eid(indices)  
#         token_ids = self._flat_token_ids(N, device)          
#         token_ids = token_ids.unsqueeze(1).expand_as(indices)  

#         avg = math.ceil((N * k) / self.n_routed_experts)
#         cap = max(1, int(1.2 * avg))

#         W, E = self.world_size, self.n_local_experts
#         send_x      = x.new_zeros((W, E, cap, D))
#         send_w      = x.new_zeros((W, E, cap, 1))
#         send_tid    = token_ids.new_full((W, E, cap), -1)    
#         fill_cursor = torch.zeros((W, E), dtype=torch.int32, device=device)

#         flat_rows = (N * k)
#         flat_dest = dest_rank.reshape(-1)     
#         flat_eid  = local_eid.reshape(-1)    
#         flat_tid  = token_ids.reshape(-1)   
#         flat_w    = weights.reshape(-1)    
#         flat_x    = x.index_select(0, flat_tid)

#         y = self.project(torch.zeros_like(x[..., 1:]))
#         counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
#         for i in range(self.experts_start_idx, self.experts_end_idx):
#             if counts[i] == 0:
#                 continue
#             expert = self.experts[i]
#             idx, top = torch.where(indices == i)
#             y[idx] += expert(x[idx]) * weights[idx, top, None]
#         z = self.shared_experts(x)
#         if world_size > 1:
#             dist.all_reduce(y)
#         ave = z + y
#         denom = (-self.manifold.l_inner(ave, ave, dim=-1, keep_dim=True)).abs().clamp_min(1e-8).sqrt()
#         out = self.manifold.c.sqrt() * ave / denom
#         if self.training:
#             return out.view(shape), indices, scores
#         else:
#             return out.view(shape)