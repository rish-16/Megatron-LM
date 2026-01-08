#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain HELM (Hyperbolic Efficient Language Model) with Qwen3 architecture.

This script trains a hyperbolic transformer model using the Lorentz manifold
for attention and MoE layers. Based on the HELM-MiCE architecture which combines:
- Lorentz Multi-head Latent Attention (HMLA)
- Lorentz Mixture of Curvature Experts (MiCE)
"""

import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import StragglerDetector

# Hyperbolic imports
from megatron.core.hypercore.manifolds import Lorentz
from megatron.core.transformer.custom_layers.custom_gpt import LorentzDeepSeekV3

stimer = StragglerDetector()


@dataclass
class HELMModelArgs:
    """Model arguments for HELM-Qwen3."""
    # Model architecture
    dim: int = 2048
    n_layers: int = 36
    n_heads: int = 16
    n_dense_layers: int = 2  # Number of initial dense FFN layers before MoE
    vocab_size: int = 151936

    # Sequence parameters
    max_seq_len: int = 4096
    original_seq_len: int = 4096

    # MLA (Multi-head Latent Attention) parameters
    q_lora_rank: int = 0  # 0 means no LoRA for Q
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # RoPE parameters
    rope_theta: float = 1000000.0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    # MoE parameters
    inter_dim: int = 11008  # FFN intermediate dimension for dense layers
    mice_inter_dim: int = 1408  # FFN intermediate dimension for MoE experts
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: str = "softmax"
    route_scale: float = 1.0
    bias_update_speed: float = 0.001

    # Hyperbolic parameters
    curvature: float = 1.0
    train_curv: bool = True
    project_emb: bool = True  # If True, project Euclidean embeddings to hyperbolic space


def add_helm_args(parser):
    """Add HELM-specific arguments to the parser."""
    group = parser.add_argument_group(title='HELM-Qwen3')

    # MLA parameters (use helm- prefix to avoid conflicts with Megatron args)
    group.add_argument('--helm-q-lora-rank', type=int, default=0,
                       help='LoRA rank for query projection (0 = no LoRA)')
    group.add_argument('--helm-kv-lora-rank', type=int, default=512,
                       help='LoRA rank for key/value projection')
    group.add_argument('--helm-qk-nope-head-dim', type=int, default=128,
                       help='Dimension of non-positional qk heads')
    group.add_argument('--helm-qk-rope-head-dim', type=int, default=64,
                       help='Dimension of rotary positional qk heads')
    group.add_argument('--helm-v-head-dim', type=int, default=128,
                       help='Dimension of value heads')

    # MoE parameters
    group.add_argument('--mice-inter-dim', type=int, default=1408,
                       help='Intermediate dimension for MoE experts')
    group.add_argument('--n-routed-experts', type=int, default=8,
                       help='Number of routed experts')
    group.add_argument('--n-shared-experts', type=int, default=1,
                       help='Number of shared experts')
    group.add_argument('--n-activated-experts', type=int, default=2,
                       help='Number of activated experts per token')
    group.add_argument('--n-expert-groups', type=int, default=1,
                       help='Number of expert groups')
    group.add_argument('--n-limited-groups', type=int, default=1,
                       help='Number of limited groups for routing')
    group.add_argument('--score-func', type=str, default='softmax',
                       choices=['softmax', 'sigmoid'],
                       help='Scoring function for expert routing')
    group.add_argument('--route-scale', type=float, default=1.0,
                       help='Scaling factor for routing weights')
    group.add_argument('--n-dense-layers', type=int, default=2,
                       help='Number of initial dense FFN layers before MoE')
    group.add_argument('--bias-update-speed', type=float, default=0.001,
                       help='Speed for bias updates in expert routing')

    # Hyperbolic parameters
    group.add_argument('--curvature', type=float, default=1.0,
                       help='Initial curvature of hyperbolic space')
    group.add_argument('--train-curvature', action='store_true', default=True,
                       help='Whether to train the curvature parameter')
    group.add_argument('--no-train-curvature', action='store_false', dest='train_curvature',
                       help='Disable training of curvature parameter')
    group.add_argument('--project-emb', action='store_true', default=True,
                       help='Project Euclidean embeddings to hyperbolic space')
    group.add_argument('--no-project-emb', action='store_false', dest='project_emb',
                       help='Use native hyperbolic embeddings')

    # RoPE scaling parameters (use helm- prefix to avoid conflicts)
    group.add_argument('--helm-beta-fast', type=int, default=32,
                       help='Beta fast for RoPE scaling')
    group.add_argument('--helm-beta-slow', type=int, default=1,
                       help='Beta slow for RoPE scaling')
    group.add_argument('--helm-mscale', type=float, default=1.0,
                       help='M-scale for attention scaling')
    group.add_argument('--helm-rope-factor', type=float, default=1.0,
                       help='RoPE scaling factor')

    return parser


def build_helm_model_args(args) -> HELMModelArgs:
    """Build HELM model arguments from training args."""
    return HELMModelArgs(
        dim=args.hidden_size,
        n_layers=args.num_layers,
        n_heads=args.num_attention_heads,
        n_dense_layers=getattr(args, 'n_dense_layers', 2),
        vocab_size=args.padded_vocab_size,
        max_seq_len=args.seq_length,
        original_seq_len=getattr(args, 'original_seq_len', args.seq_length),
        q_lora_rank=getattr(args, 'helm_q_lora_rank', 0),
        kv_lora_rank=getattr(args, 'helm_kv_lora_rank', 512),
        qk_nope_head_dim=getattr(args, 'helm_qk_nope_head_dim', 128),
        qk_rope_head_dim=getattr(args, 'helm_qk_rope_head_dim', 64),
        v_head_dim=getattr(args, 'helm_v_head_dim', 128),
        rope_theta=getattr(args, 'rotary_base', 1000000.0),
        rope_factor=getattr(args, 'helm_rope_factor', 1.0),
        beta_fast=getattr(args, 'helm_beta_fast', 32),
        beta_slow=getattr(args, 'helm_beta_slow', 1),
        mscale=getattr(args, 'helm_mscale', 1.0),
        inter_dim=args.ffn_hidden_size,
        mice_inter_dim=getattr(args, 'mice_inter_dim', 1408),
        n_routed_experts=getattr(args, 'n_routed_experts', 8),
        n_shared_experts=getattr(args, 'n_shared_experts', 1),
        n_activated_experts=getattr(args, 'n_activated_experts', 2),
        n_expert_groups=getattr(args, 'n_expert_groups', 1),
        n_limited_groups=getattr(args, 'n_limited_groups', 1),
        score_func=getattr(args, 'score_func', 'softmax'),
        route_scale=getattr(args, 'route_scale', 1.0),
        bias_update_speed=getattr(args, 'bias_update_speed', 0.001),
        curvature=getattr(args, 'curvature', 1.0),
        train_curv=getattr(args, 'train_curvature', True),
        project_emb=getattr(args, 'project_emb', True),
    )


class HELMQwen3Model(MegatronModule):
    """HELM-Qwen3 model wrapper for Megatron training.

    This model combines hyperbolic geometry with the Qwen3 architecture,
    using Lorentz manifold for attention and mixture-of-experts layers.
    """

    def __init__(self, config, pre_process=True, post_process=True):
        super().__init__(config)

        args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        # Build HELM model arguments
        helm_args = build_helm_model_args(args)

        # Create manifolds with specified curvature
        curvature = helm_args.curvature
        train_curv = helm_args.train_curv

        self.manifold_in = Lorentz(c=curvature, learnable=train_curv)
        self.manifold_hidden = Lorentz(c=curvature, learnable=train_curv)
        self.manifold_out = Lorentz(c=curvature, learnable=train_curv)

        # Build the hyperbolic transformer
        self.transformer = LorentzDeepSeekV3(
            args=helm_args,
            manifold_in=self.manifold_in,
            manifold_hidden=self.manifold_hidden,
            manifold_out=self.manifold_out,
        )

    def set_input_tensor(self, input_tensor):
        """Set input tensor for pipeline parallelism."""
        self.input_tensor = input_tensor

    def forward(self, input_ids, position_ids, attention_mask,
                labels=None, loss_mask=None, inference_params=None):
        """Forward pass.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            position_ids: Position IDs [batch, seq_len]
            attention_mask: Attention mask [batch, 1, seq_len, seq_len] or [batch, seq_len, seq_len]
            labels: Target labels for loss computation [batch, seq_len]
            loss_mask: Mask for loss computation [batch, seq_len]
            inference_params: Parameters for inference (optional)

        Returns:
            If training: loss tensor
            If inference: logits tensor
        """
        # Reshape attention mask if needed
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                # [B, 1, S, S] -> [B, S, S]
                attention_mask = attention_mask.squeeze(1)
            # Convert to boolean mask (True = masked positions)
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask < 0.5

        # Forward through transformer
        if self.training:
            logits, all_indices, all_scores = self.transformer(
                input_ids,
                start_pos=0,
                attn_mask=attention_mask
            )
        else:
            logits = self.transformer(
                input_ids,
                start_pos=0,
                attn_mask=attention_mask
            )
            all_indices, all_scores = None, None

        # Compute loss if labels provided
        if labels is not None and self.post_process:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Apply loss mask if provided
            if loss_mask is not None:
                shift_loss_mask = loss_mask[..., 1:].contiguous()
                loss = loss.view_as(shift_labels) * shift_loss_mask

            # Return per-token losses for proper averaging
            return loss.view(-1)

        return logits


def model_provider(pre_process=True, post_process=True):
    """Build the HELM-Qwen3 model."""
    args = get_args()
    print_rank_0('Building HELM-Qwen3 model ...')

    model = HELMQwen3Model(
        config=args,
        pre_process=pre_process,
        post_process=post_process,
    )

    return model


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None):
    """Loss function.

    Args:
        loss_mask: Mask for loss computation
        output_tensor: Per-token losses from forward pass
        model: The model (unused)

    Returns:
        Tuple of (loss, num_tokens, reporting_dict)
    """
    args = get_args()

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask[..., 1:].contiguous().view(-1).float()  # Shift mask to match shifted labels

    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)

    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model):
    """Forward training step."""
    args = get_args()
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            tokens, position_ids, attention_mask,
            labels=labels, loss_mask=loss_mask
        )

    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None):
    """Check if dataset should be built on this rank."""
    return (is_first_or_last_pipeline_stage(vp_stage) and
            parallel_state.get_tensor_model_parallel_rank() == 0)


def core_gpt_dataset_config_from_args(args):
    """Build dataset config from args."""
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build train/valid/test datasets."""
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    print_rank_0("> building train, validation, and test datasets for HELM-Qwen3 ...")
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        partial(is_dataset_built_on_rank, vp_stage=vp_stage),
        config
    ).build()
    print_rank_0("> finished creating HELM-Qwen3 datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # Enable distributed dataset building
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_helm_args,
    )
