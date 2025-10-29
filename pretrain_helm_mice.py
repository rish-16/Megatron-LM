#!/usr/bin/env python

"""Train HELM-MiCE model using Megatron-LM."""

import torch
from functools import partial
from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.core.enums import ModelType
from megatron.core.models.helm_mice import HELMMiCEModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder

def get_model_provider():
    """Build the model."""
    
    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        args = get_args()
        
        model = HELMMiCEModel(
            config=args,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            # HELM-MiCE specific args
            dim=args.hidden_size,
            inter_dim=args.intermediate_size,
            mice_inter_dim=args.mice_inter_dim,
            n_layers=args.num_layers,
            n_heads=args.num_attention_heads,
            n_routed_experts=args.num_routed_experts,
            n_shared_experts=args.num_shared_experts,
            n_activated_experts=args.num_activated_experts,
            n_expert_groups=args.num_expert_groups,
            score_func=args.score_function,
            route_scale=args.route_scale,
            train_curv=args.train_curvature
        )
        
        return model
    
    return model_provider

def get_forward_backward_func():
    """Get the forward backward function."""
    def forward_backward_func(data_iterator, model, optimizer, timers):
        """Forward and backward pass."""
        args = get_args()
        timers('forward').start()
        
        # Get the batch
        tokens, labels, loss_mask, attention_mask, position_ids = \
            get_batch(data_iterator)
            
        # Forward pass
        output = model(tokens, position_ids, attention_mask)
        losses = model.compute_loss(output, labels, loss_mask)
        
        # Average loss across data parallel group
        loss = average_losses_across_data_parallel_group([losses])
        
        # Backward pass
        if args.deepspeed:
            model.backward(loss)
        else:
            optimizer.zero_grad()
            if args.fp16:
                optimizer.backward(loss, update_master_grads=False)
            else:
                loss.backward()
                
        timers('backward').stop()
        
        return loss
    
    return forward_backward_func

def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()
    tokenizer = get_tokenizer()
    
    # Items and their masks.
    tokens, labels, loss_mask, attention_mask, position_ids = next(data_iterator)
    
    # Convert to torch tensors.
    if tokens is not None:
        tokens = tokens.cuda()
    if labels is not None:
        labels = labels.cuda()
    if loss_mask is not None:
        loss_mask = loss_mask.cuda()
    if attention_mask is not None:
        attention_mask = attention_mask.cuda()
    if position_ids is not None:
        position_ids = position_ids.cuda()
        
    return tokens, labels, loss_mask, attention_mask, position_ids

def get_tasks_args(parser):
    """Define custom arguments for HELM-MiCE."""
    group = parser.add_argument_group(title='HELM-MiCE')
    
    # MiCE specific args
    group.add_argument('--mice-inter-dim', type=int, default=1820,
                      help='Intermediate dimension for MiCE layers')
    group.add_argument('--num-routed-experts', type=int, default=8,
                      help='Number of routed experts for MiCE layers')
    group.add_argument('--num-shared-experts', type=int, default=1,
                      help='Number of shared experts for MiCE layers')
    group.add_argument('--num-activated-experts', type=int, default=2,
                      help='Number of activated experts in MiCE layers')
    group.add_argument('--num-expert-groups', type=int, default=1,
                      help='Number of expert groups')
    group.add_argument('--score-function', type=str, default='softmax',
                      help='Scoring function for MiCE routing')
    group.add_argument('--route-scale', type=float, default=1.0,
                      help='Scaling factor for routing scores')
    group.add_argument('--train-curvature', type=bool, default=True,
                      help='If true, sets the curvatures of the experts as trainable')
    
    return parser

def main():
    """Main training program."""
    
    # Initialize Megatron.
    initialize_megatron(extra_args_provider=get_tasks_args)
    
    # Set up model, optimizer, and learning rate scheduler
    model_provider = get_model_provider()
    forward_backward_func = get_forward_backward_func()
    
    # Build datasets
    train_dataset = BlendedMegatronDatasetBuilder(
        get_args(),
        tokenizer=get_tokenizer(),
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup)
    ).build()
    
    # Train
    pretrain(
        train_valid_test_datasets=(train_dataset, None, None),
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_backward_func=forward_backward_func
    )

if __name__ == "__main__":
    main()