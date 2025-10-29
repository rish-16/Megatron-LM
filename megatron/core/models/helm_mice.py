"""HELM-MiCE model implementation."""

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.custom_layers.custom_gpt.helm_mice import LorentzDeepSeekV3 as HELMMiCEBlock
from megatron.core.transformer.custom_layers.custom_gpt.hmla import LorentzMLA
from megatron.core.transformer.utils import get_linear_layer

class HELMMiCEModel(MegatronModule):
    """HELM-MiCE model.
    
    Combines hyperbolic attention with mixture of experts in Lorentz space.
    """
    
    def __init__(self, config, num_tokentypes=0, parallel_output=True,
                 pre_process=True, post_process=True, **kwargs):
        super().__init__(config)
        
        args = self.config
        
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.sequence_parallel = args.sequence_parallel
        
        # Embeddings
        if self.pre_process:
            self.embedding = tensor_parallel.VocabParallelEmbedding(
                args.vocab_size,
                args.hidden_size,
                init_method=self.config.init_method
            )
            
        # Transformer
        self.encoder = HELMMiCEBlock(
            dim=kwargs.get('dim', args.hidden_size),
            inter_dim=kwargs.get('inter_dim', args.intermediate_size),
            mice_inter_dim=kwargs.get('mice_inter_dim', args.mice_inter_dim),
            n_layers=kwargs.get('n_layers', args.num_layers),
            n_heads=kwargs.get('n_heads', args.num_attention_heads),
            n_routed_experts=kwargs.get('n_routed_experts', args.num_routed_experts),
            n_shared_experts=kwargs.get('n_shared_experts', args.num_shared_experts),
            n_activated_experts=kwargs.get('n_activated_experts', args.num_activated_experts),
            n_expert_groups=kwargs.get('n_expert_groups', args.num_expert_groups),
            score_func=kwargs.get('score_func', args.score_function),
            route_scale=kwargs.get('route_scale', args.route_scale),
            train_curv=kwargs.get('train_curv', args.train_curvature)
        )
        
        # Output layer
        if self.post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                args.vocab_size,
                gather_output=not self.parallel_output,
                init_method=self.config.init_method
            )
            
    def set_input_tensor(self, input_tensor):
        """Sets input tensor for pipeline parallelism."""
        self.input_tensor = input_tensor
        
    def forward(self, input_ids, position_ids, attention_mask):
        """Forward pass."""
        # Embeddings
        if self.pre_process:
            hidden_states = self.embedding(input_ids)
        else:
            hidden_states = self.input_tensor
            
        # Transformer
        hidden_states = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        # Output
        if self.post_process:
            logits = self.output_layer(hidden_states)
        else:
            logits = hidden_states
            
        return logits
        
    def compute_loss(self, logits, labels, loss_mask):
        """Compute cross entropy loss."""
        losses = tensor_parallel.vocab_parallel_cross_entropy(
            logits.contiguous().float(),
            labels.contiguous()
        )
        
        loss_mask = loss_mask.contiguous().view(-1)
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        
        return loss