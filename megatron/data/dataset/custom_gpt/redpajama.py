"""RedPajama dataset handler for hyperbolic transformer training."""

import os
import torch
from typing import Dict, List
from datasets import load_dataset
from megatron.core import mpu
from megatron.data.dataset import MegatronDataset
from megatron.data.indexed_dataset import MMapIndexedDataset

class RedPajamaDataset(MegatronDataset):
    """Dataset handler for RedPajama corpus with hyperbolic preprocessing."""
    
    def __init__(self, name, data_prefix, documents, indexed_dataset,
                 num_samples, seq_length, seed, **kwargs):
        super().__init__(name, data_prefix, documents, indexed_dataset,
                        num_samples, seq_length, seed)
                        
        # Add hyperbolic-specific preprocessing configs
        self.manifold = kwargs.get('manifold', None)
        self.curvature = kwargs.get('curvature', -1.0)
        
    def _preprocess_text(self, text: str) -> torch.Tensor:
        """Apply hyperbolic preprocessing to text.
        
        This ensures the embeddings are properly mapped to hyperbolic space
        before being fed to the model.
        """
        # First apply standard tokenization
        tokens = self.tokenizer.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # If using hyperbolic embeddings, map to the appropriate manifold
        if self.manifold is not None:
            # Map to hyperbolic space using your manifold
            tokens = self.manifold.expmap0(tokens, c=self.curvature)
            
        return tokens
        
    def __getitem__(self, idx):
        """Get preprocessed item from dataset."""
        # Get raw item using parent class method
        item = super().__getitem__(idx)
        
        # Apply hyperbolic preprocessing if needed
        if isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, str):
                    item[k] = self._preprocess_text(v)
        elif isinstance(item, str):
            item = self._preprocess_text(item)
            
        return item