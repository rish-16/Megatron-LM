#!/usr/bin/env python

"""Preprocess RedPajama dataset for hyperbolic transformer training."""

import os
import sys
import argparse
from datasets import load_dataset
from megatron.tokenizer import build_tokenizer
from megatron.data.dataset.custom_gpt.redpajama import RedPajamaDataset
from megatron.core.hypercore.manifolds import Lorentz

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    parser.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])
    parser.add_argument('--tokenizer-type', type=str, required=True,
                       help='Tokenizer type')
    parser.add_argument('--vocab-file', type=str, required=True,
                       help='Vocab file')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Chunk size for processing')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes')
    parser.add_argument('--curvature', type=float, default=-1.0,
                       help='Hyperbolic curvature')
    return parser.parse_args()

def process_redpajama():
    """Download and process RedPajama dataset."""
    args = get_args()
    
    # Initialize tokenizer
    tokenizer = build_tokenizer(args)
    
    # Initialize hyperbolic manifold
    manifold = Lorentz()
    
    # Load RedPajama from HuggingFace
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T")
    
    # Create dataset handler
    redpajama = RedPajamaDataset(
        name='redpajama',
        data_prefix=args.output_prefix,
        documents=dataset,
        indexed_dataset=None,
        num_samples=len(dataset),
        seq_length=2048,  # Adjust based on your model's context length
        seed=42,
        manifold=manifold,
        curvature=args.curvature
    )
    
    # Process and save in Megatron's binary format
    redpajama.process_and_save()

if __name__ == '__main__':
    process_redpajama()