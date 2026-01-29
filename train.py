#!/usr/bin/env python3
"""
SB-FLOW: Mask-Aware Bidirectional Schrödinger Bridge for Tabular Data Imputation

Training entry point.

Usage:
    python train.py --config configs/sbflow.yaml
    python train.py --config configs/sbflow.yaml --gpus 0
"""

import argparse
import os
import yaml

from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.engine.train_loop import train


def main():
    parser = argparse.ArgumentParser(description="Train SB-FLOW model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--gpus", type=str, default=None,
                        help="GPU IDs to use, e.g., '0' or '0,1'")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed from config")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override seed if provided
    if args.seed is not None:
        cfg['seed'] = args.seed
    
    # Set GPU
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using GPUs: {args.gpus}")
    
    # Set random seed
    seed = cfg.get('seed', 1024)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Get logger
    logger = get_logger("sbflow")
    
    # Print config summary
    print("\n" + "="*60)
    print("SB-FLOW: Mask-Aware Bidirectional Schrödinger Bridge")
    print("="*60)
    print(f"Dataset: {cfg['data']['dataname']}")
    print(f"Mask type: {cfg['mask']['type']}, Rate: {cfg['mask']['rate']}%")
    print(f"Lambda_B: {cfg['sbflow']['lambda_B']}, Lambda_cycle: {cfg['sbflow']['lambda_cycle']}")
    print(f"OT alpha: {cfg['sbflow']['ot_alpha']}")
    print("="*60 + "\n")
    
    # Run training
    train(cfg, logger, args)


if __name__ == "__main__":
    main()

