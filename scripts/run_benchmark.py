#!/usr/bin/env python3
"""
Benchmark script for SB-FLOW on real datasets.

Runs multiple seeds to compute mean ± std for MAE, RMSE, and MNAR Score.

Usage:
    python scripts/run_benchmark.py --dataset bean --mask_type MCAR --rate 30 --seeds 3
    python scripts/run_benchmark.py --dataset bean --mask_type MNAR_logistic_T2 --rate 30 --seeds 5
    python scripts/run_benchmark.py --all  # Run all combinations
"""

import os
import sys
import argparse
import json
import yaml
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_single_experiment(
    config_path: str,
    dataset: str,
    mask_type: str,
    rate: int,
    seed: int,
    split_idx: int = 0,
    gpu: str = "0",
) -> dict:
    """
    Run a single experiment with given configuration.
    
    Returns:
        dict with metrics (mae, rmse, mnar_score) or None if failed
    """
    # Create temporary config with overrides
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg['data']['dataname'] = dataset
    cfg['mask']['type'] = mask_type
    cfg['mask']['rate'] = rate
    cfg['mask']['split_idx'] = split_idx
    cfg['seed'] = seed
    
    # Use different save root for each seed
    cfg['train']['save_root'] = f"runs/SBFLOW_benchmark"
    
    # Write temp config
    temp_config = f"/tmp/sbflow_config_{dataset}_{mask_type}_{rate}_{seed}.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(cfg, f)
    
    print(f"\n{'='*60}")
    print(f"Running: {dataset} | {mask_type} | rate={rate} | seed={seed}")
    print(f"{'='*60}")
    
    # Run training
    cmd = [
        sys.executable, "train.py",
        "--config", temp_config,
        "--gpus", gpu,
        "--seed", str(seed),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=False,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"⚠️ Experiment failed with return code {result.returncode}")
            return None
        
        # Load results
        results_path = os.path.join(
            "results", dataset, f"rate{rate}", mask_type, str(split_idx), "metrics.json"
        )
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            print(f"⚠️ Results not found at {results_path}")
            return None
            
    except Exception as e:
        print(f"⚠️ Exception: {e}")
        return None
    finally:
        # Cleanup temp config
        if os.path.exists(temp_config):
            os.remove(temp_config)


def run_benchmark(
    dataset: str,
    mask_type: str,
    rate: int,
    seeds: list,
    config_path: str = "configs/sbflow.yaml",
    gpu: str = "0",
):
    """
    Run benchmark with multiple seeds and aggregate results.
    """
    all_results = []
    
    for seed in seeds:
        result = run_single_experiment(
            config_path=config_path,
            dataset=dataset,
            mask_type=mask_type,
            rate=rate,
            seed=seed,
            gpu=gpu,
        )
        if result is not None:
            all_results.append(result)
    
    if len(all_results) == 0:
        print("❌ No successful runs!")
        return None
    
    # Aggregate metrics
    mae_values = [r['metrics']['out_sample']['mae'] for r in all_results]
    rmse_values = [r['metrics']['out_sample']['rmse'] for r in all_results]
    mnar_values = [r['metrics'].get('mnar_score', 0.0) for r in all_results]
    
    summary = {
        'dataset': dataset,
        'mask_type': mask_type,
        'rate': rate,
        'n_seeds': len(all_results),
        'seeds': seeds[:len(all_results)],
        'mae': {
            'mean': float(np.mean(mae_values)),
            'std': float(np.std(mae_values)),
            'values': mae_values,
        },
        'rmse': {
            'mean': float(np.mean(rmse_values)),
            'std': float(np.std(rmse_values)),
            'values': rmse_values,
        },
        'mnar_score': {
            'mean': float(np.mean(mnar_values)),
            'std': float(np.std(mnar_values)),
            'values': mnar_values,
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    # Print summary
    print("\n" + "="*70)
    print(f"BENCHMARK SUMMARY: {dataset} | {mask_type} | rate={rate}")
    print("="*70)
    print(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'N'}")
    print("-"*70)
    print(f"{'Out-sample MAE':<20} {summary['mae']['mean']:<15.6f} {summary['mae']['std']:<15.6f} {len(all_results)}")
    print(f"{'Out-sample RMSE':<20} {summary['rmse']['mean']:<15.6f} {summary['rmse']['std']:<15.6f} {len(all_results)}")
    print(f"{'MNAR Score S':<20} {summary['mnar_score']['mean']:<15.4f} {summary['mnar_score']['std']:<15.4f} {len(all_results)}")
    print("="*70)
    
    # Save summary
    summary_dir = os.path.join("results", "benchmark_summaries")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{dataset}_{mask_type}_rate{rate}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📁 Summary saved to: {summary_path}")
    
    return summary


def run_all_benchmarks(
    datasets: list = None,
    mask_types: list = None,
    rates: list = None,
    seeds: list = None,
    gpu: str = "0",
):
    """Run full benchmark suite."""
    if datasets is None:
        datasets = ['bean', 'adult', 'default', 'letter']
    if mask_types is None:
        mask_types = ['MCAR', 'MAR', 'MNAR_logistic_T2']
    if rates is None:
        rates = [30]
    if seeds is None:
        seeds = [42, 123, 456]
    
    all_summaries = []
    
    for dataset in datasets:
        for mask_type in mask_types:
            for rate in rates:
                summary = run_benchmark(
                    dataset=dataset,
                    mask_type=mask_type,
                    rate=rate,
                    seeds=seeds,
                    gpu=gpu,
                )
                if summary:
                    all_summaries.append(summary)
    
    # Print final table
    if all_summaries:
        print("\n" + "="*90)
        print("FULL BENCHMARK RESULTS")
        print("="*90)
        print(f"{'Dataset':<12} {'Mask':<18} {'Rate':<6} {'MAE (mean±std)':<20} {'RMSE (mean±std)':<20} {'S':<10}")
        print("-"*90)
        for s in all_summaries:
            mae_str = f"{s['mae']['mean']:.4f}±{s['mae']['std']:.4f}"
            rmse_str = f"{s['rmse']['mean']:.4f}±{s['rmse']['std']:.4f}"
            s_str = f"{s['mnar_score']['mean']:.3f}"
            print(f"{s['dataset']:<12} {s['mask_type']:<18} {s['rate']:<6} {mae_str:<20} {rmse_str:<20} {s_str:<10}")
        print("="*90)
        
        # Save full results
        full_path = os.path.join("results", "benchmark_summaries", "full_benchmark.json")
        with open(full_path, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\n📁 Full results saved to: {full_path}")
    
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="SB-FLOW Benchmark Runner")
    parser.add_argument("--dataset", type=str, default="bean",
                        help="Dataset name")
    parser.add_argument("--mask_type", type=str, default="MCAR",
                        choices=["MCAR", "MAR", "MNAR_logistic_T2", "MNAR_self_pos", "MNAR_self_neg", 
                                 "MNAR_self_pos_s0p5", "MNAR_self_pos_s1p5", "MNAR_self_pos_s2p0", 
                                 "MNAR_self_pos_s3p0"],
                        help="Missing mechanism type")
    parser.add_argument("--rate", type=int, default=30,
                        help="Missing rate (percent)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds to run")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU ID")
    parser.add_argument("--all", action="store_true",
                        help="Run all benchmark combinations")
    parser.add_argument("--config", type=str, default="configs/sbflow.yaml",
                        help="Path to config file")
    
    args = parser.parse_args()
    
    if args.all:
        run_all_benchmarks(gpu=args.gpu, seeds=list(range(args.seeds)))
    else:
        seeds = [42 + i * 100 for i in range(args.seeds)]
        run_benchmark(
            dataset=args.dataset,
            mask_type=args.mask_type,
            rate=args.rate,
            seeds=seeds,
            config_path=args.config,
            gpu=args.gpu,
        )


if __name__ == "__main__":
    main()

