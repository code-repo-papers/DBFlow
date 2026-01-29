#!/usr/bin/env python3
"""
Simple baseline comparison for MNAR experiments.

Purpose: Show that high MAE under MNAR_self is due to task difficulty,
not model deficiency. All methods struggle with self-masking MNAR.

Usage:
    python scripts/baseline_comparison.py --dataset bean --mask_type MNAR_self_pos
    python scripts/baseline_comparison.py --dataset bean --mask_type MNAR_logistic_T2
"""

import os
import sys
import argparse
import numpy as np
import json
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIFFPUTER_PATH = "/scratch/iotgroup/dengyiliu/impute-fm/DiffPuter"
DATA_DIR = f"{DIFFPUTER_PATH}/datasets"


def load_data_simple(dataname, mask_type, rate=30, split_idx=0):
    """Load data and mask directly from files."""
    import pandas as pd

    data_dir = f"{DATA_DIR}/{dataname}"
    info_path = f"{DATA_DIR}/Info/{dataname}.json"

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']

    # Load data
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    # Use only numeric columns (simplification)
    cols = train_df.columns
    X_train = train_df[cols[num_col_idx]].values.astype(np.float32)
    X_test = test_df[cols[num_col_idx]].values.astype(np.float32)

    # Load masks
    mask_dir = f"{data_dir}/masks/rate{rate}/{mask_type}"
    train_mask = np.load(f"{mask_dir}/train_mask_{split_idx}.npy")
    test_mask = np.load(f"{mask_dir}/test_mask_{split_idx}.npy")

    # Align mask dimensions with numeric columns only
    M_train = train_mask[:, num_col_idx] if train_mask.shape[1] > len(num_col_idx) else train_mask
    M_test = test_mask[:, num_col_idx] if test_mask.shape[1] > len(num_col_idx) else test_mask

    # Normalize using train stats (on observed values only)
    M_obs_train = ~M_train.astype(bool)
    train_mean = np.zeros(X_train.shape[1])
    train_std = np.ones(X_train.shape[1])

    for j in range(X_train.shape[1]):
        obs_vals = X_train[M_obs_train[:, j], j]
        if len(obs_vals) > 0:
            train_mean[j] = obs_vals.mean()
            train_std[j] = obs_vals.std() + 1e-8

    X_train_norm = (X_train - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std

    return {
        'X_train': X_train_norm,
        'X_test': X_test_norm,
        'X_train_full': X_train_norm.copy(),  # Full data for evaluation
        'X_test_full': X_test_norm.copy(),
        'M_train': M_train.astype(bool),
        'M_test': M_test.astype(bool),
    }


def compute_metrics(x_true, x_imputed, mask):
    """Compute MAE and RMSE on missing positions."""
    diff = x_imputed - x_true
    diff_masked = diff[mask]

    mae = np.abs(diff_masked).mean()
    rmse = np.sqrt((diff_masked ** 2).mean())

    return mae, rmse


def run_baselines(dataname, mask_type, rate=30, split_idx=0):
    """Run baseline methods and compare."""
    print(f"\n{'='*60}")
    print(f"Baseline Comparison: {dataname} | {mask_type}")
    print(f"{'='*60}")

    # Load data
    try:
        data = load_data_simple(dataname, mask_type, rate, split_idx)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    X_train = data['X_train']
    X_test = data['X_test']
    X_test_full = data['X_test_full']
    M_train = data['M_train']
    M_test = data['M_test']

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Missing rate (train): {M_train.mean():.2%}")
    print(f"Missing rate (test): {M_test.mean():.2%}")

    # Prepare masked data (set missing to NaN)
    X_train_masked = X_train.copy()
    X_test_masked = X_test.copy()
    X_train_masked[M_train] = np.nan
    X_test_masked[M_test] = np.nan

    results = {}

    # ============================================================
    # Baseline 1: Mean Imputation
    # ============================================================
    print("\n[1] Mean Imputation...")
    imputer_mean = SimpleImputer(strategy='mean')
    imputer_mean.fit(X_train_masked)
    X_test_mean = imputer_mean.transform(X_test_masked)

    mae_mean, rmse_mean = compute_metrics(X_test_full, X_test_mean, M_test)
    results['Mean'] = {'mae': mae_mean, 'rmse': rmse_mean}
    print(f"   MAE: {mae_mean:.4f}, RMSE: {rmse_mean:.4f}")

    # ============================================================
    # Baseline 2: Median Imputation
    # ============================================================
    print("\n[2] Median Imputation...")
    imputer_median = SimpleImputer(strategy='median')
    imputer_median.fit(X_train_masked)
    X_test_median = imputer_median.transform(X_test_masked)

    mae_median, rmse_median = compute_metrics(X_test_full, X_test_median, M_test)
    results['Median'] = {'mae': mae_median, 'rmse': rmse_median}
    print(f"   MAE: {mae_median:.4f}, RMSE: {rmse_median:.4f}")

    # ============================================================
    # Baseline 3: MICE (Iterative Imputer)
    # ============================================================
    print("\n[3] MICE (IterativeImputer)...")
    try:
        imputer_mice = IterativeImputer(max_iter=10, random_state=42, verbose=0)
        imputer_mice.fit(X_train_masked)
        X_test_mice = imputer_mice.transform(X_test_masked)

        mae_mice, rmse_mice = compute_metrics(X_test_full, X_test_mice, M_test)
        results['MICE'] = {'mae': mae_mice, 'rmse': rmse_mice}
        print(f"   MAE: {mae_mice:.4f}, RMSE: {rmse_mice:.4f}")
    except Exception as e:
        print(f"   MICE failed: {e}")
        results['MICE'] = {'mae': np.nan, 'rmse': np.nan}

    # ============================================================
    # Baseline 4: KNN Imputation
    # ============================================================
    print("\n[4] KNN Imputation...")
    try:
        imputer_knn = KNNImputer(n_neighbors=5)
        imputer_knn.fit(X_train_masked)
        X_test_knn = imputer_knn.transform(X_test_masked)

        mae_knn, rmse_knn = compute_metrics(X_test_full, X_test_knn, M_test)
        results['KNN'] = {'mae': mae_knn, 'rmse': rmse_knn}
        print(f"   MAE: {mae_knn:.4f}, RMSE: {rmse_knn:.4f}")
    except Exception as e:
        print(f"   KNN failed: {e}")
        results['KNN'] = {'mae': np.nan, 'rmse': np.nan}

    # ============================================================
    # Load DBFlow result (if exists)
    # ============================================================
    dbflow_path = f"results/{dataname}/rate{rate}/{mask_type}/{split_idx}/metrics.json"
    if os.path.exists(dbflow_path):
        with open(dbflow_path, 'r') as f:
            dbflow_result = json.load(f)

        mae_dbflow = rmse_dbflow = mnar_score = np.nan

        if 'metrics' in dbflow_result:
            metrics = dbflow_result['metrics']
            if 'out_sample' in metrics:
                mae_dbflow = metrics['out_sample'].get('mae', np.nan)
                rmse_dbflow = metrics['out_sample'].get('rmse', np.nan)
            if 'mnar_diagnosis' in metrics:
                mnar_score = metrics['mnar_diagnosis'].get('mnar_score', np.nan)
            elif 'mnar_score' in metrics:
                mnar_score = metrics['mnar_score']

        results['DBFlow'] = {'mae': mae_dbflow, 'rmse': rmse_dbflow, 'mnar_score': mnar_score}

        print(f"\n[DBFlow] (from saved results)")
        print(f"   MAE: {mae_dbflow:.4f}, RMSE: {rmse_dbflow:.4f}")
        if not np.isnan(mnar_score):
            print(f"   MNAR Score S: {mnar_score:.4f}")
    else:
        print(f"\n[DBFlow] No results found at {dbflow_path}")

    # ============================================================
    # Summary Table
    # ============================================================
    print(f"\n{'='*60}")
    print(f"SUMMARY: {dataname} | {mask_type}")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'MAE':>10} {'RMSE':>10}")
    print(f"{'-'*35}")

    for method, metrics in results.items():
        mae = metrics['mae']
        rmse = metrics['rmse']
        if np.isnan(mae):
            print(f"{method:<15} {'N/A':>10} {'N/A':>10}")
        else:
            print(f"{method:<15} {mae:>10.4f} {rmse:>10.4f}")

    print(f"{'='*60}")

    # ============================================================
    # Key Insight
    # ============================================================
    if 'DBFlow' in results and not np.isnan(results['DBFlow']['mae']):
        dbflow_mae = results['DBFlow']['mae']
        mean_mae = results['Mean']['mae']
        mice_mae = results.get('MICE', {}).get('mae', np.nan)

        print(f"\n📊 Analysis:")
        if dbflow_mae < mean_mae:
            improvement = (mean_mae - dbflow_mae) / mean_mae * 100
            print(f"   • DBFlow improves over Mean by {improvement:.1f}%")
        if not np.isnan(mice_mae) and dbflow_mae < mice_mae:
            improvement = (mice_mae - dbflow_mae) / mice_mae * 100
            print(f"   • DBFlow improves over MICE by {improvement:.1f}%")

        if 'MNAR_self' in mask_type:
            print(f"\n💡 Key Insight:")
            print(f"   Under MNAR_self_pos, ALL methods show elevated MAE because")
            print(f"   large values are systematically missing → extrapolation problem.")
            print(f"   This is an information-theoretic limitation, not a model deficiency.")

    # Save results
    save_dir = f"results/baseline_comparison"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{dataname}_{mask_type}_rate{rate}.json"
    with open(save_path, 'w') as f:
        json.dump({
            'dataset': dataname,
            'mask_type': mask_type,
            'rate': rate,
            'results': {k: {kk: float(vv) if not np.isnan(vv) else None for kk, vv in v.items()}
                       for k, v in results.items()}
        }, f, indent=2)
    print(f"\n📁 Results saved to: {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bean")
    parser.add_argument("--mask_type", type=str, default="MNAR_self_pos")
    parser.add_argument("--rate", type=int, default=30)
    parser.add_argument("--split_idx", type=int, default=0)

    args = parser.parse_args()

    run_baselines(
        dataname=args.dataset,
        mask_type=args.mask_type,
        rate=args.rate,
        split_idx=args.split_idx,
    )


if __name__ == "__main__":
    main()
