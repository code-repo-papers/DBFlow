#!/bin/bash
# Benchmark SB-FLOW on real data with MNAR_logistic_T2
# Runs multiple splits and aggregates results

set -e

# Configuration
DATANAME=${1:-bean}
RATE=${2:-30}
N_SPLITS=${3:-3}
CONFIG=${4:-configs/sbflow_mnar_t2.yaml}

echo "=============================================="
echo "SB-FLOW Real Data Benchmark"
echo "=============================================="
echo "Dataset: $DATANAME"
echo "Missing Rate: $RATE%"
echo "Number of Splits: $N_SPLITS"
echo "Config: $CONFIG"
echo "=============================================="

# Step 1: Analyze MNAR_logistic_T2 mechanism
echo ""
echo "Step 1: Analyzing MNAR_logistic_T2 selection bias..."
python experiments/analyze_mnar_t2.py \
    --dataname $DATANAME \
    --rate $RATE \
    --compare_all

# Step 2: Run training for each split
echo ""
echo "Step 2: Training on multiple splits..."
for split_idx in $(seq 0 $((N_SPLITS-1))); do
    echo ""
    echo "=== Split $split_idx ==="
    
    # Create temporary config with updated split_idx
    TMP_CONFIG="configs/tmp_split_${split_idx}.yaml"
    cat $CONFIG | sed "s/split_idx: 0/split_idx: $split_idx/" > $TMP_CONFIG
    
    # Also update dataname and rate
    cat $TMP_CONFIG | sed "s/dataname: bean/dataname: $DATANAME/" > ${TMP_CONFIG}.tmp
    mv ${TMP_CONFIG}.tmp $TMP_CONFIG
    cat $TMP_CONFIG | sed "s/rate: 30/rate: $RATE/" > ${TMP_CONFIG}.tmp
    mv ${TMP_CONFIG}.tmp $TMP_CONFIG
    
    python train.py --config $TMP_CONFIG --gpus 0
    
    rm -f $TMP_CONFIG
done

# Step 3: Aggregate results
echo ""
echo "Step 3: Aggregating results..."
python - << 'EOF'
import os
import json
import numpy as np
from glob import glob

dataname = os.environ.get('DATANAME', 'bean')
rate = int(os.environ.get('RATE', '30'))

results_dir = f'results/{dataname}/rate{rate}/MNAR_logistic_T2'
json_files = sorted(glob(f'{results_dir}/*/metrics.json'))

if not json_files:
    print(f"No results found in {results_dir}")
else:
    maes, rmses, mnar_scores = [], [], []
    
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        maes.append(data['metrics']['out_sample']['mae'])
        rmses.append(data['metrics']['out_sample']['rmse'])
        mnar_scores.append(data['metrics']['mnar_score'])
    
    print("\n" + "="*70)
    print(f"BENCHMARK SUMMARY: {dataname} | MNAR_logistic_T2 | rate={rate}")
    print("="*70)
    print(f"{'Metric':<20} {'Mean':<16} {'Std':<16} {'N'}")
    print("-"*70)
    print(f"{'Out-sample MAE':<20} {np.mean(maes):<16.6f} {np.std(maes):<16.6f} {len(maes)}")
    print(f"{'Out-sample RMSE':<20} {np.mean(rmses):<16.6f} {np.std(rmses):<16.6f} {len(rmses)}")
    print(f"{'MNAR Score S':<20} {np.mean(mnar_scores):<16.4f} {np.std(mnar_scores):<16.4f} {len(mnar_scores)}")
    print("="*70)
    
    # Interpretation
    mean_s = np.mean(mnar_scores)
    abs_mean_s = abs(mean_s)
    if abs_mean_s > 0.2:
        direction = "larger" if mean_s > 0 else "smaller"
        print(f"\n⚠️  |S| = {abs_mean_s:.4f} > 0.2: MNAR pattern detected")
        print(f"   ({direction} values more likely to be missing)")
    else:
        print(f"\n✓  |S| = {abs_mean_s:.4f} ≤ 0.2: Consistent with MAR/MCAR")
EOF

echo ""
echo "Benchmark complete!"

