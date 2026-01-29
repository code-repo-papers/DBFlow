"""
Training loop for SB-FLOW.

Features:
- Per-batch MA-OT coupling computation (ensures proper optimal transport pairing)
- Per-batch cycle consistency
- EMA, early stopping, checkpoint saving
- Final imputation with metrics
"""

from typing import Dict, Any, Optional
import os
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.data.dataset import build_dataloaders_from_cfg, load_sbflow_dataset
from src.models.sbflow_model import SBFlowModel
from src.losses.sbflow_loss import sbflow_loss
from src.ot.mask_aware_ot import MaskAwareOTSampler, create_source_distribution, check_pot_available, get_available_backends
from src.metrics.regression import mae_rmse_on_mask
from src.theory.missing_mechanism import MNARDetector
from src.theory.mnar_score import compute_mnar_score, interpret_mnar_score


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    @torch.no_grad()
    def store(self, model: torch.nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
    
    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def build_model(cfg: Dict[str, Any], in_dim: int) -> SBFlowModel:
    """Build BiFlow model from config."""
    mcfg = cfg.get('model', {})
    return SBFlowModel(
        d_in=in_dim,
        d_model=int(mcfg.get('d_model', 512)),
        nlayers=int(mcfg.get('nlayers', 3)),
        dropout=float(mcfg.get('dropout', 0.1)),
        mlp_mode=str(mcfg.get('mlp_mode', 'macfm')),  # "macfm" (recommended) or "dynamic"
        use_mask_input=bool(mcfg.get('use_mask_input', False)),  # Mask as input (experimental)
    )


def train(cfg: Dict[str, Any], logger=None, args=None):
    """
    Main training function for SB-FLOW.
    
    Args:
        cfg: Configuration dictionary
        logger: Optional logger
        args: Optional command line arguments
    """
    # Check OT backend availability
    backends = get_available_backends()
    print(f"✓ OT backends available: {[k for k, v in backends.items() if v and k != 'recommended']}")
    print(f"  Recommended backend: {backends['recommended']}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Build dataloaders
    dl_train, dl_val, train_ds = build_dataloaders_from_cfg(cfg)
    in_dim = train_ds.X.shape[1]
    print(f"Input dimension: {in_dim}")
    
    # Build model
    model = build_model(cfg, in_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Training config
    train_cfg = cfg.get('train', {})
    lr = float(train_cfg.get('lr', 1e-4))
    weight_decay = float(train_cfg.get('weight_decay', 0.0))
    grad_clip = float(train_cfg.get('grad_clip', 1.0))
    use_amp = bool(train_cfg.get('amp', True))
    max_epochs = int(train_cfg.get('max_epochs', 1000))
    save_every = int(train_cfg.get('save_every', 100))
    
    # SB-FLOW specific config
    sbflow_cfg = cfg.get('sbflow', {})
    lambda_B = float(sbflow_cfg.get('lambda_B', 0.5))
    lambda_cycle = float(sbflow_cfg.get('lambda_cycle', 0.1))
    source_sigma = float(sbflow_cfg.get('source_sigma', 1.0))
    ot_enabled = bool(sbflow_cfg.get('ot_enabled', False))  # OT is slow, disable by default
    ot_alpha = float(sbflow_cfg.get('ot_alpha', 0.1))
    ot_reg = float(sbflow_cfg.get('ot_reg', 0.05))
    cycle_steps = int(sbflow_cfg.get('cycle_steps', 10))
    path_schedule = str(train_cfg.get('path_schedule', 'linear'))
    time_beta = tuple(train_cfg.get('time_beta', [1.0, 1.0]))
    
    # NEW: MACFM-style regularization (improves generalization significantly)
    lambda_stab = float(train_cfg.get('lambda_stab', 0.15))      # Stabilization on observed
    lambda_cons = float(train_cfg.get('lambda_cons', 0.01))      # Consistency regularization
    eta_cons = float(train_cfg.get('eta_cons', 0.05))            # Perturbation scale
    sigma_in = float(train_cfg.get('sigma_in', 0.05))            # Input noise on observed
    augment_target_p = float(train_cfg.get('augment_target_p', 0.15))  # Mask augmentation
    
    # Save directory
    dataname = cfg['data'].get('dataname', 'dataset')
    mask_type = cfg['mask'].get('type', 'MCAR')
    rate = int(cfg['mask'].get('rate', 30))
    split_idx = int(cfg['mask'].get('split_idx', 0))
    save_root = train_cfg.get('save_root', 'runs/SBFLOW')
    save_dir = os.path.join(save_root, dataname, f"rate{rate}", mask_type, str(split_idx))
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")
    
    # Optimizer and scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50, verbose=False)
    scaler = GradScaler(enabled=use_amp)
    
    # EMA
    ema_cfg = train_cfg.get('ema', {})
    use_ema = bool(ema_cfg.get('enabled', True))
    ema_decay = float(ema_cfg.get('decay', 0.999))
    ema = EMA(model, decay=ema_decay) if use_ema else None
    
    # Early stopping
    es_cfg = train_cfg.get('early_stopping', {})
    es_enabled = bool(es_cfg.get('enabled', True))
    es_patience = int(es_cfg.get('patience', 200))
    es_min_delta = float(es_cfg.get('min_delta', 0.0001))
    
    # OT Sampler (per-batch coupling for proper optimal transport)
    # Now supports GPU-accelerated backends for fast training
    ot_backend = str(sbflow_cfg.get('ot_backend', 'auto'))
    ot_sampler = None
    if ot_enabled:
        ot_sampler = MaskAwareOTSampler(
            alpha=ot_alpha,
            reg=ot_reg,
            source_sigma=source_sigma,
            backend=ot_backend,
        )
        print(f"✓ MA-OT coupling enabled (backend: {ot_sampler.backend})")
    else:
        print("✓ MA-OT coupling disabled (using random pairing)")
    
    # Training state
    best_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None
    
    # Training loop
    pbar = tqdm(total=max_epochs, desc="Training")
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        
        # Note: We now compute MA-OT coupling per mini-batch instead of per-epoch
        # This ensures proper optimal transport pairing without index remapping issues
        
        epoch_loss = 0.0
        epoch_loss_F = 0.0
        epoch_loss_B = 0.0
        epoch_loss_cycle = 0.0
        n_batches = 0
        
        for batch_idx, (x, m_obs, m_cond, m_target) in enumerate(dl_train):
            x = x.to(device)
            m_obs = m_obs.to(device)
            m_target = m_target.to(device)
            
            # Compute MA-OT coupling within this mini-batch (if enabled)
            # OT coupling is slow due to CPU-based Sinkhorn, so it's disabled by default
            coupling_idx = None
            if ot_sampler is not None:
                with torch.no_grad():
                    # Create source distribution for this batch
                    noise = torch.randn_like(x) * source_sigma
                    x0_batch = x * m_obs.float() + noise * m_target.float()
                    x1_batch = x  # Target is clean data
                    
                    # Compute batch-local MA-OT coupling
                    coupling_idx = ot_sampler.compute_batch_coupling(
                        x0_batch, x1_batch, m_target, m_obs
                    )
            
            opt.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=use_amp):
                loss, loss_dict = sbflow_loss(
                    model=model,
                    x_clean=x,
                    m_obs=m_obs,
                    m_target=m_target,
                    coupling_indices=coupling_idx,
                    lambda_B=lambda_B,
                    lambda_cycle=lambda_cycle,
                    source_sigma=source_sigma,
                    path_schedule=path_schedule,
                    time_beta=time_beta,
                    cycle_steps=cycle_steps,
                    # NEW: MACFM-style regularization
                    lambda_stab=lambda_stab,
                    lambda_cons=lambda_cons,
                    eta_cons=eta_cons,
                    sigma_in=sigma_in,
                    augment_target_p=augment_target_p,
                )
            
            scaler.scale(loss).backward()
            
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(opt)
            scaler.update()
            
            if use_ema and ema is not None:
                ema.update(model)
            
            epoch_loss += loss_dict['total']
            epoch_loss_F += loss_dict['loss_F']
            epoch_loss_B += loss_dict['loss_B']
            epoch_loss_cycle += loss_dict['loss_cycle']
            n_batches += 1
        
        # Average losses
        epoch_loss /= max(n_batches, 1)
        epoch_loss_F /= max(n_batches, 1)
        epoch_loss_B /= max(n_batches, 1)
        epoch_loss_cycle /= max(n_batches, 1)
        
        # Scheduler step
        scheduler.step(epoch_loss)
        
        # Check for improvement
        improved = (best_loss - epoch_loss) > es_min_delta
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Save best model
            if use_ema and ema is not None:
                ema.store(model)
                ema.copy_to(model)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                torch.save({'model': best_state, 'epoch': epoch}, os.path.join(save_dir, 'best.ckpt'))
                ema.restore(model)
            else:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                torch.save({'model': best_state, 'epoch': epoch}, os.path.join(save_dir, 'best.ckpt'))
        else:
            epochs_no_improve += 1
        
        # Periodic checkpoint
        if save_every > 0 and epoch % save_every == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch}, 
                      os.path.join(save_dir, f'epoch_{epoch}.ckpt'))
        
        # Update progress bar
        pbar.set_postfix_str(
            f"loss={epoch_loss:.4f} F={epoch_loss_F:.4f} B={epoch_loss_B:.4f} "
            f"cyc={epoch_loss_cycle:.4f} best={best_loss:.4f} no_imp={epochs_no_improve}/{es_patience}"
        )
        pbar.update(1)
        
        # Early stopping
        if es_enabled and epochs_no_improve >= es_patience:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            print(f"   Best loss: {best_loss:.6f} at epoch {best_epoch}")
            break
    
    pbar.close()
    
    # Load best model for evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"\nTraining finished. Best loss: {best_loss:.6f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {os.path.abspath(save_dir)}")
    
    # Final imputation and evaluation
    _run_final_imputation(cfg, model, train_ds, device, save_dir)


def _compute_mnar_score_direct(
    model: SBFlowModel,
    X: np.ndarray,
    M: np.ndarray,
    info: Dict[str, Any],
    device: str,
    steps: int = 50,
    source_sigma: float = 1.0,
    solver: str = "heun",
    batch_size: int = 2048,
    scale_factor: float = 1.0,
    threshold: float = 0.15,
):
    """
    Compute MNAR diagnostic score S using SINGLE TRIAL, NO RESAMPLE.

    Uses unified mnar_score module for consistent computation.

    This method directly imputes with the model (single trial, no resample)
    and computes S = E[x²|missing] / E[x²|observed] - 1.

    Key insight:
    - Under MCAR: Model learns unbiased distribution, E[x²|miss] ≈ 0.6-0.7
    - Under MNAR: Model learns from biased observations, E[x²|miss] differs
    - |S| > threshold indicates potential MNAR

    Args:
        model: Trained SBFlowModel
        X: Raw data (not imputed yet)
        M: Missing mask (True = missing)
        info: Dataset info with mean/std
        device: Device
        steps: ODE integration steps
        source_sigma: Initial noise std
        solver: ODE solver
        batch_size: Batch size for processing
        scale_factor: Scaling factor for normalization
        threshold: MNAR detection threshold

    Returns:
        MNARScoreResult with full diagnostic information
    """
    model.eval()

    mean = torch.from_numpy(info['mean']).to(device)
    std = torch.from_numpy(info['std']).to(device)

    N = X.shape[0]

    # Collect all imputed data for unified MNAR score computation
    all_x_pred = []
    all_m_obs = []
    all_m_target = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            x_batch = torch.from_numpy(X[start:end]).float().to(device)
            m_batch = torch.from_numpy(M[start:end]).to(device)

            # Normalize (match training preprocessing)
            x_norm = (x_batch - mean) / (std + 1e-8)
            if scale_factor != 1.0:
                x_norm = x_norm / scale_factor

            m_obs = (~m_batch).float()
            m_target = m_batch.float()
            x_obs = x_norm * m_obs

            # Impute with SINGLE TRIAL, NO RESAMPLE for accurate MNAR detection
            x_pred = model.impute(
                x_obs=x_obs,
                m_obs=m_obs.bool(),
                m_target=m_target.bool(),
                steps=steps,
                sigma=source_sigma,
                solver=solver,
                resample_enabled=False,  # IMPORTANT: No resample for MNAR detection
            )

            all_x_pred.append(x_pred)
            all_m_obs.append(m_obs)
            all_m_target.append(m_target)

    # Concatenate all batches
    x_pred_all = torch.cat(all_x_pred, dim=0)
    m_obs_all = torch.cat(all_m_obs, dim=0)
    m_target_all = torch.cat(all_m_target, dim=0)

    # Use unified MNAR score computation
    mnar_result = compute_mnar_score(
        x_pred_all, m_obs_all.bool(), m_target_all.bool(),
        normalize=True,
        threshold=threshold,
    )

    # Print diagnostic info
    print(f"   E[x²|missing] = {mnar_result.e_x2_missing:.4f}")
    print(f"   E[x²|observed] = {mnar_result.e_x2_observed:.4f}")
    print(f"   Selection Bias Ratio = {mnar_result.selection_bias_ratio:.4f}")
    print(f"   Confidence: {mnar_result.confidence}")

    return mnar_result


def _run_final_imputation(
    cfg: Dict[str, Any],
    model: SBFlowModel,
    train_ds,
    device: str,
    save_dir: str,
):
    """Run final imputation on train/test sets and compute metrics."""
    print("\n" + "="*60)
    print("Running final imputation...")
    print("="*60)
    
    # Load raw data
    data_cfg = cfg['data']
    mask_cfg = cfg['mask']
    
    X_train, X_test, M_train, M_test, info = load_sbflow_dataset(
        data_root=data_cfg.get('data_root', 'datasets'),
        dataname=data_cfg['dataname'],
        mask_type=mask_cfg.get('type', 'MCAR'),
        rate=int(mask_cfg.get('rate', 30)),
        split_idx=int(mask_cfg.get('split_idx', 0)),
    )
    
    mean = torch.from_numpy(info['mean']).to(device)
    std = torch.from_numpy(info['std']).to(device)
    num_num = info.get('num_num', X_train.shape[1])
    
    # Imputation config
    fm_cfg = cfg.get('fm', {})
    steps = int(fm_cfg.get('steps', 50))
    trials = int(fm_cfg.get('trials', 10))
    solver = str(fm_cfg.get('solver', 'heun'))
    aggregation = str(fm_cfg.get('aggregation', 'single'))
    
    # Resample config - DISABLED by default for accurate MNAR detection
    # MAE/RMSE uses trials=50 + mean aggregation
    # MNAR Score computed separately with single trial, no resample
    resample_cfg = fm_cfg.get('resample', {})
    resample_enabled = bool(resample_cfg.get('enabled', False))  # Disabled for MNAR detection
    resample_interval = int(resample_cfg.get('interval', 5))
    resample_sigma = float(resample_cfg.get('sigma', 0.6))
    
    sbflow_cfg = cfg.get('sbflow', {})
    source_sigma = float(sbflow_cfg.get('source_sigma', 1.0))
    
    model.eval()
    
    # Get robust normalization params (match training)
    clip_outliers = float(data_cfg.get('clip_outliers', 0.0))
    scale_factor = float(data_cfg.get('scale_factor', 1.0))
    
    def impute_dataset(X: np.ndarray, M: np.ndarray, batch_size: int = 1024, desc: str = "Imputing"):
        """Impute a full dataset."""
        N, D = X.shape
        X_imputed = np.array(X, copy=True)
        
        sum_abs, sum_sq, total = 0.0, 0.0, 0
        n_batches = (N + batch_size - 1) // batch_size
        
        print(f"\n{desc}: {N} samples, {trials} trials, {steps} steps...")
        
        with torch.no_grad():
            for batch_i, start in enumerate(range(0, N, batch_size)):
                if batch_i % 5 == 0:  # Progress every 5 batches
                    print(f"  Batch {batch_i+1}/{n_batches}...", end='\r')
                end = min(start + batch_size, N)
                
                x_batch = torch.from_numpy(X[start:end]).to(device)
                m_batch = torch.from_numpy(M[start:end]).to(device)
                
                # Normalize (match training: z-score, then optional clip/scale)
                x_norm = (x_batch - mean) / (std + 1e-8)
                
                # Apply same robust normalization as training
                if clip_outliers > 0:
                    # Use training set's percentiles for clipping
                    lower = torch.from_numpy(
                        np.percentile((X_train - info['mean']) / (info['std'] + 1e-8), clip_outliers * 100, axis=0).astype(np.float32)
                    ).to(device)
                    upper = torch.from_numpy(
                        np.percentile((X_train - info['mean']) / (info['std'] + 1e-8), (1 - clip_outliers) * 100, axis=0).astype(np.float32)
                    ).to(device)
                    x_norm = torch.clamp(x_norm, lower, upper)
                
                if scale_factor != 1.0:
                    x_norm = x_norm / scale_factor
                
                m_obs = (~m_batch).float()
                m_target = m_batch.float()
                
                # Impute with configured settings
                # aggregation="single" preserves variance (recommended)
                # resample_enabled adds noise during ODE for additional variance preservation
                x_pred = model.impute_with_trials(
                    x_obs=x_norm,
                    m_obs=m_obs.bool(),
                    m_target=m_target.bool(),
                    steps=steps,
                    trials=trials,
                    sigma=source_sigma,
                    solver=solver,
                    aggregation=aggregation,
                    resample_enabled=resample_enabled,
                    resample_interval=resample_interval,
                    resample_sigma=resample_sigma,
                )
                
                # Compute metrics in normalized space
                if num_num > 0:
                    diff = (x_pred[:, :num_num] - x_norm[:, :num_num]) * m_target[:, :num_num]
                    active = m_target[:, :num_num]
                else:
                    diff = (x_pred - x_norm) * m_target
                    active = m_target
                
                sum_abs += torch.abs(diff).sum().item()
                sum_sq += (diff ** 2).sum().item()
                total += active.sum().item()
                
                # Denormalize and store (reverse scale_factor first, then std/mean)
                x_denorm = x_pred
                if scale_factor != 1.0:
                    x_denorm = x_denorm * scale_factor  # Reverse scaling
                x_denorm = x_denorm * (std + 1e-8) + mean
                x_denorm_np = x_denorm.cpu().numpy()
                m_np = m_batch.cpu().numpy()
                X_imputed[start:end][m_np] = x_denorm_np[m_np]
        
        mae = sum_abs / max(total, 1)
        rmse = np.sqrt(sum_sq / max(total, 1))
        
        return X_imputed, mae, rmse
    
    # Impute train and test
    dataname = cfg['data']['dataname']
    
    # Handle news dataset edge case
    if dataname == 'news' and X_test.shape[0] > 6265:
        X_test = np.delete(X_test, 6265, axis=0)
        M_test = np.delete(M_test, 6265, axis=0)
    
    X_train_imputed, mae_train, rmse_train = impute_dataset(X_train, M_train, desc="Imputing train set")
    X_test_imputed, mae_test, rmse_test = impute_dataset(X_test, M_test, desc="Imputing test set")
    
    # ===== Compute MNAR Score S on test set =====
    # IMPORTANT: Use single trial, NO resample for accurate MNAR detection
    print("\nComputing MNAR diagnostic score (single trial, no resample)...")
    mnar_result = _compute_mnar_score_direct(
        model, X_test, M_test, info, device,
        steps=steps, source_sigma=source_sigma, solver=solver,
        scale_factor=scale_factor,
        threshold=0.15,
    )

    # Save results
    mask_type = cfg['mask'].get('type', 'MCAR')
    rate = int(cfg['mask'].get('rate', 30))
    split_idx = int(cfg['mask'].get('split_idx', 0))
    results_dir = os.path.join('results', dataname, f'rate{rate}', mask_type, str(split_idx))
    os.makedirs(results_dir, exist_ok=True)

    train_csv = os.path.join(results_dir, 'imputed_train.csv')
    test_csv = os.path.join(results_dir, 'imputed_test.csv')

    pd.DataFrame(X_train_imputed).to_csv(train_csv, index=False)
    pd.DataFrame(X_test_imputed).to_csv(test_csv, index=False)

    # Print results
    print(f"\nSaved imputed train CSV to: {os.path.abspath(train_csv)}")
    print(f"Saved imputed test CSV to: {os.path.abspath(test_csv)}")
    print(f"\n{'='*60}")
    print("FINAL RESULTS (Normalized Space)")
    print(f"{'='*60}")
    print(f"In-sample  MAE:  {mae_train:.6f}")
    print(f"In-sample  RMSE: {rmse_train:.6f}")
    print(f"Out-of-sample MAE:  {mae_test:.6f}")
    print(f"Out-of-sample RMSE: {rmse_test:.6f}")
    print(f"{'='*60}")
    print(f"MNAR Diagnostic Score S (test): {mnar_result.mnar_score:.4f}")
    if mnar_result.is_mnar_detected:
        print(f"⚠️  |S| = {abs(mnar_result.mnar_score):.4f} > {mnar_result.threshold}: MNAR detected!")
        print(f"   Direction: {mnar_result.direction}")
        print(f"   Confidence: {mnar_result.confidence}")
        if mnar_result.direction == "positive":
            print(f"   Interpretation: Larger values are more likely to be missing")
        elif mnar_result.direction == "negative":
            print(f"   Interpretation: Smaller values are more likely to be missing")
    else:
        print(f"✓  |S| = {abs(mnar_result.mnar_score):.4f} ≤ {mnar_result.threshold}: Consistent with MCAR/MAR")
    print(f"{'='*60}")
    print(f"Results directory: {os.path.abspath(results_dir)}")
    
    # Save metrics JSON
    metrics = {
        'seed': cfg.get('seed', None),
        'dataset': dataname,
        'mask': {'type': mask_type, 'rate': rate, 'split_idx': split_idx},
        'metrics': {
            'in_sample': {'mae': mae_train, 'rmse': rmse_train},
            'out_sample': {'mae': mae_test, 'rmse': rmse_test},
            'mnar_diagnosis': mnar_result.to_dict(),  # Full MNAR diagnostic info
        },
        'config': {
            'lambda_B': cfg.get('sbflow', {}).get('lambda_B', 0.5),
            'lambda_cycle': cfg.get('sbflow', {}).get('lambda_cycle', 0.1),
            'ot_alpha': cfg.get('sbflow', {}).get('ot_alpha', 0.1),
        },
        'save_dir': os.path.abspath(save_dir),
        'results_dir': os.path.abspath(results_dir),
        'timestamp': int(time.time()),
    }

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

