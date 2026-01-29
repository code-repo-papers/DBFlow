"""Regression metrics for imputation evaluation."""

from typing import Dict, Optional, List
import torch
import numpy as np


def mae_rmse_on_mask(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: torch.Tensor,
    num_col_idx: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute MAE and RMSE on masked (missing) positions.
    
    Args:
        y_true: Ground truth values (B, D)
        y_pred: Predicted values (B, D)
        mask: Target mask (B, D) - 1 where we compute metrics
        num_col_idx: Optional list of numeric column indices
    
    Returns:
        Dictionary with 'mae' and 'rmse' values
    """
    mask_f = mask.float()
    
    # If numeric column indices specified, only evaluate on those
    if num_col_idx is not None and len(num_col_idx) > 0:
        y_true = y_true[:, num_col_idx]
        y_pred = y_pred[:, num_col_idx]
        mask_f = mask_f[:, num_col_idx]
    
    # Compute differences only on masked positions
    diff = (y_pred - y_true) * mask_f
    
    n_elements = mask_f.sum().item()
    if n_elements == 0:
        return {'mae': 0.0, 'rmse': 0.0}
    
    # MAE
    mae = torch.abs(diff).sum().item() / n_elements
    
    # RMSE
    mse = (diff ** 2).sum().item() / n_elements
    rmse = np.sqrt(mse)
    
    return {'mae': mae, 'rmse': rmse}


def compute_imputation_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    M: np.ndarray,
    num_col_idx: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute imputation metrics on numpy arrays.
    
    Args:
        X_true: Ground truth data (N, D)
        X_pred: Imputed data (N, D)
        M: Missing mask (N, D) - True where missing
        num_col_idx: Optional numeric column indices
    
    Returns:
        Dictionary with 'mae' and 'rmse'
    """
    if num_col_idx is not None and len(num_col_idx) > 0:
        X_true = X_true[:, num_col_idx]
        X_pred = X_pred[:, num_col_idx]
        M = M[:, num_col_idx]
    
    M_float = M.astype(np.float32)
    diff = (X_pred - X_true) * M_float
    
    n_elements = M_float.sum()
    if n_elements == 0:
        return {'mae': 0.0, 'rmse': 0.0}
    
    mae = np.abs(diff).sum() / n_elements
    mse = (diff ** 2).sum() / n_elements
    rmse = np.sqrt(mse)
    
    return {'mae': float(mae), 'rmse': float(rmse)}

