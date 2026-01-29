"""
Missing Mechanism Analysis via Backward Bridge

This module implements MNAR detection based on the theoretical framework
in docs/theory.md.

Key Results (from toy model analysis):
======================================

Proposition 1: For backward velocity on missing positions,
    E[||v_B^miss||^2 | x, m] = ||x_miss||^2 + σ²|m|

Proposition 2: Under MNAR, there is correlation between ||v_B^miss||^2
and ||x_miss||^2 that doesn't exist under MAR.

Proposition 3: The MNAR score S = Corr(||v_B^miss||^2, ||x_miss||^2) is
approximately 0 under MAR and non-zero under MNAR.

Implementation Notes:
====================
- MNARDetector: Uses unified mnar_score module for selection bias detection
- HeuristicPropensityEstimator: Heuristic estimator motivated by the theory
- All "propensity estimation" is labeled as heuristic, not identification

NOTE: Core MNAR score computation has been moved to mnar_score.py for consistency.
This module provides additional analysis via backward velocity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

# Import unified MNAR score computation
from .mnar_score import (
    compute_mnar_score,
    compute_mnar_score_batched,
    compute_feature_wise_mnar,
    MNARScoreResult,
)


# Keep MNARTestResult for backward compatibility, but prefer MNARScoreResult
@dataclass
class MNARTestResult:
    """Result of MNAR detection test (legacy, use MNARScoreResult instead)."""
    mnar_score: float  # Correlation statistic
    is_mnar_detected: bool  # Whether |S| > threshold
    threshold: float  # Detection threshold used
    details: Dict[str, float]  # Additional statistics


class MNARDetector:
    """
    MNAR Detection via Backward Bridge Velocity Analysis.
    
    Based on Proposition 2-3 from the theoretical framework:
    Under MNAR, the backward velocity magnitude on missing positions
    correlates with the actual missing values in a way that doesn't
    occur under MAR.
    
    Test statistic: S = Corr(||v_B^miss||^2, ||x_miss||^2)
    - Under MAR: E[S] ≈ 0 (no selection bias)
    - Under MNAR: E[S] ≠ 0 (selection bias creates correlation)
    """
    
    def __init__(
        self, 
        model, 
        sigma: float = 1.0,
        t_eval: float = 0.99,
        threshold: float = 0.15,
    ):
        """
        Args:
            model: Trained SBFlowModel with backward_bridge
            sigma: Source distribution noise scale
            t_eval: Time point for velocity evaluation (close to 1)
            threshold: Detection threshold for |S|
        """
        self.model = model
        self.sigma = sigma
        self.t_eval = t_eval
        self.threshold = threshold
    
    @torch.no_grad()
    def compute_backward_velocity_stats(
        self,
        x: torch.Tensor,
        m_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute backward velocity magnitude on missing positions.
        
        From Proposition 1:
        E[||v_B^miss||^2 | x, m] = ||x_miss||^2 + σ²|m|
        
        Args:
            x: Complete/imputed data (B, D)
            m_target: Missing mask (B, D), 1 = missing
        
        Returns:
            v_magnitude: ||v_B^miss||^2 per sample (B,)
            x_magnitude: ||x_miss||^2 per sample (B,)
        """
        B, D = x.shape
        device = x.device
        
        m_obs = 1.0 - m_target.float()
        t = torch.full((B, 1), self.t_eval, device=device)
        
        # Get backward velocity
        v_back = self.model.backward_bridge(x, m_obs, m_obs, t)
        
        # Velocity magnitude on missing positions only
        v_miss = v_back * m_target.float()
        v_magnitude = (v_miss ** 2).sum(dim=-1)  # (B,)
        
        # Missing values magnitude
        x_miss = x * m_target.float()
        x_magnitude = (x_miss ** 2).sum(dim=-1)  # (B,)
        
        return v_magnitude, x_magnitude
    
    @torch.no_grad()
    def compute_mnar_score(
        self,
        x: torch.Tensor,
        m_target: torch.Tensor,
    ) -> MNARTestResult:
        """
        Compute MNAR detection score using Selection Bias Detection.

        Uses unified mnar_score module for core computation, adds backward
        velocity analysis for additional diagnostics.

        MNAR Score:
            S = E[x² | missing] / E[x² | observed] - 1
            S ≈ 0 under MCAR/MAR
            |S| > threshold suggests MNAR (either direction!)

        Args:
            x: Complete/imputed data (B, D)
            m_target: Missing mask (B, D), 1 = missing

        Returns:
            MNARTestResult with score and detection decision
        """
        B, D = x.shape
        m_obs = 1.0 - m_target.float()

        # ========== Use unified MNAR score computation ==========
        mnar_result = compute_mnar_score(
            x, m_obs.bool(), m_target.bool(),
            normalize=True,
            threshold=self.threshold,
        )

        # ========== Additional: Per-dimension analysis ==========
        feature_analysis = compute_feature_wise_mnar(x, m_obs.bool(), m_target.bool())

        # ========== Additional: Backward velocity stats ==========
        v_mag, x_mag = self.compute_backward_velocity_stats(x, m_target)
        v_centered = v_mag - v_mag.mean()
        x_centered = x_mag - x_mag.mean()
        overall_correlation = (
            (v_centered * x_centered).sum() /
            (torch.sqrt((v_centered ** 2).sum() * (x_centered ** 2).sum()) + 1e-8)
        ).item()

        # Determine bias direction
        if mnar_result.mnar_score > self.threshold:
            bias_direction = "positive (larger values missing)"
        elif mnar_result.mnar_score < -self.threshold:
            bias_direction = "negative (smaller values missing)"
        else:
            bias_direction = "none (consistent with MAR/MCAR)"

        details = {
            'selection_bias_ratio': mnar_result.selection_bias_ratio,
            'mnar_score': mnar_result.mnar_score,
            'abs_mnar_score': abs(mnar_result.mnar_score),
            'bias_direction': bias_direction,
            'e_x2_missing': mnar_result.e_x2_missing,
            'e_x2_observed': mnar_result.e_x2_observed,
            'mean_dim_bias': feature_analysis['mean_score'],
            'mean_abs_dim_bias': np.mean(np.abs(feature_analysis['feature_scores'][feature_analysis['feature_valid']])) if feature_analysis['n_valid_features'] > 0 else 0.0,
            'std_dim_bias': feature_analysis['std_score'],
            'n_dims_analyzed': feature_analysis['n_valid_features'],
            'overall_correlation': overall_correlation,
            'n_missing_total': mnar_result.n_missing,
            'n_observed_total': mnar_result.n_observed,
            'confidence': mnar_result.confidence,
        }

        return MNARTestResult(
            mnar_score=mnar_result.mnar_score,
            is_mnar_detected=mnar_result.is_mnar_detected,
            threshold=self.threshold,
            details=details,
        )


class HeuristicPropensityEstimator:
    """
    Heuristic Propensity Estimator based on Backward Velocity.
    
    WARNING: This is a HEURISTIC estimator, NOT a theoretically justified
    identification procedure. The relationship between backward velocity
    and propensity is motivated by intuition (see Section 6 of theory.md)
    but NOT proven.
    
    Use for:
    - Exploratory analysis
    - Sensitivity analysis  
    - As a feature for downstream models
    
    Do NOT use for:
    - Claiming propensity identification
    - Inverse probability weighting (without additional validation)
    """
    
    def __init__(
        self, 
        model, 
        sigma: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Args:
            model: Trained SBFlowModel
            sigma: Source noise scale
            temperature: Softmax temperature for propensity estimation
        """
        self.model = model
        self.sigma = sigma
        self.temperature = temperature
    
    @torch.no_grad()
    def estimate_propensity_heuristic(
        self,
        x: torch.Tensor,
        m_target: torch.Tensor,
        t_eval: float = 0.99,
    ) -> torch.Tensor:
        """
        Heuristic propensity score based on backward velocity.
        
        Formula (heuristic, NOT identification):
        π_hat(m|x) ∝ exp(-||v_B^miss||^2 / (2θ²))
        
        Motivation: Larger backward velocity might indicate less likely
        missing patterns. This is an ASSUMPTION, not a theorem.
        
        Args:
            x: Data (B, D)
            m_target: Missing mask (B, D)
            t_eval: Time for velocity evaluation
        
        Returns:
            propensity: Heuristic scores (B,) in [0, 1]
        """
        B, D = x.shape
        device = x.device
        
        m_obs = 1.0 - m_target.float()
        t = torch.full((B, 1), t_eval, device=device)
        
        # Get backward velocity
        v_back = self.model.backward_bridge(x, m_obs, m_obs, t)
        
        # Velocity magnitude on missing positions
        v_miss = v_back * m_target.float()
        v_magnitude = (v_miss ** 2).sum(dim=-1)  # (B,)
        
        # Heuristic transform (NOT theoretically justified)
        # Larger velocity → smaller propensity (heuristic assumption)
        propensity = torch.exp(-v_magnitude / (2 * self.temperature ** 2))
        
        # Normalize to [0, 1]
        propensity = propensity / (propensity.max() + 1e-8)
        
        return propensity


def validate_proposition_1(
    v_magnitude: torch.Tensor,
    x_magnitude: torch.Tensor, 
    n_missing: torch.Tensor,
    sigma: float = 1.0,
) -> Dict[str, float]:
    """
    Validate Proposition 1: E[||v_B^miss||^2] = ||x_miss||^2 + σ²|m|
    
    Args:
        v_magnitude: ||v_B^miss||^2 per sample
        x_magnitude: ||x_miss||^2 per sample  
        n_missing: Number of missing features per sample
        sigma: Source noise scale
    
    Returns:
        Validation metrics
    """
    # Predicted by Proposition 1
    predicted = x_magnitude + sigma ** 2 * n_missing
    
    # Actual
    actual = v_magnitude
    
    # Compute correlation (should be high if proposition holds)
    pred_centered = predicted - predicted.mean()
    act_centered = actual - actual.mean()
    
    correlation = (pred_centered * act_centered).sum() / (
        torch.sqrt((pred_centered ** 2).sum() * (act_centered ** 2).sum()) + 1e-8
    )
    
    # MSE
    mse = ((predicted - actual) ** 2).mean()
    
    # Relative error
    rel_error = (torch.abs(predicted - actual) / (actual + 1e-8)).mean()
    
    return {
        'correlation': correlation.item(),
        'mse': mse.item(),
        'relative_error': rel_error.item(),
        'predicted_mean': predicted.mean().item(),
        'actual_mean': actual.mean().item(),
    }


# =============================================================================
# Theoretical Statements (for documentation)
# =============================================================================

PROPOSITION_1 = """
Proposition 1 (Backward Velocity on Missing Positions)

Under the linear-Gaussian toy model (Assumptions A1-A3 in theory.md):

(a) E[v_B^miss | x, m] = -x_miss

(b) E[||v_B^miss||^2 | x, m] = ||x_miss||^2 + σ²|m|

Proof: See Appendix A of theory.md.
"""

PROPOSITION_2 = """
Proposition 2 (Correlation Structure under MAR vs MNAR)

Under MAR: After controlling for |m|, there is no additional structure 
in which specific features are missing.

Under MNAR with π(m_j|x) = σ(αx_j): Features with large |x_j| are 
preferentially missing (if α > 0), creating selection bias.

This bias propagates to ||v_B^miss||^2 through the ||x_miss||^2 term.
"""

PROPOSITION_3 = """
Proposition 3 (MNAR Detection)

Define S = Corr(||v_B^miss||^2, ||x_miss||^2).

(a) Under MCAR: E[S] ≈ 0
(b) Under MNAR with π(m_j|x) = σ(αx_j): E[S] ≠ 0

The test "reject MAR if |S| > threshold" provides a practical 
MNAR diagnostic.

Caveat: In real data, x_miss is unknown and must be imputed.
"""
