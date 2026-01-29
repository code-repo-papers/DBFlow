"""
Unified MNAR Score Computation Module

This module provides a single, consistent interface for computing MNAR
(Missing Not At Random) diagnostic scores across the BiFlow codebase.

Key metric: Selection Bias Score S
    S = E[x² | missing] / E[x² | observed] - 1

    - S ≈ 0: Consistent with MCAR/MAR (no selection bias)
    - S > 0: Larger values more likely missing (positive MNAR)
    - S < 0: Smaller values more likely missing (negative MNAR)
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class MNARScoreResult:
    """Result container for MNAR score computation."""
    mnar_score: float                    # S = ratio - 1
    selection_bias_ratio: float          # E[x²|miss] / E[x²|obs]
    e_x2_missing: float                  # E[x² | missing]
    e_x2_observed: float                 # E[x² | observed]
    is_mnar_detected: bool               # |S| > threshold
    threshold: float                     # Detection threshold used
    normalized: bool                     # Whether input was normalized
    confidence: str                      # "high", "medium", "low"
    direction: str                       # "positive", "negative", "none"
    n_missing: int                       # Number of missing entries
    n_observed: int                      # Number of observed entries

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'mnar_score': self.mnar_score,
            'selection_bias_ratio': self.selection_bias_ratio,
            'e_x2_missing': self.e_x2_missing,
            'e_x2_observed': self.e_x2_observed,
            'is_mnar_detected': self.is_mnar_detected,
            'threshold': self.threshold,
            'normalized': self.normalized,
            'confidence': self.confidence,
            'direction': self.direction,
            'n_missing': self.n_missing,
            'n_observed': self.n_observed,
        }


def compute_mnar_score(
    x: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    normalize: bool = True,
    threshold: float = 0.15,
    eps: float = 1e-8,
) -> MNARScoreResult:
    """
    Unified MNAR score computation.

    Computes the Selection Bias Score S which measures whether missing
    values have systematically different magnitudes than observed values.

    Args:
        x: Data tensor (B, D) - can be imputed or complete
        m_obs: Observed mask (B, D) - 1 where observed, 0 where missing
        m_target: Missing/target mask (B, D) - 1 where missing, 0 where observed
        normalize: Whether to z-score normalize before computing (recommended)
        threshold: Detection threshold for |S| (default 0.15)
        eps: Numerical stability constant

    Returns:
        MNARScoreResult with all diagnostic information

    Example:
        >>> result = compute_mnar_score(x_imputed, m_obs, m_target)
        >>> if result.is_mnar_detected:
        ...     print(f"⚠️ MNAR detected! S = {result.mnar_score:.3f}")
    """
    # Ensure float tensors
    x = x.float()
    m_obs_f = m_obs.float()
    m_target_f = m_target.float()

    # Optional normalization (recommended for multi-scale features)
    if normalize:
        # Per-feature z-score normalization
        x_mean = x.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True) + eps
        x_norm = (x - x_mean) / x_std
    else:
        x_norm = x

    # Compute x²
    x_squared = x_norm ** 2

    # Count entries
    n_missing = m_target_f.sum().item()
    n_observed = m_obs_f.sum().item()

    # Compute E[x² | missing] and E[x² | observed]
    e_x2_missing = (x_squared * m_target_f).sum() / (n_missing + eps)
    e_x2_observed = (x_squared * m_obs_f).sum() / (n_observed + eps)

    # Selection bias ratio and MNAR score
    selection_bias_ratio = (e_x2_missing / (e_x2_observed + eps)).item()
    mnar_score = selection_bias_ratio - 1.0

    # Detection decision (use absolute value for bidirectional MNAR)
    abs_score = abs(mnar_score)
    is_mnar_detected = abs_score > threshold

    # Determine direction
    if mnar_score > threshold:
        direction = "positive"  # Larger values more likely missing
    elif mnar_score < -threshold:
        direction = "negative"  # Smaller values more likely missing
    else:
        direction = "none"      # No significant bias

    # Confidence estimation based on sample size and score magnitude
    min_entries = min(n_missing, n_observed)
    if min_entries < 100:
        confidence = "low"      # Small sample, unreliable
    elif abs_score > 0.5 and min_entries > 500:
        confidence = "high"     # Strong signal, large sample
    elif abs_score > 0.2 and min_entries > 200:
        confidence = "medium"
    else:
        confidence = "low"

    return MNARScoreResult(
        mnar_score=mnar_score,
        selection_bias_ratio=selection_bias_ratio,
        e_x2_missing=e_x2_missing.item(),
        e_x2_observed=e_x2_observed.item(),
        is_mnar_detected=is_mnar_detected,
        threshold=threshold,
        normalized=normalize,
        confidence=confidence,
        direction=direction,
        n_missing=int(n_missing),
        n_observed=int(n_observed),
    )


def compute_mnar_score_batched(
    x: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-sample MNAR scores for a batch.

    Useful for analyzing which samples have stronger MNAR signals.

    Args:
        x: Data tensor (B, D)
        m_obs: Observed mask (B, D)
        m_target: Missing mask (B, D)
        normalize: Whether to normalize
        eps: Numerical stability

    Returns:
        per_sample_scores: (B,) MNAR score per sample
        per_sample_n_missing: (B,) number of missing features per sample
    """
    x = x.float()
    m_obs_f = m_obs.float()
    m_target_f = m_target.float()

    if normalize:
        x_mean = x.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True) + eps
        x_norm = (x - x_mean) / x_std
    else:
        x_norm = x

    x_squared = x_norm ** 2

    # Per-sample computation
    n_missing = m_target_f.sum(dim=-1)  # (B,)
    n_observed = m_obs_f.sum(dim=-1)    # (B,)

    e_x2_missing = (x_squared * m_target_f).sum(dim=-1) / (n_missing + eps)  # (B,)
    e_x2_observed = (x_squared * m_obs_f).sum(dim=-1) / (n_observed + eps)   # (B,)

    per_sample_scores = e_x2_missing / (e_x2_observed + eps) - 1.0

    return per_sample_scores, n_missing


def compute_feature_wise_mnar(
    x: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    min_samples: int = 30,
) -> Dict[str, np.ndarray]:
    """
    Compute per-feature MNAR scores.

    Analyzes each feature independently to identify which features
    might have MNAR patterns.

    Args:
        x: Data tensor (B, D)
        m_obs: Observed mask (B, D)
        m_target: Missing mask (B, D)
        min_samples: Minimum samples required for reliable estimate

    Returns:
        Dictionary with per-feature statistics
    """
    B, D = x.shape

    feature_scores = []
    feature_valid = []

    for j in range(D):
        # Masks for this feature
        mask_miss_j = m_target[:, j] > 0.5
        mask_obs_j = m_obs[:, j] > 0.5

        n_miss = mask_miss_j.sum().item()
        n_obs = mask_obs_j.sum().item()

        if n_miss >= min_samples and n_obs >= min_samples:
            # Compute E[x²] for this feature
            e_x2_miss = (x[mask_miss_j, j] ** 2).mean().item()
            e_x2_obs = (x[mask_obs_j, j] ** 2).mean().item()

            score_j = e_x2_miss / (e_x2_obs + 1e-8) - 1.0
            feature_scores.append(score_j)
            feature_valid.append(True)
        else:
            feature_scores.append(0.0)
            feature_valid.append(False)

    return {
        'feature_scores': np.array(feature_scores),
        'feature_valid': np.array(feature_valid),
        'mean_score': np.mean([s for s, v in zip(feature_scores, feature_valid) if v]),
        'std_score': np.std([s for s, v in zip(feature_scores, feature_valid) if v]),
        'n_valid_features': sum(feature_valid),
    }


# =============================================================================
# Issue 1 Solution: Bidirectional MNAR Score
# =============================================================================

@dataclass
class BidirectionalMNARResult:
    """Result container for bidirectional MNAR score computation."""
    # Combined score
    score_bidirectional: float           # S_bi = S_F + lambda * S_B

    # Individual scores
    score_forward: float                 # S_F: forward-based score
    score_backward: float                # S_B: backward velocity-based score

    # Detection
    is_mnar_detected: bool
    threshold: float
    lambda_weight: float                 # Weight for backward score

    # Confidence and direction
    confidence: str
    direction: str

    # Diagnostics
    forward_result: MNARScoreResult      # Full forward result
    v_backward_miss_sq_mean: float       # E[||v_B^miss||²]
    v_backward_obs_sq_mean: float        # E[||v_B^obs||²] (reference)

    def to_dict(self) -> Dict:
        return {
            'score_bidirectional': self.score_bidirectional,
            'score_forward': self.score_forward,
            'score_backward': self.score_backward,
            'is_mnar_detected': self.is_mnar_detected,
            'threshold': self.threshold,
            'lambda_weight': self.lambda_weight,
            'confidence': self.confidence,
            'direction': self.direction,
            'v_backward_miss_sq_mean': self.v_backward_miss_sq_mean,
            'v_backward_obs_sq_mean': self.v_backward_obs_sq_mean,
        }


def compute_backward_score(
    v_backward: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    """
    Compute backward velocity-based MNAR score.

    Under MNAR with α > 0, missing values have systematically larger magnitudes,
    which leads to larger backward velocity magnitudes on missing positions.

    S_B = E[||v_B||² | missing] / E[||v_B||² | observed] - 1

    Args:
        v_backward: Backward velocity (B, D)
        m_obs: Observed mask (B, D)
        m_target: Missing mask (B, D)
        eps: Numerical stability

    Returns:
        (S_B, v_miss_sq_mean, v_obs_sq_mean)
    """
    v_backward = v_backward.float()
    m_obs_f = m_obs.float()
    m_target_f = m_target.float()

    # Compute squared velocity magnitudes
    v_sq = v_backward ** 2

    # Count entries
    n_missing = m_target_f.sum().item()
    n_observed = m_obs_f.sum().item()

    # E[||v_B||² | missing] and E[||v_B||² | observed]
    v_miss_sq_mean = (v_sq * m_target_f).sum() / (n_missing + eps)
    v_obs_sq_mean = (v_sq * m_obs_f).sum() / (n_observed + eps)

    # Backward MNAR score
    score_backward = (v_miss_sq_mean / (v_obs_sq_mean + eps) - 1.0).item()

    return score_backward, v_miss_sq_mean.item(), v_obs_sq_mean.item()


def compute_bidirectional_mnar_score(
    x_imputed: torch.Tensor,
    v_backward: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    lambda_weight: float = 0.5,
    normalize: bool = True,
    threshold: float = 0.15,
    eps: float = 1e-8,
) -> BidirectionalMNARResult:
    """
    Compute bidirectional MNAR score combining forward and backward signals.

    This addresses the reviewer concern that MNAR detection only uses forward output.
    The bidirectional score combines:
    - Forward score S_F: based on imputed value magnitudes
    - Backward score S_B: based on backward velocity magnitudes

    REVISED Formula: S_bi = S_F - λ * S_B

    Key insight from experiments:
    - S_F: transitions from ~0 (MCAR) to positive (MNAR)
    - S_B: always negative, becomes MORE negative under MNAR

    By using subtraction (S_F - λ*S_B), we:
    - Add |S_B| contribution when S_B < 0 (which is always true)
    - Under MCAR: S_F ≈ 0, S_B ≈ -0.35, so S_bi ≈ 0.17
    - Under MNAR: S_F > 0, S_B << 0, so S_bi = S_F + λ|S_B| (amplified)

    To avoid false positives at MCAR, we use a calibrated threshold.

    Args:
        x_imputed: Imputed data (B, D)
        v_backward: Backward velocity at some time t (B, D)
        m_obs: Observed mask (B, D)
        m_target: Missing mask (B, D)
        lambda_weight: Weight for backward score (default 0.5)
        normalize: Whether to normalize for forward score
        threshold: Detection threshold for |S_bi|
        eps: Numerical stability

    Returns:
        BidirectionalMNARResult with all diagnostics
    """
    # Forward score (existing method)
    forward_result = compute_mnar_score(
        x_imputed, m_obs, m_target,
        normalize=normalize,
        threshold=threshold,
        eps=eps,
    )
    score_forward = forward_result.mnar_score

    # Backward score (new)
    score_backward, v_miss_sq, v_obs_sq = compute_backward_score(
        v_backward, m_obs, m_target, eps
    )

    # REVISED: Use subtraction since S_B is always negative
    # S_bi = S_F - λ * S_B = S_F + λ * |S_B| when S_B < 0
    score_bidirectional = score_forward - lambda_weight * score_backward

    # Detection using bidirectional score
    # Use higher threshold to account for baseline S_B contribution
    bi_threshold = threshold + lambda_weight * 0.35  # ~0.35 is MCAR baseline for |S_B|
    is_mnar_detected = score_bidirectional > bi_threshold

    # Direction based on combined score
    if score_bidirectional > bi_threshold:
        direction = "positive"
    elif score_bidirectional < -threshold:
        direction = "negative"
    else:
        direction = "none"

    # Confidence based on both signals
    # Strong MNAR: S_F > 0 AND S_B very negative (< -0.5)
    forward_signal = score_forward > 0.05
    backward_signal = score_backward < -0.5  # More negative than MCAR baseline

    if forward_signal and backward_signal:
        confidence = "high"  # Both directions indicate MNAR
    elif forward_signal or backward_signal:
        confidence = "medium"  # At least one direction has signal
    else:
        confidence = "low"

    return BidirectionalMNARResult(
        score_bidirectional=score_bidirectional,
        score_forward=score_forward,
        score_backward=score_backward,
        is_mnar_detected=is_mnar_detected,
        threshold=bi_threshold,  # Return the adjusted threshold
        lambda_weight=lambda_weight,
        confidence=confidence,
        direction=direction,
        forward_result=forward_result,
        v_backward_miss_sq_mean=v_miss_sq,
        v_backward_obs_sq_mean=v_obs_sq,
    )


# =============================================================================
# Issue 3 Solution: Bootstrap Confidence Interval
# =============================================================================

@dataclass
class BootstrapMNARResult:
    """Result container for bootstrap MNAR hypothesis test."""
    # Point estimate
    mnar_score: float

    # Confidence interval
    ci_lower: float
    ci_upper: float
    confidence_level: float

    # Hypothesis test
    p_value: float
    is_mnar_detected: bool              # CI doesn't contain 0

    # Bootstrap statistics
    bootstrap_mean: float
    bootstrap_std: float
    n_bootstrap: int

    # Direction
    direction: str

    def to_dict(self) -> Dict:
        return {
            'mnar_score': self.mnar_score,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'confidence_level': self.confidence_level,
            'p_value': self.p_value,
            'is_mnar_detected': self.is_mnar_detected,
            'bootstrap_mean': self.bootstrap_mean,
            'bootstrap_std': self.bootstrap_std,
            'n_bootstrap': self.n_bootstrap,
            'direction': self.direction,
        }


def bootstrap_mnar_score(
    x: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    normalize: bool = True,
    eps: float = 1e-8,
    seed: Optional[int] = None,
) -> BootstrapMNARResult:
    """
    Compute MNAR Score with bootstrap confidence interval and p-value.

    This addresses the reviewer concern about lack of statistical guarantees.
    Provides:
    - Confidence interval for the MNAR Score
    - P-value for hypothesis test H0: S = 0 (MAR) vs H1: S ≠ 0 (MNAR)
    - Detection decision with Type I error control

    Algorithm:
    1. Resample (with replacement) from samples
    2. Compute MNAR Score on each bootstrap sample
    3. Construct CI from bootstrap distribution
    4. P-value = 2 * min(fraction of scores > 0, fraction of scores < 0)

    Args:
        x: Data tensor (B, D) - imputed or complete
        m_obs: Observed mask (B, D)
        m_target: Missing mask (B, D)
        n_bootstrap: Number of bootstrap samples
        confidence_level: CI level (default 0.95)
        normalize: Whether to normalize
        eps: Numerical stability
        seed: Random seed for reproducibility

    Returns:
        BootstrapMNARResult with CI, p-value, and detection decision
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    B = x.shape[0]
    alpha = 1.0 - confidence_level

    # Original score
    original_result = compute_mnar_score(x, m_obs, m_target, normalize=normalize, eps=eps)
    original_score = original_result.mnar_score

    # Bootstrap
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = torch.randint(0, B, (B,), device=x.device)
        x_boot = x[idx]
        m_obs_boot = m_obs[idx]
        m_target_boot = m_target[idx]

        # Compute score on bootstrap sample
        result_boot = compute_mnar_score(
            x_boot, m_obs_boot, m_target_boot,
            normalize=normalize, eps=eps
        )
        bootstrap_scores.append(result_boot.mnar_score)

    bootstrap_scores = np.array(bootstrap_scores)

    # Confidence interval (percentile method)
    ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

    # P-value for two-sided test H0: S = 0
    # Under H0, scores should be centered around 0
    # P-value = 2 * min(P(S > 0), P(S < 0)) under bootstrap distribution
    frac_positive = np.mean(bootstrap_scores > 0)
    frac_negative = np.mean(bootstrap_scores < 0)
    p_value = 2 * min(frac_positive, frac_negative)

    # Detection: CI doesn't contain 0
    is_mnar_detected = (ci_lower > 0) or (ci_upper < 0)

    # Direction
    if ci_lower > 0:
        direction = "positive"
    elif ci_upper < 0:
        direction = "negative"
    else:
        direction = "none"

    return BootstrapMNARResult(
        mnar_score=original_score,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        p_value=p_value,
        is_mnar_detected=is_mnar_detected,
        bootstrap_mean=np.mean(bootstrap_scores),
        bootstrap_std=np.std(bootstrap_scores),
        n_bootstrap=n_bootstrap,
        direction=direction,
    )


def mnar_hypothesis_test(
    x: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    normalize: bool = True,
) -> Dict:
    """
    Formal hypothesis test for MNAR detection.

    H0: Data is MAR/MCAR (S = 0)
    H1: Data is MNAR (S ≠ 0)

    Args:
        x: Data tensor (B, D)
        m_obs: Observed mask (B, D)
        m_target: Missing mask (B, D)
        alpha: Significance level (default 0.05)
        n_bootstrap: Number of bootstrap samples
        normalize: Whether to normalize

    Returns:
        Dictionary with test results
    """
    result = bootstrap_mnar_score(
        x, m_obs, m_target,
        n_bootstrap=n_bootstrap,
        confidence_level=1.0 - alpha,
        normalize=normalize,
    )

    # Test decision
    reject_null = result.p_value < alpha

    return {
        'test_statistic': result.mnar_score,
        'p_value': result.p_value,
        'alpha': alpha,
        'reject_null': reject_null,
        'conclusion': 'MNAR detected' if reject_null else 'Cannot reject MAR/MCAR',
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
        'bootstrap_result': result,
    }


# =============================================================================
# Issue 2: Theoretical bound (for paper - numerical verification)
# =============================================================================

def verify_quantitative_bound(
    alpha_values: list = [0.1, 0.2, 0.5, 1.0, 2.0],
    n_samples: int = 10000,
    d: int = 10,
    missing_rate: float = 0.3,
    seed: int = 42,
) -> Dict:
    """
    Numerically verify the theoretical bound: S ≈ 2α/√(2π) for small α.

    For paper: validates Theorem (Quantitative MNAR Score Bound).

    Args:
        alpha_values: MNAR strength values to test
        n_samples: Number of samples
        d: Feature dimension
        missing_rate: Target missing rate
        seed: Random seed

    Returns:
        Dictionary with theoretical vs empirical comparison
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = []

    for alpha in alpha_values:
        # Generate Gaussian data
        x = torch.randn(n_samples, d)

        # Generate MNAR mask: π(m=1|x) = σ(α*x + β)
        # Calibrate β to achieve target missing rate
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        # Binary search for β
        beta_low, beta_high = -10, 10
        for _ in range(50):
            beta = (beta_low + beta_high) / 2
            logits = alpha * x + beta
            probs = torch.sigmoid(logits)
            rate = probs.mean().item()
            if rate < missing_rate:
                beta_high = beta
            else:
                beta_low = beta

        # Generate mask
        logits = alpha * x + beta
        probs = torch.sigmoid(logits)
        m_target = (torch.rand_like(probs) < probs).float()
        m_obs = 1 - m_target

        # Compute empirical MNAR Score
        result = compute_mnar_score(x, m_obs, m_target, normalize=False)
        empirical_S = result.mnar_score

        # Theoretical bound: S ≈ 2α/√(2π) for small α
        theoretical_S = 2 * alpha / np.sqrt(2 * np.pi)

        results.append({
            'alpha': alpha,
            'empirical_S': empirical_S,
            'theoretical_S': theoretical_S,
            'ratio': empirical_S / theoretical_S if theoretical_S != 0 else float('inf'),
            'absolute_error': abs(empirical_S - theoretical_S),
        })

    return {
        'results': results,
        'summary': {
            'mean_ratio': np.mean([r['ratio'] for r in results]),
            'std_ratio': np.std([r['ratio'] for r in results]),
        }
    }


def interpret_mnar_score(result: MNARScoreResult) -> str:
    """
    Generate human-readable interpretation of MNAR score.

    Args:
        result: MNARScoreResult from compute_mnar_score

    Returns:
        Interpretation string
    """
    s = result.mnar_score

    lines = [
        f"MNAR Diagnostic Report",
        f"=" * 40,
        f"Selection Bias Score S = {s:.4f}",
        f"E[x²|missing] = {result.e_x2_missing:.4f}",
        f"E[x²|observed] = {result.e_x2_observed:.4f}",
        f"Samples: {result.n_missing} missing, {result.n_observed} observed",
        f"",
    ]

    if result.is_mnar_detected:
        lines.append(f"⚠️  MNAR DETECTED (|S| = {abs(s):.4f} > {result.threshold})")

        if result.direction == "positive":
            lines.append("   Interpretation: Larger values are more likely to be missing.")
            lines.append("   Example: High-income individuals less likely to report income.")
        else:
            lines.append("   Interpretation: Smaller values are more likely to be missing.")
            lines.append("   Example: Low-performing students less likely to report scores.")

        lines.append(f"   Confidence: {result.confidence}")
        lines.append("")
        lines.append("   ⚠️  Imputation results may be biased!")
        lines.append("   Consider: sensitivity analysis, MNAR-aware methods")
    else:
        lines.append(f"✓  No MNAR detected (|S| = {abs(s):.4f} ≤ {result.threshold})")
        lines.append("   Data appears consistent with MCAR/MAR assumption.")
        lines.append("   Standard imputation methods should be reliable.")

    return "\n".join(lines)
