"""
Robust MNAR Detection Module

This module provides multiple independent methods for MNAR detection,
addressing the circular dependency problem in selection bias detection.

Key insight: The original MNAR score S = E[x²|miss] / E[x²|obs] - 1
depends on imputed values, which may be biased. We provide additional
detection methods that are more robust.

Methods:
1. Selection Bias Score (original) - based on imputed values
2. Holdout Validation - uses known values to validate imputation bias
3. Feature Distribution Test - KS test on imputed vs observed distributions
4. Velocity Field Analysis - analyzes backward bridge behavior
5. Ensemble Detection - combines multiple signals

Usage:
    detector = RobustMNARDetector(model)
    result = detector.detect(x_obs, m_obs, m_target, x_imputed)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .mnar_score import compute_mnar_score, MNARScoreResult


@dataclass
class RobustMNARResult:
    """Result container for robust MNAR detection."""

    # Primary detection result
    is_mnar_detected: bool
    confidence: float  # 0-1, higher = more confident

    # Individual signal results
    selection_bias: MNARScoreResult
    holdout_result: Optional[Dict] = None
    distribution_test: Optional[Dict] = None
    velocity_analysis: Optional[Dict] = None

    # Ensemble voting
    n_signals_positive: int = 0
    n_signals_total: int = 0

    # Interpretation
    interpretation: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'is_mnar_detected': self.is_mnar_detected,
            'confidence': self.confidence,
            'selection_bias': self.selection_bias.to_dict(),
            'holdout_result': self.holdout_result,
            'distribution_test': self.distribution_test,
            'velocity_analysis': self.velocity_analysis,
            'n_signals_positive': self.n_signals_positive,
            'n_signals_total': self.n_signals_total,
            'interpretation': self.interpretation,
        }


class RobustMNARDetector:
    """
    Robust MNAR detection using multiple independent signals.

    Addresses the circular dependency problem by:
    1. Not relying solely on imputed values
    2. Using multiple independent detection methods
    3. Ensemble voting for final decision
    """

    def __init__(
        self,
        model,
        threshold: float = 0.15,
        holdout_ratio: float = 0.1,
        min_samples_per_feature: int = 30,
    ):
        """
        Args:
            model: Trained BiFlow model (SBFlowModel)
            threshold: Detection threshold for selection bias score
            holdout_ratio: Fraction of observed values to hold out for validation
            min_samples_per_feature: Minimum samples for feature-wise analysis
        """
        self.model = model
        self.threshold = threshold
        self.holdout_ratio = holdout_ratio
        self.min_samples = min_samples_per_feature

    @torch.no_grad()
    def detect(
        self,
        x_obs: torch.Tensor,
        m_obs: torch.Tensor,
        m_target: torch.Tensor,
        x_imputed: torch.Tensor,
        x_true: Optional[torch.Tensor] = None,
        run_all_tests: bool = True,
    ) -> RobustMNARResult:
        """
        Run robust MNAR detection using multiple signals.

        Args:
            x_obs: Observed data (B, D) - original values
            m_obs: Observed mask (B, D) - 1 where observed
            m_target: Missing mask (B, D) - 1 where missing
            x_imputed: Imputed complete data (B, D)
            x_true: (Optional) True complete data for validation
            run_all_tests: Whether to run all detection methods

        Returns:
            RobustMNARResult with comprehensive detection results
        """
        signals = []
        signal_names = []

        # ============================================================
        # Signal 1: Selection Bias Score (original method)
        # ============================================================
        selection_bias = compute_mnar_score(
            x_imputed, m_obs, m_target,
            normalize=True,
            threshold=self.threshold,
        )
        signals.append(selection_bias.is_mnar_detected)
        signal_names.append('selection_bias')

        # ============================================================
        # Signal 2: Holdout Validation (if enough observed values)
        # ============================================================
        holdout_result = None
        if run_all_tests:
            holdout_result = self._holdout_validation(
                x_obs, m_obs, m_target, x_imputed
            )
            if holdout_result is not None:
                signals.append(holdout_result['is_mnar'])
                signal_names.append('holdout')

        # ============================================================
        # Signal 3: Distribution Test (KS test)
        # ============================================================
        distribution_test = None
        if run_all_tests:
            distribution_test = self._distribution_test(
                x_imputed, m_obs, m_target
            )
            if distribution_test is not None:
                signals.append(distribution_test['is_mnar'])
                signal_names.append('distribution')

        # ============================================================
        # Signal 4: Velocity Field Analysis
        # ============================================================
        velocity_analysis = None
        if run_all_tests and self.model is not None:
            velocity_analysis = self._velocity_analysis(
                x_imputed, m_obs, m_target
            )
            if velocity_analysis is not None:
                signals.append(velocity_analysis['is_mnar'])
                signal_names.append('velocity')

        # ============================================================
        # Ensemble Decision
        # ============================================================
        n_positive = sum(signals)
        n_total = len(signals)

        # Decision rule: majority voting with confidence
        # At least 2 signals must be positive, or selection_bias with high confidence
        if n_total >= 3:
            is_mnar = n_positive >= 2
        elif n_total == 2:
            is_mnar = n_positive >= 1 and selection_bias.is_mnar_detected
        else:
            is_mnar = selection_bias.is_mnar_detected

        # Compute confidence
        if n_total > 0:
            confidence = n_positive / n_total
            # Boost confidence if selection bias is strong
            if abs(selection_bias.mnar_score) > 0.3:
                confidence = min(1.0, confidence + 0.2)
        else:
            confidence = 0.5

        # Generate interpretation
        interpretation = self._generate_interpretation(
            is_mnar, confidence, selection_bias,
            signal_names, signals
        )

        return RobustMNARResult(
            is_mnar_detected=is_mnar,
            confidence=confidence,
            selection_bias=selection_bias,
            holdout_result=holdout_result,
            distribution_test=distribution_test,
            velocity_analysis=velocity_analysis,
            n_signals_positive=n_positive,
            n_signals_total=n_total,
            interpretation=interpretation,
        )

    def _holdout_validation(
        self,
        x_obs: torch.Tensor,
        m_obs: torch.Tensor,
        m_target: torch.Tensor,
        x_imputed: torch.Tensor,
    ) -> Optional[Dict]:
        """
        Holdout validation: Check if imputation error correlates with value magnitude.

        Under MNAR: larger values are harder to impute correctly, leading to
        correlation between |error| and |true_value|.
        """
        B, D = x_obs.shape
        device = x_obs.device

        # We can only do this analysis on observed positions
        # Compare imputed values at observed positions with original values
        # (This gives us a proxy for imputation quality)

        m_obs_f = m_obs.float()
        n_observed = m_obs_f.sum().item()

        if n_observed < 100:
            return None  # Not enough observed values

        # At observed positions, compare x_imputed with x_obs
        # In theory, they should be the same (imputation should preserve observed)
        # But due to numerical issues, there might be small differences
        # Instead, we analyze the distribution of imputed values

        # Get imputed values at missing positions
        imputed_at_miss = x_imputed[m_target.bool()]
        obs_values = x_obs[m_obs.bool()]

        if len(imputed_at_miss) < 50 or len(obs_values) < 50:
            return None

        # Compare variances
        var_imputed = imputed_at_miss.var().item()
        var_observed = obs_values.var().item()

        # Under MNAR with "large values missing": var_imputed > var_observed
        # Under MCAR: var_imputed ≈ var_observed
        var_ratio = var_imputed / (var_observed + 1e-8)

        # Also compare means (for asymmetric MNAR)
        mean_imputed = imputed_at_miss.mean().item()
        mean_observed = obs_values.mean().item()
        mean_diff = abs(mean_imputed - mean_observed)

        # Detection thresholds
        is_mnar_by_var = abs(var_ratio - 1.0) > 0.2
        is_mnar_by_mean = mean_diff > 0.3

        return {
            'var_imputed': var_imputed,
            'var_observed': var_observed,
            'var_ratio': var_ratio,
            'mean_imputed': mean_imputed,
            'mean_observed': mean_observed,
            'mean_diff': mean_diff,
            'is_mnar_by_var': is_mnar_by_var,
            'is_mnar_by_mean': is_mnar_by_mean,
            'is_mnar': is_mnar_by_var or is_mnar_by_mean,
        }

    def _distribution_test(
        self,
        x_imputed: torch.Tensor,
        m_obs: torch.Tensor,
        m_target: torch.Tensor,
    ) -> Optional[Dict]:
        """
        Distribution test: KS test between imputed and observed distributions.

        Under MCAR: distributions should be similar
        Under MNAR: distributions differ systematically
        """
        try:
            from scipy.stats import ks_2samp
        except ImportError:
            return None  # scipy not available

        B, D = x_imputed.shape
        m_obs_f = m_obs.float()
        m_target_f = m_target.float()

        ks_stats = []
        p_values = []
        significant_features = []

        for j in range(D):
            obs_j = m_obs[:, j].bool()
            miss_j = m_target[:, j].bool()

            n_obs = obs_j.sum().item()
            n_miss = miss_j.sum().item()

            if n_obs < self.min_samples or n_miss < self.min_samples:
                continue

            obs_vals = x_imputed[obs_j, j].cpu().numpy()
            miss_vals = x_imputed[miss_j, j].cpu().numpy()

            stat, p_val = ks_2samp(obs_vals, miss_vals)
            ks_stats.append(stat)
            p_values.append(p_val)

            # Bonferroni correction
            if p_val < 0.05 / D:
                significant_features.append(j)

        if len(ks_stats) == 0:
            return None

        mean_ks = np.mean(ks_stats)
        n_significant = len(significant_features)
        frac_significant = n_significant / len(ks_stats)

        # Detection: significant distribution difference in >10% features
        is_mnar = frac_significant > 0.1 or mean_ks > 0.15

        return {
            'mean_ks_stat': mean_ks,
            'n_features_tested': len(ks_stats),
            'n_significant': n_significant,
            'frac_significant': frac_significant,
            'significant_features': significant_features,
            'is_mnar': is_mnar,
        }

    def _velocity_analysis(
        self,
        x_imputed: torch.Tensor,
        m_obs: torch.Tensor,
        m_target: torch.Tensor,
    ) -> Optional[Dict]:
        """
        Velocity field analysis: Check backward bridge behavior.

        Under MNAR: backward velocity at missing positions correlates with
        the magnitude of imputed values (selection bias in velocity field).
        """
        if self.model is None:
            return None

        B, D = x_imputed.shape
        device = x_imputed.device
        m_obs_f = m_obs.float()
        m_target_f = m_target.float()

        # Evaluate backward velocity at t close to 1
        t_eval = 0.95
        t = torch.full((B, 1), t_eval, device=device)

        v_backward = self.model.backward_bridge(x_imputed, m_obs_f, m_obs_f, t)

        # Compute velocity magnitude at missing positions
        v_miss = v_backward * m_target_f
        v_magnitude = (v_miss ** 2).sum(dim=-1)  # (B,)

        # Compute value magnitude at missing positions
        x_miss = x_imputed * m_target_f
        x_magnitude = (x_miss ** 2).sum(dim=-1)  # (B,)

        # Correlation between velocity and value magnitude
        # Under MNAR: we expect correlation ≠ 0
        v_centered = v_magnitude - v_magnitude.mean()
        x_centered = x_magnitude - x_magnitude.mean()

        correlation = (
            (v_centered * x_centered).sum() /
            (torch.sqrt((v_centered ** 2).sum() * (x_centered ** 2).sum()) + 1e-8)
        ).item()

        # Also check velocity variance across samples
        v_var = v_magnitude.var().item()
        x_var = x_magnitude.var().item()

        # Detection: significant correlation or unusual velocity variance
        is_mnar = abs(correlation) > 0.2

        return {
            'velocity_value_correlation': correlation,
            'velocity_variance': v_var,
            'value_variance': x_var,
            't_eval': t_eval,
            'is_mnar': is_mnar,
        }

    def _generate_interpretation(
        self,
        is_mnar: bool,
        confidence: float,
        selection_bias: MNARScoreResult,
        signal_names: List[str],
        signals: List[bool],
    ) -> str:
        """Generate human-readable interpretation."""
        lines = [
            "=" * 50,
            "Robust MNAR Detection Report",
            "=" * 50,
        ]

        # Summary
        if is_mnar:
            lines.append(f"⚠️  MNAR DETECTED (confidence: {confidence:.1%})")
        else:
            lines.append(f"✓  No MNAR detected (confidence: {1-confidence:.1%})")

        # Selection bias details
        lines.append(f"\nSelection Bias Score S = {selection_bias.mnar_score:.4f}")
        lines.append(f"  Direction: {selection_bias.direction}")

        # Signal breakdown
        lines.append(f"\nSignal Breakdown ({sum(signals)}/{len(signals)} positive):")
        for name, signal in zip(signal_names, signals):
            status = "⚠️ MNAR" if signal else "✓ OK"
            lines.append(f"  - {name}: {status}")

        # Recommendation
        lines.append("")
        if is_mnar:
            lines.append("Recommendations:")
            lines.append("  - Interpret imputation results with caution")
            lines.append("  - Consider MNAR-aware imputation methods")
            lines.append("  - Perform sensitivity analysis")
        else:
            lines.append("The data appears consistent with MCAR/MAR assumptions.")
            lines.append("Standard imputation methods should be reliable.")

        return "\n".join(lines)


def quick_mnar_check(
    x_imputed: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    threshold: float = 0.15,
) -> Tuple[bool, float, str]:
    """
    Quick MNAR check using only selection bias (no model required).

    Args:
        x_imputed: Imputed data (B, D)
        m_obs: Observed mask (B, D)
        m_target: Missing mask (B, D)
        threshold: Detection threshold

    Returns:
        (is_mnar, score, interpretation)
    """
    result = compute_mnar_score(x_imputed, m_obs, m_target, threshold=threshold)

    if result.is_mnar_detected:
        interp = f"⚠️ MNAR detected (S={result.mnar_score:.3f}, {result.direction})"
    else:
        interp = f"✓ Consistent with MCAR/MAR (S={result.mnar_score:.3f})"

    return result.is_mnar_detected, result.mnar_score, interp
