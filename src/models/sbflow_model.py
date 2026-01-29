"""
BiFlow: Dual-Bridge Model for Bidirectional Flow Matching

The model consists of two independent networks:
- Forward Bridge (F): x0 → x1 (imputation direction)
- Backward Bridge (B): x1 → x0 (missing mechanism modeling)

Both bridges share the same architecture but have independent parameters.

Key feature: One model, two outputs
- Imputed data (from forward bridge)
- MNAR diagnostic score (from selection bias analysis)
"""

from typing import Optional

import torch
import torch.nn as nn
from .tabular_models import TabularFlowNet
from src.theory.mnar_score import (
    compute_mnar_score,
    compute_mnar_score_batched,
    compute_bidirectional_mnar_score,
    bootstrap_mnar_score,
    MNARScoreResult,
    BidirectionalMNARResult,
    BootstrapMNARResult,
)


class SBFlowModel(nn.Module):
    """
    BiFlow: Dual-Bridge model for Bidirectional Flow Matching.

    Contains two independent TabularFlowNet:
    - forward_net: learns to impute (noise → clean)
    - backward_net: learns the reverse (clean → noise), modeling missing mechanism

    Key feature: One model, two outputs (imputation + MNAR diagnosis)
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 512,
        nlayers: int = 3,
        dropout: float = 0.1,
        use_mask_input: bool = False,
        **kwargs
    ):
        """
        Args:
            d_in: Input dimension (number of features)
            d_model: Hidden dimension
            nlayers: Number of MLP layers
            dropout: Dropout rate
            use_mask_input: Whether to use mask as additional input conditioning
        """
        super().__init__()

        self.use_mask_input = use_mask_input

        # Forward bridge: x0 (noisy/incomplete) → x1 (clean/complete)
        self.forward_net = TabularFlowNet(
            d_in=d_in,
            d_model=d_model,
            nlayers=nlayers,
            dropout=dropout,
            use_mask_input=use_mask_input,
            **kwargs
        )

        # Backward bridge: x1 (clean) → x0 (noisy/incomplete)
        self.backward_net = TabularFlowNet(
            d_in=d_in,
            d_model=d_model,
            nlayers=nlayers,
            dropout=dropout,
            use_mask_input=use_mask_input,
            **kwargs
        )

        self.d_in = d_in
        self.d_model = d_model
    
    def forward_bridge(
        self,
        x: torch.Tensor,
        m_obs: torch.Tensor,
        m_cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward bridge: predict velocity for x0 → x1 direction.
        
        Args:
            x: Interpolated state x_t (B, D)
            m_obs: Observed mask (B, D)
            m_cond: Conditioning mask (B, D)
            t: Time in [0, 1] (B, 1)
        
        Returns:
            v_F: Forward velocity (B, D)
        """
        return self.forward_net(x, m_obs, m_cond, t)
    
    def backward_bridge(
        self,
        x: torch.Tensor,
        m_obs: torch.Tensor,
        m_cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Backward bridge: predict velocity for x1 → x0 direction.
        
        Args:
            x: Interpolated state x_{1-t} (B, D)
            m_obs: Observed mask (B, D)
            m_cond: Conditioning mask (B, D)
            t: Time in [0, 1] (B, 1), note: use 1-t for backward
        
        Returns:
            v_B: Backward velocity (B, D)
        """
        return self.backward_net(x, m_obs, m_cond, t)
    
    def forward(
        self,
        x: torch.Tensor,
        m_obs: torch.Tensor,
        m_cond: torch.Tensor,
        t: torch.Tensor,
        direction: str = "forward",
    ) -> torch.Tensor:
        """
        Unified forward pass.
        
        Args:
            x: Input state (B, D)
            m_obs: Observed mask (B, D)
            m_cond: Conditioning mask (B, D)
            t: Time (B, 1)
            direction: "forward" or "backward"
        
        Returns:
            v: Predicted velocity (B, D)
        """
        if direction == "forward":
            return self.forward_bridge(x, m_obs, m_cond, t)
        elif direction == "backward":
            return self.backward_bridge(x, m_obs, m_cond, t)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    @torch.no_grad()
    def impute(
        self,
        x_obs: torch.Tensor,
        m_obs: torch.Tensor,
        m_target: torch.Tensor,
        steps: int = 50,
        sigma: float = 1.0,
        solver: str = "heun",
        resample_enabled: bool = False,
        resample_interval: int = 5,
        resample_sigma: float = 0.1,
    ) -> torch.Tensor:
        """
        Impute missing values using forward bridge ODE solver with optional resampling.
        
        Resampling (Repaint-style) Mechanism:
        -------------------------------------
        To prevent variance collapse during ODE integration, we periodically inject
        small noise on missing positions. The noise magnitude decays with time to
        ensure convergence at t=1.
        
        Key idea from DiffPuter/Repaint:
        - At intermediate steps, add noise: x += noise * m_target * (1 - t)
        - Only affects missing positions (observed values are projected back)
        - Noise decays as t → 1 to ensure clean output
        
        Args:
            x_obs: Observed data (B, D) - values at observed positions
            m_obs: Observed mask (B, D)
            m_target: Missing mask (B, D)
            steps: Number of ODE steps
            sigma: Initial noise std on missing positions
            solver: ODE solver ("euler" or "heun")
            resample_enabled: Whether to enable resampling (default: False)
            resample_interval: Inject noise every K steps (default: 5)
            resample_sigma: Base noise scale for resampling (default: 0.1)
        
        Returns:
            x_imputed: Imputed data (B, D)
        """
        device = x_obs.device
        B, D = x_obs.shape
        
        # Initialize: observed + noise on missing
        noise = torch.randn(B, D, device=device) * sigma
        x_cur = x_obs * m_obs.float() + noise * m_target.float()
        
        # Time grid
        t_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
        dt = 1.0 / steps
        
        m_obs_f = m_obs.float()
        m_target_f = m_target.float()
        
        for i in range(steps):
            t = t_grid[i:i+1].expand(B, 1)
            t_val = i / steps  # Current time as scalar
            
            if solver == "euler":
                v = self.forward_bridge(x_cur, m_obs_f, m_obs_f, t)
                x_cur = x_cur + dt * v
                # Project: keep observed values fixed
                x_cur = x_cur * m_target_f + x_obs * m_obs_f
            elif solver == "heun":
                # Heun's method (2nd order) with proper projection handling
                # Step 1: Compute first velocity estimate
                v1 = self.forward_bridge(x_cur, m_obs_f, m_obs_f, t)
                
                # Step 2: Euler prediction (without projection for velocity estimation)
                x_pred = x_cur + dt * v1
                
                # Step 3: Compute second velocity estimate at predicted point
                t_next = t_grid[i+1:i+2].expand(B, 1)
                v2 = self.forward_bridge(x_pred, m_obs_f, m_obs_f, t_next)
                
                # Step 4: Heun update (average of velocities)
                x_cur = x_cur + 0.5 * dt * (v1 + v2)
                
                # Step 5: Project only once at the end to maintain observed values
                x_cur = x_cur * m_target_f + x_obs * m_obs_f
            else:
                raise ValueError(f"Unknown solver: {solver}")
            
            # =============================================
            # Resampling (Repaint-style) for variance preservation
            # =============================================
            # Inject noise periodically on missing positions only
            # Noise magnitude decays with time: σ_resample * (1 - t)
            # This preserves diversity while ensuring convergence
            if resample_enabled and (i + 1) % resample_interval == 0 and i < steps - 1:
                # Time-decaying noise injection
                decay_factor = 1.0 - t_val
                noise_resample = torch.randn(B, D, device=device) * resample_sigma * decay_factor
                
                # Add noise only on missing positions
                x_cur = x_cur + noise_resample * m_target_f
                
                # Re-project to ensure observed values remain unchanged
                x_cur = x_cur * m_target_f + x_obs * m_obs_f
        
        return x_cur
    
    @torch.no_grad()
    def impute_with_trials(
        self,
        x_obs: torch.Tensor,
        m_obs: torch.Tensor,
        m_target: torch.Tensor,
        steps: int = 50,
        trials: int = 10,
        sigma: float = 1.0,
        solver: str = "heun",
        aggregation: str = "single",  # Changed default from "mean" to "single"
        resample_enabled: bool = False,
        resample_interval: int = 5,
        resample_sigma: float = 0.1,
    ) -> torch.Tensor:
        """
        Impute with multiple trials.
        
        IMPORTANT: Simple averaging of multiple trials reduces variance!
        Each trial starts from different noise, averaging pushes values toward mean.
        
        Aggregation strategies:
        - "single": Use single trial (trials parameter ignored, preserves variance)
        - "mean": Average all trials (reduces variance, not recommended)
        - "median": Take element-wise median (better variance preservation)
        
        Resampling (for variance preservation):
        - resample_enabled: Inject noise periodically during ODE integration
        - resample_interval: How often to inject (every K steps)
        - resample_sigma: Base noise magnitude (decays with time)
        
        Args:
            x_obs: Observed data (B, D)
            m_obs: Observed mask (B, D)
            m_target: Missing mask (B, D)
            steps: Number of ODE steps
            trials: Number of random trials
            sigma: Initial noise std
            solver: ODE solver
            aggregation: How to aggregate trials ("single", "mean", "median")
            resample_enabled: Enable resampling for variance preservation
            resample_interval: Inject noise every K steps
            resample_sigma: Base noise scale for resampling
        
        Returns:
            x_imputed: Imputed data (B, D)
        """
        if aggregation == "single" or trials == 1:
            # Single trial: best for preserving variance
            return self.impute(
                x_obs, m_obs, m_target, steps, sigma, solver,
                resample_enabled, resample_interval, resample_sigma
            )
        
        B, D = x_obs.shape
        all_imputed = []
        
        for _ in range(trials):
            imputed = self.impute(
                x_obs, m_obs, m_target, steps, sigma, solver,
                resample_enabled, resample_interval, resample_sigma
            )
            all_imputed.append(imputed)
        
        all_imputed = torch.stack(all_imputed, dim=0)  # (trials, B, D)
        
        if aggregation == "mean":
            # Warning: This reduces variance!
            return all_imputed.mean(dim=0)
        elif aggregation == "median":
            # Median preserves variance better than mean
            return all_imputed.median(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    @torch.no_grad()
    def impute_and_diagnose(
        self,
        x_obs: torch.Tensor,
        m_obs: torch.Tensor,
        m_target: torch.Tensor,
        steps: int = 50,
        trials: int = 10,
        sigma: float = 1.0,
        solver: str = "heun",
        t_diagnose: float = 0.99,
        aggregation: str = "single",
        resample_enabled: bool = False,
        resample_interval: int = 5,
        resample_sigma: float = 0.1,
        mnar_threshold: float = 0.15,
    ) -> dict:
        """
        Dual-output inference: imputation + MNAR diagnosis.

        This is the KEY method that implements BiFlow's contribution:
        - Forward bridge → imputed values
        - Selection bias detection → MNAR diagnostic

        One forward pass gives you:
        1. Imputed data for downstream tasks
        2. MNAR diagnostic signal via selection bias detection

        MNAR Detection Method (unified via mnar_score module):
        - Under MNAR: E[x² | missing] ≠ E[x² | observed] (selection bias)
        - Under MCAR: E[x² | missing] ≈ E[x² | observed] (no bias)
        - MNAR Score S = E[x² | missing] / E[x² | observed] - 1

        Args:
            x_obs: Observed data (B, D) - values at observed positions
            m_obs: Observed mask (B, D) - 1 where observed
            m_target: Missing mask (B, D) - 1 where missing
            steps: Number of ODE steps for imputation
            trials: Number of trials for averaging
            sigma: Source noise scale
            solver: ODE solver type
            t_diagnose: Time point for backward velocity (for reference)
            aggregation: How to aggregate trials ("single", "mean", "median")
            resample_enabled: Enable resampling for variance preservation
            resample_interval: Inject noise every K steps
            resample_sigma: Base noise scale for resampling
            mnar_threshold: Threshold for MNAR detection (default 0.15)

        Returns:
            dict with:
                - 'x_imputed': Imputed complete data (B, D)
                - 'mnar_result': MNARScoreResult object with full diagnostics
                - 'mnar_score': Selection bias S (scalar) - key MNAR diagnostic
                - 'is_mnar_detected': Boolean indicating MNAR detection
                - 'per_sample_scores': Per-sample bias signal (B,)
                - 'v_backward': Backward velocity (B, D)
        """
        B, D = x_obs.shape
        device = x_obs.device
        m_obs_f = m_obs.float()
        m_target_f = m_target.float()

        # ========================================
        # Step 1: Forward Bridge → Imputation
        # ========================================
        x_imputed = self.impute_with_trials(
            x_obs, m_obs, m_target, steps, trials, sigma, solver,
            aggregation, resample_enabled, resample_interval, resample_sigma
        )

        # ========================================
        # Step 2: Compute MNAR Score (unified interface)
        # ========================================
        mnar_result = compute_mnar_score(
            x_imputed, m_obs, m_target,
            normalize=True,  # Always normalize for consistency
            threshold=mnar_threshold,
        )

        # Per-sample MNAR scores
        per_sample_scores, n_missing = compute_mnar_score_batched(
            x_imputed, m_obs, m_target, normalize=True
        )

        # ========================================
        # Step 3: Backward velocity for additional diagnostics
        # ========================================
        t = torch.full((B, 1), t_diagnose, device=device)
        v_backward = self.backward_bridge(x_imputed, m_obs_f, m_obs_f, t)

        return {
            # Primary outputs
            'x_imputed': x_imputed,
            'mnar_result': mnar_result,  # Full result object

            # Convenience accessors (from mnar_result)
            'mnar_score': mnar_result.mnar_score,
            'is_mnar_detected': mnar_result.is_mnar_detected,
            'selection_bias_ratio': mnar_result.selection_bias_ratio,
            'e_x2_missing': mnar_result.e_x2_missing,
            'e_x2_observed': mnar_result.e_x2_observed,
            'mnar_confidence': mnar_result.confidence,
            'mnar_direction': mnar_result.direction,

            # Per-sample analysis
            'per_sample_scores': per_sample_scores,
            'n_missing': n_missing,

            # Backward velocity diagnostics
            'v_backward': v_backward,
            't_diagnose': t_diagnose,
        }
    
    @torch.no_grad()
    def impute_adaptive(
        self,
        x_obs: torch.Tensor,
        m_obs: torch.Tensor,
        m_target: torch.Tensor,
        steps: int = 50,
        sigma: float = 1.0,
        solver: str = "heun",
        target_variance: float = 1.0,
        max_iterations: int = 8,
        tolerance: float = 0.05,
    ) -> tuple:
        """
        Adaptive imputation that automatically adjusts resample_sigma to achieve
        target ABSOLUTE variance on imputed values.
        
        Key insight: For normalized data (mean=0, std=1), the imputed values
        should have variance ≈ 1.0 regardless of missing mechanism.
        
        This differs from matching observed variance because:
        - Under MCAR: Var(miss) ≈ Var(obs) ≈ 1.0
        - Under MNAR: Var(miss) > Var(obs), but we still want Var(imputed) ≈ 1.0
        
        Algorithm:
        1. Start with sigma=0.6 (good for MCAR)
        2. Impute and measure Var(imputed)
        3. Binary search for sigma that gives Var(imputed) ≈ target_variance
        
        Args:
            x_obs: Observed data (B, D) - should be normalized!
            m_obs: Observed mask (B, D)
            m_target: Missing mask (B, D)
            steps: ODE integration steps
            sigma: Initial source noise
            solver: ODE solver
            target_variance: Target variance for imputed values (default 1.0)
            max_iterations: Max calibration iterations
            tolerance: Acceptable variance error
        
        Returns:
            (x_imputed, best_sigma): Imputed data and the optimal sigma used
        """
        device = x_obs.device
        
        # Binary search for optimal sigma
        sigma_low, sigma_high = 0.3, 1.5
        best_sigma = 0.6
        best_x = None
        best_var_diff = float('inf')
        
        for iteration in range(max_iterations):
            test_sigma = (sigma_low + sigma_high) / 2
            
            x_imp = self.impute(
                x_obs, m_obs, m_target, steps, sigma, solver,
                resample_enabled=True,
                resample_interval=5,
                resample_sigma=test_sigma,
            )
            
            # Compute ABSOLUTE variance of imputed values
            imp_vals = x_imp[m_target.bool()]
            if imp_vals.numel() > 0:
                imp_var = imp_vals.var().item()
            else:
                imp_var = 1.0
            
            var_diff = abs(imp_var - target_variance)
            
            # Check if we found a good sigma
            if var_diff < tolerance:
                return x_imp, test_sigma
            
            # Update best
            if var_diff < best_var_diff:
                best_var_diff = var_diff
                best_sigma = test_sigma
                best_x = x_imp
            
            # Binary search update
            if imp_var < target_variance:
                sigma_low = test_sigma  # Need more noise
            else:
                sigma_high = test_sigma  # Need less noise
        
        # Return best found
        if best_x is None:
            best_x = self.impute(
                x_obs, m_obs, m_target, steps, sigma, solver,
                resample_enabled=True, resample_interval=5, resample_sigma=best_sigma,
            )
        
        return best_x, best_sigma

    @torch.no_grad()
    def get_mnar_diagnosis(
        self,
        x_imputed: torch.Tensor,
        m_obs: torch.Tensor,
        m_target: torch.Tensor,
        threshold: float = 0.15,
    ) -> dict:
        """
        Get MNAR diagnosis for already-imputed data.

        Uses unified mnar_score module for consistent computation.

        Args:
            x_imputed: Complete data (imputed) (B, D)
            m_obs: Original observed mask (B, D)
            m_target: Original missing mask (B, D)
            threshold: Detection threshold for |S| score (default 0.15)

        Returns:
            dict with diagnosis results including MNARScoreResult
        """
        from src.theory.mnar_score import compute_mnar_score, interpret_mnar_score

        # Use unified MNAR score computation
        mnar_result = compute_mnar_score(
            x_imputed, m_obs, m_target,
            normalize=True,
            threshold=threshold,
        )

        # Generate interpretation
        interpretation = interpret_mnar_score(mnar_result)

        return {
            'mnar_result': mnar_result,
            'mnar_score': mnar_result.mnar_score,
            'selection_bias_ratio': mnar_result.selection_bias_ratio,
            'e_x2_missing': mnar_result.e_x2_missing,
            'e_x2_observed': mnar_result.e_x2_observed,
            'is_mnar_detected': mnar_result.is_mnar_detected,
            'mnar_direction': mnar_result.direction,
            'mnar_confidence': mnar_result.confidence,
            'threshold': threshold,
            'interpretation': interpretation,
        }

    @torch.no_grad()
    def robust_mnar_detection(
        self,
        x_obs: torch.Tensor,
        m_obs: torch.Tensor,
        m_target: torch.Tensor,
        x_imputed: Optional[torch.Tensor] = None,
        steps: int = 50,
        sigma: float = 1.0,
        solver: str = "heun",
        threshold: float = 0.15,
        run_all_tests: bool = True,
    ) -> dict:
        """
        Robust MNAR detection using multiple independent signals.

        This method addresses the circular dependency problem in MNAR detection
        by combining multiple detection strategies:
        1. Selection bias score (original)
        2. Holdout validation
        3. Distribution tests (KS test)
        4. Velocity field analysis

        Args:
            x_obs: Observed data (B, D)
            m_obs: Observed mask (B, D)
            m_target: Missing mask (B, D)
            x_imputed: Pre-imputed data (optional, will impute if not provided)
            steps: ODE steps for imputation
            sigma: Source noise scale
            solver: ODE solver
            threshold: Detection threshold
            run_all_tests: Whether to run all detection methods

        Returns:
            dict with RobustMNARResult and interpretation
        """
        from src.theory.robust_mnar_detection import RobustMNARDetector

        # Impute if not provided
        if x_imputed is None:
            x_imputed = self.impute(
                x_obs, m_obs, m_target,
                steps=steps, sigma=sigma, solver=solver,
                resample_enabled=False,  # No resample for accurate detection
            )

        # Create detector and run
        detector = RobustMNARDetector(
            model=self,
            threshold=threshold,
        )

        result = detector.detect(
            x_obs, m_obs, m_target, x_imputed,
            run_all_tests=run_all_tests,
        )

        return {
            'robust_result': result,
            'is_mnar_detected': result.is_mnar_detected,
            'confidence': result.confidence,
            'mnar_score': result.selection_bias.mnar_score,
            'n_signals_positive': result.n_signals_positive,
            'n_signals_total': result.n_signals_total,
            'interpretation': result.interpretation,
            'x_imputed': x_imputed,
        }

    @torch.no_grad()
    def bidirectional_mnar_detection(
        self,
        x_obs: torch.Tensor,
        m_obs: torch.Tensor,
        m_target: torch.Tensor,
        steps: int = 50,
        sigma: float = 1.0,
        solver: str = "heun",
        lambda_weight: float = 0.5,
        threshold: float = 0.15,
        t_backward: float = 0.5,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> dict:
        """
        Bidirectional MNAR detection with bootstrap confidence interval.

        This method addresses reviewer concerns by:
        1. Using BOTH forward and backward bridges for detection (Issue 1)
        2. Providing statistical guarantees via bootstrap CI (Issue 3)

        The bidirectional score combines:
        - S_F: Forward score from imputed value magnitudes
        - S_B: Backward score from backward velocity magnitudes
        - S_bi = S_F + λ * S_B

        Args:
            x_obs: Observed data (B, D)
            m_obs: Observed mask (B, D)
            m_target: Missing mask (B, D)
            steps: ODE integration steps
            sigma: Source noise scale
            solver: ODE solver
            lambda_weight: Weight for backward score (default 0.5)
            threshold: Detection threshold
            t_backward: Time point for backward velocity evaluation
            n_bootstrap: Number of bootstrap samples for CI
            confidence_level: Confidence level for CI (default 0.95)

        Returns:
            dict with:
                - 'x_imputed': Imputed data
                - 'bidirectional_result': BidirectionalMNARResult
                - 'bootstrap_result': BootstrapMNARResult (for forward score)
                - 'score_forward': S_F
                - 'score_backward': S_B
                - 'score_bidirectional': S_bi
                - 'is_mnar_detected': Detection decision
                - 'p_value': Bootstrap p-value
                - 'ci': Confidence interval
        """
        B, D = x_obs.shape
        device = x_obs.device
        m_obs_f = m_obs.float()
        m_target_f = m_target.float()

        # Step 1: Impute using forward bridge
        x_imputed = self.impute(
            x_obs, m_obs, m_target,
            steps=steps, sigma=sigma, solver=solver,
        )

        # Step 2: Get backward velocity
        t = torch.full((B, 1), t_backward, device=device)
        v_backward = self.backward_bridge(x_imputed, m_obs_f, m_obs_f, t)

        # Step 3: Compute bidirectional MNAR score
        bi_result = compute_bidirectional_mnar_score(
            x_imputed, v_backward, m_obs, m_target,
            lambda_weight=lambda_weight,
            normalize=True,
            threshold=threshold,
        )

        # Step 4: Bootstrap CI for forward score (statistical guarantee)
        bootstrap_result = bootstrap_mnar_score(
            x_imputed, m_obs, m_target,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            normalize=True,
        )

        return {
            # Primary outputs
            'x_imputed': x_imputed,
            'bidirectional_result': bi_result,
            'bootstrap_result': bootstrap_result,

            # Key scores
            'score_forward': bi_result.score_forward,
            'score_backward': bi_result.score_backward,
            'score_bidirectional': bi_result.score_bidirectional,

            # Detection
            'is_mnar_detected': bi_result.is_mnar_detected,
            'confidence': bi_result.confidence,
            'direction': bi_result.direction,

            # Statistical guarantees
            'p_value': bootstrap_result.p_value,
            'ci': (bootstrap_result.ci_lower, bootstrap_result.ci_upper),
            'ci_contains_zero': (bootstrap_result.ci_lower <= 0 <= bootstrap_result.ci_upper),

            # Backward diagnostics
            'v_backward': v_backward,
            't_backward': t_backward,
        }

