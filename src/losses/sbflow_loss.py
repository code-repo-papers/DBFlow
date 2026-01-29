"""
SB-FLOW Loss Functions

Components:
1. Forward FM Loss: L_F - velocity matching for x0 → x1
2. Backward FM Loss: L_B - velocity matching for x1 → x0  
3. Cycle Consistency Loss: L_cycle - F(B(x)) ≈ x constraint

Total Loss: L = L_F + λ_B * L_B + λ_cycle * L_cycle
"""

from typing import Optional, Tuple
import math
import torch


def _get_path_schedule(schedule_type: str = "linear", gamma: Optional[float] = None):
    """
    Get path schedule functions s(t) and s'(t).
    
    For linear: s(t) = t, s'(t) = 1
    For cosine: s(t) = 0.5 * (1 - cos(πt)), s'(t) = π/2 * sin(πt)
    """
    if schedule_type == "linear":
        return (lambda t: t, lambda t: torch.ones_like(t))
    elif schedule_type == "power":
        g = float(gamma) if gamma is not None else 2.0
        return (lambda t: torch.pow(t, g), lambda t: g * torch.pow(t, g - 1))
    elif schedule_type == "cosine":
        return (
            lambda t: 0.5 * (1 - torch.cos(math.pi * t)),
            lambda t: (math.pi / 2) * torch.sin(math.pi * t),
        )
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")


def forward_fm_loss(
    model,
    x0: torch.Tensor,
    x1: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    t: torch.Tensor,
    path_schedule: str = "linear",
) -> torch.Tensor:
    """
    Forward bridge flow matching loss.
    
    Learns velocity field for x0 → x1 direction.
    
    Args:
        model: SBFlowModel
        x0: Source samples (B, D) - observed + noise on missing
        x1: Target samples (B, D) - clean complete data
        m_obs: Observed mask (B, D)
        m_target: Missing/target mask (B, D)
        t: Time samples (B, 1) - must be provided (sampled in sbflow_loss)
        path_schedule: Time schedule type
    
    Returns:
        loss: Forward FM loss
    """
    # Get schedule functions
    s_fn, s_prime_fn = _get_path_schedule(path_schedule)
    s_t = s_fn(t)
    
    # Interpolate: x_t = (1 - s(t)) * x0 + s(t) * x1
    x_t = (1 - s_t) * x0 + s_t * x1
    
    # Target velocity: v_target = x1 - x0 (for linear path with s'(t)=1)
    # For general schedule: v_target = s'(t) * (x1 - x0)
    s_prime_t = s_prime_fn(t)
    v_target = s_prime_t * (x1 - x0)
    
    # Predict velocity using forward bridge
    v_pred = model.forward_bridge(x_t, m_obs.float(), m_obs.float(), t)
    
    # Loss: MSE weighted by target mask (focus on missing positions)
    sq_diff = (v_pred - v_target) ** 2
    
    # Weight by target mask
    m_t = m_target.float()
    loss = (sq_diff * m_t).sum() / (m_t.sum() + 1e-6)
    
    return loss


def backward_fm_loss(
    model,
    X: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    t: torch.Tensor,
    path_schedule: str = "linear",
) -> torch.Tensor:
    """
    Backward bridge flow matching loss (MACFM-style mask-aware).
    
    Learns velocity field for data → noise direction (models missing mechanism).
    This helps SB-FLOW understand how data becomes missing (MNAR modeling).
    
    Uses same mask-aware interpolation as forward:
    - x_t = m_obs * X + m_target * ((1-s(t)) * X + s(t) * eps)
    - At t=0: x_0 = X (clean data)
    - At t=1: x_1 = m_obs * X + m_target * eps (data with noise on missing)
    
    Args:
        model: SBFlowModel
        X: Clean data (B, D)
        m_obs: Observed mask (B, D)
        m_target: Missing/target mask (B, D)
        t: Time samples (B, 1)
        path_schedule: Time schedule type
    
    Returns:
        loss: Backward FM loss
    """
    # Get schedule functions
    s_fn, s_prime_fn = _get_path_schedule(path_schedule)
    s_t = s_fn(t)
    s_prime_t = s_prime_fn(t)
    
    m_obs_f = m_obs.float()
    m_target_f = m_target.float()
    
    # Sample noise for backward direction
    eps = torch.randn_like(X)
    
    # Mask-aware interpolation for backward: X → eps on missing positions
    # x_t = m_obs * X + m_target * ((1-s(t)) * X + s(t) * eps)
    x_t = m_obs_f * X + m_target_f * ((1 - s_t) * X + s_t * eps)
    
    # Target velocity: d/dt of x_t on missing positions
    # v_target = m_target * s'(t) * (eps - X)
    v_target = m_target_f * s_prime_t * (eps - X)
    
    # Predict velocity using backward bridge
    v_pred = model.backward_bridge(x_t, m_obs_f, m_obs_f, t)
    
    # Loss on missing positions
    loss = ((v_pred - v_target) ** 2 * m_target_f).sum() / (m_target_f.sum() + 1e-6)
    
    return loss


def cycle_consistency_loss(
    model,
    X: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    steps: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Cycle consistency loss: B(F(eps)) ≈ eps (on missing positions)
    
    This encourages the forward and backward bridges to be inverses.
    
    Forward: eps → X (noise to data)
    Backward: X → eps (data to noise)
    Cycle: eps → X → eps_recon, expect eps_recon ≈ eps
    
    Args:
        model: SBFlowModel
        X: Clean data (B, D)
        m_obs: Observed mask (B, D)
        m_target: Missing mask (B, D)
        steps: ODE integration steps
        sigma: Noise scale
    
    Returns:
        loss: Cycle consistency loss
    """
    B, D = X.shape
    device = X.device
    
    m_obs_f = m_obs.float()
    m_target_f = m_target.float()
    
    # Start from noise on missing positions
    eps = torch.randn_like(X)
    x_start = m_obs_f * X + m_target_f * eps
    
    dt = 1.0 / steps
    
    # Forward pass: eps → X_pred (t: 0 → 1)
    x_cur = x_start.clone()
    for i in range(steps):
        t = torch.full((B, 1), i / steps, device=device)
        v = model.forward_bridge(x_cur, m_obs_f, m_obs_f, t)
        x_cur = x_cur + dt * v
        # Keep observed positions fixed
        x_cur = m_obs_f * X + m_target_f * x_cur
    X_pred = x_cur
    
    # Backward pass: X_pred → eps_recon (t: 0 → 1)
    x_cur = X_pred.clone()
    for i in range(steps):
        t = torch.full((B, 1), i / steps, device=device)
        v = model.backward_bridge(x_cur, m_obs_f, m_obs_f, t)
        x_cur = x_cur + dt * v
        # Keep observed positions fixed
        x_cur = m_obs_f * X + m_target_f * x_cur
    eps_recon = x_cur
    
    # Cycle loss: reconstruction error on missing positions
    # Compare reconstructed noise with original noise
    sq_diff = (eps_recon - x_start) ** 2
    loss = (sq_diff * m_target_f).sum() / (m_target_f.sum() + 1e-6)
    
    return loss


def sbflow_loss(
    model,
    x_clean: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    coupling_indices: Optional[torch.Tensor] = None,
    lambda_B: float = 0.5,
    lambda_cycle: float = 0.1,
    source_sigma: float = 1.0,
    path_schedule: str = "linear",
    time_beta: Tuple[float, float] = (1.0, 1.0),
    cycle_steps: int = 10,
    t_eps: float = 1e-3,
    # NEW: Regularization techniques from MACFM
    lambda_stab: float = 0.0,
    lambda_cons: float = 0.0,
    eta_cons: float = 0.05,
    sigma_in: float = 0.0,
    augment_target_p: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Full SB-FLOW loss function with MACFM-style regularization.
    
    L = L_F + λ_B * L_B + λ_cycle * L_cycle + λ_stab * L_stab + λ_cons * L_cons
    
    NEW regularization (aligned with MACFM):
    - lambda_stab: Stabilization loss - keeps velocity near 0 on observed positions
    - lambda_cons: Consistency regularization - similar predictions for perturbed inputs
    - sigma_in: Input noise on observed positions (decays with time)
    - augment_target_p: Mask augmentation probability
    
    Args:
        model: SBFlowModel
        x_clean: Clean complete data (B, D)
        m_obs: Observed mask (B, D)
        m_target: Missing/target mask (B, D)
        coupling_indices: Optional OT coupling indices (B,)
        lambda_B: Weight for backward loss
        lambda_cycle: Weight for cycle consistency loss
        source_sigma: Noise std for source distribution
        path_schedule: Time schedule type
        time_beta: Beta distribution parameters
        cycle_steps: ODE steps for cycle consistency
        t_eps: Small epsilon for time clamping
        lambda_stab: Stabilization loss weight (NEW)
        lambda_cons: Consistency regularization weight (NEW)
        eta_cons: Perturbation scale for consistency (NEW)
        sigma_in: Input noise on observed positions (NEW)
        augment_target_p: Probability of augmenting target mask (NEW)
    
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual loss components
    """
    B, D = x_clean.shape
    device = x_clean.device
    
    # Sample shared time for F and B
    a, b = time_beta
    if a <= 0 or b <= 0:
        t = torch.rand(B, 1, device=device)
    else:
        t = torch.distributions.Beta(float(a), float(b)).sample((B, 1)).to(device)
    t = t.clamp(min=t_eps, max=1 - t_eps)
    
    # Get schedule functions
    s_fn, s_prime_fn = _get_path_schedule(path_schedule)
    s_t = s_fn(t)
    
    # === MASK AUGMENTATION (from MACFM) ===
    # Randomly move some observed positions to target (improves generalization)
    # Convert to bool for bitwise operations
    m_obs_bool = m_obs.bool() if m_obs.dtype != torch.bool else m_obs
    m_target_bool = m_target.bool() if m_target.dtype != torch.bool else m_target
    
    if augment_target_p > 0:
        aug_mask = torch.rand(B, D, device=device) < augment_target_p
        # Only augment positions that are currently observed
        aug_mask = aug_mask & m_obs_bool
        m_obs_aug = m_obs_bool & (~aug_mask)
        m_target_aug = m_target_bool | aug_mask
    else:
        m_obs_aug = m_obs_bool
        m_target_aug = m_target_bool
    
    m_obs_f = m_obs_aug.float()
    m_target_f = m_target_aug.float()
    
    # === INPUT NOISE ON OBSERVED POSITIONS (from MACFM) ===
    X = x_clean.clone()
    if sigma_in > 0:
        obs_noise = sigma_in * torch.randn_like(X) * (1 - s_t)
        X = X + obs_noise * m_obs_f
    
    # === MACFM-STYLE MASK-AWARE INTERPOLATION ===
    # Key insight: Keep observed positions fixed at X, only interpolate on missing positions
    # This is different from standard FM which interpolates everywhere
    eps = torch.randn_like(X)  # Pure noise as source (not obs + noise)
    
    # Apply OT coupling if provided (affects target pairing)
    if coupling_indices is not None:
        X_paired = X[coupling_indices]
        m_target_paired = m_target_aug[coupling_indices]
        m_obs_paired = m_obs_aug[coupling_indices]
        m_target_f_paired = m_target_paired.float()
        m_obs_f_paired = m_obs_paired.float()
    else:
        X_paired = X
        m_target_paired = m_target_aug
        m_obs_paired = m_obs_aug
        m_target_f_paired = m_target_f
        m_obs_f_paired = m_obs_f
    
    # === MASK-AWARE x_t (MACFM style) ===
    # x_t = m_obs * X + m_target * (s(t) * X + (1 - s(t)) * eps)
    # Observed positions stay at X, missing positions interpolate from eps to X
    x_t = m_obs_f_paired * X_paired + m_target_f_paired * (s_t * X_paired + (1 - s_t) * eps)
    
    # === Target velocity (MACFM style) ===
    # v_true = m_target * s'(t) * (X - eps)
    # Only missing positions have non-zero target velocity
    s_prime_t = s_prime_fn(t)
    v_target = m_target_f_paired * s_prime_t * (X_paired - eps)
    
    # === Forward prediction ===
    v_pred = model.forward_bridge(x_t, m_obs_f_paired, m_obs_f_paired, t)
    
    # === Forward loss (on target positions) ===
    # v_target already has mask applied (zeros on observed), so compute MSE directly
    m_t = m_target_f_paired
    loss_F = ((v_pred - v_target) ** 2 * m_t).sum() / (m_t.sum() + 1e-6)
    
    # === STABILIZATION LOSS (from MACFM) ===
    # Keep velocity near 0 on observed positions (critical for MACFM performance!)
    loss_stab = torch.tensor(0.0, device=device)
    if lambda_stab > 0:
        m_c = m_obs_f_paired
        loss_stab = (v_pred.pow(2) * m_c).sum() / (m_c.sum() + 1e-6)
    
    # === CONSISTENCY LOSS (from MACFM) ===
    # Similar predictions for perturbed inputs
    loss_cons = torch.tensor(0.0, device=device)
    if lambda_cons > 0:
        x_t_pert = x_t + eta_cons * torch.randn_like(x_t) * (1 - s_t)
        v_pred_pert = model.forward_bridge(x_t_pert, m_obs_f, m_obs_f, t)
        loss_cons = ((v_pred_pert - v_pred) ** 2 * m_t).sum() / (m_t.sum() + 1e-6)
    
    # === Backward loss (models missing mechanism, helps MNAR detection) ===
    loss_B = torch.tensor(0.0, device=device)
    if lambda_B > 0:
        loss_B = backward_fm_loss(
            model, X_paired, m_obs_paired, m_target_paired,
            t=t, path_schedule=path_schedule
        )
    
    # === Cycle consistency loss (regularization, ensures F and B are inverses) ===
    loss_cycle = torch.tensor(0.0, device=device)
    if lambda_cycle > 0:
        # Use eps as starting point (same as training)
        loss_cycle = cycle_consistency_loss(
            model, X, m_obs_aug, m_target_aug,
            steps=cycle_steps, sigma=source_sigma
        )
    # Total loss
    total_loss = (
        loss_F 
        + lambda_B * loss_B 
        + lambda_cycle * loss_cycle
        + lambda_stab * loss_stab
        + lambda_cons * loss_cons
    )
    
    loss_dict = {
        'loss_F': loss_F.item(),
        'loss_B': loss_B.item(),
        'loss_cycle': loss_cycle.item() if isinstance(loss_cycle, torch.Tensor) else loss_cycle,
        'loss_stab': loss_stab.item() if isinstance(loss_stab, torch.Tensor) else loss_stab,
        'loss_cons': loss_cons.item() if isinstance(loss_cons, torch.Tensor) else loss_cons,
        'total': total_loss.item(),
    }
    
    return total_loss, loss_dict

