"""
Mask-Aware Optimal Transport for BiFlow

Core innovation: Design mask-conditioned cost function for OT coupling
- Missing positions get higher weight in the cost
- Observed positions get lower weight (alpha factor)

Supports multiple backends:
- 'pot': CPU-based POT library (original, slower)
- 'gpu': GPU-accelerated pure PyTorch (recommended)
- 'greedy': Fast greedy matching (approximate but very fast)
"""

import torch
import numpy as np
from typing import Literal, Optional

try:
    import ot as pot  # POT library for optimal transport
    HAS_POT = True
except ImportError:
    HAS_POT = False

# Import GPU OT module
try:
    from .gpu_optimal_transport import GPUMaskAwareOT, MiniBatchOT, HAS_GEOMLOSS
    HAS_GPU_OT = True
except ImportError:
    HAS_GPU_OT = False
    HAS_GEOMLOSS = False


def check_pot_available(raise_error: bool = False) -> bool:
    """
    Check if any OT backend is available.

    With GPU OT support, POT is no longer strictly required.

    Args:
        raise_error: If True, raises ImportError when no OT backend is available.

    Returns:
        True if any OT backend is available, False otherwise.

    Raises:
        ImportError: If raise_error=True and no OT backend is available.
    """
    has_any = HAS_POT or HAS_GPU_OT

    if not has_any and raise_error:
        raise ImportError(
            "No OT backend available. Install POT (pip install POT) "
            "or ensure gpu_optimal_transport.py is accessible."
        )

    return has_any


def get_available_backends() -> dict:
    """
    Get information about available OT backends.

    Returns:
        dict with backend availability and recommendations
    """
    return {
        'pot': HAS_POT,
        'gpu': HAS_GPU_OT,
        'geomloss': HAS_GEOMLOSS if HAS_GPU_OT else False,
        'greedy': True,  # Always available (pure Python)
        'recommended': 'gpu' if HAS_GPU_OT else ('pot' if HAS_POT else 'greedy'),
    }


def mask_aware_ot_cost(
    x0: torch.Tensor,
    x1: torch.Tensor,
    m_target: torch.Tensor,
    m_obs: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Compute mask-aware OT cost matrix.
    
    The key insight: missing positions should contribute more to the cost,
    since that's where we want the OT coupling to be optimal.
    
    Args:
        x0: Source samples (B, D) - observed values + noise on missing
        x1: Target samples (B, D) - clean complete data
        m_target: Missing mask (B, D) - 1 where missing
        m_obs: Observed mask (B, D) - 1 where observed
        alpha: Weight factor for observed positions (default 0.1)
    
    Returns:
        cost_matrix: (B, B) pairwise cost matrix
    """
    B, D = x0.shape
    device = x0.device
    
    # Position weights: missing=1.0, observed=alpha
    # We use mean of m_target across batch for stable weighting
    weights = m_target.float() + alpha * m_obs.float()  # (B, D)
    
    # Compute pairwise weighted L2 distance
    # diff[i,j,k] = x0[i,k] - x1[j,k]
    diff = x0.unsqueeze(1) - x1.unsqueeze(0)  # (B, B, D)
    
    # Weight the squared differences
    # Use weights from source (x0) for consistency
    weighted_sq_diff = (diff ** 2) * weights.unsqueeze(1)  # (B, B, D)
    
    # Sum over features to get cost matrix
    cost_matrix = weighted_sq_diff.sum(dim=-1)  # (B, B)
    
    return cost_matrix


def compute_ot_coupling(
    cost_matrix: torch.Tensor,
    reg: float = 0.05,
    method: str = "sinkhorn",
) -> torch.Tensor:
    """
    Compute OT coupling matrix using Sinkhorn algorithm.
    
    Args:
        cost_matrix: (B, B) pairwise cost matrix
        reg: Entropy regularization parameter
        method: OT method ("sinkhorn" or "emd")
    
    Returns:
        coupling: (B, B) optimal transport plan
    """
    if not HAS_POT:
        # Fallback: random permutation if POT not available
        B = cost_matrix.shape[0]
        perm = torch.randperm(B)
        coupling = torch.zeros(B, B)
        coupling[torch.arange(B), perm] = 1.0 / B
        return coupling
    
    B = cost_matrix.shape[0]
    device = cost_matrix.device
    
    # Uniform marginals
    a = np.ones(B) / B
    b = np.ones(B) / B
    
    # Convert to numpy for POT
    M = cost_matrix.detach().cpu().numpy()
    
    # Normalize cost matrix for numerical stability
    M = M / (M.max() + 1e-8)
    
    if method == "sinkhorn":
        # Sinkhorn is differentiable and faster
        coupling = pot.sinkhorn(a, b, M, reg=reg)
    elif method == "emd":
        # Exact EMD (slower but exact)
        coupling = pot.emd(a, b, M)
    else:
        raise ValueError(f"Unknown OT method: {method}")
    
    return torch.from_numpy(coupling).float().to(device)


def sample_coupling_indices(coupling: torch.Tensor, n_samples: int = None) -> torch.Tensor:
    """
    Sample paired indices from OT coupling matrix.
    
    Args:
        coupling: (B, B) OT coupling matrix
        n_samples: Number of samples (default: B)
    
    Returns:
        indices: (n_samples,) indices for pairing x0[i] with x1[indices[i]]
    """
    B = coupling.shape[0]
    if n_samples is None:
        n_samples = B
    
    # Normalize rows to get conditional distribution P(j|i)
    row_sums = coupling.sum(dim=1, keepdim=True) + 1e-8
    conditional = coupling / row_sums
    
    # Sample one target index for each source
    indices = torch.multinomial(conditional, num_samples=1).squeeze(-1)
    
    return indices


class MaskAwareOTSampler:
    """
    Manager for per-epoch OT coupling computation.

    Supports multiple backends:
    - 'pot': CPU-based POT library (original, slower)
    - 'gpu': GPU-accelerated pure PyTorch (recommended)
    - 'greedy': Fast greedy matching (approximate but very fast)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        reg: float = 0.05,
        method: str = "sinkhorn",
        source_sigma: float = 1.0,
        backend: Literal['auto', 'pot', 'gpu', 'greedy'] = 'auto',
    ):
        """
        Args:
            alpha: Weight factor for observed positions in cost
            reg: Sinkhorn regularization parameter
            method: OT method for POT backend ("sinkhorn" or "emd")
            source_sigma: Noise std for source distribution
            backend: OT backend to use
                - 'auto': Use GPU if available, else POT, else greedy
                - 'pot': CPU-based POT library
                - 'gpu': GPU-accelerated PyTorch
                - 'greedy': Fast greedy matching
        """
        self.alpha = alpha
        self.reg = reg
        self.method = method
        self.source_sigma = source_sigma

        # Select backend
        if backend == 'auto':
            if HAS_GPU_OT:
                self.backend = 'gpu'
            elif HAS_POT:
                self.backend = 'pot'
            else:
                self.backend = 'greedy'
        else:
            self.backend = backend

        # Initialize GPU OT if using it
        self._gpu_ot = None
        if self.backend == 'gpu' and HAS_GPU_OT:
            self._gpu_ot = GPUMaskAwareOT(
                alpha=alpha,
                reg=reg,
                backend='sinkhorn',  # Use pure PyTorch sinkhorn
            )
        elif self.backend == 'greedy':
            if HAS_GPU_OT:
                self._gpu_ot = GPUMaskAwareOT(alpha=alpha, backend='greedy')

        # Cached coupling (computed per epoch)
        self.coupling = None
        self.coupling_indices = None
        self.n_samples = 0

    def compute_epoch_coupling(
        self,
        X: torch.Tensor,
        M_target: torch.Tensor,
        M_obs: torch.Tensor,
        batch_size: int = 2048,
    ):
        """
        Compute OT coupling for entire dataset at epoch start.

        For large datasets, we compute coupling on mini-batches and aggregate.

        Args:
            X: Full dataset (N, D) - clean complete data
            M_target: Missing mask (N, D)
            M_obs: Observed mask (N, D)
            batch_size: Batch size for coupling computation
        """
        N, D = X.shape
        device = X.device
        self.n_samples = N

        # Create source distribution: observed + noise on missing
        noise = torch.randn_like(X) * self.source_sigma
        X0 = X * M_obs.float() + noise * M_target.float()
        X1 = X  # Target is clean data

        if N <= batch_size:
            # Small dataset: compute full coupling
            if self._gpu_ot is not None:
                self.coupling_indices = self._gpu_ot.compute_coupling(
                    X0, X1, M_target, M_obs
                )
                self.coupling = None
            else:
                cost = mask_aware_ot_cost(X0, X1, M_target, M_obs, self.alpha)
                self.coupling = compute_ot_coupling(cost, self.reg, self.method)
                self.coupling_indices = sample_coupling_indices(self.coupling, N)
        else:
            # Large dataset: random pairing with local OT refinement
            # We'll compute coupling per mini-batch during training
            self.coupling = None
            self.coupling_indices = torch.randperm(N, device=device)

    def get_paired_indices(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Get OT-paired target indices for a batch.

        Args:
            batch_indices: (B,) indices of current batch

        Returns:
            paired_indices: (B,) indices for pairing
        """
        if self.coupling_indices is not None:
            return self.coupling_indices[batch_indices]
        else:
            # Fallback: return same indices (identity coupling)
            return batch_indices

    def compute_batch_coupling(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        m_target: torch.Tensor,
        m_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute OT coupling within a mini-batch.

        This is an alternative to per-epoch coupling for online updates.
        Uses GPU-accelerated OT when available.

        Args:
            x0: Source batch (B, D)
            x1: Target batch (B, D)
            m_target: Missing mask (B, D)
            m_obs: Observed mask (B, D)

        Returns:
            indices: (B,) paired indices within batch
        """
        if self._gpu_ot is not None:
            # Use GPU-accelerated OT
            return self._gpu_ot.compute_coupling(x0, x1, m_target, m_obs)
        else:
            # Fallback to POT
            cost = mask_aware_ot_cost(x0, x1, m_target, m_obs, self.alpha)
            coupling = compute_ot_coupling(cost, self.reg, self.method)
            indices = sample_coupling_indices(coupling)
            return indices


def create_source_distribution(
    x_clean: torch.Tensor,
    m_obs: torch.Tensor,
    m_target: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Create data-coupled source distribution for SB-Flow.
    
    x0 = x_clean * m_obs + sigma * noise * m_target
    
    This preserves observed values and adds noise only on missing positions.
    
    Args:
        x_clean: Clean complete data (B, D)
        m_obs: Observed mask (B, D) - 1 where observed
        m_target: Missing mask (B, D) - 1 where missing
        sigma: Noise standard deviation
    
    Returns:
        x0: Source samples (B, D)
    """
    noise = torch.randn_like(x_clean) * sigma
    x0 = x_clean * m_obs.float() + noise * m_target.float()
    return x0

