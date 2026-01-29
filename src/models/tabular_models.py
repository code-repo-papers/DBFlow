"""
Tabular data backbone models for BiFlow.
Aligned with impute-FM / MACFM architecture.

Supports optional mask input for conditioning the model on missing patterns.
"""

import torch
import torch.nn as nn
from typing import Literal


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous time t."""
    
    def __init__(self, num_channels: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, t_cont: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_cont: (B,) continuous time values in [0, 1]
        
        Returns:
            embeddings: (B, num_channels) sinusoidal embeddings
        """
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=t_cont.device)
        denom = (self.num_channels // 2 - (1 if self.endpoint else 0))
        denom = max(1, int(denom))
        freqs = freqs / denom
        freqs = (1 / self.max_positions) ** freqs
        x = t_cont.ger(freqs.to(t_cont.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class TabularBackboneV2(nn.Module):
    """
    MLP backbone for tabular data with time conditioning.

    Architecture: x_proj + time_embed → MLP → hidden features

    Two MLP modes available:
    - mlp_mode="macfm": Fixed 3-layer structure matching MACFM (recommended)
    - mlp_mode="dynamic": Configurable nlayers with dropout (original)

    Optional mask input:
    - use_mask_input=True: Concatenate mask to input for conditioning
    - use_mask_input=False: Standard input without mask (default)
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 512,
        nlayers: int = 3,
        dropout: float = 0.1,
        swap_sincos: bool = True,
        mlp_mode: Literal["macfm", "dynamic"] = "macfm",
        use_mask_input: bool = False,
        **kwargs
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.swap_sincos = swap_sincos
        self.mlp_mode = mlp_mode
        self.use_mask_input = use_mask_input

        # Input projection (with optional mask concatenation)
        input_dim = d_in * 2 if use_mask_input else d_in
        self.input_proj = nn.Linear(input_dim, d_model)

        # Time embedding
        self.time_map = PositionalEmbedding(d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # MLP layers - two modes available
        if mlp_mode == "macfm":
            # Fixed structure matching MACFM exactly (recommended for best performance)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 2),      # 512 → 1024
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model * 2),  # 1024 → 1024
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model),      # 1024 → 512
                nn.SiLU(),
            )
        else:
            # Dynamic structure with configurable nlayers and dropout
            layers = []
            for i in range(nlayers):
                layers.extend([
                    nn.Linear(d_model, d_model * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Linear(d_model * 2, d_model),
                    nn.SiLU(),
                ])
            self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        m_obs: torch.Tensor,
        m_cond: torch.Tensor,
        t_cont: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (B, D)
            m_obs: Observed mask (B, D) - used if use_mask_input=True
            m_cond: Conditioning mask (B, D) - reserved for future use
            t_cont: Continuous time (B, 1) or (B,)

        Returns:
            h: Hidden features (B, d_model)
        """
        B = x.shape[0]

        # Handle time dimension
        t = t_cont.squeeze(-1) if t_cont.dim() > 1 else t_cont

        # Time embedding with optional sin/cos swap
        t_emb = self.time_map(t)
        if self.swap_sincos:
            t_emb = t_emb.reshape(B, 2, -1).flip(1).reshape(B, -1)
        t_emb = self.time_embed(t_emb)

        # Input projection (with optional mask)
        if self.use_mask_input:
            # Concatenate mask to input: [x, m_obs] -> (B, 2D)
            x_with_mask = torch.cat([x, m_obs.float()], dim=-1)
            h = self.input_proj(x_with_mask)
        else:
            h = self.input_proj(x)

        h = h + t_emb
        h = self.mlp(h)

        return h


class TabularFlowHead(nn.Module):
    """Output head for velocity prediction."""
    
    def __init__(self, d_model: int, d_out: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_out),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden features (B, d_model)
        
        Returns:
            v: Predicted velocity (B, d_out)
        """
        return self.head(h)


class TabularFlowNet(nn.Module):
    """
    Complete flow network: backbone + head.

    Predicts velocity v given (x_t, masks, t).

    Args:
        mlp_mode: "macfm" (fixed 3-layer, recommended) or "dynamic" (configurable)
        use_mask_input: Whether to use mask as additional input
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 512,
        nlayers: int = 3,
        dropout: float = 0.1,
        mlp_mode: Literal["macfm", "dynamic"] = "macfm",
        use_mask_input: bool = False,
        **kwargs
    ):
        super().__init__()
        self.backbone = TabularBackboneV2(
            d_in=d_in,
            d_model=d_model,
            nlayers=nlayers,
            dropout=dropout,
            mlp_mode=mlp_mode,
            use_mask_input=use_mask_input,
            **kwargs
        )
        self.head = TabularFlowHead(d_model, d_in)
        self.use_mask_input = use_mask_input
    
    def forward(
        self,
        x: torch.Tensor,
        m_obs: torch.Tensor,
        m_cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input (B, D)
            m_obs: Observed mask (B, D)
            m_cond: Conditioning mask (B, D)
            t: Time (B, 1)
        
        Returns:
            v: Predicted velocity (B, D)
        """
        h = self.backbone(x, m_obs, m_cond, t)
        return self.head(h)

