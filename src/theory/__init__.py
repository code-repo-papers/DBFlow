"""
Theoretical components for MA-BSB.

Provides:
- MNAR detection via backward bridge (Proposition 3)
- Heuristic propensity estimation (Section 6)
- Proposition validation utilities

See docs/theory.md for full theoretical framework.
"""

from .missing_mechanism import (
    MNARDetector,
    MNARTestResult,
    HeuristicPropensityEstimator,
    validate_proposition_1,
)

__all__ = [
    'MNARDetector',
    'MNARTestResult', 
    'HeuristicPropensityEstimator',
    'validate_proposition_1',
]

