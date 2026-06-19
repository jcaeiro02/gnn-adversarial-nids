"""
Adversarial attack implementations for GNN-based NIDS.

Features:
- FGSM (Fast Gradient Sign Method): single-step attack
- PGD (Projected Gradient Descent): iterative attack with projection
- Both support selective node attacks (malicious nodes only)
- Feature clipping and bounded perturbations
"""

from .fgsm import fgsm_attack
from .pgd import pgd_attack

__all__ = ["fgsm_attack", "pgd_attack"]
