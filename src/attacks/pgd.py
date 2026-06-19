"""
PGD (Projected Gradient Descent) adversarial attack for PyTorch Geometric graphs.

Iterative feature-space evasion attack: applies multiple FGSM-like steps with
projection to maintain bounded perturbation.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


def pgd_attack(
    model: nn.Module,
    data: Data,
    epsilon: float = 0.05,
    alpha: float = 0.01,
    steps: int = 10,
    attack_only_malicious: bool = True,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> Data:
    """
    Apply PGD (Projected Gradient Descent) attack to a PyTorch Geometric graph.

    Iteratively applies FGSM-style updates to node features, then projects
    perturbations back into an L∞ epsilon-ball around the original features.

    Args:
        model: A GNN model (e.g., GCN_NIDS, GAT_NIDS). Must be differentiable
               and support forward pass on PyG Data objects.
        data: PyTorch Geometric Data object with:
            - data.x: node features (num_nodes, num_features)
            - data.edge_index: edge indices (2, num_edges)
            - data.y: node labels (num_nodes,), binary where 0=benign, 1=attack
        epsilon: L∞ bound on perturbation magnitude.
        alpha: Step size for gradient ascent in each iteration.
        steps: Number of PGD iterations.
        attack_only_malicious: If True, only perturb nodes where data.y == 1.
                              If False, perturb all nodes.
        clip_min: If provided, clamp adversarial features to be >= clip_min.
        clip_max: If provided, clamp adversarial features to be <= clip_max.

    Returns:
        A new PyTorch Geometric Data object with adversarial node features.
        Preserves edge_index, y, and all other attributes.

    Raises:
        ValueError: If epsilon, alpha, or steps are invalid (<=0).

    Notes:
        If attack_only_malicious=True and no malicious nodes exist, the original 
        graph is returned unchanged.
    """
    # Validate parameters
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    model.eval()

    # Store original features and initialize adversarial features
    x_original = data.x.clone().detach()
    device = data.x.device
    x_adv = x_original.clone().detach().to(device)

    # Get the attack mask once
    if attack_only_malicious:
        mask = data.y == 1
        if mask.sum() == 0:
            # No malicious nodes to attack; return original data
            data_adv = data.clone()
            return data_adv

    # Iterative attack
    for _ in range(steps):
        x_adv = x_adv.detach().to(device).requires_grad_(True)

        # Create a temporary data object with current adversarial features
        data_adv = data.clone()
        data_adv.x = x_adv

        # Forward pass
        with torch.enable_grad():
            logits = model(data_adv)

            # Compute loss on targeted nodes
            if attack_only_malicious:
                loss = F.cross_entropy(logits[mask], data.y[mask])
            else:
                loss = F.cross_entropy(logits, data.y)

            # Compute gradient
            model.zero_grad(set_to_none=True)
            loss.backward()
            grad_sign = x_adv.grad.sign()

        # Update with gradient ascent and projection
        with torch.no_grad():
            if attack_only_malicious:
                x_adv_updated = x_adv.detach().clone()
                x_adv_updated[mask] = x_adv_updated[mask] + alpha * grad_sign[mask]
                x_adv = x_adv_updated
            else:
                x_adv = x_adv.detach() + alpha * grad_sign

            # Project back into epsilon-ball
            perturbation = x_adv - x_original
            perturbation = torch.clamp(perturbation, min=-epsilon, max=epsilon)
            x_adv = x_original + perturbation

            # Optional feature clipping
            if clip_min is not None:
                x_adv = torch.clamp(x_adv, min=clip_min)
            if clip_max is not None:
                x_adv = torch.clamp(x_adv, max=clip_max)

    # Create final adversarial data object
    data_adv = data.clone()
    data_adv.x = x_adv.detach()
    return data_adv