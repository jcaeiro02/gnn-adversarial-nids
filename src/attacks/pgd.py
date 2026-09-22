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
    epsilon: float = 0.03,
    alpha: Optional[float] = None,
    steps: int = 20,
    attack_only_malicious: bool = True,
    random_start: bool = True,
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
        epsilon: Maximum L∞ bound on perturbation magnitude.
        alpha: Step size for gradient ascent in each iteration.
        steps: Number of PGD iterations.
        attack_only_malicious: If True, only perturb nodes where data.y == 1.
                              If False, perturb all nodes.
        random_start: If True, initialize attacked nodes at a random point
                      inside the L∞ epsilon-ball before the iterative updates.
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
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if alpha is None:
        alpha = 2.5 * epsilon / steps
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    model.eval()

    # Store original features
    device = data.x.device
    x_original = data.x.clone().detach().to(device)
    x_adv = x_original.clone()

    # Get the attack mask once
    if attack_only_malicious:
        mask = data.y == 1
        if mask.sum() == 0:
            # No malicious nodes to attack; return original data
            data_adv = data.clone()
            return data_adv
    else:
        mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)


    # ---------------------------------------------------------
    # Random initialization inside the L∞ epsilon-ball
    # ---------------------------------------------------------
    if random_start:
        with torch.no_grad():
            random_noise = torch.empty_like(x_adv).uniform_(
                -epsilon,
                epsilon,
            )

            # Apply random initialization only to attacked nodes
            x_adv[mask] = x_adv[mask] + random_noise[mask]

            # Optional feature clipping
            if clip_min is not None:
                x_adv[mask] = torch.clamp(
                    x_adv[mask],
                    min=clip_min,
                )

            if clip_max is not None:
                x_adv[mask] = torch.clamp(
                    x_adv[mask],
                    max=clip_max,
                )

            # Ensure random initialization remains inside epsilon-ball
            perturbation = x_adv - x_original
            perturbation = torch.clamp(
                perturbation,
                min=-epsilon,
                max=epsilon,
            )

            x_adv = x_original + perturbation

            # Guarantee untouched nodes remain unchanged
            x_adv[~mask] = x_original[~mask]


    # Iterative attack
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)

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
                x_adv[mask] = torch.clamp(x_adv[mask], min=clip_min)

            if clip_max is not None:
                x_adv[mask] = torch.clamp(x_adv[mask], max=clip_max)

            # Guarantee untouched nodes remain unchanged
            x_adv[~mask] = x_original[~mask]

    # Create final adversarial data object
    data_adv = data.clone()
    data_adv.x = x_adv.detach()
    return data_adv