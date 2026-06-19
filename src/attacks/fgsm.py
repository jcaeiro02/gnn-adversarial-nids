"""
FGSM (Fast Gradient Sign Method) adversarial attack for PyTorch Geometric graphs.

Feature-space evasion attack: perturbs only node features, not graph structure.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


def fgsm_attack(
    model: nn.Module,
    data: Data,
    epsilon: float = 0.05,
    attack_only_malicious: bool = True,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> Data:
    """
    Apply FGSM (Fast Gradient Sign Method) attack to a PyTorch Geometric graph.

    Computes the sign of the gradient of the loss with respect to node features
    and moves each node in the direction of increasing loss by epsilon.

    Args:
        model: A GNN model (e.g., GCN_NIDS, GAT_NIDS). Must be differentiable
               and support forward pass on PyG Data objects.
        data: PyTorch Geometric Data object with:
            - data.x: node features (num_nodes, num_features)
            - data.edge_index: edge indices (2, num_edges)
            - data.y: node labels (num_nodes,), binary where 0=benign, 1=attack
        epsilon: Attack strength; perturbation magnitude in L∞ norm.
        attack_only_malicious: If True, only perturb nodes where data.y == 1.
                              If False, perturb all nodes.
        clip_min: If provided, clamp adversarial features to be >= clip_min.
        clip_max: If provided, clamp adversarial features to be <= clip_max.

    Returns:
        A new PyTorch Geometric Data object with adversarial node features.
        Preserves edge_index, y, and all other attributes.

    Notes:
        If attack_only_malicious=True and no malicious nodes exist, the original
        graph is returned unchanged.
    """
    model.eval()

    # Clone the data object and prepare features for gradient computation
    data_adv = data.clone()
    device = data_adv.x.device
    
    # Detach and enable gradient tracking on node features
    x_adv = data_adv.x.detach().clone().to(device).requires_grad_(True)
    data_adv.x = x_adv

    # Forward pass
    with torch.enable_grad():
        logits = model(data_adv)

        # Compute loss on targeted nodes
        if attack_only_malicious:
            mask = data_adv.y == 1
            # Handle empty mask gracefully
            if mask.sum() == 0:
                # No malicious nodes to attack; return original data
                data_adv.x = data_adv.x.detach()
                return data_adv
            loss = F.cross_entropy(logits[mask], data_adv.y[mask])
        else:
            loss = F.cross_entropy(logits, data_adv.y)

        # Compute gradient and extract its sign
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad_sign = x_adv.grad.sign()

    # Apply perturbation
    with torch.no_grad():
        if attack_only_malicious:
            mask = data_adv.y == 1
            x_perturbed = x_adv.detach().clone()
            x_perturbed[mask] = x_perturbed[mask] + epsilon * grad_sign[mask]
        else:
            x_perturbed = x_adv.detach() + epsilon * grad_sign

        # Optional feature clipping
        if clip_min is not None:
            x_perturbed = torch.clamp(x_perturbed, min=clip_min)
        if clip_max is not None:
            x_perturbed = torch.clamp(x_perturbed, max=clip_max)

    # Create output data object with adversarial features
    data_adv.x = x_perturbed
    return data_adv

