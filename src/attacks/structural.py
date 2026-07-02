"""
Structural adversarial attacks for PyTorch Geometric graphs.

This module implements topology-based evasion attacks by perturbing the graph
structure (edge_index) while preserving node features.
"""

import torch
from torch_geometric.data import Data
from torch.nn.functional import cosine_similarity


def edge_removal_attack(
    data: Data,
    perturbation_rate: float = 0.10,
    attack_only_malicious: bool = True,
) -> Data:
    """
    Apply a structural edge removal attack.

    Randomly removes a percentage of graph edges. Optionally restricts the
    perturbation to edges incident to malicious nodes.

    Args:
        data: PyTorch Geometric Data object containing:
            - data.edge_index
            - data.edge_attr (optional)
            - data.y
        perturbation_rate:
            Approximate fraction of candidate existing edges to remove.
            When attack_only_malicious=True, candidate edges are those incident to
            malicious nodes. Reverse edges are handled consistently when present.
        attack_only_malicious:
            If True, only remove edges connected to malicious nodes.

    Returns:
        A new PyTorch Geometric Data object with a perturbed edge_index.

    Raises:
        ValueError:
            If perturbation_rate is outside (0,1].
    """

    # ------------------------------------------------------------------
    # Validate parameters
    # ------------------------------------------------------------------

    if perturbation_rate <= 0 or perturbation_rate > 1:
        raise ValueError(
            f"perturbation_rate must be in (0,1], got {perturbation_rate}"
        )

    # ------------------------------------------------------------------
    # Clone graph
    # ------------------------------------------------------------------

    data_adv = data.clone()

    edge_index = data.edge_index.clone()

    edge_attr = None
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_attr = data.edge_attr.clone()

    # Identify candidate edges for removal
    if attack_only_malicious:
        node_mask = data.y == 1

        if node_mask.sum() == 0:
            return data_adv

        src = edge_index[0]
        dst = edge_index[1]

        candidate_edges = node_mask[src] | node_mask[dst]
    else:
        candidate_edges = torch.ones(
            edge_index.size(1),
            dtype=torch.bool,
            device=edge_index.device
        )

    # Get indices of candidate edges
    candidate_idx = candidate_edges.nonzero(as_tuple=False).view(-1)

    # If there are no candidate edges, return original graph
    if candidate_idx.numel() == 0:
        return data_adv

    # Number of edges to remove.
    # Candidate edges include both directions.
    # Remove only half so that removing reverse edges results in approximately perturbation_rate.  
    num_remove = int((candidate_idx.numel() * perturbation_rate) / 2)

    # Guarantee at least 1 edge is removed
    num_remove = max(1, num_remove)

    # Randomly select candidate edges to remove
    random_order = torch.randperm(candidate_idx.numel(), device=edge_index.device)
    remove_idx = candidate_idx[random_order[:num_remove]]

    # Also remove reverse edges to preserve bidirectional consistency
    edges_to_remove = set()

    for idx in remove_idx:
        idx = int(idx.item())

        src = int(edge_index[0, idx].item())
        dst = int(edge_index[1, idx].item())

        edges_to_remove.add((src, dst))
        edges_to_remove.add((dst, src))


    # Create keep mask: start by keeping all edges
    # keep_edges = [True, True, True, True, True]
    keep_edges = torch.ones(
        edge_index.size(1),
        dtype=torch.bool,
        device=edge_index.device
    )

    # Mark selected edges and their reverse counterparts for removal
    for i in range(edge_index.size(1)):
        src = int(edge_index[0, i].item())
        dst = int(edge_index[1, i].item())

        if (src, dst) in edges_to_remove:
            keep_edges[i] = False

    # Apply mask to edge_index
    data_adv.edge_index = edge_index[:, keep_edges]

    # Apply same mask to edge_attr, if it exists
    if edge_attr is not None:
        data_adv.edge_attr = edge_attr[keep_edges]

    return data_adv


def edge_addition_attack(
    data: Data,
    perturbation_rate: float = 0.10,
    attack_only_malicious: bool = True,
    avoid_duplicates: bool = True,
) -> Data:
    """
    Apply a structural edge addition attack.

    Randomly inserts new edges into the graph. Optionally restricts new
    connections to malicious nodes.

    Args:
        data: PyTorch Geometric Data object.
        perturbation_rate:
            Approximate fraction of the current total number of edges to add.
            New edges originate from malicious nodes when attack_only_malicious=True.
            Reverse edges are added consistently when the graph is treated as undirected.
        attack_only_malicious:
            If True, only create new edges incident to malicious nodes.
        avoid_duplicates:
            Prevent insertion of existing edges.

    Returns:
        A new PyTorch Geometric Data object with additional edges.

    Raises:
        ValueError:
            If perturbation_rate is outside (0,1].
    """

    # ------------------------------------------------------------------
    # Validate parameters
    # ------------------------------------------------------------------

    if perturbation_rate <= 0 or perturbation_rate > 1:
        raise ValueError(
            f"perturbation_rate must be in (0,1], got {perturbation_rate}"
        )

    # ------------------------------------------------------------------
    # Clone graph
    # ------------------------------------------------------------------

    data_adv = data.clone()

    edge_index = data.edge_index.clone()

    edge_attr = None
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_attr = data.edge_attr.clone()

    # Identify candidate source nodes
    if attack_only_malicious:
        node_mask = data.y == 1

        if node_mask.sum() == 0:
            return data_adv
        
        source_nodes = node_mask.nonzero(as_tuple=False).view(-1)
    else:
        source_nodes = torch.arange(data.num_nodes, device=data.edge_index.device)

    # Number of directed edge pairs to add
    num_add = int((edge_index.size(1) * perturbation_rate) / 2)
    num_add = max(1, num_add)

    # All nodes are possible destinations
    destination_nodes = torch.arange(
        data.num_nodes,
        device=edge_index.device
    )

    # Store existing edges to avoid duplicates
    existing_edges = set()
    if avoid_duplicates:
        for i in range(edge_index.size(1)):
            src = int(edge_index[0, i].item())
            dst = int(edge_index[1, i].item())
            existing_edges.add((src, dst))

    # Generate new valid edges
    new_edges_list = []
    max_attempts = num_add * 10
    attempts = 0

    while len(new_edges_list) < num_add and attempts < max_attempts:
        attempts += 1

        src_pos = torch.randint(
            0,
            source_nodes.numel(),
            (1,),
            device=edge_index.device,
        )
        src = int(source_nodes[src_pos].item())

        dst_pos = torch.randint(
            0,
            destination_nodes.numel(),
            (1,),
            device=edge_index.device,
        )
        dst = int(destination_nodes[dst_pos].item())

        # Avoid self-loops
        if src == dst:
            continue

        edge = (src, dst)
        reverse_edge = (dst, src)

        # Avoid duplicate edges
        if avoid_duplicates and (edge in existing_edges or reverse_edge in existing_edges):
            continue

        new_edges_list.append(edge)

        if avoid_duplicates:
            existing_edges.add(edge)
            existing_edges.add(reverse_edge)

    if len(new_edges_list) == 0:
        return data_adv

    # Convert list of edges to PyG edge_index format [2, num_new_edges]
    new_edges = torch.tensor(
        new_edges_list,
        dtype=torch.long,
        device=edge_index.device,
    ).t().contiguous()

    # Add reverse edges to preserve the bidirectional graph structure
    reverse_edges = new_edges.flip(0)
    new_edges = torch.cat([new_edges, reverse_edges], dim=1)

    # Add new edges to graph
    data_adv.edge_index = torch.cat([edge_index, new_edges], dim=1)

    # Compute edge_attr for new edges using cosine similarity
    if edge_attr is not None:
        new_src = new_edges[0]
        new_dst = new_edges[1]

        new_edge_attr = cosine_similarity(
            data.x[new_src],
            data.x[new_dst],
            dim=1,
        ).view(-1, 1)

        data_adv.edge_attr = torch.cat(
            [edge_attr, new_edge_attr.to(edge_attr.dtype)],
            dim=0,
        )

    return data_adv