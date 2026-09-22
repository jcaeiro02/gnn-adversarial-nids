"""
Random structural perturbations for PyTorch Geometric graphs.

This module implements topology-based perturbations by modifying the graph
structure (edge_index) while preserving node features.

The implemented perturbations are random rather than optimised adversarial
attacks. When attack_only_malicious=True, perturbations are restricted to
connections involving malicious nodes.
"""

import torch
from torch_geometric.data import Data
from torch.nn.functional import cosine_similarity


def _get_candidate_pairs(
    data: Data,
    attack_only_malicious: bool,
):
    """
    Return the set of unique undirected existing edge pairs that define
    the perturbation budget.

    When attack_only_malicious=True, only edges incident to at least one
    malicious node are considered.
    """

    edge_index = data.edge_index

    src = edge_index[0]
    dst = edge_index[1]

    if attack_only_malicious:
        node_mask = data.y == 1

        if node_mask.sum() == 0:
            return set()

        candidate_mask = node_mask[src] | node_mask[dst]
    else:
        candidate_mask = torch.ones(
            edge_index.size(1),
            dtype=torch.bool,
            device=edge_index.device,
        )

    candidate_edges = edge_index[:, candidate_mask]

    candidate_pairs = set()

    for i in range(candidate_edges.size(1)):
        u = int(candidate_edges[0, i].item())
        v = int(candidate_edges[1, i].item())

        if u == v:
            continue

        # Canonical representation of an undirected edge
        pair = (min(u, v), max(u, v))
        candidate_pairs.add(pair)

    return candidate_pairs


def edge_removal_attack(
    data: Data,
    perturbation_rate: float = 0.10,
    attack_only_malicious: bool = True,
) -> Data:
    """
    Apply random structural edge removal.

    A fraction of existing unique edge pairs is randomly removed.
    When attack_only_malicious=True, the perturbation budget is defined
    over existing edges incident to malicious nodes.

    Args:
        data:
            PyTorch Geometric Data object.

        perturbation_rate:
            Fraction of candidate undirected edge pairs to remove.

        attack_only_malicious:
            If True, only edges incident to malicious nodes are candidates.

    Returns:
        A cloned Data object with the selected edges removed.

    Raises:
        ValueError:
            If perturbation_rate is outside (0, 1].
    """

    if perturbation_rate <= 0 or perturbation_rate > 1:
        raise ValueError(
            f"perturbation_rate must be in (0,1], got {perturbation_rate}"
        )

    data_adv = data.clone()

    edge_index = data.edge_index.clone()

    edge_attr = None
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_attr = data.edge_attr.clone()

    # --------------------------------------------------------------
    # Candidate existing edge pairs
    # --------------------------------------------------------------

    candidate_pairs = list(
        _get_candidate_pairs(
            data,
            attack_only_malicious=attack_only_malicious,
        )
    )

    if len(candidate_pairs) == 0:
        return data_adv

    # Number of unique undirected connections to remove
    num_remove = int(len(candidate_pairs) * perturbation_rate)
    num_remove = max(1, num_remove)
    num_remove = min(num_remove, len(candidate_pairs))

    # Randomly select connections
    random_order = torch.randperm(len(candidate_pairs))

    selected_pairs = {
        candidate_pairs[int(i)]
        for i in random_order[:num_remove]
    }

    # --------------------------------------------------------------
    # Remove both directed representations of each selected pair
    # --------------------------------------------------------------

    keep_edges = torch.ones(
        edge_index.size(1),
        dtype=torch.bool,
        device=edge_index.device,
    )

    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())

        pair = (min(u, v), max(u, v))

        if pair in selected_pairs:
            keep_edges[i] = False

    data_adv.edge_index = edge_index[:, keep_edges]

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
    Apply random structural edge addition.

    New connections are randomly created while using the same perturbation
    budget definition as edge removal.

    When attack_only_malicious=True, the perturbation budget is calculated
    from the number of existing edges incident to malicious nodes, and new
    connections originate from malicious nodes.

    Args:
        data:
            PyTorch Geometric Data object.

        perturbation_rate:
            Fraction of the baseline candidate edge count to add.

        attack_only_malicious:
            If True, new connections originate from malicious nodes.

        avoid_duplicates:
            If True, existing connections cannot be added again.

    Returns:
        A cloned Data object with additional bidirectional edges.

    Raises:
        ValueError:
            If perturbation_rate is outside (0, 1].
    """

    if perturbation_rate <= 0 or perturbation_rate > 1:
        raise ValueError(
            f"perturbation_rate must be in (0,1], got {perturbation_rate}"
        )

    data_adv = data.clone()

    edge_index = data.edge_index.clone()

    edge_attr = None
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_attr = data.edge_attr.clone()

    # --------------------------------------------------------------
    # Define source nodes
    # --------------------------------------------------------------

    if attack_only_malicious:
        node_mask = data.y == 1

        if node_mask.sum() == 0:
            return data_adv

        source_nodes = node_mask.nonzero(
            as_tuple=False
        ).view(-1)

    else:
        source_nodes = torch.arange(
            data.num_nodes,
            device=edge_index.device,
        )

    destination_nodes = torch.arange(
        data.num_nodes,
        device=edge_index.device,
    )

    # --------------------------------------------------------------
    # Use the SAME denominator as edge removal
    # --------------------------------------------------------------

    candidate_pairs = _get_candidate_pairs(
        data,
        attack_only_malicious=attack_only_malicious,
    )

    if len(candidate_pairs) == 0:
        return data_adv

    num_add = int(len(candidate_pairs) * perturbation_rate)
    num_add = max(1, num_add)

    # --------------------------------------------------------------
    # Existing undirected connections
    # --------------------------------------------------------------

    existing_pairs = set()

    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())

        if u == v:
            continue

        existing_pairs.add(
            (min(u, v), max(u, v))
        )

    # --------------------------------------------------------------
    # Generate new connections
    # --------------------------------------------------------------

    new_pairs = set()

    max_attempts = max(num_add * 20, 100)
    attempts = 0

    while len(new_pairs) < num_add and attempts < max_attempts:
        attempts += 1

        src_pos = torch.randint(
            0,
            source_nodes.numel(),
            (1,),
            device=edge_index.device,
        )

        dst_pos = torch.randint(
            0,
            destination_nodes.numel(),
            (1,),
            device=edge_index.device,
        )

        src = int(source_nodes[src_pos].item())
        dst = int(destination_nodes[dst_pos].item())

        # Avoid self-loops
        if src == dst:
            continue

        pair = (min(src, dst), max(src, dst))

        # Avoid existing connections
        if avoid_duplicates and pair in existing_pairs:
            continue

        # Avoid adding the same new connection twice
        if pair in new_pairs:
            continue

        new_pairs.add(pair)

    # Verify that the requested perturbation budget was achieved
    if len(new_pairs) < num_add:
        raise RuntimeError(
            f"Could only add {len(new_pairs)} of "
            f"{num_add} requested logical edges."
        )

    # --------------------------------------------------------------
    # Convert new pairs into bidirectional PyG edges
    # --------------------------------------------------------------

    new_edges_list = []

    for u, v in new_pairs:
        new_edges_list.append((u, v))
        new_edges_list.append((v, u))

    new_edges = torch.tensor(
        new_edges_list,
        dtype=torch.long,
        device=edge_index.device,
    ).t().contiguous()

    data_adv.edge_index = torch.cat(
        [edge_index, new_edges],
        dim=1,
    )

    # --------------------------------------------------------------
    # Edge attributes for new connections
    # --------------------------------------------------------------

    if edge_attr is not None:
        new_src = new_edges[0]
        new_dst = new_edges[1]

        new_edge_attr = cosine_similarity(
            data.x[new_src],
            data.x[new_dst],
            dim=1,
        ).view(-1, 1)

        data_adv.edge_attr = torch.cat(
            [
                edge_attr,
                new_edge_attr.to(edge_attr.dtype),
            ],
            dim=0,
        )

    return data_adv