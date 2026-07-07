import numpy as np

from src.analysis.neighbor_churn import compute_neighbor_churn, rebuild_knn_graph


def test_rebuild_and_compute_neighbor_churn():
    x_original = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    x_adv = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    original_edge_index, _ = rebuild_knn_graph(x_original, k=1, bidirectional=False)
    adv_edge_index, _ = rebuild_knn_graph(x_adv, k=1, bidirectional=False)

    churn = compute_neighbor_churn(original_edge_index, adv_edge_index)

    assert original_edge_index.shape[0] == 2
    assert adv_edge_index.shape[0] == 2
    assert churn == 0.0

    explicit_original = np.array([[0, 1], [1, 2]], dtype=np.int64)
    explicit_attacked = np.array([[0, 1], [2, 2]], dtype=np.int64)
    changed_churn = compute_neighbor_churn(explicit_original, explicit_attacked)

    assert 0.0 < changed_churn <= 1.0
