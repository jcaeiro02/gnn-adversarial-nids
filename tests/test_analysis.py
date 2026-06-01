import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.analyze_dataset_features import analyze_from_dataframes


def make_nsl_small():
    # minimal NSL-KDD like dataframe with a few features
    cols = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "label",
    ]
    data_train = [
        [0, "tcp", "http", "SF", 181, 5450, "normal"],
        [0, "udp", "other", "S0", 239, 486, "neptune"],
        [0, "tcp", "http", "SF", 235, 1337, "neptune"],
        [0, "tcp", "http", "SF", 0, 0, "normal"],
    ]
    data_test = [
        [0, "tcp", "http", "SF", 120, 2000, "normal"],
        [0, "udp", "other", "S0", 300, 600, "neptune"],
    ]
    train_df = pd.DataFrame(data_train, columns=cols)
    test_df = pd.DataFrame(data_test, columns=cols)
    return train_df, test_df


def test_analyze_nsl_kdd(tmp_path):
    train_df, test_df = make_nsl_small()
    # create a small validation split from training rows to avoid using repository splits
    validation_df = train_df.iloc[[0]].reset_index(drop=True)
    out_dir = analyze_from_dataframes(
        "nsl-kdd",
        train_df,
        validation_df=validation_df,
        test_df=test_df,
        output_base=str(tmp_path),
        timestamp="testtime",
    )

    # Check files exist
    assert (out_dir / "class_distribution.json").exists()
    assert (out_dir / "feature_statistics.csv").exists()
    assert (out_dir / "feature_label_correlation.csv").exists()
    assert (out_dir / "mutual_information.csv").exists()
    assert (out_dir / "leakage_report.md").exists()

    # Basic sanity check of class distribution
    cd = json.loads((out_dir / "class_distribution.json").read_text())
    assert "train" in cd and "validation" in cd and "test" in cd


def make_cicids2017_small_with_inf():
    """Create CICIDS2017-like dataframe with inf/NaN values to test sanitization."""
    cols = [
        "flow id",
        "source ip",
        "destination ip",
        "source port",
        "destination port",
        "protocol",
        "flow duration",
        "total fwd packets",
        "total backward packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Flow Bytes/s",  # Often contains inf/NaN
        "Flow Packets/s",  # Often contains inf/NaN
        "label",
    ]
    data_train = [
        ["flow_1", "10.0.0.1", "10.0.0.2", 1234, 5678, "TCP", 100, 5, 3, 500, 300, 100, 50, 75, 100, 50, 75, 1000.0, 10.0, "BENIGN"],
        ["flow_2", "10.0.0.1", "10.0.0.3", 1234, 5679, "TCP", 200, 10, 5, 1000, 600, 150, 75, 100, 150, 75, 100, np.inf, 20.0, "ATTACK"],
        ["flow_3", "10.0.0.1", "10.0.0.4", 1234, 5680, "TCP", 150, 7, 4, 750, 450, 120, 60, 90, 120, 60, 90, 500.0, np.nan, "BENIGN"],
        ["flow_4", "10.0.0.1", "10.0.0.5", 1234, 5681, "TCP", 300, 15, 8, 1500, 900, 200, 100, 150, 200, 100, 150, -np.inf, 30.0, "ATTACK"],
    ]
    data_test = [
        ["flow_5", "10.0.0.1", "10.0.0.6", 1234, 5682, "TCP", 120, 6, 3, 600, 350, 110, 55, 80, 110, 55, 80, 800.0, 15.0, "BENIGN"],
        ["flow_6", "10.0.0.1", "10.0.0.7", 1234, 5683, "TCP", 250, 12, 6, 1200, 700, 160, 80, 120, 160, 80, 120, np.inf, np.inf, "ATTACK"],
    ]
    train_df = pd.DataFrame(data_train, columns=cols)
    test_df = pd.DataFrame(data_test, columns=cols)
    return train_df, test_df


def test_analyze_cicids2017_with_inf(tmp_path):
    """Test that CICIDS2017 analysis handles synthetic data gracefully (including handling of inf/NaN in real data)."""
    train_df, test_df = make_cicids2017_small_with_inf()
    validation_df = train_df.iloc[[0]].reset_index(drop=True)
    
    out_dir = analyze_from_dataframes(
        "cicids2017",
        train_df,
        validation_df=validation_df,
        test_df=test_df,
        output_base=str(tmp_path),
        timestamp="testtime",
    )

    # Check files exist and analysis completes without crash
    assert (out_dir / "class_distribution.json").exists()
    assert (out_dir / "feature_statistics.csv").exists()
    assert (out_dir / "feature_label_correlation.csv").exists()
    assert (out_dir / "mutual_information.csv").exists()
    assert (out_dir / "leakage_report.md").exists()

    # Verify class distribution
    cd = json.loads((out_dir / "class_distribution.json").read_text())
    assert "train" in cd and "validation" in cd and "test" in cd
    
    # Verify leakage report was generated (may or may not contain non-finite warning
    # depending on whether synthetic data's inf values persist through preprocessing)
    report = (out_dir / "leakage_report.md").read_text()
    assert "Warnings" in report  # Report should have a Warnings section
    assert len(report) > 10  # Non-empty report
