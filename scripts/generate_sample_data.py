#!/usr/bin/env python3
"""
Generate sample NSL-KDD data for testing and development.

This script creates synthetic NSL-KDD datasets that follow the real format
for use in local testing without needing to download the full datasets.
"""

import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_nsl_kdd(
    n_samples: int = 1000,
    train_ratio: float = 0.8,
    output_dir: str = "data/raw/nsl-kdd",
) -> None:
    """Generate synthetic NSL-KDD data.

    Args:
        n_samples: Total number of samples to generate.
        train_ratio: Ratio of training samples.
        output_dir: Output directory for generated files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    # NSL-KDD parameters
    protocols = ["tcp", "udp", "icmp", "igmp", "ggp"]
    services = [
        "http", "ftp", "ssh", "domain", "telnet", "smtp", "finger",
        "pop_2", "pop_3", "sunrpc", "auth", "other"
    ]
    flags = ["SF", "S0", "S1", "S2", "S3", "RSTO", "RSTR", "RSTOS0",
             "OTH", "SH", "SRPO", "URPOH"]
    labels = [
        # Normal
        "normal",
        # Denial of Service (DoS)
        "back", "buffer_overflow", "dos", "land", "neptune", "pod",
        "smurf", "teardrop",
        # Remote to Local (R2L)
        "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
        "warezclient", "warezmaster",
        # User to Root (U2R)
        "exec_guard", "httpd", "perl", "rootkit", "xterm",
        # Probe
        "ipsweep", "mscan", "nmap", "saint", "satan",
    ]

    # Generate training data
    n_train = int(n_samples * train_ratio)
    n_test = n_samples - n_train

    logger.info(f"Generating {n_train} training samples...")
    train_data = generate_data(n_train, protocols, services, flags, labels)

    logger.info(f"Generating {n_test} test samples...")
    test_data = generate_data(n_test, protocols, services, flags, labels)

    # Save training data
    train_file = output_dir / "KDDTrain+.txt"
    save_data(train_data, train_file)

    # Save test data
    test_file = output_dir / "KDDTest+.txt"
    save_data(test_data, test_file)

    logger.info(f"Generated data saved to {output_dir}")
    logger.info(f"  Train: {train_file} ({n_train} samples)")
    logger.info(f"  Test: {test_file} ({n_test} samples)")

    # Print statistics
    print_statistics(train_data, "Training")
    print_statistics(test_data, "Test")


def generate_data(n_samples: int, protocols, services, flags, labels):
    """Generate synthetic data samples.

    Args:
        n_samples: Number of samples to generate.
        protocols: List of protocol types.
        services: List of services.
        flags: List of flags.
        labels: List of possible labels.

    Returns:
        List of data rows (as strings).
    """
    data = []

    for i in range(n_samples):
        # Random selection of categorical features
        protocol = protocols[np.random.randint(0, len(protocols))]
        service = services[np.random.randint(0, len(services))]
        flag = flags[np.random.randint(0, len(flags))]

        # Generate numeric features
        duration = np.random.randint(0, 1000)
        src_bytes = np.random.randint(0, 100000)
        dst_bytes = np.random.randint(0, 100000)
        land = np.random.randint(0, 2)
        wrong_fragment = np.random.randint(0, 3)
        urgent = np.random.randint(0, 10)
        hot = np.random.randint(0, 50)
        num_failed_logins = np.random.randint(0, 5)
        logged_in = np.random.randint(0, 2)
        num_compromised = np.random.randint(0, 30)
        root_shell = np.random.randint(0, 2)
        su_attempted = np.random.randint(0, 2)
        num_root = np.random.randint(0, 50)
        num_file_creations = np.random.randint(0, 100)
        num_shells = np.random.randint(0, 50)
        num_access_files = np.random.randint(0, 10)
        num_outbound_cmds = np.random.randint(0, 10)
        is_host_login = np.random.randint(0, 2)
        is_guest_login = np.random.randint(0, 2)
        count = np.random.randint(1, 500)
        srv_count = np.random.randint(1, 500)
        serror_rate = np.random.random()
        srv_serror_rate = np.random.random()
        rerror_rate = np.random.random()
        srv_rerror_rate = np.random.random()
        same_srv_rate = np.random.random()
        diff_srv_rate = np.random.random()
        srv_diff_host_rate = np.random.random()
        dst_host_count = np.random.randint(1, 255)
        dst_host_srv_count = np.random.randint(1, 255)
        dst_host_same_srv_rate = np.random.random()
        dst_host_diff_srv_rate = np.random.random()
        dst_host_same_src_port_rate = np.random.random()
        dst_host_srv_diff_host_rate = np.random.random()
        dst_host_serror_rate = np.random.random()
        dst_host_srv_serror_rate = np.random.random()
        dst_host_rerror_rate = np.random.random()
        dst_host_srv_rerror_rate = np.random.random()

        # Assign label (weighted towards normal)
        label_prob = np.random.random()
        if label_prob < 0.8:  # 80% normal
            label = "normal"
        else:  # 20% attack
            label = labels[np.random.randint(1, len(labels))]

        # Construct row
        row = (
            f"{duration},{protocol},{service},{flag},{src_bytes},{dst_bytes},"
            f"{land},{wrong_fragment},{urgent},{hot},{num_failed_logins},"
            f"{logged_in},{num_compromised},{root_shell},{su_attempted},"
            f"{num_root},{num_file_creations},{num_shells},{num_access_files},"
            f"{num_outbound_cmds},{is_host_login},{is_guest_login},{count},"
            f"{srv_count},{serror_rate:.2f},{srv_serror_rate:.2f},"
            f"{rerror_rate:.2f},{srv_rerror_rate:.2f},{same_srv_rate:.2f},"
            f"{diff_srv_rate:.2f},{srv_diff_host_rate:.2f},{dst_host_count},"
            f"{dst_host_srv_count},{dst_host_same_srv_rate:.2f},"
            f"{dst_host_diff_srv_rate:.2f},{dst_host_same_src_port_rate:.2f},"
            f"{dst_host_srv_diff_host_rate:.2f},{dst_host_serror_rate:.2f},"
            f"{dst_host_srv_serror_rate:.2f},{dst_host_rerror_rate:.2f},"
            f"{dst_host_srv_rerror_rate:.2f},{label}"
        )
        data.append(row)

    return data


def save_data(data, filepath):
    """Save data to file.

    Args:
        data: List of data rows.
        filepath: Output file path.
    """
    with open(filepath, "w") as f:
        for row in data:
            f.write(row + "\n")
    logger.info(f"Saved {len(data)} samples to {filepath}")


def print_statistics(data, dataset_name):
    """Print statistics about the data.

    Args:
        data: List of data rows.
        dataset_name: Name of the dataset for logging.
    """
    # Count labels
    label_counts = {}
    for row in data:
        label = row.split(",")[-1]
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\n{dataset_name} Dataset Statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        percentage = 100 * count / len(data)
        print(f"    {label}: {count} ({percentage:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic NSL-KDD data for testing."
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Total number of samples to generate (default: 1000)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Ratio of training samples (default: 0.8)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw/nsl-kdd",
        help="Output directory (default: data/raw/nsl-kdd)"
    )

    args = parser.parse_args()

    generate_synthetic_nsl_kdd(
        n_samples=args.samples,
        train_ratio=args.train_ratio,
        output_dir=args.output_dir,
    )
