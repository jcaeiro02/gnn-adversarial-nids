"""
Dataset download module for GNN Adversarial NIDS.

This module handles downloading and storing NSL-KDD datasets.
"""

import os
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List

# Configure logging
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Downloads and validates NSL-KDD datasets."""

    # NSL-KDD dataset URLs
    NSL_KDD_URLS = {
        "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
        "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
    }

    def __init__(self, base_dir: str = "data/raw/nsl-kdd"):
        """Initialize the downloader.

        Args:
            base_dir: Base directory to store downloaded datasets.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DatasetDownloader initialized with base_dir: {self.base_dir}")

    def download_file(self, url: str, filename: str, force: bool = False) -> bool:
        """Download a single file from URL.

        Args:
            url: URL to download from.
            filename: Filename to save as.
            force: Force re-download if file exists.

        Returns:
            True if download successful, False otherwise.
        """
        filepath = self.base_dir / filename

        # Check if file already exists
        if filepath.exists() and not force:
            logger.info(f"File already exists: {filepath}")
            return True

        try:
            logger.info(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, filepath)
            logger.info(f"Successfully downloaded: {filepath}")
            return True
        except urllib.error.URLError as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {filename}: {e}")
            return False

    def validate_file(self, filename: str) -> bool:
        """Validate downloaded file.

        Args:
            filename: Filename to validate.

        Returns:
            True if file exists and has content, False otherwise.
        """
        filepath = self.base_dir / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return False
        if filepath.stat().st_size == 0:
            logger.warning(f"File is empty: {filepath}")
            return False
        logger.info(f"File validation passed: {filepath}")
        return True

    def download_nsl_kdd(self, force: bool = False) -> bool:
        """Download NSL-KDD training and testing datasets.

        Args:
            force: Force re-download of existing files.

        Returns:
            True if both files downloaded successfully, False otherwise.
        """
        logger.info("Starting NSL-KDD download...")
        success = True

        for filename, url in self.NSL_KDD_URLS.items():
            if not self.download_file(url, filename, force):
                success = False
                logger.error(f"Failed to download {filename}")
            elif not self.validate_file(filename):
                success = False
                logger.error(f"Validation failed for {filename}")

        if success:
            logger.info("NSL-KDD download completed successfully")
        else:
            logger.warning("NSL-KDD download completed with errors")

        return success

    def download_cicids2017(self, base_dir: Optional[str] = None) -> bool:
        """Validate manual placement of CICIDS2017 CSV files.

        CICIDS2017 is not automatically downloaded from Kaggle. Users must place
        one or more CSV files under the target directory before processing.

        Args:
            base_dir: Optional directory where CSV files are located.
                If not provided, uses self.base_dir.

        Returns:
            True if at least one CSV file exists, False otherwise.
        """
        target_dir = Path(base_dir) if base_dir else self.base_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        csv_files = list(target_dir.glob("*.csv"))
        if len(csv_files) == 0:
            logger.warning(
                "No CICIDS2017 CSV files found in %s. "
                "Please download the CICIDS2017 CSV files manually and place them in this directory: %s",
                target_dir,
            )
            logger.info(
                "Expected path pattern: data/raw/cicids2017/*.csv"
            )
            logger.info(
                "Example: download CICIDS2017 from Kaggle, extract CSV files, "
                "and copy them into the target directory."
            )
            return False

        logger.info(
            "Found %d CICIDS2017 CSV file(s) in %s",
            len(csv_files),
            target_dir,
        )
        return True

    def download_all(self, force: bool = False) -> bool:
        """Download all available datasets.

        Args:
            force: Force re-download of existing files.

        Returns:
            True if all downloads successful, False otherwise.
        """
        logger.info("Starting download of all datasets...")
        success = self.download_nsl_kdd(force=force)
        cicids_success = DatasetDownloader(base_dir=Path(self.base_dir).parent / "cicids2017").download_cicids2017()
        return success and cicids_success

    def get_dataset_path(self, dataset_name: str) -> Optional[Path]:
        """Get path to a downloaded dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Path to dataset if it exists, None otherwise.
        """
        filepath = self.base_dir / dataset_name
        if filepath.exists():
            return filepath
        logger.warning(f"Dataset not found: {dataset_name}")
        return None

    def list_datasets(self) -> List[str]:
        """List all available downloaded datasets.

        Returns:
            List of dataset filenames.
        """
        datasets = [f.name for f in self.base_dir.glob("*.txt")] + [f.name for f in self.base_dir.glob("*.csv")]
        logger.info(f"Found {len(datasets)} datasets")
        return datasets


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    downloader = DatasetDownloader()
    success = downloader.download_nsl_kdd()
    if success:
        datasets = downloader.list_datasets()
        print(f"Available datasets: {datasets}")
