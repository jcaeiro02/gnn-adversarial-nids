"""
Data preprocessing module for NSL-KDD dataset.

This module handles loading, preprocessing, and encoding the NSL-KDD dataset
for GNN-based network intrusion detection.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

logger = logging.getLogger(__name__)


class NSLKDDPreprocessor:
    """Preprocessor for NSL-KDD dataset.

    Handles loading, encoding, normalization, and label conversion.
    """

    # NSL-KDD feature definitions (41 features)
    FEATURE_COLUMNS = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ]
    LABEL_COLUMN = "label"
    DIFFICULTY_COLUMN = "difficulty"

    # Categorical columns to encode
    CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

    # Attack types (all labels except 'normal')
    ATTACK_TYPES = {
        "normal": 0,
        # Denial of Service (DoS) attacks
        "back": 1,
        "buffer_overflow": 1,
        "dos": 1,
        "land": 1,
        "neptune": 1,
        "pod": 1,
        "smurf": 1,
        "teardrop": 1,
        # Remote to Local (R2L) attacks
        "ftp_write": 1,
        "guess_passwd": 1,
        "imap": 1,
        "multihop": 1,
        "phf": 1,
        "spy": 1,
        "warezclient": 1,
        "warezmaster": 1,
        # User to Root (U2R) attacks
        "exec_guard": 1,
        "httpd": 1,
        "perl": 1,
        "rootkit": 1,
        "xterm": 1,
        # Probe attacks
        "ipsweep": 1,
        "mscan": 1,
        "nmap": 1,
        "saint": 1,
        "satan": 1,
    }

    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        logger.info("NSLKDDPreprocessor initialized")

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load NSL-KDD dataset from file.

        Args:
            filepath: Path to the NSL-KDD data file.

        Returns:
            Loaded DataFrame with columns named appropriately.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If data loading fails.
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Dataset file not found: {filepath}")

            logger.info(f"Loading data from {filepath}...")
            # Load without header and inspect columns
            df = pd.read_csv(filepath, header=None)
            column_count = df.shape[1]

            if column_count == len(self.FEATURE_COLUMNS) + 1:
                df.columns = self.FEATURE_COLUMNS + [self.LABEL_COLUMN]
            elif column_count == len(self.FEATURE_COLUMNS) + 2:
                df.columns = self.FEATURE_COLUMNS + [self.LABEL_COLUMN, self.DIFFICULTY_COLUMN]
            else:
                raise ValueError(
                    f"Expected 42 or 43 columns for NSL-KDD data, got {column_count} columns. "
                    f"Found file {filepath} with shape {df.shape}."
                )

            # Remove extra spaces from string columns if present
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

            logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
            return df

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise ValueError(f"Failed to load data from {filepath}: {e}")

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing, NaN, inf, and -inf values.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        logger.info("Handling missing values...")

        # Replace inf and -inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Count missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found {missing_counts.sum()} missing values")
            logger.debug(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")

        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled {col} with median: {median_val}")

        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown"
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Filled {col} with mode: {mode_val}")

        return df

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features.

        Args:
            df: Input DataFrame.
            fit: If True, fit the encoders. If False, use existing encoders.

        Returns:
            DataFrame with encoded categorical columns.

        Raises:
            ValueError: If fit=False but encoders not fitted.
        """
        logger.info("Encoding categorical features...")

        for col in self.CATEGORICAL_COLUMNS:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in data")
                continue

            if fit:
                # Fit new encoder
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.label_encoders[col] = encoder
                logger.info(f"Fitted encoder for {col}: {len(encoder.classes_)} classes")
            else:
                # Use existing encoder
                if col not in self.label_encoders:
                    raise ValueError(f"Encoder for {col} not fitted. Call preprocess(fit=True) first.")
                encoder = self.label_encoders[col]
                values = df[col].astype(str).tolist()
                try:
                    df[col] = encoder.transform(values)
                except ValueError:
                    unknown_index = len(encoder.classes_)
                    mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
                    df[col] = [mapping.get(val, unknown_index) for val in values]
                    logger.info(
                        f"Transformed {col} using fitted encoder; unseen labels mapped to index {unknown_index}"
                    )
                else:
                    logger.info(f"Transformed {col} using fitted encoder")

        return df

    def _convert_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Convert labels to binary (normal=0, attack=1).

        Args:
            df: Input DataFrame with label column.

        Returns:
            Tuple of (df without label column, label series).
        """
        logger.info("Converting labels to binary classification...")

        labels = df["label"].copy()

        # Convert to binary: normal=0, all attacks=1
        binary_labels = labels.apply(
            lambda x: self.ATTACK_TYPES.get(x.lower(), 1)
        )

        # Count label distribution
        unique_counts = binary_labels.value_counts()
        logger.info(f"Label distribution after conversion:\n{unique_counts}")

        # Check for labels not explicitly listed in ATTACK_TYPES
        unknown_mask = ~labels.str.lower().isin(self.ATTACK_TYPES.keys())
        if unknown_mask.any():
            unlisted_labels = labels[unknown_mask].unique()
            logger.warning(
                f"Found {unknown_mask.sum()} labels not explicitly listed in ATTACK_TYPES; "
                f"safely mapped to attack class 1. Unlisted labels: {list(unlisted_labels)}"
            )
            logger.debug(f"Unlisted labels: {unlisted_labels}")

        df = df.drop([self.LABEL_COLUMN, self.DIFFICULTY_COLUMN], axis=1, errors="ignore")
        return df, binary_labels

    def _normalize_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Normalize numeric features using StandardScaler.

        Args:
            X: Feature matrix (DataFrame).
            fit: If True, fit the scaler. If False, use existing scaler.

        Returns:
            Normalized feature matrix as numpy array.

        Raises:
            ValueError: If fit=False but scaler not fitted.
        """
        logger.info("Normalizing features...")

        if fit:
            self.scaler = StandardScaler()
            X_normalized = self.scaler.fit_transform(X)
            logger.info(f"Fitted StandardScaler on {X.shape[1]} features")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call preprocess(fit=True) first.")
            X_normalized = self.scaler.transform(X)
            logger.info(f"Transformed using fitted scaler")

        return X_normalized

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess NSL-KDD data.

        Args:
            df: Loaded NSL-KDD DataFrame.
            fit: If True, fit encoders and scaler. If False, use existing ones.

        Returns:
            Tuple of (feature matrix X, labels y).

        Raises:
            ValueError: If fit=False but preprocessor not fitted.
        """
        logger.info(f"Starting preprocessing (fit={fit})...")

        # Step 1: Handle missing values
        df = self._handle_missing_values(df)

        # Step 2: Encode categorical features
        df = self._encode_categorical(df, fit=fit)

        # Step 3: Convert labels
        df_features, y = self._convert_labels(df)

        # Step 4: Normalize features
        X = self._normalize_features(df_features, fit=fit)

        # Store feature names for later reference
        self.feature_names = df_features.columns.tolist()

        logger.info(f"Preprocessing complete: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"Feature names: {self.feature_names}")

        return X, y.values

    def save_preprocessed(
        self, X: np.ndarray, y: np.ndarray, output_dir: str
    ) -> Tuple[str, str]:
        """Save preprocessed data and preprocessor state.

        Args:
            X: Feature matrix.
            y: Labels.
            output_dir: Directory to save outputs.

        Returns:
            Tuple of (X filepath, y filepath).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save feature matrix
            X_path = output_dir / "X_preprocessed.npy"
            np.save(X_path, X)
            logger.info(f"Saved X to {X_path}")

            # Save labels
            y_path = output_dir / "y_preprocessed.npy"
            np.save(y_path, y)
            logger.info(f"Saved y to {y_path}")

            # Save preprocessor state
            preprocessor_path = output_dir / "preprocessor_state.pkl"
            state = {
                "scaler": self.scaler,
                "label_encoders": self.label_encoders,
                "feature_names": self.feature_names,
            }
            with open(preprocessor_path, "wb") as f:
                pickle.dump(state, f)
            logger.info(f"Saved preprocessor state to {preprocessor_path}")

            return str(X_path), str(y_path)

        except Exception as e:
            logger.error(f"Error saving preprocessed data: {e}")
            raise

    def load_preprocessed(self, state_path: str) -> None:
        """Load preprocessor state from saved file.

        Args:
            state_path: Path to saved preprocessor state.
        """
        try:
            state_path = Path(state_path)
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            self.scaler = state["scaler"]
            self.label_encoders = state["label_encoders"]
            self.feature_names = state["feature_names"]
            logger.info(f"Loaded preprocessor state from {state_path}")
        except Exception as e:
            logger.error(f"Error loading preprocessor state: {e}")
            raise


class CICIDS2017Preprocessor:
    """Preprocessor for CICIDS2017 dataset.

    This preprocessor loads one or more CSV files from a CICIDS2017 raw directory,
    filters out non-numeric identifiers, encodes labels into binary classes, and
    normalizes numeric flow features with StandardScaler.
    """

    LABEL_CANDIDATES = {"label", "flow label"}
    IGNORE_COLUMNS = {
        "flow id",
        "source ip",
        "destination ip",
        "timestamp",
        "simillarhttp",
    }

    def __init__(self):
        self.scaler = None
        self.feature_names = None
        logger.info("CICIDS2017Preprocessor initialized")

    def load_data(self, raw_dir: str) -> pd.DataFrame:
        """Load CICIDS2017 CSV files from the raw directory.

        Args:
            raw_dir: Directory containing one or more CICIDS2017 CSV files.

        Returns:
            Concatenated DataFrame from all CSV files.
        """
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(f"CICIDS2017 raw directory not found: {raw_dir}")

        csv_files = sorted(raw_dir.glob("*.csv"))
        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No CICIDS2017 CSV files found in {raw_dir}. "
                "Place one or more *.csv files in this directory."
            )

        data_frames = []
        for csv_path in csv_files:
            logger.info(f"Loading CICIDS2017 CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
            data_frames.append(df)

        combined = pd.concat(data_frames, ignore_index=True)
        logger.info(f"Loaded {len(combined)} CICIDS2017 samples from {len(csv_files)} file(s)")
        return combined

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        return df

    def _find_label_column(self, columns: pd.Index) -> str:
        for col in columns:
            if isinstance(col, str) and col.strip().lower() in self.LABEL_CANDIDATES:
                return col
        raise ValueError(
            "CICIDS2017 label column not found. Expected 'Label' or 'Flow Label'."
        )

    def _drop_identifier_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [
            col
            for col in df.columns
            if isinstance(col, str) and col.strip().lower() in self.IGNORE_COLUMNS
        ]
        if cols_to_drop:
            logger.info(f"Dropping identifier columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop, errors="ignore")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled missing values in {col} with median: {median_val}")

        return df

    def _convert_labels(self, labels: pd.Series) -> pd.Series:
        labels = labels.astype(str).str.strip().str.upper()
        binary_labels = labels.apply(lambda x: 0 if x == "BENIGN" else 1).astype(np.int64)

        unknown_labels = labels[~labels.isin({"BENIGN"})]
        if len(unknown_labels) > 0:
            logger.info(
                "Converted %d attack labels to binary 1 and %d benign labels to binary 0",
                int((binary_labels == 1).sum()),
                int((binary_labels == 0).sum()),
            )

        return binary_labels

    def _select_numeric_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        label_column = self._find_label_column(df.columns)
        labels = df[label_column].copy()

        df_features = df.drop(columns=[label_column], errors="ignore")
        df_features = self._drop_identifier_columns(df_features)

        # Convert all remaining columns to numeric where possible.
        for col in df_features.columns:
            if df_features[col].dtype == object:
                df_features[col] = pd.to_numeric(df_features[col], errors="coerce")

        # Keep only numeric features and drop fully empty columns.
        df_features = df_features.select_dtypes(include=[np.number])
        df_features = df_features.loc[:, df_features.notna().any(axis=0)]

        if df_features.shape[1] == 0:
            raise ValueError("No numeric CICIDS2017 features found after preprocessing.")

        return df_features, labels

    def _normalize_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        if fit:
            self.scaler = StandardScaler()
            X_normalized = self.scaler.fit_transform(X)
            logger.info(f"Fitted StandardScaler on {X.shape[1]} CICIDS2017 features")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call preprocess(fit=True) first.")
            X_normalized = self.scaler.transform(X)
            logger.info("Transformed CICIDS2017 features using fitted scaler")
        return X_normalized

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> tuple[np.ndarray, np.ndarray]:
        df = self._normalize_column_names(df)
        df = self._handle_missing_values(df)
        X_df, y_series = self._select_numeric_features(df)

        self.feature_names = X_df.columns.tolist()
        X = self._normalize_features(X_df, fit=fit).astype(np.float32)
        y = self._convert_labels(y_series).to_numpy(dtype=np.int64)

        logger.info(
            f"CICIDS2017 preprocessing complete: X shape={X.shape}, y shape={y.shape}"
        )
        return X, y

    def save_preprocessed(self, X: np.ndarray, y: np.ndarray, output_dir: str) -> tuple[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X_path = output_dir / "X_preprocessed.npy"
        np.save(X_path, X)
        logger.info(f"Saved X to {X_path}")

        y_path = output_dir / "y_preprocessed.npy"
        np.save(y_path, y)
        logger.info(f"Saved y to {y_path}")

        preprocessor_path = output_dir / "preprocessor_state.pkl"
        state = {
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }
        with open(preprocessor_path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved preprocessor state to {preprocessor_path}")

        return str(X_path), str(y_path)

    def load_preprocessed(self, state_path: str) -> None:
        try:
            state_path = Path(state_path)
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            self.scaler = state["scaler"]
            self.feature_names = state["feature_names"]
            logger.info(f"Loaded CICIDS2017 preprocessor state from {state_path}")
        except Exception as e:
            logger.error(f"Error loading CICIDS2017 preprocessor state: {e}")
            raise


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    preprocessor = NSLKDDPreprocessor()
    # df = preprocessor.load_data("data/raw/nsl-kdd/KDDTrain+.txt")
    # X, y = preprocessor.preprocess(df, fit=True)
    # print(f"Preprocessed data shape: X={X.shape}, y={y.shape}")
