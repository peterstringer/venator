"""Supervised linear and MLP probe detectors for jailbreak detection.

These detectors follow the approach from Anthropic's "Cost-Effective
Constitutional Classifiers via Representation Re-use" (Cunningham et al.,
2025), which showed that linear probes on intermediate model activations can
match the performance of much larger dedicated classifiers.

Unlike the unsupervised detectors (PCA+Mahalanobis, Isolation Forest,
Autoencoder), these REQUIRE labeled examples of both benign and jailbreak
prompts during training. They are designed for use with the SEMI_SUPERVISED
split mode.

Why this works:
    - LLM activations already encode rich representations of input semantics.
    - Jailbreaks activate distinct computational patterns vs benign prompts.
    - A linear boundary in activation space can separate these patterns.
    - Even with few labeled examples, the activation space is informative.

Two variants are provided:
    - LinearProbeDetector: Logistic regression (minimal capacity, fast).
    - MLPProbeDetector: 2-layer MLP (more capacity, PyTorch-based).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from venator.detection.base import AnomalyDetector

logger = logging.getLogger(__name__)


class LinearProbeDetector(AnomalyDetector):
    """Linear probe classifier on LLM activations for jailbreak detection.

    A SUPERVISED detector that trains logistic regression on labeled
    activation vectors. Score = predicted probability of jailbreak class
    (higher = more anomalous, consistent with other detectors).

    Pipeline:
        1. (Optional) PCA reduction on the combined training activations.
        2. Logistic regression on labeled activations.
        3. Score = P(jailbreak | activation).

    Attributes:
        n_components: Target PCA dimensions (None to skip PCA).
        C: Logistic regression regularization strength.
        class_weight: How to handle class imbalance ("balanced" recommended).
        max_iter: Maximum iterations for logistic regression.
        pca_: Fitted PCA model (set after fit).
        clf_: Fitted logistic regression (set after fit).
    """

    def __init__(
        self,
        n_components: int | None = 50,
        C: float = 1.0,
        class_weight: str = "balanced",
        max_iter: int = 1000,
    ) -> None:
        if n_components is not None and n_components < 1:
            raise ValueError(f"n_components must be >= 1 or None, got {n_components}")
        if C <= 0:
            raise ValueError(f"C must be > 0, got {C}")

        self.n_components = n_components
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter

        # Set after fit()
        self.pca_: PCA | None = None
        self.clf_: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> LinearProbeDetector:
        """Fit on labeled activation vectors.

        IMPORTANT: This detector is supervised and REQUIRES labels.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Binary labels of shape (n_samples,). 0=benign, 1=jailbreak.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If y is None, X is wrong shape, or labels invalid.
        """
        if y is None:
            raise ValueError(
                "LinearProbeDetector is supervised and requires labels. "
                "Pass y with 0=benign, 1=jailbreak."
            )
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(f"Need at least 2 training samples, got {n_samples}")

        y = np.asarray(y, dtype=np.int64)
        if len(y) != n_samples:
            raise ValueError(
                f"X has {n_samples} samples but y has {len(y)} labels"
            )
        unique_labels = set(np.unique(y))
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be 0 or 1, got unique values: {unique_labels}")
        if len(unique_labels) < 2:
            raise ValueError(
                "Need both benign (0) and jailbreak (1) labels for supervised "
                f"training. Got only label(s): {unique_labels}"
            )

        # 1. Optional PCA
        if self.n_components is not None:
            effective = min(self.n_components, n_samples - 1, n_features)
            if effective < self.n_components:
                warnings.warn(
                    f"Reduced PCA components from {self.n_components} to "
                    f"{effective} (n_samples={n_samples}, n_features={n_features})"
                )
            self.pca_ = PCA(n_components=effective)
            X = self.pca_.fit_transform(X)
        else:
            self.pca_ = None

        # 2. Logistic regression
        self.clf_ = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=42,
            solver="lbfgs",
        )
        self.clf_.fit(X, y)

        n_jailbreak = int(np.sum(y == 1))
        logger.info(
            "Fitted LinearProbe: %d samples (%d jailbreak), "
            "PCA=%s, C=%.2f",
            n_samples,
            n_jailbreak,
            effective if self.n_components is not None else "None",
            self.C,
        )

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return jailbreak probability as anomaly score.

        Higher = more likely jailbreak (consistent with other detectors).

        Args:
            X: Data to score, shape (n_samples, n_features).

        Returns:
            Jailbreak probabilities of shape (n_samples,) in [0, 1].
        """
        if self.clf_ is None:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        if self.pca_ is not None:
            X = self.pca_.transform(X)

        # predict_proba returns [P(benign), P(jailbreak)]
        return self.clf_.predict_proba(X)[:, 1]

    def get_probe_direction(self) -> np.ndarray:
        """Return the learned linear direction in activation space.

        This is the weight vector of the logistic regression — the direction
        along which benign and jailbreak activations differ most.
        Useful for interpretability and visualization.

        If PCA was applied, projects back to the original activation space.

        Returns:
            Unit vector of shape (n_features,) in the original space.
        """
        if self.clf_ is None:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")

        direction = self.clf_.coef_[0]  # shape: (n_pca_dims,)
        if self.pca_ is not None:
            direction = self.pca_.inverse_transform(direction.reshape(1, -1)).flatten()
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        return direction

    def save(self, path: Path | str) -> None:
        """Save PCA and logistic regression to a directory."""
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.pca_ is not None:
            joblib.dump(self.pca_, path / "pca.joblib")
        joblib.dump(self.clf_, path / "logistic_regression.joblib")
        np.savez(
            path / "config.npz",
            n_components=np.array(self.n_components if self.n_components is not None else -1),
            C=np.array(self.C),
            class_weight=np.array(self.class_weight),
            max_iter=np.array(self.max_iter),
        )
        logger.info("Saved LinearProbeDetector to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> LinearProbeDetector:
        """Load a saved LinearProbeDetector from disk."""
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        data = np.load(path / "config.npz")

        n_components_raw = int(data["n_components"])
        detector = cls(
            n_components=n_components_raw if n_components_raw >= 0 else None,
            C=float(data["C"]),
            class_weight=str(data["class_weight"]),
            max_iter=int(data["max_iter"]),
        )

        pca_path = path / "pca.joblib"
        if pca_path.exists():
            detector.pca_ = joblib.load(pca_path)
        detector.clf_ = joblib.load(path / "logistic_regression.joblib")

        logger.info("Loaded LinearProbeDetector from %s", path)
        return detector


# ---------------------------------------------------------------------------
# MLP Probe
# ---------------------------------------------------------------------------


class _MLPClassifier(nn.Module):
    """Small 2-layer MLP for binary classification.

    Architecture: input_dim -> hidden1 -> hidden2 -> 1 (sigmoid).
    No BatchNorm — training sets may be too small for stable statistics.
    """

    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPProbeDetector(AnomalyDetector):
    """Small MLP probe (2-layer) on activations for jailbreak detection.

    A SUPERVISED detector that trains a small neural network on labeled
    activation vectors. Sits between the linear probe (minimal capacity)
    and the autoencoder (unsupervised) in terms of model complexity.

    Architecture: hidden_dim -> 128 -> 32 -> 1 (sigmoid)
    Trained with BCE loss on labeled data with early stopping.

    Attributes:
        n_components: Target PCA dimensions (None to skip PCA).
        hidden1: Width of first hidden layer.
        hidden2: Width of second hidden layer.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        early_stopping_patience: Epochs without val improvement before stopping.
        pca_: Fitted PCA model (set after fit).
        model_: Trained MLP (set after fit).
    """

    def __init__(
        self,
        n_components: int | None = 50,
        hidden1: int = 128,
        hidden2: int = 32,
        epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 20,
    ) -> None:
        if n_components is not None and n_components < 1:
            raise ValueError(f"n_components must be >= 1 or None, got {n_components}")
        if hidden1 < 1:
            raise ValueError(f"hidden1 must be >= 1, got {hidden1}")
        if hidden2 < 1:
            raise ValueError(f"hidden2 must be >= 1, got {hidden2}")

        self.n_components = n_components
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        # Set after fit()
        self.pca_: PCA | None = None
        self.model_: _MLPClassifier | None = None
        self._effective_input_dim: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> MLPProbeDetector:
        """Fit on labeled activation vectors.

        Uses a 90/10 train/val split for early stopping.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Binary labels of shape (n_samples,). 0=benign, 1=jailbreak.

        Returns:
            self, for method chaining.
        """
        if y is None:
            raise ValueError(
                "MLPProbeDetector is supervised and requires labels. "
                "Pass y with 0=benign, 1=jailbreak."
            )
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(f"Need at least 2 training samples, got {n_samples}")

        y = np.asarray(y, dtype=np.int64)
        if len(y) != n_samples:
            raise ValueError(f"X has {n_samples} samples but y has {len(y)} labels")
        unique_labels = set(np.unique(y))
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be 0 or 1, got unique values: {unique_labels}")
        if len(unique_labels) < 2:
            raise ValueError(
                "Need both benign (0) and jailbreak (1) labels for supervised "
                f"training. Got only label(s): {unique_labels}"
            )

        # --- PCA ---
        if self.n_components is not None:
            effective = min(self.n_components, n_samples - 1, n_features)
            if effective < self.n_components:
                warnings.warn(
                    f"Reduced PCA components from {self.n_components} to "
                    f"{effective} (n_samples={n_samples}, n_features={n_features})"
                )
            self.pca_ = PCA(n_components=effective)
            X = self.pca_.fit_transform(X)
            self._effective_input_dim = effective
        else:
            self.pca_ = None
            self._effective_input_dim = n_features

        X = X.astype(np.float32)
        y_float = y.astype(np.float32)

        # --- Train/val split (90/10) ---
        rng = np.random.default_rng(42)
        indices = rng.permutation(n_samples)
        n_val = max(1, int(n_samples * 0.1))
        n_train = n_samples - n_val

        X_train = torch.from_numpy(X[indices[:n_train]])
        y_train = torch.from_numpy(y_float[indices[:n_train]]).unsqueeze(1)
        X_val = torch.from_numpy(X[indices[n_train:]])
        y_val = torch.from_numpy(y_float[indices[n_train:]]).unsqueeze(1)

        # --- Build MLP ---
        effective_h1 = min(self.hidden1, self._effective_input_dim)
        effective_h2 = min(self.hidden2, effective_h1)

        self.model_ = _MLPClassifier(
            input_dim=self._effective_input_dim,
            hidden1=effective_h1,
            hidden2=effective_h2,
        )

        # Class weights for BCE
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # --- Training loop with early stopping ---
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.model_.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n_train)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n_train, self.batch_size):
                batch_x = X_train_shuffled[i : i + self.batch_size]
                batch_y = y_train_shuffled[i : i + self.batch_size]

                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_val)
                val_loss = criterion(val_logits, y_val).item()
            self.model_.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    k: v.clone() for k, v in self.model_.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (val_loss=%.6f)",
                        epoch + 1,
                        best_val_loss,
                    )
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()

        n_jailbreak = int(np.sum(y == 1))
        logger.info(
            "Fitted MLPProbe: %d samples (%d jailbreak), "
            "arch=%d->%d->%d->1, best_val_loss=%.6f",
            n_samples,
            n_jailbreak,
            self._effective_input_dim,
            effective_h1,
            effective_h2,
            best_val_loss,
        )

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return jailbreak probability as anomaly score.

        Higher = more likely jailbreak (consistent with other detectors).

        Args:
            X: Data to score, shape (n_samples, n_features).

        Returns:
            Jailbreak probabilities of shape (n_samples,) in [0, 1].
        """
        if self.model_ is None:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        if self.pca_ is not None:
            X = self.pca_.transform(X)
        X = X.astype(np.float32)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X))
            probs = torch.sigmoid(logits).squeeze(1)

        return probs.numpy()

    def save(self, path: Path | str) -> None:
        """Save PCA and MLP model to a directory."""
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.pca_ is not None:
            joblib.dump(self.pca_, path / "pca.joblib")
        torch.save(self.model_.state_dict(), path / "mlp_probe.pt")
        np.savez(
            path / "config.npz",
            n_components=np.array(self.n_components if self.n_components is not None else -1),
            hidden1=np.array(self.hidden1),
            hidden2=np.array(self.hidden2),
            epochs=np.array(self.epochs),
            batch_size=np.array(self.batch_size),
            learning_rate=np.array(self.learning_rate),
            early_stopping_patience=np.array(self.early_stopping_patience),
            effective_input_dim=np.array(self._effective_input_dim),
            effective_h1=np.array(min(self.hidden1, self._effective_input_dim)),
            effective_h2=np.array(
                min(self.hidden2, min(self.hidden1, self._effective_input_dim))
            ),
        )
        logger.info("Saved MLPProbeDetector to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> MLPProbeDetector:
        """Load a saved MLPProbeDetector from disk."""
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        data = np.load(path / "config.npz")

        n_components_raw = int(data["n_components"])
        detector = cls(
            n_components=n_components_raw if n_components_raw >= 0 else None,
            hidden1=int(data["hidden1"]),
            hidden2=int(data["hidden2"]),
            epochs=int(data["epochs"]),
            batch_size=int(data["batch_size"]),
            learning_rate=float(data["learning_rate"]),
            early_stopping_patience=int(data["early_stopping_patience"]),
        )

        pca_path = path / "pca.joblib"
        if pca_path.exists():
            detector.pca_ = joblib.load(pca_path)

        effective_input = int(data["effective_input_dim"])
        effective_h1 = int(data["effective_h1"])
        effective_h2 = int(data["effective_h2"])
        detector._effective_input_dim = effective_input

        detector.model_ = _MLPClassifier(
            input_dim=effective_input,
            hidden1=effective_h1,
            hidden2=effective_h2,
        )
        detector.model_.load_state_dict(
            torch.load(path / "mlp_probe.pt", weights_only=True)
        )
        detector.model_.eval()

        logger.info("Loaded MLPProbeDetector from %s", path)
        return detector
