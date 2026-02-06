"""Autoencoder-based anomaly detector.

Learns a compressed representation of benign activations via reconstruction.
Anomaly score is the reconstruction error — jailbreak activations that deviate
from the learned manifold produce higher reconstruction loss. Uses PyTorch (CPU).

Architecture (operates on PCA-reduced features, so input_dim ~ 50):
    Encoder: input_dim → hidden_dim → latent_dim
    Decoder: latent_dim → hidden_dim → input_dim

Training:
    - 90/10 train/val split of the training data
    - MSE loss
    - Adam optimizer
    - Early stopping on validation loss
    - No BatchNorm (too few samples for stable statistics)

Why this complements the other detectors:
    - Learns a nonlinear manifold (vs Mahalanobis's Gaussian assumption).
    - Reconstruction error captures deviation from the learned normal pattern.
    - Different failure modes → decorrelated errors in the ensemble.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from venator.detection.base import AnomalyDetector

logger = logging.getLogger(__name__)


class _Autoencoder(nn.Module):
    """Simple symmetric autoencoder for anomaly detection.

    No BatchNorm — training set is too small for stable running stats.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderDetector(AnomalyDetector):
    """Autoencoder-based anomaly detection using reconstruction error.

    The autoencoder learns to compress and reconstruct normal activations.
    Jailbreak activations should reconstruct poorly, producing high
    reconstruction error which serves as the anomaly score.

    Attributes:
        n_components: Target number of PCA dimensions (input to autoencoder).
        latent_dim: Bottleneck dimension in the autoencoder.
        hidden_dim: Width of encoder/decoder hidden layers.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size for training.
        learning_rate: Adam learning rate.
        early_stopping_patience: Epochs without val improvement before stopping.
        pca_: Fitted PCA model (set after fit).
        model_: Trained autoencoder (set after fit).
    """

    def __init__(
        self,
        n_components: int = 50,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 20,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        self.n_components = n_components
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        # Set after fit()
        self.pca_: PCA | None = None
        self.model_: _Autoencoder | None = None

    def fit(self, X: np.ndarray) -> AutoencoderDetector:
        """Fit PCA and train the autoencoder on normal activation vectors.

        Uses a 90/10 train/val split of X for early stopping.

        Args:
            X: Training data of shape (n_samples, n_features).
               Must contain ONLY benign/normal activations.

        Returns:
            self, for method chaining.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"Need at least 2 training samples, got {n_samples}"
            )

        # --- PCA ---
        effective_components = min(self.n_components, n_samples - 1, n_features)
        if effective_components < self.n_components:
            warnings.warn(
                f"Reduced PCA components from {self.n_components} to "
                f"{effective_components} (n_samples={n_samples}, "
                f"n_features={n_features})"
            )

        self.pca_ = PCA(n_components=effective_components)
        X_reduced = self.pca_.fit_transform(X).astype(np.float32)

        # --- Train/val split (90/10) ---
        n_val = max(1, int(n_samples * 0.1))
        n_train = n_samples - n_val
        indices = np.random.default_rng(42).permutation(n_samples)
        X_train = torch.from_numpy(X_reduced[indices[:n_train]])
        X_val = torch.from_numpy(X_reduced[indices[n_train:]])

        # --- Build autoencoder ---
        # Adjust dims if effective_components is smaller than expected
        effective_hidden = min(self.hidden_dim, effective_components)
        effective_latent = min(self.latent_dim, effective_hidden)

        self.model_ = _Autoencoder(
            input_dim=effective_components,
            hidden_dim=effective_hidden,
            latent_dim=effective_latent,
        )

        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate
        )
        criterion = nn.MSELoss()

        # --- Training loop with early stopping ---
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.model_.train()
        for epoch in range(self.epochs):
            # Shuffle training data
            perm = torch.randperm(n_train)
            X_train_shuffled = X_train[perm]

            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n_train, self.batch_size):
                batch = X_train_shuffled[i : i + self.batch_size]
                optimizer.zero_grad()
                recon = self.model_(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self.model_.eval()
            with torch.no_grad():
                val_recon = self.model_(X_val)
                val_loss = criterion(val_recon, X_val).item()
            self.model_.train()

            # Early stopping
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

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()

        logger.info(
            "Fitted Autoencoder: %d→%d→%d→%d→%d, best_val_loss=%.6f, "
            "n_train=%d",
            effective_components,
            effective_hidden,
            effective_latent,
            effective_hidden,
            effective_components,
            best_val_loss,
            n_samples,
        )

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction error as anomaly score.

        Args:
            X: Data to score, shape (n_samples, n_features).

        Returns:
            MSE reconstruction errors of shape (n_samples,).
            Higher = more anomalous.
        """
        if self.pca_ is None or self.model_ is None:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        X_reduced = self.pca_.transform(X).astype(np.float32)
        X_tensor = torch.from_numpy(X_reduced)

        self.model_.eval()
        with torch.no_grad():
            recon = self.model_(X_tensor)
            # Per-sample MSE
            mse = ((recon - X_tensor) ** 2).mean(dim=1)

        return mse.numpy()

    def save(self, path: Path | str) -> None:
        """Save PCA model and autoencoder state to a directory.

        Args:
            path: Directory to save model files into.
        """
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pca_, path / "pca.joblib")
        torch.save(self.model_.state_dict(), path / "autoencoder.pt")
        np.savez(
            path / "config.npz",
            n_components=np.array(self.n_components),
            latent_dim=np.array(self.latent_dim),
            hidden_dim=np.array(self.hidden_dim),
            epochs=np.array(self.epochs),
            batch_size=np.array(self.batch_size),
            learning_rate=np.array(self.learning_rate),
            early_stopping_patience=np.array(self.early_stopping_patience),
            # Store effective dims so we can rebuild the architecture
            effective_input_dim=np.array(self.pca_.n_components_),
            effective_hidden_dim=np.array(
                min(self.hidden_dim, self.pca_.n_components_)
            ),
            effective_latent_dim=np.array(
                min(self.latent_dim, min(self.hidden_dim, self.pca_.n_components_))
            ),
        )
        logger.info("Saved AutoencoderDetector to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> AutoencoderDetector:
        """Load a saved AutoencoderDetector from disk.

        Args:
            path: Directory containing saved model files.

        Returns:
            A fitted AutoencoderDetector.
        """
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        data = np.load(path / "config.npz")

        detector = cls(
            n_components=int(data["n_components"]),
            latent_dim=int(data["latent_dim"]),
            hidden_dim=int(data["hidden_dim"]),
            epochs=int(data["epochs"]),
            batch_size=int(data["batch_size"]),
            learning_rate=float(data["learning_rate"]),
            early_stopping_patience=int(data["early_stopping_patience"]),
        )
        detector.pca_ = joblib.load(path / "pca.joblib")

        # Rebuild architecture with effective dims
        effective_input = int(data["effective_input_dim"])
        effective_hidden = int(data["effective_hidden_dim"])
        effective_latent = int(data["effective_latent_dim"])

        detector.model_ = _Autoencoder(
            input_dim=effective_input,
            hidden_dim=effective_hidden,
            latent_dim=effective_latent,
        )
        detector.model_.load_state_dict(
            torch.load(path / "autoencoder.pt", weights_only=True)
        )
        detector.model_.eval()

        logger.info("Loaded AutoencoderDetector from %s", path)
        return detector
