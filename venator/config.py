"""Venator configuration using pydantic-settings.

Central configuration for model paths, extraction parameters, detection thresholds,
and directory paths. Values can be overridden via environment variables prefixed with
VENATOR_ (e.g., VENATOR_PCA_DIMS=100).
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class VenatorConfig(BaseSettings):
    """Configuration for the Venator jailbreak detection pipeline."""

    model_config = {"env_prefix": "VENATOR_"}

    # --- Model ---
    model_id: str = Field(
        default="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        description="HuggingFace model ID for MLX inference",
    )

    # --- Activation extraction ---
    extraction_layers: list[int] = Field(
        default=[4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        description="Transformer layers to extract hidden states from (wide range to find optimal via ablations)",
    )
    extraction_batch_size: int = Field(
        default=1,
        description="Batch size for activation extraction (1 is safest for memory)",
    )

    # --- Dimensionality reduction ---
    pca_dims: int = Field(
        default=50,
        description="Number of PCA components. Keeps sample-to-feature ratio healthy.",
    )

    # --- Detection ---
    anomaly_threshold_percentile: float = Field(
        default=95.0,
        description="Percentile on validation scores for anomaly threshold (set on benign-only data)",
    )

    # --- Ensemble weights (decorrelated errors per ELK paper) ---
    weight_pca_mahalanobis: float = Field(
        default=2.0, description="Ensemble weight for PCA + Mahalanobis detector"
    )
    weight_isolation_forest: float = Field(
        default=1.5, description="Ensemble weight for Isolation Forest detector"
    )
    weight_autoencoder: float = Field(
        default=1.0, description="Ensemble weight for Autoencoder detector"
    )

    # --- Paths ---
    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    models_dir: Path = Field(default=Path("models"), description="Saved detector models directory")

    # --- Reproducibility ---
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    @property
    def prompts_dir(self) -> Path:
        return self.data_dir / "prompts"

    @property
    def activations_dir(self) -> Path:
        return self.data_dir / "activations"

    @property
    def ensemble_weights(self) -> dict[str, float]:
        return {
            "pca_mahalanobis": self.weight_pca_mahalanobis,
            "isolation_forest": self.weight_isolation_forest,
            "autoencoder": self.weight_autoencoder,
        }


# Singleton config instance — import this throughout the codebase
config = VenatorConfig()
