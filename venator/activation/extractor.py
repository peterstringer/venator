"""MLX-based hidden state extraction from transformer models.

Hooks into transformer layer forward passes to capture hidden state activations.
Uses MLX for Apple Silicon-optimized inference with 4-bit quantized models.

The extractor manually walks through the model's transformer layers, capturing
the output hidden state at specified layers. Each layer's output is mean-pooled
across the token/sequence dimension to produce a single activation vector per
prompt. This captures HOW the model computes, not just WHAT it outputs.

Design choice (ELK paper): Middle layers (12-20 for Mistral-7B's 32 layers)
generalize best for detecting behavioral anomalies — early layers capture
syntax, late layers capture surface-level output features, but middle layers
encode the abstract computation that differs between benign and adversarial inputs.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn

from venator.config import VenatorConfig, config as default_config

logger = logging.getLogger(__name__)


def _find_embedding_layer(inner_model: nn.Module) -> nn.Module:
    """Locate the token embedding layer, handling naming differences across mlx-lm versions.

    mlx-lm model architectures use different attribute names for the embedding layer
    depending on version and model type (e.g. 'embed_tokens' for Llama-style,
    'tok_embeddings' for older Mistral standalone).
    """
    for attr in ("embed_tokens", "tok_embeddings"):
        if hasattr(inner_model, attr):
            return getattr(inner_model, attr)
    raise AttributeError(
        f"Cannot find embedding layer on {type(inner_model).__name__}. "
        f"Expected 'embed_tokens' or 'tok_embeddings'. "
        f"Available attributes: {[a for a in dir(inner_model) if not a.startswith('_')]}"
    )


def _find_norm_layer(inner_model: nn.Module) -> nn.Module | None:
    """Locate the final RMSNorm layer (optional — used if we want post-norm activations)."""
    for attr in ("norm", "final_layernorm"):
        if hasattr(inner_model, attr):
            return getattr(inner_model, attr)
    return None


class ActivationExtractor:
    """Extracts hidden state activations from an LLM using MLX.

    For each input prompt, runs a forward pass through the model and captures
    the hidden states at specified transformer layers. Returns the mean-pooled
    activation vector across token positions for each layer.

    Args:
        model_id: HuggingFace model ID or local path for mlx-lm.
        layers: Transformer layer indices to extract from. Defaults to config.
        config: VenatorConfig instance. Defaults to the global singleton.
    """

    def __init__(
        self,
        model_id: str | None = None,
        layers: list[int] | None = None,
        config: VenatorConfig = default_config,
    ) -> None:
        self._config = config
        self._model_id = model_id or config.model_id
        self._target_layers = set(layers or config.extraction_layers)

        # Lazy-loaded on first use to avoid slow import/download at construction time
        self._model: nn.Module | None = None
        self._tokenizer = None

        logger.info(
            "ActivationExtractor initialized (model=%s, layers=%s)",
            self._model_id,
            sorted(self._target_layers),
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the model and tokenizer on first use."""
        if self._model is not None:
            return

        import mlx_lm

        logger.info("Loading model %s ...", self._model_id)
        t0 = time.perf_counter()

        self._model, self._tokenizer = mlx_lm.load(self._model_id)

        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.1fs", elapsed)

        # Validate that requested layers exist
        inner = self._get_inner_model()
        n_layers = len(inner.layers)
        invalid = [l for l in self._target_layers if l < 0 or l >= n_layers]
        if invalid:
            raise ValueError(
                f"Requested layers {sorted(invalid)} are out of range for a "
                f"{n_layers}-layer model. Valid range: 0..{n_layers - 1}."
            )

    def _get_inner_model(self) -> nn.Module:
        """Return the inner transformer model (unwrap the Model wrapper).

        mlx-lm wraps the core transformer (e.g. Mistral/LlamaModel) inside
        a Model class that has .model attribute. Some architectures nest
        differently, so we probe for the common patterns.
        """
        model = self._model
        # Standard mlx-lm wrapper: Model.model -> LlamaModel/Mistral
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model
        # Direct model (no wrapper)
        if hasattr(model, "layers"):
            return model
        raise AttributeError(
            f"Cannot locate transformer layers on {type(model).__name__}. "
            "Expected model.model.layers or model.layers."
        )

    @property
    def hidden_dim(self) -> int:
        """Dimensionality of the model's hidden states (e.g. 4096 for Mistral-7B)."""
        self._ensure_model_loaded()
        inner = self._get_inner_model()
        embed = _find_embedding_layer(inner)
        # nn.Embedding.weight has shape (vocab_size, hidden_dim)
        return embed.weight.shape[1]

    @property
    def n_layers(self) -> int:
        """Total number of transformer layers in the model."""
        self._ensure_model_loaded()
        return len(self._get_inner_model().layers)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, prompt: str) -> list[int]:
        """Tokenize a prompt using the model's chat template for instruction-tuned models.

        Wraps the prompt as a user message and applies the chat template so the
        activations reflect the model's actual processing of instructions.
        """
        import mlx.core as mx  # noqa: F811

        messages = [{"role": "user", "content": prompt}]

        # apply_chat_template returns token IDs when tokenize=True
        if hasattr(self._tokenizer, "apply_chat_template"):
            token_ids = self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
        else:
            # Fallback: plain tokenization without chat template
            logger.warning("Tokenizer has no apply_chat_template — using plain encode")
            token_ids = self._tokenizer.encode(prompt)

        if not token_ids:
            raise ValueError(f"Tokenization produced empty sequence for prompt: {prompt!r}")

        return token_ids

    # ------------------------------------------------------------------
    # Forward pass with activation capture
    # ------------------------------------------------------------------

    def _forward_with_capture(self, token_ids: list[int]) -> dict[int, np.ndarray]:
        """Run a forward pass, capturing hidden states at target layers.

        Manually iterates through the transformer layers so we can intercept
        intermediate outputs. Each captured activation is mean-pooled across
        the sequence dimension to produce a single vector per layer.

        Args:
            token_ids: Token IDs from the tokenizer.

        Returns:
            Dict mapping layer index -> activation vector of shape (hidden_dim,).
        """
        import mlx.core as mx
        import mlx.nn as nn  # noqa: F811

        inner = self._get_inner_model()
        embed_layer = _find_embedding_layer(inner)

        # Embed tokens: (1, seq_len) -> (1, seq_len, hidden_dim)
        inputs = mx.array([token_ids])
        h = embed_layer(inputs)

        # Causal attention mask — needed when seq_len > 1
        seq_len = h.shape[1]
        mask = None
        if seq_len > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

        # Walk through transformer layers, capturing target outputs
        activations: dict[int, np.ndarray] = {}
        cache = [None] * len(inner.layers)

        for i, layer in enumerate(inner.layers):
            h, cache[i] = layer(h, mask=mask, cache=cache[i])

            if i in self._target_layers:
                # Mean-pool across sequence dimension: (1, seq_len, hidden_dim) -> (hidden_dim,)
                # No padding tokens here (batch_size=1, all tokens are real)
                pooled = mx.mean(h, axis=1).squeeze(axis=0)
                mx.eval(pooled)
                act = np.asarray(pooled).astype(np.float32)

                # Validate: no NaN/Inf propagation (fail fast per code style)
                if not np.all(np.isfinite(act)):
                    raise RuntimeError(
                        f"Non-finite activations at layer {i} "
                        f"(NaN={np.isnan(act).sum()}, Inf={np.isinf(act).sum()}). "
                        "This may indicate numerical instability in the model."
                    )

                activations[i] = act

        return activations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_single(self, prompt: str) -> dict[int, np.ndarray]:
        """Extract activations for a single prompt.

        Args:
            prompt: The text prompt to process.

        Returns:
            Dict mapping layer_index -> activation_vector.
            Each vector has shape (hidden_dim,), e.g. (4096,) for Mistral-7B.

        Raises:
            ValueError: If prompt is empty or tokenization fails.
            RuntimeError: If activations contain NaN/Inf values.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        self._ensure_model_loaded()

        token_ids = self._tokenize(prompt)
        logger.debug(
            "Extracting activations: %d tokens, layers %s",
            len(token_ids),
            sorted(self._target_layers),
        )

        t0 = time.perf_counter()
        activations = self._forward_with_capture(token_ids)
        elapsed = time.perf_counter() - t0

        logger.debug("Extraction completed in %.3fs", elapsed)
        return activations

    def extract_batch(
        self,
        prompts: list[str],
        show_progress: bool = True,
    ) -> dict[int, np.ndarray]:
        """Extract activations for multiple prompts.

        Processes one prompt at a time (MLX handles memory efficiently with
        single-sequence inference on Apple Silicon). Results are stacked into
        arrays with shape (n_prompts, hidden_dim) per layer.

        Args:
            prompts: List of text prompts to process.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            Dict mapping layer_index -> np.ndarray of shape (n_prompts, hidden_dim).

        Raises:
            ValueError: If prompts list is empty or contains empty strings.
        """
        if not prompts:
            raise ValueError("Prompts list must not be empty")

        self._ensure_model_loaded()

        # Collect per-prompt activations
        all_activations: dict[int, list[np.ndarray]] = {
            layer: [] for layer in sorted(self._target_layers)
        }

        iterator = prompts
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(prompts, desc="Extracting activations", unit="prompt")

        t0 = time.perf_counter()
        for idx, prompt in enumerate(iterator):
            try:
                single = self.extract_single(prompt)
                for layer_idx, act in single.items():
                    all_activations[layer_idx].append(act)
            except Exception:
                logger.error("Failed to extract activations for prompt %d: %r", idx, prompt[:100])
                raise

            # Periodically clear MLX cache to manage memory
            if (idx + 1) % 50 == 0:
                import mlx.core as mx

                mx.metal.clear_cache()
                logger.debug("Cleared MLX metal cache after %d prompts", idx + 1)

        elapsed = time.perf_counter() - t0
        n = len(prompts)
        logger.info(
            "Batch extraction complete: %d prompts in %.1fs (%.2fs/prompt)",
            n,
            elapsed,
            elapsed / n,
        )

        # Stack into (n_prompts, hidden_dim) arrays
        return {
            layer: np.stack(acts, axis=0) for layer, acts in all_activations.items()
        }
