# Venator

Jailbreak detection via unsupervised anomaly detection on LLM hidden state activations.

Venator learns what "normal" model computation looks like from benign prompts, then flags jailbreak attempts as anomalous activation patterns — without ever seeing jailbreaks during training.

## Core Idea

Jailbreak attempts produce statistically distinguishable activation patterns in LLM hidden states compared to benign prompts. By training anomaly detectors on benign-only activations extracted from middle transformer layers, we can detect novel jailbreaks without enumerating attack types.

This builds on findings from the ELK literature (Mallen et al. 2024, Burns et al. 2022):
- Middle-layer activations generalize best for behavioral anomaly detection (0.94+ AUROC)
- Ensembles with decorrelated errors provide the most robust detection
- LLMs maintain context-independent knowledge representations even when outputs deviate

## Setup

**Requirements:** Python 3.11+, Apple Silicon Mac (for MLX acceleration)

```bash
# Clone and install
git clone <repo-url> && cd venator
pip install -e ".[dev,dashboard]"
```

## Quickstart

### Option 1: Streamlit Dashboard (recommended)

```bash
streamlit run venator/dashboard/app.py
```

The dashboard guides you through a 7-step pipeline:
1. **Data** — Collect/upload benign and jailbreak prompts
2. **Extract** — Run activation extraction from Mistral-7B
3. **Split** — Create train/val/test splits (jailbreaks test-only)
4. **Train** — Train anomaly detectors on benign activations
5. **Evaluate** — Test metrics, ROC curves, score distributions
6. **Detect** — Live single-prompt jailbreak detection
7. **Ablations** — Compare layers, PCA dims, detectors

### Option 2: CLI Scripts

```bash
# 1. Collect prompts
python scripts/collect_prompts.py --output data/prompts/

# 2. Extract activations
python scripts/extract_activations.py \
    --prompts data/prompts/benign_train.jsonl \
    --output data/activations/benign_train.h5 \
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --layers 12 14 16 18 20

# 3. Create splits
python scripts/create_splits.py --prompts data/prompts/ --output data/prompts/

# 4. Train detector
python scripts/train_detector.py \
    --train-activations data/activations/benign_train.h5 \
    --val-activations data/activations/benign_val.h5 \
    --detector pca_mahalanobis \
    --output models/detector_v1/

# 5. Evaluate
python scripts/evaluate.py \
    --model-dir models/detector_v1/ \
    --test-benign data/activations/benign_test.h5 \
    --test-jailbreak data/activations/jailbreak_test.h5 \
    --output results/eval_v1.json
```

## Architecture

```
venator/
├── venator/
│   ├── config.py              # Pydantic-settings configuration
│   ├── activation/            # MLX-based hidden state extraction + HDF5 storage
│   ├── data/                  # Prompt datasets and train/val/test splitting
│   ├── detection/             # Anomaly detectors (PCA+Mahalanobis, IsolationForest, Autoencoder, Ensemble)
│   ├── pipeline.py            # End-to-end orchestration
│   └── dashboard/             # Streamlit 7-step pipeline UI
├── scripts/                   # CLI entry points
├── data/                      # Prompts and activation HDF5 files
├── models/                    # Saved detector models
└── tests/                     # pytest suite
```

## Detection Ensemble

Three detectors with decorrelated errors, combined via weighted ensemble:

| Detector | Weight | Approach |
|----------|--------|----------|
| PCA + Mahalanobis | 2.0 | Gaussian distance in PCA space (primary, ELK-validated) |
| Isolation Forest | 1.5 | Tree-based, no distributional assumptions |
| Autoencoder | 1.0 | Reconstruction error from learned compression |

## Methodology

Strict unsupervised anomaly detection — no jailbreaks in training:

- **Train**: Benign prompts only (learn "normal" activation distribution)
- **Validation**: Benign prompts only (threshold tuning)
- **Test**: Held-out benign + jailbreak prompts (evaluation)
- **Primary metric**: AUROC (threshold-independent)

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=venator
```

## References

- Mallen et al., "Eliciting Latent Knowledge from Quirky Language Models" (arXiv:2312.01037)
- Burns et al., "Discovering Latent Knowledge Without Supervision" (arXiv:2212.03827)
- Chao et al., "JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs" (arXiv:2404.01318)
