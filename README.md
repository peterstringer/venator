# Venator

**Jailbreak detection via unsupervised anomaly detection on LLM hidden state activations.**

Venator learns what "normal" model computation looks like from benign prompts, then flags jailbreak attempts as anomalous activation patterns — without ever seeing jailbreaks during training.

## Core Hypothesis

> Jailbreak attempts produce statistically distinguishable activation patterns in LLM hidden states compared to benign prompts, detectable via unsupervised anomaly detection.

This builds on findings from the ELK literature (Mallen et al. 2024, Burns et al. 2022):

- Middle-layer activations generalize best for behavioral anomaly detection (0.94+ AUROC)
- Ensembles with decorrelated errors provide the most robust detection
- LLMs maintain context-independent knowledge representations even when outputs deviate

**Why this matters for AI safety:** Jailbreaks bypass safety training in unpredictable ways — we can't enumerate all attacks. Unsupervised detection catches novel attacks by learning "normal" rather than "bad". Probing internal activations captures *how the model computes*, not just *what it outputs*.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           Venator Pipeline               │
                        └─────────────────────────────────────────┘
                                         │
         ┌───────────────┬───────────────┼───────────────┬────────────────┐
         ▼               ▼               ▼               ▼                ▼
   ┌───────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  ┌────────────┐
   │  Collect   │  │  Extract   │  │   Split    │  │   Train   │  │  Evaluate  │
   │  Prompts   │  │ Activations│  │  Data      │  │ Detectors │  │  & Detect  │
   └─────┬─────┘  └─────┬──────┘  └─────┬──────┘  └─────┬─────┘  └─────┬──────┘
         │               │               │               │              │
         ▼               ▼               ▼               ▼              ▼
   ┌───────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  ┌────────────┐
   │  Benign   │  │   MLX +    │  │  Train:    │  │ PCA+Maha  │  │  AUROC,    │
   │  + Jail-  │  │ Mistral-7B │  │  benign    │  │ IForest   │  │  PR curves │
   │  break    │  │ Hidden     │  │  Val:      │  │ AutoEnc   │  │  Prompt    │
   │  JSONL    │  │ States     │  │  benign    │  │ Ensemble  │  │  Explorer  │
   │           │  │ → HDF5     │  │  Test:     │  │           │  │            │
   │           │  │            │  │  benign +  │  │           │  │            │
   │           │  │            │  │  jailbreak │  │           │  │            │
   └───────────┘  └────────────┘  └────────────┘  └───────────┘  └────────────┘
```

```
venator/
├── venator/
│   ├── config.py                 # Pydantic-settings configuration
│   ├── pipeline.py               # End-to-end orchestration
│   ├── activation/
│   │   ├── extractor.py          # MLX-based hidden state extraction
│   │   └── storage.py            # HDF5 activation storage
│   ├── data/
│   │   ├── prompts.py            # Prompt dataset management
│   │   └── splits.py             # Train/val/test splitting
│   ├── detection/
│   │   ├── base.py               # AnomalyDetector ABC
│   │   ├── pca_mahalanobis.py    # PCA + Mahalanobis distance
│   │   ├── isolation_forest.py   # Tree-based anomaly detection
│   │   ├── autoencoder.py        # Reconstruction-based detection
│   │   ├── ensemble.py           # Weighted detector combination
│   │   └── metrics.py            # AUROC, precision, recall, curves
│   └── dashboard/                # Streamlit 7-step pipeline UI
│       ├── app.py                # Main entry point + navigation
│       ├── state.py              # Session state management
│       ├── pages/                # One page per pipeline stage
│       └── components/           # Reusable charts, tables
├── scripts/                      # CLI entry points
├── docs/                         # Methodology and results write-up
├── examples/                     # Quickstart scripts
├── data/                         # Prompts and activation HDF5 files
├── models/                       # Saved detector models
└── tests/                        # 279 pytest tests
```

## Setup

**Requirements:** Python 3.11+, Apple Silicon Mac (for MLX acceleration)

```bash
# Clone and install
git clone <repo-url> && cd venator
pip install -e ".[dev,dashboard]"
```

**Dependencies:**

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| LLM Inference | MLX + mlx-lm | Optimized for Apple Silicon, native Metal acceleration |
| Model | Mistral-7B-Instruct-v0.3 (4-bit) | Fits M4 Pro memory, instruction-tuned |
| Numerical | NumPy, SciPy, scikit-learn | Standard scientific Python stack |
| Deep Learning | PyTorch (CPU) | Autoencoder detector only |
| Storage | HDF5 (h5py) | Efficient activation matrix storage |
| Dashboard | Streamlit + Plotly | Interactive pipeline UI |
| Config | Pydantic Settings | Type-safe, environment variable overrides |

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

### Option 2: Python API

```python
from venator.pipeline import VenatorPipeline
from venator.data.splits import SplitManager

# Full pipeline
pipeline = VenatorPipeline.from_config()
store = pipeline.extract_and_store(prompts, "data/activations/all.h5")
splits = SplitManager().create_splits(store)
pipeline.train(store, splits)
metrics = pipeline.evaluate(store, splits)

# Single-prompt detection
result = pipeline.detect("What causes the seasons to change?")
print(f"Score: {result['ensemble_score']:.4f}, Anomaly: {result['is_anomaly']}")
```

See [examples/quickstart.py](examples/quickstart.py) for a complete minimal example.

### Option 3: CLI Scripts

```bash
# 1. Collect prompts
python scripts/collect_prompts.py --output data/prompts/

# 2. Extract activations
python scripts/extract_activations.py \
    --prompts data/prompts/benign.jsonl \
    --output data/activations/store.h5 \
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --layers 12 14 16 18 20

# 3. Create splits
python scripts/create_splits.py --store data/activations/store.h5 --output data/

# 4. Train detector ensemble
python scripts/train_detector.py \
    --store data/activations/store.h5 \
    --splits data/splits.json \
    --output models/detector_v1/

# 5. Evaluate
python scripts/evaluate.py \
    --store data/activations/store.h5 \
    --splits data/splits.json \
    --model-dir models/detector_v1/ \
    --output results/eval_v1.json

# 6. Run ablation studies
python scripts/run_ablations.py \
    --store data/activations/store.h5 \
    --splits data/splits.json \
    --output results/ablations/
```

## Detection Ensemble

Three detectors with decorrelated errors, combined via weighted ensemble:

| Detector | Weight | Approach | Error Profile |
|----------|--------|----------|---------------|
| PCA + Mahalanobis | 2.0 | Gaussian distance in PCA space | Assumes normality; misses non-Gaussian outliers |
| Isolation Forest | 1.5 | Tree-based partitioning | No distributional assumption; different failure modes |
| Autoencoder | 1.0 | Reconstruction error | Learned nonlinear manifold; complements linear PCA |

**Why an ensemble?** From the ELK paper: *"An ELK method can be useful even when it is no more accurate than other sources, as long as its errors are decorrelated."* Each detector captures different aspects of "normality" — their disagreements make the ensemble more robust than any individual detector.

**Score normalization:** Raw scores from different detectors are on different scales (Mahalanobis distances vs. isolation depths vs. reconstruction errors). Each detector's scores are normalized to [0, 1] via percentile rank against training data before weighted combination.

## Methodology

Strict unsupervised anomaly detection — see [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full write-up.

**Key principles:**

- **Train**: Benign prompts only — learn the "normal" activation distribution
- **Validation**: Benign prompts only — set threshold at the 95th percentile
- **Test**: Held-out benign + jailbreak prompts — evaluate detection
- **Primary metric**: AUROC (threshold-independent, standard for anomaly detection)

**Critical constraints:**
1. No anomalies in training data
2. No threshold selection on test data
3. Healthy sample-to-feature ratio (PCA to 50 dims from 4096)
4. Score normalization before ensemble combination
5. Report AUROC as primary metric

## Results

See [docs/RESULTS.md](docs/RESULTS.md) for detailed results, ablation studies, and failure analysis.

**Summary** (placeholder — run the pipeline to generate actual numbers):

| Metric | Value |
|--------|-------|
| AUROC | _TBD_ |
| AUPRC | _TBD_ |
| Precision @95th pctile | _TBD_ |
| Recall @95th pctile | _TBD_ |
| FPR @95th pctile | _TBD_ |

## Configuration

All settings can be overridden via environment variables (prefix: `VENATOR_`):

| Setting | Default | Env Var |
|---------|---------|---------|
| Model | Mistral-7B-Instruct-v0.3 (4-bit) | `VENATOR_MODEL_ID` |
| Extraction layers | [12, 14, 16, 18, 20] | `VENATOR_EXTRACTION_LAYERS` |
| PCA dimensions | 50 | `VENATOR_PCA_DIMS` |
| Threshold percentile | 95.0 | `VENATOR_ANOMALY_THRESHOLD_PERCENTILE` |
| Random seed | 42 | `VENATOR_RANDOM_SEED` |

## Testing

```bash
# Run all tests (279 tests)
pytest tests/ -v

# With coverage
pytest tests/ --cov=venator

# Specific module
pytest tests/test_detectors.py -v
```

Test suite covers:
- Unit tests for each detector on synthetic Gaussian data
- Integration tests for the full pipeline with mock activations
- Methodology tests verifying no data leakage
- Roundtrip tests for save/load of all detectors

## References

- Mallen et al., ["Eliciting Latent Knowledge from Quirky Language Models"](https://arxiv.org/abs/2312.01037) (2024) — ELK paper; middle-layer probing, decorrelated ensembles
- Burns et al., ["Discovering Latent Knowledge Without Supervision"](https://arxiv.org/abs/2212.03827) (2022) — CCS; unsupervised truth discovery from activations
- Chao et al., ["JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs"](https://arxiv.org/abs/2404.01318) (2024) — Jailbreak dataset and evaluation framework

## License

MIT
